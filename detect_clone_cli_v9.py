#!/usr/bin/env python3
"""
Hybrid semantic + structural (AST / lexical) clone detection with FAISS + fingerprint gating (v8.2).

What’s new vs v8.0:
- Language-agnostic token streams via Tree-sitter AST serialization (works for Java/JS/C/C++).
- Java-specific tokenizer + generic C-style fallback if AST parser is unavailable.
- Short-file fingerprint hard gate is applied **only when both files have token streams**.
- Strip C-style comments for Java/JS/C/C++ before embedding (Type-1 robustness).
- Expose --short-token-gate (default=5).

Background / references (for your thesis):
  - Winnowing local fingerprinting (MOSS) — Schleimer et al., SIGMOD’03 (guarantees ≥1 k-gram in substrings of length ≥ w+k−1). 
  - Dolos (language-agnostic plagiarism detection): Tree-sitter → AST tokens → k-grams → winnowing; report similarity/total-overlap/longest fragment. 
  - Sentence-Transformers pooling; BGE-Code via trust_remote_code=True (last-token pooling).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import re
import sys
import tokenize
import keyword
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------- Optional deps (AST via tree-sitter) ----------------
try:
    from tree_sitter import Parser  # modern API
    _TS_AVAILABLE = True
except Exception:
    Parser = None  # type: ignore
    _TS_AVAILABLE = False

_ts_get_language = None
_ts_get_parser = None
try:
    from tree_sitter_language_pack import (  # type: ignore
        get_language as _ts_get_language,
        get_parser as _ts_get_parser,
    )
    _TS_LANGPACK = True
except Exception:
    _TS_LANGPACK = False
    try:
        from tree_sitter_languages import (  # type: ignore
            get_language as _ts_get_language_legacy,
            get_parser as _ts_get_parser_legacy,
        )
        _ts_get_language = _ts_get_language_legacy
        _ts_get_parser = _ts_get_parser_legacy if "_ts_get_parser_legacy" in globals() else None
        _TS_LANGPACK = True
    except Exception:
        _TS_LANGPACK = False

# --------------------------- Core deps -----------------------------------------
try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None  # type: ignore

# --------------------------- I/O & utils ---------------------------------------
def load_files(directory: str, extensions: Optional[List[str]]) -> List[str]:
    collected: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            if fname.startswith('.'):
                continue
            if extensions and not any(fname.endswith(ext) for ext in extensions):
                continue
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                collected.append(path)
    return sorted(collected)

def read_text(path: str) -> str:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
        return ""

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

# --------------------------- Embedding pre-processing --------------------------
def strip_py_comments_and_docstrings(src: str) -> str:
    """
    Remove comments and top-level/inline docstrings from Python code while preserving tokens.
    """
    if not src:
        return ""
    out_tokens: List[str] = []
    try:
        reader = io.StringIO(src).readline
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(reader):
            ttype, tstr, (sl, sc), (el, ec), _ = tok
            if ttype in (tokenize.COMMENT, tokenize.NL):
                continue
            # drop likely docstrings (a STRING at indented position)
            if ttype == tokenize.STRING and prev_toktype == tokenize.INDENT:
                prev_toktype = ttype
                last_lineno, last_col = el, ec
                continue
            if sl > last_lineno:
                last_col = 0
            if sc > last_col:
                out_tokens.append(" ")
            out_tokens.append(tstr)
            prev_toktype = ttype
            last_lineno, last_col = el, ec
    except Exception:
        return src
    return "".join(out_tokens)

def strip_c_style_comments(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)   # /* ... */
    s = re.sub(r"//.*?$", "", s, flags=re.M)      # // ...
    return s

def normalize_for_embedding(path: str, text: str, strip_comments: bool = True) -> str:
    if not strip_comments or not text:
        return text or ""
    p = path.lower()
    if p.endswith(".py"):
        return strip_py_comments_and_docstrings(text)
    if p.endswith((".java", ".c", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".js", ".jsx", ".ts", ".tsx")):
        return strip_c_style_comments(text)
    return text

# --------------------------- Embeddings ----------------------------------------
def _chunk_text(s: str, window_chars: int = 3000, stride_chars: int = 1500) -> List[str]:
    if not s:
        return []
    if len(s) <= window_chars:
        return [s]
    chunks = []
    start = 0
    while start < len(s):
        end = min(len(s), start + window_chars)
        chunks.append(s[start:end])
        if end == len(s):
            break
        start += stride_chars
    return chunks

def embed_files(
    file_paths: List[str],
    model: SentenceTransformer,
    *,
    batch_size: int = 16,
    window_chars: int = 3000,
    stride_chars: int = 1500,
    strip_comments: bool = True,
) -> torch.Tensor:
    """
    Encode each file as a length-weighted mean of chunk embeddings; return L2-normalized rows.
    Uses last-token pooling for BGE-Code via trust_remote_code=True (configured at model load).
    """
    dim = model.get_sentence_embedding_dimension()
    per_file: List[torch.Tensor] = []

    for p in file_paths:
        text_raw = read_text(p)
        text = normalize_for_embedding(p, text_raw, strip_comments=strip_comments)
        chunks = _chunk_text(text, window_chars, stride_chars)
        if not chunks:
            per_file.append(torch.zeros(dim))
            continue
        emb = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        # Length-weighted pooling
        weights = torch.tensor([len(c) for c in chunks], device=emb.device, dtype=emb.dtype)
        weights = (weights / max(float(weights.sum().item()), 1e-8)).unsqueeze(1)
        file_vec = (emb * weights).sum(dim=0)
        per_file.append(file_vec)

    all_emb = torch.stack(per_file, dim=0)
    all_emb = torch.nn.functional.normalize(all_emb, p=2, dim=1)  # cosine-compatible
    return all_emb

# --------------------------- Structural features (AST) --------------------------
_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby", ".go": "go", ".php": "php", ".rs": "rust",
    ".kt": "kotlin", ".swift": "swift",
}

def guess_ts_language_from_ext(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path.lower())
    return _EXT_TO_LANG.get(ext)

def build_ts_parser(lang_name: str) -> Optional[Parser]:
    if not (_TS_AVAILABLE and _TS_LANGPACK):
        return None
    if _ts_get_parser is not None:
        try:
            return _ts_get_parser(lang_name)  # ready-to-use Parser
        except Exception as e:
            print(f"[AST] get_parser('{lang_name}') failed: {e}", file=sys.stderr)
    try:
        lang = _ts_get_language(lang_name) if _ts_get_language else None
        if lang is None:
            print(f"[AST] no language for '{lang_name}'", file=sys.stderr); return None
        try:
            return Parser(lang)
        except TypeError:
            p = Parser()
            p.set_language(lang)  # type: ignore[attr-defined]
            return p
    except Exception as e:
        print(f"[AST] parser init failed for '{lang_name}': {e}", file=sys.stderr)
        return None

# stable hashing for structural buckets
def _stable_bucket(key: str, dim: int) -> int:
    h = hashlib.sha1(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % dim

def _depth_bucket(depth: int) -> str:
    if depth <= 2: return "d:0-2"
    if depth <= 5: return "d:3-5"
    if depth <= 9: return "d:6-9"
    return "d:10+"

def structural_vector_from_ast(code: str, parser: Optional[Parser], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not parser or not code:
        return vec
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node  # type: ignore
    except Exception:
        return vec
    stack = [(root, 0, None)]
    while stack:
        node, depth, parent_t = stack.pop()
        try:
            t = node.type  # type: ignore[attr-defined]
        except Exception:
            continue
        vec[_stable_bucket(f"n:{t}", dim)] += 1.0
        if parent_t is not None:
            vec[_stable_bucket(f"e:{parent_t}>{t}", dim)] += 1.0
        vec[_stable_bucket(_depth_bucket(depth), dim)] += 1.0
        try:
            children = node.children
        except Exception:
            children = []
        for ch in children:
            stack.append((ch, depth + 1, t))
    return vec

# --------------------------- Lexical (tokens) -----------------------------------
def py_token_stream(src: str) -> List[str]:
    toks: List[str] = []
    if not src:
        return toks
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            tt = tok.type
            val = tok.string
            if tt in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            if tt == tokenize.STRING:
                continue
            if tt == tokenize.NUMBER:
                toks.append("NUM"); continue
            if tt == tokenize.NAME:
                toks.append(val if val in keyword.kwlist else "ID"); continue
            if val.strip():
                toks.append(val)
    except Exception:
        pass
    return toks

JAVA_KEYWORDS = {
    "abstract","assert","boolean","break","byte","case","catch","char","class","const",
    "continue","default","do","double","else","enum","extends","final","finally","float",
    "for","goto","if","implements","import","instanceof","int","interface","long","native",
    "new","package","private","protected","public","return","short","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient","try","void",
    "volatile","while","var","record","sealed","permits","non-sealed"
}

def java_token_stream(src: str) -> List[str]:
    """Lexer-level tokenization for Java; normalize IDs/NUM/STR; drop comments."""
    if not src: return []
    s = strip_c_style_comments(src)
    s = re.sub(r'"([^"\\]|\\.)*"', 'STR', s)              # strings
    s = re.sub(r"'(\\.|.)'", 'CHR', s)                    # char literal
    s = re.sub(r"\b\d[0-9_]*([.][0-9_]+)?([eE][+-]?\d+)?[fFdDlL]?\b", "NUM", s)
    pattern = r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||<<|>>|>>>|::|[{}()\[\];,.\+\-\*/%<>=!&|^~?:]"
    toks: List[str] = []
    for m in re.finditer(pattern, s):
        t = m.group(0)
        if t[0].isalpha() or t[0] == "_":
            toks.append(t if t in JAVA_KEYWORDS else "ID")
        else:
            toks.append(t)
    return toks

def c_like_fallback_token_stream(src: str) -> List[str]:
    """Generic fallback for C/C++/JS-like languages when AST parser isn't available."""
    if not src: return []
    s = strip_c_style_comments(src)
    s = re.sub(r'"([^"\\]|\\.)*"', 'STR', s)
    s = re.sub(r"'(\\.|.)'", 'CHR', s)
    s = re.sub(r"\b\d[0-9_]*([.][0-9_]+)?([eE][+-]?\d+)?[fFdDlL]?\b", "NUM", s)
    pattern = r"[A-Za-z_]\w*|==|!=|<=|>=|&&|\|\||<<|>>|>>>|::|[{}()\[\];,.\+\-\*/%<>=!&|^~?:]"
    toks: List[str] = []
    for m in re.finditer(pattern, s):
        t = m.group(0)
        if t[0].isalpha() or t[0] == "_":
            toks.append("ID")
        else:
            toks.append(t)
    return toks

def ts_token_stream(code: str, parser: Optional[Parser], lang_name: str = "lang") -> List[str]:
    """
    Language-agnostic AST serialization via Tree-sitter node types.
    Map identifiers/literals to ID/NUM/STR, drop comments.
    """
    if not parser or not code: return []
    try:
        tree = parser.parse(code.encode("utf-8"))
        root = tree.root_node  # type: ignore
    except Exception:
        return []
    out: List[str] = []
    stack = [root]
    while stack:
        n = stack.pop()
        t = getattr(n, "type", "")
        # Skip comments
        if "comment" in t:
            continue
        # Coarsen common lexical categories
        lt = t.lower()
        if "string" in lt: out.append("STR"); continue
        if "char" in lt and "literal" in lt: out.append("CHR"); continue
        if any(x in lt for x in ("number", "integer", "float", "decimal")): out.append("NUM"); continue
        if t in ("identifier", "scoped_identifier", "type_identifier"): out.append("ID"); continue
        out.append(f"{lang_name}:{t}")
        try:
            for ch in n.children:
                stack.append(ch)
        except Exception:
            pass
    return out

def token_ngram_vector(tokens: List[str], n: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens or len(tokens) < n:
        return vec
    for i in range(len(tokens) - n + 1):
        key = "⊕".join(tokens[i:i+n])
        idx = _stable_bucket(f"t:{key}", dim)
        vec[idx] += 1.0
    return vec

# --------------------------- Fingerprinting (k-grams + winnowing) ---------------
def kgram_hashes(tokens: List[str], k: int) -> List[int]:
    if not tokens or len(tokens) < k:
        return []
    hashes: List[int] = []
    for i in range(len(tokens) - k + 1):
        kgram = "§".join(tokens[i:i+k])
        h = hashlib.sha1(kgram.encode("utf-8")).digest()
        hashes.append(int.from_bytes(h[:8], "little"))
    return hashes

def winnow(hashes: List[int], w: int) -> List[Tuple[int,int]]:
    if w <= 1 or len(hashes) <= w:
        return [(h, i) for i, h in enumerate(hashes)]
    fps: List[Tuple[int,int]] = []
    last_pos = -1
    for start in range(0, len(hashes) - w + 1):
        window = hashes[start:start+w]
        m = min(window)
        pos = start + window.index(m)
        if pos != last_pos:
            fps.append((m, pos))
            last_pos = pos
    return fps

def fp_sequence(tokens: List[str], k: int, w: int) -> Tuple[List[int], Counter]:
    h = kgram_hashes(tokens, k)
    fp = winnow(h, w) if w > 0 else [(x, i) for i, x in enumerate(h)]
    seq = [x for (x, _pos) in fp]
    return seq, Counter(seq)

def jaccard_from_counters(a: Counter, b: Counter) -> float:
    if not a and not b:
        return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / max(union, 1)

def total_overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())

def longest_common_run(seqA: List[int], seqB: List[int]) -> int:
    if not seqA or not seqB:
        return 0
    N, M = len(seqA), len(seqB)
    dp = [0]*(M+1)
    best = 0
    for i in range(1, N+1):
        prev = 0
        for j in range(1, M+1):
            tmp = dp[j]
            if seqA[i-1] == seqB[j-1]:
                dp[j] = prev + 1
                if dp[j] > best: best = dp[j]
            else:
                dp[j] = 0
            prev = tmp
    return best

# --------------------------- Structural pipeline (AST + lex vec + FP) ----------
def apply_tfidf(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0: return mat
    N = mat.shape[0]
    df = np.count_nonzero(mat, axis=0)
    idf = np.log((N + 1.0) / (df + 1.0)) + 1.0
    return l2_normalize_rows(mat * idf.astype(mat.dtype, copy=False))

def stop_topk(mat: np.ndarray, k: int) -> np.ndarray:
    if mat.size == 0 or k <= 0: return mat
    totals = mat.sum(axis=0)
    if k >= totals.size: return np.zeros_like(mat)
    topk_idx = np.argpartition(totals, -(k))[-k:]
    mat[:, topk_idx] = 0.0
    return mat

def compute_structural_features(
    file_paths: List[str],
    *,
    ast_dim: int,
    use_ast: bool,
    ast_tfidf: bool,
    ast_stop_topk: int,
    lex_dim: int,
    lex_n: int,
    use_lex: bool,
    lex_mode: str,   # 'token' or 'char'
    fp_k: int,
    fp_w: int,
    use_fp: bool,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[int]], List[Counter], List[int]]:
    N = len(file_paths)
    ast_mat = np.zeros((N, 0), dtype=np.float32)
    lex_mat = np.zeros((N, 0), dtype=np.float32)
    tokens_by_file: List[List[str]] = []
    fp_seq_by_file: List[List[int]] = []
    fp_ctr_by_file: List[Counter] = []
    tok_counts: List[int] = []

    # AST parsers
    parser_cache: Dict[str, Optional[Parser]] = {}
    if use_ast and _TS_AVAILABLE and _TS_LANGPACK:
        langs_needed = set(filter(None, (guess_ts_language_from_ext(p) for p in file_paths)))
        for ln in langs_needed:
            if ln:
                parser_cache[ln] = build_ts_parser(ln)

    ast_rows, lex_rows = [], []
    for path in file_paths:
        text = read_text(path)
        p_lower = path.lower()

        # AST
        if use_ast and parser_cache:
            lang_name = guess_ts_language_from_ext(path)
            parser = parser_cache.get(lang_name) if (lang_name in parser_cache) else None
            ast_vec = structural_vector_from_ast(text, parser, ast_dim) if parser else np.zeros(ast_dim, np.float32)
            ast_rows.append(ast_vec)

        # --- Token streams (for lex + fingerprints) ---
        toks: List[str] = []
        lang_name = guess_ts_language_from_ext(path)
        parser = parser_cache.get(lang_name) if (parser_cache and lang_name in parser_cache) else None

        if p_lower.endswith(".py"):
            toks = py_token_stream(text)
        elif p_lower.endswith(".java"):
            toks = java_token_stream(text)
        elif p_lower.endswith((".js", ".jsx", ".ts", ".tsx", ".c", ".h", ".hpp", ".cc", ".cxx", ".cpp")):
            if parser is not None:
                toks = ts_token_stream(text, parser, lang_name or "lang")
            else:
                toks = c_like_fallback_token_stream(text)
        else:
            if parser is not None:
                toks = ts_token_stream(text, parser, lang_name or "lang")

        tokens_by_file.append(toks)
        tok_counts.append(len(toks))

        # Lex vec
        if use_lex:
            if lex_mode in ("token", "py-token") and toks:
                lex_rows.append(token_ngram_vector(toks, lex_n, lex_dim))
            else:
                s = text if text else ""
                if not s or len(s) < lex_n:
                    lex_rows.append(np.zeros(lex_dim, np.float32))
                else:
                    vec = np.zeros(lex_dim, np.float32)
                    for i in range(len(s)-lex_n+1):
                        key = s[i:i+lex_n]
                        vec[_stable_bucket(f"g:{key}", lex_dim)] += 1.0
                    lex_rows.append(vec)

        # Fingerprints
        if use_fp and toks:
            seq, ctr = fp_sequence(toks, fp_k, fp_w)
            fp_seq_by_file.append(seq)
            fp_ctr_by_file.append(ctr)
        else:
            fp_seq_by_file.append([])
            fp_ctr_by_file.append(Counter())

    # stack AST
    if use_ast and ast_rows:
        ast_mat = np.vstack(ast_rows)
        ast_mat = stop_topk(ast_mat, ast_stop_topk)
        ast_mat = apply_tfidf(ast_mat) if ast_tfidf else l2_normalize_rows(ast_mat)

    # stack lex
    if use_lex and lex_rows:
        lex_mat = np.vstack(lex_rows)
        lex_mat = apply_tfidf(lex_mat)

    return ast_mat, lex_mat, tokens_by_file, fp_seq_by_file, fp_ctr_by_file, tok_counts

# --------------------------- FAISS helpers -------------------------------------
def build_faiss_index_np(mat: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise RuntimeError("faiss-cpu is required. pip install faiss-cpu")
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine for L2-normalized vectors
    index.add(mat.astype('float32', copy=False))
    return index

# --------------------------- Search: top-M neighbors ---------------------------
def prefilter_neighbors(
    base: np.ndarray,
    *,
    topM: int = 50,
) -> Tuple[List[List[int]], List[List[float]]]:
    """
    Returns neighbors indices and distances per row using top-M FAISS search.
    Excludes self-matches in the returned lists.
    """
    index = build_faiss_index_np(base)
    D, I = index.search(base, min(topM, base.shape[0]))
    neigh_idx: List[List[int]] = []
    neigh_sim: List[List[float]] = []
    for i in range(base.shape[0]):
        idx = []
        sim = []
        for j, d in zip(I[i].tolist(), D[i].tolist()):
            if j == -1 or j == i:
                continue
            idx.append(j)
            sim.append(float(d))
        neigh_idx.append(idx)
        neigh_sim.append(sim)
    return neigh_idx, neigh_sim

# --------------------------- Fusion + scoring ----------------------------------
def compute_pairs_late(
    file_paths: List[str],
    embed_np_final: np.ndarray,
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    w_embed: float,
    w_ast: float,
    w_lex: float,
    final_threshold: float,
    min_ast_sim: float,
    min_lex_sim: float,
    # fingerprint artifacts for gating
    fp_seq: List[List[int]],
    fp_ctr: List[Counter],
    min_fp_sim: float,
    min_fp_total: int,
    min_fp_longest: int,
    # neighbor graph from prefilter
    neigh_idx: List[List[int]],
    mutual_nearest: bool,
    structure_positive_only: bool = True,
    embed_superpass: float = 0.985,
    short_token_gate: int = 5,
    token_counts: Optional[List[int]] = None,
    semantic_light_gate: bool = False,
) -> List[Tuple[str, str, float]]:
    ast_active = ast_np.size > 0 and np.any(ast_np) and w_ast > 0.0
    lex_active = lex_np.size > 0 and np.any(lex_np) and w_lex > 0.0

    n = embed_np_final.shape[0]
    # build reverse neighbor sets for mutual-nearest filtering
    neigh_sets = [set(row) for row in neigh_idx]
    pairs: List[Tuple[str, str, float]] = []

    for i in range(n):
        a_i = ast_np[i] if ast_active else None
        l_i = lex_np[i] if lex_active else None

        for j in neigh_idx[i]:
            if j == i or i > j:
                continue
            # mutual nearest pruning (optional)
            if mutual_nearest and i not in neigh_sets[j]:
                continue

            sim_e = float(np.dot(embed_np_final[i], embed_np_final[j]))  # cosine
            num, den = w_embed * sim_e, w_embed

            sim_a = float(np.dot(a_i, ast_np[j])) if ast_active else 0.0
            sim_l = float(np.dot(l_i, lex_np[j])) if lex_active else 0.0

            if structure_positive_only:
                sim_a = max(sim_a, 0.0); sim_l = max(sim_l, 0.0)

            # --- fingerprint gate (and short-file hard gate) ---
            fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j]) if fp_ctr else 0.0
            fpT = total_overlap(fp_ctr[i], fp_ctr[j]) if fp_ctr else 0
            fpL = longest_common_run(fp_seq[i], fp_seq[j]) if fp_seq else 0

            # short-file hard gate — apply ONLY if both token streams exist
            has_tokens = token_counts is not None and (token_counts[i] > 0 and token_counts[j] > 0)
            is_short = has_tokens and (token_counts[i] < short_token_gate or token_counts[j] < short_token_gate)
            if is_short:
                if not (fpS >= min_fp_sim and fpT >= min_fp_total and fpL >= min_fp_longest):
                    if sim_e < embed_superpass:  # allow extreme semantic superpass
                        continue

            # structural or semantic+light fingerprint gate
            gate_fail = True
            if ast_active and sim_a >= min_ast_sim: gate_fail = False
            if lex_active and sim_l >= min_lex_sim: gate_fail = False
            if (not ast_active and not lex_active):
                if semantic_light_gate:
                    # require a *tiny* fingerprint signal unless embed is extreme
                    if (fpS >= min_fp_sim and fpT >= max(1, min_fp_total // 2)) or sim_e >= embed_superpass:
                        gate_fail = False
                else:
                    gate_fail = False  # pure semantic mode: no structure requirement
            if gate_fail:
                continue

            if ast_active: num += w_ast * sim_a; den += w_ast
            if lex_active: num += w_lex * sim_l; den += w_lex

            s = num / max(den, 1e-12)

            if s >= final_threshold:
                pairs.append((file_paths[i], file_paths[j], s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

# =========================== AUTO PROFILE (1-knob UX) ==========================
def _auto_profile(args, N: int):
    """
    Translate --mode and --min-tokens to internal params with safe defaults.
    """
    mt = max(2, int(args.min_tokens))

    # --- fingerprint parameters from min-tokens (MOSS-like) ---
    args.fp_k = max(4, min(7, mt))
    args.fp_w = max(4, args.fp_k - 1)
    args.min_fp_total   = max(2, args.fp_k - 1)
    args.min_fp_longest = max(1, (args.fp_k - 2) // 2)
    args.min_fp_sim = 0.05 + 0.01 * max(0, args.fp_k - 5)   # 0.05..0.07 typical

    # --- channels & weights ---
    args.w_embed = 1.0
    args.w_ast   = 0.35
    args.w_lex   = 0.10
    args.no_ast  = False
    args.no_lex  = False
    args.no_fp   = False
    args.lex_mode = "token"
    args.lex_n    = 3
    args.ast_dim  = 2048
    args.lex_dim  = 4096
    args.ast_tfidf = True

    # --- thresholds ---
    args.prefilter_topM = max(5, int(args.prefilter_topM))  # sane minimum

    # final threshold by mode
    if args.mode == "semantic":
        args.threshold = 0.40 if getattr(args, "threshold", None) is None else args.threshold
        args.no_ast = True; args.no_lex = True; args.w_ast = 0.0; args.w_lex = 0.0
        args.min_ast_sim = 0.0; args.min_lex_sim = 0.0
        args.semantic_light_gate = False
    elif args.mode == "semantic-plus":
        args.threshold = 0.38 if getattr(args, "threshold", None) is None else args.threshold
        args.no_ast = True; args.no_lex = True; args.w_ast = 0.0; args.w_lex = 0.0
        args.min_ast_sim = 0.0; args.min_lex_sim = 0.0
        args.semantic_light_gate = True   # <-- key difference
    elif args.mode == "structural":
        args.threshold = 0.30 if getattr(args, "threshold", None) is None else args.threshold
        args.w_embed = 0.0
        args.no_ast = False; args.no_lex = False
        args.min_ast_sim = 0.03; args.min_lex_sim = 0.03
        args.semantic_light_gate = False
    else:  # hybrid (recommended)
        args.threshold = 0.32 if getattr(args, "threshold", None) is None else args.threshold
        args.min_ast_sim = 0.00; args.min_lex_sim = 0.00
        args.semantic_light_gate = False

    # Tiny cohorts: slightly ease thresholds.
    if N < 8:
        args.threshold = max(0.28, args.threshold - 0.02)

    # embed superpass is conservative
    args.embed_superpass = 0.985

# --------------------------- CLI / main ----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid semantic+structural clone detection (FAISS) + fingerprint gate (v8.2)")
    # Minimal knobs users need:
    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--extensions", nargs="*", required=True, help="Extensions to include, e.g., .py .java .js .cpp")
    ap.add_argument("--mode", choices=["hybrid", "semantic", "semantic-plus", "structural"], default="hybrid",
                    help="Scoring mode (hybrid recommended; semantic-plus keeps a light FP gate)")
    ap.add_argument("--min-tokens", type=int, default=5, help="Primary knob (affects fingerprinting)")

    # Expert / optional:
    ap.add_argument("--model", default="BAAI/bge-code-v1",
                    help="SentenceTransformer model (BGE-Code recommended)")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    ap.add_argument("--prefilter-topM", type=int, default=50, help="Top-M neighbors to prefetch per file")
    ap.add_argument("--mutual-nearest", action="store_true", help="Keep pairs only if mutual neighbors")
    ap.add_argument("--topk", type=int, default=0, help="Print only top-K pairs (0=all)")
    ap.add_argument("--debug-components", action="store_true",
                    help="Print per-pair components (e, ast, lex, fp_sim/total/longest, final)")
    ap.add_argument("--no-strip-comments", action="store_true",
                    help="Do NOT strip comments/docstrings before embedding (default is to strip)")
    ap.add_argument("--threshold", type=float, default=None, help="Override final similarity threshold")
    ap.add_argument("--short-token-gate", type=int, default=5,
                    help="Short-token hard gate (tokens per file). Only applied when both files have tokens.")

    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        ap.error(f"{root} is not a valid directory")

    files = load_files(root, args.extensions)
    if not files:
        print(f"No files found in {root} matching extensions {args.extensions}", file=sys.stderr)
        return
    N = len(files)
    print(f"Found {N} files. Loading model '{args.model}' …")

    # Dependencies
    if SentenceTransformer is None or torch is None:
        print("Error: requires 'sentence_transformers' and 'torch'.", file=sys.stderr); sys.exit(1)
    if faiss is None:
        print("Error: requires 'faiss-cpu'.", file=sys.stderr); sys.exit(1)

    _auto_profile(args, N)
    print(f"[AUTO] mode={args.mode}, min_tokens={args.min_tokens} → "
          f"fp_k={args.fp_k}, fp_w={args.fp_w}, min_fp_sim={args.min_fp_sim:.3f}, "
          f"min_fp_total={args.min_fp_total}, min_fp_longest={args.min_fp_longest}, "
          f"topM={args.prefilter_topM}, threshold={args.threshold}, superpass={args.embed_superpass}, "
          f"short_token_gate={args.short_token_gate}, mutual_nearest={args.mutual_nearest}")

    # Embeddings
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # trust_remote_code=True is critical for BGE-Code pooling
        model = SentenceTransformer(
            args.model,
            device=device,
            trust_remote_code=True
        )
    except Exception as exc:
        print(f"Error loading model {args.model}: {exc}", file=sys.stderr); sys.exit(1)

    emb_t = embed_files(
        files, model, batch_size=args.batch_size,
        strip_comments=(not args.no_strip_comments)
    )
    embed_np_raw = emb_t.detach().cpu().numpy().astype('float32')  # L2-normalized

    # Center ONLY if safe and useful
    use_center = (N >= 20)
    embed_np_centered = None
    if use_center:
        mu = embed_np_raw.mean(axis=0, keepdims=True).astype(embed_np_raw.dtype)
        centered = embed_np_raw - mu
        centered = l2_normalize_rows(centered)
        if np.min(np.linalg.norm(centered, axis=1)) > 1e-6:
            embed_np_centered = centered
        else:
            use_center = False

    # Prefilter always uses RAW (more robust for tiny cohorts)
    prefilter_np = embed_np_raw
    embed_for_final = embed_np_centered if use_center else embed_np_raw

    # Structural & fingerprints
    use_ast = (args.mode in ("hybrid", "structural")) and (not args.no_ast if hasattr(args, "no_ast") else True)
    use_lex = (args.mode in ("hybrid", "structural")) and (not args.no_lex if hasattr(args, "no_lex") else True)
    use_fp  = True

    if use_ast and not (_TS_AVAILABLE and _TS_LANGPACK):
        print("Note: tree-sitter / language pack not available; AST features limited to fallback.\n"
              "      Install: pip install 'tree-sitter>=0.25,<0.26' tree-sitter-language-pack", file=sys.stderr)
        # keep use_ast True for vector shape consistency; structural vector_from_ast will be zeros if no parser

    ast_np, lex_np, tokens_by_file, fp_seq, fp_ctr, tok_counts = compute_structural_features(
        files,
        ast_dim=2048, use_ast=use_ast, ast_tfidf=True, ast_stop_topk=0,
        lex_dim=4096, lex_n=3, use_lex=use_lex, lex_mode="token",
        fp_k=args.fp_k, fp_w=args.fp_w, use_fp=use_fp,
    )

    ast_active = use_ast and (ast_np.size > 0) and np.any(ast_np) and (args.w_ast > 0.0)
    lex_active = use_lex and (lex_np.size > 0) and np.any(lex_np) and (args.w_lex > 0.0)

    # Effective weights
    w_e = float(args.w_embed if args.mode != "structural" else 0.0)
    w_a = float(args.w_ast) if ast_active else 0.0
    w_l = float(args.w_lex) if lex_active else 0.0

    # Zero-out structures if inactive
    if not ast_active:
        ast_np = np.zeros((embed_np_raw.shape[0], 0), dtype=np.float32)
    if not lex_active:
        lex_np = np.zeros((embed_np_raw.shape[0], 0), dtype=np.float32)

    print(f"Channels → embed:{'yes' if w_e>0 else 'no'}  ast:{'yes' if ast_active else 'no'}  lex:{'yes' if lex_active else 'no'}")
    print(f"Weights  → w_e={w_e}  w_ast={w_a}  w_lex={w_l}")
    print(f"Features → embed:{embed_for_final.shape[1]}  ast:{ast_np.shape[1]}  lex:{lex_np.shape[1]}")
    print(f"Prefilter: topM={args.prefilter_topM} | Final threshold={args.threshold} | Short-token gate={args.short_token_gate} tokens")

    # Prefilter neighbors (top-M)
    neigh_idx, _neigh_sim = prefilter_neighbors(prefilter_np, topM=args.prefilter_topM)

    pairs = compute_pairs_late(
        files, embed_for_final, ast_np, lex_np,
        w_embed=w_e, w_ast=w_a, w_lex=w_l,
        final_threshold=args.threshold,
        min_ast_sim=getattr(args, "min_ast_sim", 0.0),
        min_lex_sim=getattr(args, "min_lex_sim", 0.0),
        fp_seq=fp_seq if use_fp else [ [] for _ in files ],
        fp_ctr=fp_ctr if use_fp else [ Counter() for _ in files ],
        min_fp_sim=args.min_fp_sim,
        min_fp_total=args.min_fp_total,
        min_fp_longest=args.min_fp_longest,
        neigh_idx=neigh_idx,
        mutual_nearest=args.mutual_nearest,
        structure_positive_only=True,
        embed_superpass=args.embed_superpass,
        short_token_gate=args.short_token_gate,
        token_counts=tok_counts,
        semantic_light_gate=(args.mode == "semantic-plus"),
    )

    if not pairs:
        print(f"No pairs met the similarity threshold {args.threshold}")
        return

    out_pairs = pairs[:args.topk] if args.topk and args.topk > 0 else pairs
    print(f"\nDetected {len(pairs)} candidate clone pairs (showing {len(out_pairs)}):\n")
    if args.debug_components:
        # recompute components for reporting
        for p1, p2, sim in out_pairs:
            i = files.index(p1); j = files.index(p2)
            sim_e = float(np.dot(embed_for_final[i], embed_for_final[j]))
            sim_a = float(np.dot(ast_np[i], ast_np[j])) if ast_np.shape[1] > 0 else 0.0
            sim_l = float(np.dot(lex_np[i], lex_np[j])) if lex_np.shape[1] > 0 else 0.0
            fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j])
            fpT = total_overlap(fp_ctr[i], fp_ctr[j])
            fpL = longest_common_run(fp_seq[i], fp_seq[j])
            print(f"{sim:.4f}\t{p1}\t{p2}\t(e={sim_e:.4f}, ast={sim_a:.4f}, lex={sim_l:.4f}, fp_sim={fpS:.3f}, fp_tot={fpT}, fp_long={fpL})")
    else:
        for p1, p2, sim in out_pairs:
            print(f"{sim:.4f}\t{p1}\t{p2}")

if __name__ == "__main__":
    main()
