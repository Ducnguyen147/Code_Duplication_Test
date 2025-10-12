#!/usr/bin/env python3
"""
Hybrid semantic + structural (AST / lexical) clone detection with FAISS + fingerprint gating (v6).

- FIX: remove unsupported `truncation=True` in SentenceTransformer.encode()
- Robust embeddings: chunk long files and mean-pool chunk embeddings
- "Easy mode" UX: one knob (--min-tokens) + optional --profile/--mode

Install:
  pip install "tree-sitter>=0.25,<0.26" tree-sitter-language-pack
  pip install sentence-transformers torch faiss-cpu

Background:
  - Winnowing local fingerprinting (MOSS) — Schleimer et al., SIGMOD’03.
  - Dolos pipeline & metrics (similarity, total overlap, longest fragment).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import sys
import tokenize
import keyword
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------- Optional deps (AST) -------------------------------
try:
    from tree_sitter import Parser  # modern API: Parser(language, ...)
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
            get_parser as _ts_get_parser_legacy,  # may exist on newer wheels
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

# --------------------------- Embeddings ----------------------------------------
def _chunk_text(s: str, window_chars: int = 3000, stride_chars: int = 1500) -> List[str]:
    """Simple character-based chunking (robust; tokenizer-free)."""
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
    batch_size: int = 16,
    window_chars: int = 3000,
    stride_chars: int = 1500,
) -> torch.Tensor:
    """
    Encode each file as the mean of chunk embeddings.
    NOTE: We do NOT pass `truncation=True` to encode(); it's not a supported kwarg.
          Long inputs are truncated internally by the tokenizer to model.max_seq_length.  # see docs
    """
    dim = model.get_sentence_embedding_dimension()
    per_file: List[torch.Tensor] = []

    for p in file_paths:
        text = read_text(p)
        chunks = _chunk_text(text, window_chars, stride_chars)
        if not chunks:
            per_file.append(torch.zeros(dim))
            continue

        emb = model.encode(
            chunks,
            batch_size=batch_size,
            convert_to_tensor=True,            # valid kwarg
            show_progress_bar=False,           # valid kwarg
            # no 'truncation' kwarg here!
        )
        # mean-pool across chunks
        file_vec = emb.mean(dim=0)
        per_file.append(file_vec)

    all_emb = torch.stack(per_file, dim=0)
    # L2-normalize rows so that inner product == cosine (works with FAISS IP index)
    all_emb = torch.nn.functional.normalize(all_emb, p=2, dim=1)
    return all_emb

# --------------------------- Structural features (AST) --------------------------
_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".cs": "c_sharp",
    ".rb": "ruby", ".go": "go", ".php": "php", ".rs": "rust", ".kt": "kotlin", ".swift": "swift",
}

def guess_ts_language_from_ext(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
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
            print(f"[AST] no language for '{lang_name}'", file=sys.stderr)
            return None
        try:
            return Parser(lang)  # modern API: Parser(language, ...)
        except TypeError:
            p = Parser()  # legacy path
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
    """Normalized Python tokens:
       - keep keywords as-is; NAMEs -> 'ID'; NUMBER -> 'NUM'
       - skip strings/comments/whitespace/indent/dedent/newlines
       - keep operators/punct as their literal
    """
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
    """Return list of (hash, position) fingerprints using MOSS winnowing (min per window)."""
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
    """Return ordered fingerprint sequence (by pos) and a Counter for set operations."""
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
    """Longest common contiguous substring length over fingerprint sequences."""
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

def mean_center(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0: return mat
    mu = mat.mean(axis=0, keepdims=True).astype(mat.dtype)
    return l2_normalize_rows(mat - mu)

def compute_structural_features(
    file_paths: List[str],
    *,
    ast_dim: int,
    use_ast: bool,
    ast_tfidf: bool,
    ast_stop_topk: int,
    ast_center: bool,
    lex_dim: int,
    lex_n: int,
    use_lex: bool,
    lex_mode: str,   # 'char' or 'py-token'
    fp_k: int,
    fp_w: int,
    use_fp: bool,
) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[List[int]], List[Counter]]:
    N = len(file_paths)
    ast_mat = np.zeros((N, 0), dtype=np.float32)
    lex_mat = np.zeros((N, 0), dtype=np.float32)
    tokens_by_file: List[List[str]] = []
    fp_seq_by_file: List[List[int]] = []
    fp_ctr_by_file: List[Counter] = []

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

        # AST
        if use_ast and parser_cache:
            lang_name = guess_ts_language_from_ext(path)
            parser = parser_cache.get(lang_name) if lang_name in parser_cache else None
            ast_vec = structural_vector_from_ast(text, parser, ast_dim) if parser else np.zeros(ast_dim, np.float32)
            ast_rows.append(ast_vec)

        # tokens for Python (lex vec + fingerprints)
        toks: List[str] = []
        if path.endswith(".py"):
            toks = py_token_stream(text)
        tokens_by_file.append(toks)

        # Lex vec
        if use_lex:
            if lex_mode == "py-token" and toks:
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
        ast_mat = mean_center(ast_mat) if ast_center else ast_mat

    # stack lex
    if use_lex and lex_rows:
        lex_mat = np.vstack(lex_rows)
        lex_mat = apply_tfidf(lex_mat)

    return ast_mat, lex_mat, tokens_by_file, fp_seq_by_file, fp_ctr_by_file

# --------------------------- FAISS helpers -------------------------------------
def build_faiss_index_np(mat: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise RuntimeError("faiss-cpu is required. pip install faiss-cpu")
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; with L2-normalized vectors => cosine similarity
    index.add(mat.astype('float32', copy=False))
    return index

# --------------------------- Fusion --------------------------------------------
def build_hybrid_vectors_concat(
    embed_np: np.ndarray,
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    w_embed: float,
    w_ast: float,
    w_lex: float,
) -> np.ndarray:
    parts, active_w = [], 0.0
    parts.append(np.sqrt(max(w_embed, 0.0)) * embed_np); active_w += max(w_embed, 0.0)
    ast_active = ast_np.size > 0 and np.any(ast_np) and w_ast > 0.0
    if ast_active:
        parts.append(np.sqrt(w_ast) * ast_np); active_w += w_ast
    lex_active = lex_np.size > 0 and np.any(lex_np) and w_lex > 0.0
    if lex_active:
        parts.append(np.sqrt(w_lex) * lex_np); active_w += w_lex
    V = np.concatenate(parts, axis=1)
    return (V / math.sqrt(max(active_w, 1e-12))).astype('float32', copy=False)

def compute_pairs_concat(file_paths: List[str], hybrid_np: np.ndarray, threshold: float):
    index = build_faiss_index_np(hybrid_np)
    lims, dists, labels = index.range_search(hybrid_np, float(threshold))
    n = hybrid_np.shape[0]; pairs = []
    for i in range(n):
        for j_idx in range(lims[i], lims[i+1]):
            j = labels[j_idx]
            if j == i or i > j:
                continue
            pairs.append((file_paths[i], file_paths[j], float(dists[j_idx])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

def compute_pairs_late(
    file_paths: List[str],
    embed_np_final: np.ndarray,
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    w_embed: float,
    w_ast: float,
    w_lex: float,
    prefilter_threshold: float,
    final_threshold: float,
    min_ast_sim: float,
    min_lex_sim: float,
    # fingerprint artifacts for gating
    fp_seq: List[List[int]],
    fp_ctr: List[Counter],
    min_fp_sim: float,
    min_fp_total: int,
    min_fp_longest: int,
    prefilter_np: Optional[np.ndarray] = None,
    structure_positive_only: bool = True,
    embed_superpass: float = 0.95,  # if embedding >= this, allow pass even if fp gate fails
) -> List[Tuple[str, str, float]]:
    base = prefilter_np if prefilter_np is not None else embed_np_final
    emb_index = build_faiss_index_np(base)
    lims, _dists, labels = emb_index.range_search(base, float(prefilter_threshold))

    ast_active = ast_np.size > 0 and np.any(ast_np) and w_ast > 0.0
    lex_active = lex_np.size > 0 and np.any(lex_np) and w_lex > 0.0

    n = embed_np_final.shape[0]
    pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        start, end = lims[i], lims[i+1]
        a_i = ast_np[i] if ast_active else None
        l_i = lex_np[i] if lex_active else None

        for k in range(start, end):
            j = labels[k]
            if j == i or i > j:
                continue

            sim_e = float(np.dot(embed_np_final[i], embed_np_final[j]))  # cosine (vectors are L2-normalized)
            num, den = w_embed * sim_e, w_embed

            sim_a = float(np.dot(a_i, ast_np[j])) if ast_active else 0.0
            sim_l = float(np.dot(l_i, lex_np[j])) if lex_active else 0.0

            if structure_positive_only:
                sim_a = max(sim_a, 0.0)
                sim_l = max(sim_l, 0.0)

            # Structural gate
            gate_fail = True
            if ast_active and sim_a >= min_ast_sim: gate_fail = False
            if lex_active and sim_l >= min_lex_sim: gate_fail = False
            if (not ast_active and not lex_active): gate_fail = False
            if gate_fail and sim_e < embed_superpass:
                continue

            if ast_active: num += w_ast * sim_a; den += w_ast
            if lex_active: num += w_lex * sim_l; den += w_lex

            s = num / max(den, 1e-12)

            # -------- Fingerprint gate (Dolos/MOSS-style) --------
            if fp_ctr:
                fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j])
                fpT = total_overlap(fp_ctr[i], fp_ctr[j])
                fpL = longest_common_run(fp_seq[i], fp_seq[j])

                if (sim_e < embed_superpass) and not (fpS >= min_fp_sim and fpT >= min_fp_total and fpL >= min_fp_longest):
                    continue  # fail fp gate

            if s >= final_threshold:
                pairs.append((file_paths[i], file_paths[j], s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

# =========================== EASY MODE (one knob UX) ===========================
def _apply_easy_mode(args):
    """
    Map a single --min-tokens knob (+ optional --profile/--mode) to all internals.
    Leaves expert flags available but overridden unless the user explicitly sets them.
    """
    mt = max(2, int(args.min_tokens))  # safety

    # --- fingerprint parameters from min-tokens (MOSS-like) ---
    args.fp_k = max(4, min(7, mt))
    args.fp_w = max(4, args.fp_k - 1)
    args.min_fp_total   = max(2, args.fp_k - 1)
    args.min_fp_longest = max(1, (args.fp_k - 2) // 2)
    args.min_fp_sim = 0.05 + 0.01 * max(0, args.fp_k - 5)   # 0.05..0.07 typical

    # --- channels & weights ---
    args.fusion = "late"
    args.w_embed = 1.0
    args.w_ast   = 0.35
    args.w_lex   = 0.10
    args.no_ast  = False
    args.no_lex  = False
    args.no_fp   = False
    args.lex_mode = "py-token"
    args.lex_n    = 3
    args.ast_dim  = 2048
    args.lex_dim  = 4096
    args.ast_tfidf = True
    args.ast_center = False
    args.center = False
    args.allow_negative_structure = False
    args.min_ast_sim = 0.00
    args.min_lex_sim = 0.00
    args.prefilter_raw_below = 50

    # --- scoring thresholds via profiles ---
    prof = getattr(args, "profile", "balanced")
    if prof == "tight":
        args.prefilter_threshold = 0.85
        args.threshold = 0.28
        args.embed_superpass = 0.90
    elif prof == "loose":
        args.prefilter_threshold = 0.75
        args.threshold = 0.80
        args.embed_superpass = 0.80
    else:  # balanced
        args.prefilter_threshold = 0.20
        args.threshold = 0.24
        args.embed_superpass = 0.85

    # --- mode (what contributes to final score) ---
    mode = getattr(args, "mode", "hybrid")
    if mode == "semantic":
        args.no_ast = True
        args.no_lex = True
        args.w_ast  = 0.0
        args.w_lex  = 0.0
    elif mode == "structural":
        args.w_embed = 0.0
        args.no_ast  = False
        args.no_lex  = False
        args.min_ast_sim = max(args.min_ast_sim, 0.03)
        args.min_lex_sim = max(args.min_lex_sim, 0.03)

    # If user forgot extensions, default to Python
    if not getattr(args, "extensions", None):
        args.extensions = [".py"]

# --------------------------- CLI / main ----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid semantic+structural clone detection (FAISS) + fingerprint gate (v6)")

    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--model", default="mchochlov/codebert-base-cd-ft",
                    help="SentenceTransformer model (clone-detection fine-tune recommended)")
    ap.add_argument("--extensions", nargs="*", default=None,
                    help="Extensions to include, e.g., .py .java (default: all files)")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")

    # Fusion & thresholds
    ap.add_argument("--fusion", choices=["concat", "late"], default="late",
                    help="Hybrid fusion: 'concat' (early) or 'late' (prefilter + re-score)")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Final similarity threshold [0,1] to report clones")
    ap.add_argument("--prefilter-threshold", type=float, default=0.75,
                    help="(late) Embedding cosine threshold for FAISS prefilter")

    # Weights
    ap.add_argument("--w-embed", type=float, default=1.0, help="Weight for embedding channel")
    ap.add_argument("--w-ast", type=float, default=0.6, help="Weight for AST structural channel")
    ap.add_argument("--w-lex", type=float, default=0.1, help="Weight for lexical channel")

    # AST options
    ap.add_argument("--no-ast", action="store_true", help="Disable AST features")
    ap.add_argument("--ast-dim", type=int, default=2048, help="Dimensionality for hashed AST features")
    ap.add_argument("--ast-tfidf", action="store_true", help="Apply TF-IDF to AST counts (recommended)")
    ap.add_argument("--ast-no-tfidf", dest="ast_tfidf", action="store_false", help="Disable AST TF-IDF")
    ap.set_defaults(ast_tfidf=True)
    ap.add_argument("--ast-stop-topk", type=int, default=0,
                    help="Drop the top-K most frequent AST dims before TF-IDF (0 for tiny sets)")
    ap.add_argument("--ast-center", action="store_true", help="Mean-center AST vectors after TF-IDF")
    ap.add_argument("--ast-no-center", dest="ast_center", action="store_false", help="Disable AST centering")
    ap.set_defaults(ast_center=False)  # safer default for tiny N

    # Lexical vector options
    ap.add_argument("--no-lex", action="store_true", help="Disable lexical vector features")
    ap.add_argument("--lex-dim", type=int, default=4096, help="Dimensionality for lexical vectors")
    ap.add_argument("--lex-n", type=int, default=3, help="n for n-grams (char or token)")
    ap.add_argument("--lex-mode", choices=["char", "py-token"], default="py-token", help="Lexical vector mode")

    # Fingerprint options (Dolos/MOSS-style)
    ap.add_argument("--no-fp", action="store_true", help="Disable fingerprint gating")
    ap.add_argument("--fp-k", type=int, default=5, help="Token k-gram size for fingerprints")
    ap.add_argument("--fp-w", type=int, default=4, help="Winnowing window (0=keep all k-grams)")
    ap.add_argument("--min-fp-sim", type=float, default=0.12, help="Min Jaccard of fingerprints to accept")
    ap.add_argument("--min-fp-total", type=int, default=4, help="Min intersecting fingerprints to accept")
    ap.add_argument("--min-fp-longest", type=int, default=2, help="Min longest common fingerprint run")

    # Robustness / hubness for embeddings
    ap.add_argument("--no-center", dest="center", action="store_false",
                    help="Disable mean-centering of embedding vectors (default: enabled)")
    ap.set_defaults(center=True)
    ap.add_argument("--prefilter-raw-below", type=int, default=50,
                    help="Use UNcentered embeddings for prefilter when N < this")
    ap.add_argument("--min-ast-sim", type=float, default=0.02, help="(late) Gate: minimum AST cosine")
    ap.add_argument("--min-lex-sim", type=float, default=0.02, help="(late) Gate: minimum lexical cosine")
    ap.add_argument("--allow-negative-structure", action="store_true",
                    help="Allow negative AST/lex cosine to reduce score (default: clipped to 0)")
    ap.add_argument("--embed-superpass", type=float, default=0.95,
                    help="If embedding cosine >= this, allow pass even if fp gate fails")

    # Output
    ap.add_argument("--topk", type=int, default=0, help="Print only top-K pairs (0=all)")
    ap.add_argument("--debug-components", action="store_true",
                    help="Print per-pair components (sim_embed, sim_ast, sim_lex, fp_sim/total/longest, sim_final)")

    # -------------------- EASY MODE (1 knob + presets) --------------------
    ap.add_argument("--easy", action="store_true",
                    help="Simplified UX: set only --min-tokens (+ optional --profile/--mode).")
    ap.add_argument("--min-tokens", type=int, default=5,
                    help="Primary knob. Scales fingerprint gate and sensible thresholds.")
    ap.add_argument("--profile", choices=["tight","balanced","loose"], default="balanced",
                    help="Precision/recall trade-off. tight=fewer FPs, loose=more recall.")
    ap.add_argument("--mode", choices=["hybrid","semantic","structural"], default="hybrid",
                    help="Which channels score in final stage. 'hybrid' recommended.")

    args = ap.parse_args()

    # If easy mode, translate the one knob into all internal params
    if args.easy:
        _apply_easy_mode(args)
        print(f"[EASY] profile={args.profile}, mode={args.mode}, min_tokens={args.min_tokens} → "
              f"fp_k={args.fp_k}, fp_w={args.fp_w}, min_fp_sim={args.min_fp_sim:.3f}, "
              f"min_fp_total={args.min_fp_total}, min_fp_longest={args.min_fp_longest}, "
              f"prefilter={args.prefilter_threshold}, threshold={args.threshold}, superpass={args.embed_superpass}")

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        ap.error(f"{root} is not a valid directory")

    files = load_files(root, args.extensions)
    if not files:
        print(f"No files found in {root} matching extensions {args.extensions}", file=sys.stderr)
        return

    print(f"Found {len(files)} files. Loading model '{args.model}' …")

    # Dependencies
    if SentenceTransformer is None or torch is None:
        print("Error: requires 'sentence_transformers' and 'torch'.", file=sys.stderr); sys.exit(1)
    if faiss is None:
        print("Error: requires 'faiss-cpu'.", file=sys.stderr); sys.exit(1)

    # Embeddings
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)
    except Exception as exc:
        print(f"Error loading model {args.model}: {exc}", file=sys.stderr); sys.exit(1)

    emb_t = embed_files(files, model, batch_size=args.batch_size)
    embed_np_raw = emb_t.detach().cpu().numpy().astype('float32')
    embed_np = embed_np_raw
    if args.center:
        mu = embed_np_raw.mean(axis=0, keepdims=True).astype(embed_np_raw.dtype)
        embed_np = l2_normalize_rows(embed_np_raw - mu)

    N = len(files)
    use_raw_for_prefilter = args.center and (N < max(args.prefilter_raw_below, 1))
    prefilter_np = embed_np_raw if use_raw_for_prefilter else embed_np
    embed_for_final = embed_np_raw if use_raw_for_prefilter else embed_np

    # Structural & fingerprints
    use_ast = not args.no_ast
    use_lex = not args.no_lex
    use_fp  = not args.no_fp
    if use_ast and not (_TS_AVAILABLE and _TS_LANGPACK):
        print("Note: tree-sitter / language pack not available; AST features disabled.\n"
              "      Install: pip install 'tree-sitter>=0.25,<0.26' tree-sitter-language-pack", file=sys.stderr)
        use_ast = False

    ast_np, lex_np, tokens_by_file, fp_seq, fp_ctr = compute_structural_features(
        files,
        ast_dim=args.ast_dim, use_ast=use_ast,
        ast_tfidf=args.ast_tfidf, ast_stop_topk=args.ast_stop_topk, ast_center=args.ast_center,
        lex_dim=args.lex_dim, lex_n=args.lex_n, use_lex=use_lex, lex_mode=args.lex_mode,
        fp_k=args.fp_k, fp_w=args.fp_w, use_fp=use_fp,
    )

    ast_active = use_ast and (ast_np.size > 0) and np.any(ast_np) and (args.w_ast > 0.0)
    lex_active = use_lex and (lex_np.size > 0) and np.any(lex_np) and (args.w_lex > 0.0)

    # Effective weights
    w_e = float(args.w_embed)
    w_a = float(args.w_ast) if ast_active else 0.0
    w_l = float(args.w_lex) if lex_active else 0.0

    if not ast_active:
        ast_np = np.zeros((embed_np.shape[0], 0), dtype=np.float32)
    if not lex_active:
        lex_np = np.zeros((embed_np.shape[0], 0), dtype=np.float32)

    print(f"Channels active → embed:{'yes' if w_e>0 else 'no'}  ast:{'yes' if ast_active else 'no'}  lex:{'yes' if lex_active else 'no'}")
    print(f"Weights (effective) → w_e={w_e}  w_ast={w_a}  w_lex={w_l}")
    print(f"Features → embed:{embed_np.shape[1]}  ast:{ast_np.shape[1]}  lex:{lex_np.shape[1]}")
    print(f"Fusion: {args.fusion}, threshold={args.threshold}, prefilter_threshold={args.prefilter_threshold}"
          f"{' (prefilter uses UNcentered)' if use_raw_for_prefilter else ''}")
    if use_ast:
        print(f"AST TF-IDF: {args.ast_tfidf}  stop-topK: {args.ast_stop_topk}  centered: {args.ast_center}  (N={N})")

    # Compute pairs
    if args.fusion == "concat":
        hybrid_np = build_hybrid_vectors_concat(embed_np, ast_np, lex_np, w_e, w_a, w_l)
        pairs = compute_pairs_concat(files, hybrid_np, args.threshold)
    else:
        pairs = compute_pairs_late(
            files, embed_for_final, ast_np, lex_np,
            w_embed=w_e, w_ast=w_a, w_lex=w_l,
            prefilter_threshold=args.prefilter_threshold,
            final_threshold=args.threshold,
            min_ast_sim=args.min_ast_sim,
            min_lex_sim=args.min_lex_sim,
            fp_seq=fp_seq if use_fp else [ [] for _ in files ],
            fp_ctr=fp_ctr if use_fp else [ Counter() for _ in files ],
            min_fp_sim=args.min_fp_sim,
            min_fp_total=args.min_fp_total,
            min_fp_longest=args.min_fp_longest,
            prefilter_np=prefilter_np,
            structure_positive_only=(not args.allow_negative_structure),
            embed_superpass=args.embed_superpass,
        )

    if not pairs:
        print(f"No pairs met the similarity threshold {args.threshold}")
        return

    out_pairs = pairs[:args.topk] if args.topk and args.topk > 0 else pairs
    print(f"\nDetected {len(pairs)} candidate clone pairs (showing {len(out_pairs)}):\n")
    if args.debug_components and args.fusion == "late":
        # Recompute components for shown pairs (reporting)
        for p1, p2, sim in out_pairs:
            i = files.index(p1); j = files.index(p2)
            sim_e = float(np.dot(embed_np[i], embed_np[j]))
            sim_a = float(np.dot(ast_np[i], ast_np[j])) if ast_np.shape[1] > 0 else 0.0
            sim_l = float(np.dot(lex_np[i], lex_np[j])) if lex_np.shape[1] > 0 else 0.0
            fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j]) if (not args.no_fp) else 0.0
            fpT = total_overlap(fp_ctr[i], fp_ctr[j]) if (not args.no_fp) else 0
            fpL = longest_common_run(fp_seq[i], fp_seq[j]) if (not args.no_fp) else 0
            print(f"{sim:.4f}\t{p1}\t{p2}\t(e={sim_e:.4f}, ast={sim_a:.4f}, lex={sim_l:.4f}, fp_sim={fpS:.3f}, fp_tot={fpT}, fp_long={fpL})")
    else:
        for p1, p2, sim in out_pairs:
            print(f"{sim:.4f}\t{p1}\t{p2}")

if __name__ == "__main__":
    main()
