#!/usr/bin/env python3
"""
Hybrid semantic + structural clone detection with FAISS and soft fingerprint/structure scoring (v8.2).

Key ideas:
- Prefilter can be 'range' (radius), 'topk', or 'all' pairs (for small cohorts).
- Soft gating: pairs without fingerprint/structure evidence are down-weighted, not dropped.
- Evidence uses Dolos-style metrics: Jaccard, total overlap (normalized), longest fragment (normalized).
- Exposes CLI overrides for thresholds and modes.

Refs: Winnowing/MOSS (Schleimer et al., SIGMOD'03), Greedy String Tiling min-match (JPlag),
      Dolos (fingerprints, total overlap, longest fragment).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tokenize
import keyword
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------- Optional deps (AST) -------------------------------
try:
    from tree_sitter import Parser  # type: ignore
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
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None  # type: ignore

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
    if not s:
        return []
    if len(s) <= window_chars:
        return [s]
    out, start = [], 0
    while start < len(s):
        end = min(len(s), start + window_chars)
        out.append(s[start:end])
        if end == len(s): break
        start += stride_chars
    return out

def embed_files(
    file_paths: List[str],
    model: SentenceTransformer,
    batch_size: int = 16,
    window_chars: int = 3000,
    stride_chars: int = 1500,
) -> torch.Tensor:
    dim = model.get_sentence_embedding_dimension()
    per_file: List[torch.Tensor] = []
    for p in file_paths:
        text = read_text(p)
        chunks = _chunk_text(text, window_chars, stride_chars)
        if not chunks:
            per_file.append(torch.zeros(dim))
            continue
        emb = model.encode(
            chunks, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        per_file.append(emb.mean(dim=0))
    all_emb = torch.stack(per_file, dim=0)
    return torch.nn.functional.normalize(all_emb, p=2, dim=1)  # cosine-compatible

# --------------------------- Structural (AST) -----------------------------------
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
    if not src: return toks
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            tt, val = tok.type, tok.string
            if tt in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.COMMENT):
                continue
            if tt == tokenize.STRING:  # ignore strings
                continue
            if tt == tokenize.NUMBER:
                toks.append("NUM"); continue
            if tt == tokenize.NAME:
                toks.append(val if val in keyword.kwlist else "ID"); continue
            if val.strip(): toks.append(val)
    except Exception:
        pass
    return toks

def token_ngram_vector(tokens: List[str], n: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not tokens or len(tokens) < n: return vec
    for i in range(len(tokens) - n + 1):
        idx = _stable_bucket("t:" + "⊕".join(tokens[i:i+n]), dim)
        vec[idx] += 1.0
    return vec

# --------------------------- Fingerprinting -------------------------------------
def kgram_hashes(tokens: List[str], k: int) -> List[int]:
    if not tokens or len(tokens) < k: return []
    out: List[int] = []
    for i in range(len(tokens) - k + 1):
        kgram = "§".join(tokens[i:i+k])
        h = hashlib.sha1(kgram.encode("utf-8")).digest()
        out.append(int.from_bytes(h[:8], "little"))
    return out

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
    if not a and not b: return 0.0
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / max(union, 1)

def total_overlap(a: Counter, b: Counter) -> int:
    return sum((a & b).values())

def longest_common_run(seqA: List[int], seqB: List[int]) -> int:
    if not seqA or not seqB: return 0
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

# --------------------------- Structural pipeline --------------------------------
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
    lex_mode: str,
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

    # parsers
    parser_cache: Dict[str, Optional[Parser]] = {}
    if use_ast and _TS_AVAILABLE and _TS_LANGPACK:
        langs_needed = set(filter(None, (guess_ts_language_from_ext(p) for p in file_paths)))
        for ln in langs_needed:
            if ln: parser_cache[ln] = build_ts_parser(ln)

    ast_rows, lex_rows = [], []
    for path in file_paths:
        text = read_text(path)

        if use_ast and parser_cache:
            lang_name = guess_ts_language_from_ext(path)
            parser = parser_cache.get(lang_name) if (lang_name in parser_cache) else None
            ast_vec = structural_vector_from_ast(text, parser, ast_dim) if parser else np.zeros(ast_dim, np.float32)
            ast_rows.append(ast_vec)

        toks: List[str] = []
        if path.endswith(".py"): toks = py_token_stream(text)
        tokens_by_file.append(toks)
        tok_counts.append(len(toks))

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
                        idx = _stable_bucket(f"g:{s[i:i+lex_n]}", lex_dim)
                        vec[idx] += 1.0
                    lex_rows.append(vec)

        if use_fp and toks:
            seq, ctr = fp_sequence(toks, fp_k, fp_w)
            fp_seq_by_file.append(seq); fp_ctr_by_file.append(ctr)
        else:
            fp_seq_by_file.append([]); fp_ctr_by_file.append(Counter())

    if use_ast and ast_rows:
        ast_mat = np.vstack(ast_rows)
        ast_mat = stop_topk(ast_mat, ast_stop_topk)
        ast_mat = apply_tfidf(ast_mat) if ast_tfidf else l2_normalize_rows(ast_mat)

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

# --------------------------- Evidence & scoring --------------------------------
def _norm_div(x: float, denom: float) -> float:
    return x / max(denom, 1e-9)

def evidence_factor(
    fpS: float, fpT: int, fpL: int,
    lenA: int, lenB: int,
    short_pair: bool,
    mode: str = "soft"
) -> float:
    """Map fingerprint evidence to a multiplicative factor in [base, 1]."""
    # normalize overlap: symmetric F1-like
    totA, totB = float(lenA), float(lenB)
    fpT_norm = _norm_div(2.0 * fpT, totA + totB) if (totA + totB) > 0 else 0.0
    # normalize longest run
    fpL_norm = _norm_div(fpL, max(1.0, min(totA, totB)))
    evid = max(fpS, fpT_norm, fpL_norm)  # [0,1]

    if mode == "hard":
        # not used here; gating is handled elsewhere if selected
        base = 0.0
    elif mode == "medium":
        base = 0.35 if short_pair else 0.45
    else:  # soft
        base = 0.40 if short_pair else 0.60

    return float(max(base, min(1.0, base + (1.0 - base) * evid)))

def compute_pairs_late(
    file_paths: List[str],
    embed_np_final: np.ndarray,
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    *,
    w_embed: float,
    w_ast: float,
    w_lex: float,
    prefilter_mode: str,
    prefilter_threshold: float,
    prefilter_topk: int,
    final_threshold: float,
    min_ast_sim: float,
    min_lex_sim: float,
    # fingerprint artifacts
    fp_seq: List[List[int]],
    fp_ctr: List[Counter],
    min_fp_sim: float,
    min_fp_total: int,
    min_fp_longest: int,
    # scoring style
    gate_mode: str = "soft",
    structure_positive_only: bool = True,
    embed_superpass: float = 0.995,
    short_token_gate: int = 50,
    token_counts: Optional[List[int]] = None,
    prefilter_np: Optional[np.ndarray] = None,
) -> List[Tuple[str, str, float]]:
    base = prefilter_np if prefilter_np is not None else embed_np_final
    n = embed_np_final.shape[0]

    # Prepare neighbor candidates
    candidate_pairs: List[Tuple[int, int]] = []
    if prefilter_mode == "all":
        for i in range(n):
            for j in range(i+1, n):
                candidate_pairs.append((i, j))
    elif prefilter_mode == "topk":
        index = build_faiss_index_np(base)
        k = max(2, min(prefilter_topk, n))  # includes self; we will skip self
        sims, idxs = index.search(base, k)
        seen = set()
        for i in range(n):
            for j in idxs[i]:
                if j == i or j < 0: continue
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in seen:
                    seen.add((a, b)); candidate_pairs.append((a, b))
    else:  # range
        index = build_faiss_index_np(base)
        lims, _dists, labels = index.range_search(base, float(prefilter_threshold))
        for i in range(n):
            start, end = lims[i], lims[i+1]
            for k in range(start, end):
                j = labels[k]
                if j == i: continue
                a, b = (i, j) if i < j else (j, i)
                if a != b:
                    candidate_pairs.append((a, b))
        # deduplicate
        candidate_pairs = sorted(list(set(candidate_pairs)))

    ast_active = ast_np.size > 0 and np.any(ast_np) and w_ast > 0.0
    lex_active = lex_np.size > 0 and np.any(lex_np) and w_lex > 0.0

    pairs: List[Tuple[str, str, float]] = []
    for (i, j) in candidate_pairs:
        a_i = ast_np[i] if ast_active else None
        l_i = lex_np[i] if lex_active else None

        sim_e = float(np.dot(embed_np_final[i], embed_np_final[j]))
        num, den = w_embed * sim_e, w_embed

        sim_a = float(np.dot(a_i, ast_np[j])) if ast_active else 0.0
        sim_l = float(np.dot(l_i, lex_np[j])) if lex_active else 0.0
        if structure_positive_only:
            sim_a = max(sim_a, 0.0); sim_l = max(sim_l, 0.0)

        # Fingerprint stats
        fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j]) if fp_ctr else 0.0
        fpT = total_overlap(fp_ctr[i], fp_ctr[j]) if fp_ctr else 0
        fpL = longest_common_run(fp_seq[i], fp_seq[j]) if fp_seq else 0
        lenA, lenB = len(fp_seq[i]), len(fp_seq[j])

        # Decide if the pair is "short"
        short_pair = False
        if token_counts is not None:
            short_pair = (token_counts[i] < short_token_gate or token_counts[j] < short_token_gate)

        # ---- Gate behaviour ----
        if gate_mode == "hard":
            # Hard gate like v8: require structure or fingerprints (or extreme semantic superpass)
            struct_ok = False
            if ast_active and sim_a >= min_ast_sim: struct_ok = True
            if lex_active and sim_l >= min_lex_sim: struct_ok = True
            need_fp = short_pair or (ast_active or lex_active) and not struct_ok or (not ast_active and not lex_active)
            if need_fp:
                if not (fpS >= min_fp_sim and fpT >= min_fp_total and fpL >= min_fp_longest):
                    if sim_e < embed_superpass:
                        continue
            # continue to scoring
            pass

        # Soft / medium: compute evidence factor in [base,1]
        ef = evidence_factor(fpS, fpT, fpL, lenA, lenB, short_pair, gate_mode)

        # Weighted fusion (+ structure)
        if ast_active: num += w_ast * sim_a; den += w_ast
        if lex_active: num += w_lex * sim_l; den += w_lex
        s_raw = num / max(den, 1e-12)

        # Final score scaled by evidence (soft penalty if little evidence)
        s = float(s_raw * ef)

        if s >= final_threshold:
            pairs.append((file_paths[i], file_paths[j], s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs

# =========================== AUTO PROFILE ======================================
def _auto_profile(args, N: int):
    mt = max(2, int(args.min_tokens))

    # fingerprints (slightly lenient)
    args.fp_k = max(4, min(9, mt))
    args.fp_w = max(4, args.fp_k - 1)
    args.min_fp_total   = max(args.fp_k - 1, 4)
    args.min_fp_longest = max(1, args.fp_k - 2)
    args.min_fp_sim     = 0.05 + 0.01 * max(0, args.fp_k - 5)

    # weights
    args.w_embed = 1.0
    args.w_ast   = 0.35
    args.w_lex   = 0.10
    args.no_ast  = False
    args.no_lex  = False

    # structural vectors
    args.lex_mode = "py-token"
    args.lex_n    = 4
    args.ast_dim  = 2048
    args.lex_dim  = 4096
    args.ast_tfidf = True
    args.ast_stop_topk = 96

    # thresholds (lenient, continuous scores)
    if args.mode == "semantic":
        args.threshold = 0.30
        args.prefilter_threshold = 0.20
        args.no_ast = True; args.no_lex = True; args.w_ast = 0.0; args.w_lex = 0.0
        args.min_ast_sim = 0.0; args.min_lex_sim = 0.0
    elif args.mode == "structural":
        args.threshold = 0.28
        args.prefilter_threshold = 0.20
        args.w_embed = 0.0
        args.min_ast_sim = 0.03; args.min_lex_sim = 0.03
    else:  # hybrid
        args.threshold = 0.30
        args.prefilter_threshold = 0.20
        args.min_ast_sim = 0.02
        args.min_lex_sim = 0.02

    args.embed_superpass = 0.995

# --------------------------- CLI / main ----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hybrid semantic+structural clone detection with soft evidence scaling (v8.2)"
    )
    # Essentials
    ap.add_argument("--dir", required=True, help="Root directory to scan")
    ap.add_argument("--extensions", nargs="*", required=True, help="Extensions to include, e.g., .py .java")
    ap.add_argument("--mode", choices=["hybrid", "semantic", "structural"], default="hybrid",
                    help="Scoring mode (hybrid recommended)")
    ap.add_argument("--min-tokens", type=int, default=7, help="Primary knob (affects fingerprinting)")

    # Prefilter controls
    ap.add_argument("--prefilter-mode", choices=["range","topk","all"], default="range",
                    help="Candidate generation: 'range' by cosine radius, 'topk' per item, or 'all' pairs")
    ap.add_argument("--prefilter-radius", type=float, default=None,
                    help="Override cosine radius for 'range' mode (L2-normed IP). Default from profile.")
    ap.add_argument("--prefilter-topk", type=int, default=50,
                    help="K neighbors per item for 'topk' mode")

    # Threshold override
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override final threshold; use 0 to print all pairs and rely on scores only")
    ap.add_argument("--topk", type=int, default=0, help="Print only top-K pairs (0=all)")

    # Evidence / gating style
    ap.add_argument("--gate-mode", choices=["soft","medium","hard"], default="soft",
                    help="How strictly fingerprints/structure affect acceptance (soft=penalize only)")

    # Embedding / perf
    ap.add_argument("--model", default="mchochlov/codebert-base-cd-ft",
                    help="SentenceTransformer model for embeddings")
    ap.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")

    # Debug & extras
    ap.add_argument("--debug-components", action="store_true",
                    help="Print per-pair components and evidence (e, ast, lex, fp_sim/total/longest, factor)")
    ap.add_argument("--short-token-gate", type=int, default=50,
                    help="Token-count threshold for 'short' pair heuristics in evidence scaling")
    ap.add_argument("--reranker-model", default="",
                    help="Optional CrossEncoder for final precision (pairwise classifier)")
    ap.add_argument("--reranker-threshold", type=float, default=0.50,
                    help="Probability threshold for reranker go/no-go")

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

    if SentenceTransformer is None or torch is None:
        print("Error: requires 'sentence_transformers' and 'torch'.", file=sys.stderr); sys.exit(1)
    if faiss is None and args.prefilter_mode in ("range","topk"):
        print("Error: requires 'faiss-cpu' for range/topk prefilter.", file=sys.stderr); sys.exit(1)

    _auto_profile(args, N)
    if args.threshold is not None:
        args.threshold = float(args.threshold)
    # else keep profile default
    if args.prefilter_radius is not None:
        args.prefilter_threshold = float(args.prefilter_radius)

    print(f"[AUTO] mode={args.mode}, min_tokens={args.min_tokens} → "
          f"fp_k={args.fp_k}, fp_w={args.fp_w}, min_fp_sim={args.min_fp_sim:.3f}, "
          f"min_fp_total={args.min_fp_total}, min_fp_longest={args.min_fp_longest}, "
          f"prefilter={args.prefilter_threshold}, threshold={args.threshold}, gate={args.gate_mode}")

    # Embeddings
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)
    except Exception as exc:
        print(f"Error loading model {args.model}: {exc}", file=sys.stderr); sys.exit(1)

    emb_t = embed_files(files, model, batch_size=args.batch_size)
    embed_np_raw = emb_t.detach().cpu().numpy().astype('float32')  # L2-normalized

    # No centering for small N — keep robust defaults
    prefilter_np = embed_np_raw
    embed_for_final = embed_np_raw

    # Structural & fingerprints
    use_ast = True if args.mode in ("hybrid", "structural") else False
    use_lex = True if args.mode in ("hybrid", "structural") else False
    use_fp  = True

    if use_ast and not (_TS_AVAILABLE and _TS_LANGPACK):
        print("Note: tree-sitter / language pack not available; AST features disabled.\n"
              "      Install: pip install 'tree-sitter>=0.25,<0.26' tree-sitter-language-pack", file=sys.stderr)
        use_ast = False

    ast_np, lex_np, tokens_by_file, fp_seq, fp_ctr, tok_counts = compute_structural_features(
        files,
        ast_dim=2048, use_ast=use_ast, ast_tfidf=True, ast_stop_topk=96,
        lex_dim=4096, lex_n=4, use_lex=use_lex, lex_mode="py-token",
        fp_k=args.fp_k, fp_w=args.fp_w, use_fp=use_fp,
    )

    ast_active = use_ast and (ast_np.size > 0) and np.any(ast_np) and (args.w_ast > 0.0)
    lex_active = use_lex and (lex_np.size > 0) and np.any(lex_np) and (args.w_lex > 0.0)

    w_e = float(args.w_embed if args.mode != "structural" else 0.0)
    w_a = float(args.w_ast) if ast_active else 0.0
    w_l = float(args.w_lex) if lex_active else 0.0

    if not ast_active: ast_np = np.zeros((embed_np_raw.shape[0], 0), dtype=np.float32)
    if not lex_active: lex_np = np.zeros((embed_np_raw.shape[0], 0), dtype=np.float32)

    print(f"Channels → embed:{'yes' if w_e>0 else 'no'}  ast:{'yes' if ast_active else 'no'}  lex:{'yes' if lex_active else 'no'}")
    print(f"Weights → w_e={w_e}  w_ast={w_a}  w_lex={w_l}")
    print(f"Features → embed:{embed_for_final.shape[1]}  ast:{ast_np.shape[1]}  lex:{lex_np.shape[1]}")
    print(f"Prefilter: mode={args.prefilter_mode} | radius={args.prefilter_threshold} | TopK={args.prefilter_topk} | Threshold={args.threshold}")

    pairs = compute_pairs_late(
        files, embed_for_final, ast_np, lex_np,
        w_embed=w_e, w_ast=w_a, w_lex=w_l,
        prefilter_mode=args.prefilter_mode,
        prefilter_threshold=args.prefilter_threshold,
        prefilter_topk=args.prefilter_topk,
        final_threshold=(0.0 if args.threshold is None else args.threshold),
        min_ast_sim=getattr(args, "min_ast_sim", 0.0),
        min_lex_sim=getattr(args, "min_lex_sim", 0.0),
        fp_seq=fp_seq if use_fp else [ [] for _ in files ],
        fp_ctr=fp_ctr if use_fp else [ Counter() for _ in files ],
        min_fp_sim=args.min_fp_sim,
        min_fp_total=args.min_fp_total,
        min_fp_longest=args.min_fp_longest,
        gate_mode=args.gate_mode,
        structure_positive_only=True,
        embed_superpass=args.embed_superpass,
        short_token_gate=args.short_token_gate,
        token_counts=tok_counts,
        prefilter_np=prefilter_np,
    )

    # Optional reranker
    if args.reranker_model:
        if CrossEncoder is None:
            print("Warning: sentence-transformers CrossEncoder not available; skipping reranker.", file=sys.stderr)
        else:
            try:
                print(f"Loading cross-encoder reranker '{args.reranker_model}' …")
                device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
                reranker = CrossEncoder(args.reranker_model, device=device)
                pairs_text = [(read_text(p1), read_text(p2)) for (p1, p2, _s) in pairs]
                probs = reranker.predict(pairs_text)
                pairs = [(p1, p2, s) for (p1, p2, s), pr in zip(pairs, probs) if float(pr) >= float(args.reranker_threshold)]
            except Exception as exc:
                print(f"Warning: could not load/use reranker '{args.reranker_model}': {exc}", file=sys.stderr)

    if not pairs:
        print(f"No pairs met the similarity threshold {args.threshold if args.threshold is not None else 0.0}")
        return

    out_pairs = pairs[:args.topk] if args.topk and args.topk > 0 else pairs
    print(f"\nDetected {len(pairs)} candidate clone pairs (showing {len(out_pairs)}):\n")
    if args.debug_components:
        for p1, p2, sim in out_pairs:
            i = files.index(p1); j = files.index(p2)
            sim_e = float(np.dot(embed_for_final[i], embed_for_final[j]))
            sim_a = float(np.dot(ast_np[i], ast_np[j])) if ast_np.shape[1] > 0 else 0.0
            sim_l = float(np.dot(lex_np[i], lex_np[j])) if lex_np.shape[1] > 0 else 0.0
            fpS = jaccard_from_counters(fp_ctr[i], fp_ctr[j])
            fpT = total_overlap(fp_ctr[i], fp_ctr[j])
            fpL = longest_common_run(fp_seq[i], fp_seq[j])
            lenA, lenB = len(fp_seq[i]), len(fp_seq[j])
            short_pair = (tok_counts[i] < args.short_token_gate or tok_counts[j] < args.short_token_gate)
            ef = evidence_factor(fpS, fpT, fpL, lenA, lenB, short_pair, args.gate_mode)
            print(f"{sim:.4f}\t{p1}\t{p2}\t(e={sim_e:.4f}, ast={sim_a:.4f}, lex={sim_l:.4f}, "
                  f"fp_sim={fpS:.3f}, fp_tot={fpT}, fp_long={fpL}, evid={ef:.3f})")
    else:
        for p1, p2, sim in out_pairs:
            print(f"{sim:.4f}\t{p1}\t{p2}")

if __name__ == "__main__":
    main()
