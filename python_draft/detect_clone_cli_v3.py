#!/usr/bin/env python3
"""
Hybrid semantic + structural (AST/lexical) code clone detection with FAISS.

Channels
- Embeddings (CodeBERT cd-ft) for semantics.
- AST histograms via tree-sitter for structure (Type-1 robust).
- Optional lexical char n-grams (Type-1 friendly for Python via comment/string stripping).

Fusion
- Early (concat): build a single weighted vector and range-search it.
- Late: FAISS prefilter on embeddings, then re-score with weighted average of per-channel cosines,
        with a small structural gate to reduce false positives.

Run "python detect_clones_hybrid_cli.py -h" for options.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import tokenize
from typing import Dict, List, Optional, Tuple

import numpy as np

# --------------------------- Optional deps (AST) -------------------------------
try:
    from tree_sitter import Parser  # modern API: Parser(language, ...)
    _TS_AVAILABLE = True
except Exception:
    Parser = None  # type: ignore
    _TS_AVAILABLE = False

# Prefer maintained language pack; fallback to legacy wheels if available
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
        # Some older distributions also expose get_parser/get_language
        from tree_sitter_languages import (  # type: ignore
            get_language as _ts_get_language_legacy,
        )
        _ts_get_language = _ts_get_language_legacy
        _ts_get_parser = None
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
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


# --------------------------- Embeddings ----------------------------------------
def embed_files(
    file_paths: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> torch.Tensor:
    texts = [read_text(p) for p in file_paths]
    embs: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        emb = model.encode(batch, convert_to_tensor=True, truncation=True)
        embs.append(emb)
    all_emb = torch.cat(embs, dim=0)
    all_emb = torch.nn.functional.normalize(all_emb, p=2, dim=1)
    return all_emb


# --------------------------- Structural features -------------------------------
_EXT_TO_LANG: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".go": "go",
    ".php": "php",
    ".rs": "rust",
    ".kt": "kotlin",
    ".swift": "swift",
}

def guess_ts_language_from_ext(path: str) -> Optional[str]:
    _, ext = os.path.splitext(path)
    return _EXT_TO_LANG.get(ext)


def build_ts_parser(lang_name: str) -> Optional[Parser]:
    """Return a tree-sitter Parser for lang_name, or None with a diagnostic."""
    if not (_TS_AVAILABLE and _TS_LANGPACK):
        return None
    # Try direct helper first (returns a ready-to-use Parser)
    if _ts_get_parser is not None:
        try:
            return _ts_get_parser(lang_name)
        except Exception as e:
            print(f"[AST] get_parser('{lang_name}') failed: {e}", file=sys.stderr)
    # Fallback: obtain Language, then construct Parser for modern/legacy APIs
    try:
        lang = _ts_get_language(lang_name) if _ts_get_language else None
        if lang is None:
            print(f"[AST] no language for '{lang_name}'", file=sys.stderr)
            return None
        try:
            return Parser(lang)  # modern API: Parser(language, ...)
        except TypeError:
            p = Parser()  # legacy API path (rare)
            p.set_language(lang)  # type: ignore[attr-defined]
            return p
    except Exception as e:
        print(f"[AST] parser init failed for '{lang_name}': {e}", file=sys.stderr)
        return None


def _hash_index(key: str, dim: int) -> int:
    return (abs(hash(key)) % dim)


def _depth_bucket(depth: int) -> str:
    if depth <= 2: return "d:0-2"
    if depth <= 5: return "d:3-5"
    if depth <= 9: return "d:6-9"
    return "d:10+"


def structural_vector_from_ast(code: str, parser: Optional[Parser], dim: int) -> np.ndarray:
    """Hash-trick AST bag: node types, parent->child edges, and depth buckets."""
    vec = np.zeros(dim, dtype=np.float32)
    if not parser or not code:
        return vec
    try:
        tree = parser.parse(bytes(code, "utf8"))
        root = tree.root_node  # type: ignore
    except Exception:
        return vec

    stack = [(root, 0, None)]  # (node, depth, parent_type)
    while stack:
        node, depth, parent_t = stack.pop()
        try:
            t = node.type  # type: ignore[attr-defined]
        except Exception:
            continue
        vec[_hash_index(f"n:{t}", dim)] += 1.0
        if parent_t is not None:
            vec[_hash_index(f"e:{parent_t}>{t}", dim)] += 1.0
        vec[_hash_index(_depth_bucket(depth), dim)] += 1.0
        try:
            children = node.children
        except Exception:
            children = []
        for ch in children:
            stack.append((ch, depth + 1, t))
    return vec


# --- Optional lexical: char n-grams (with Python comment/string stripping) -----
def strip_py_comments_strings(src: str) -> str:
    if not src: return src
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT):
                continue
            out.append(tok.string)
        return " ".join(out)
    except Exception:
        return src  # fallback

def char_ngram_vector(text: str, n: int, dim: int, max_len: int = 30000) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if not text:
        return vec
    s = text[:max_len]
    if len(s) < n:
        return vec
    for i in range(len(s) - n + 1):
        key = s[i:i+n]
        vec[_hash_index(f"g:{key}", dim)] += 1.0
    return vec


def compute_structural_features(
    file_paths: List[str],
    ast_dim: int,
    use_ast: bool,
    lex_dim: int,
    lex_n: int,
    use_lex: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    N = len(file_paths)
    ast_mat = np.zeros((N, 0), dtype=np.float32)
    lex_mat = np.zeros((N, 0), dtype=np.float32)

    parser_cache: Dict[str, Optional[Parser]] = {}
    if use_ast and _TS_AVAILABLE and _TS_LANGPACK:
        langs_needed = set(filter(None, (guess_ts_language_from_ext(p) for p in file_paths)))
        for ln in langs_needed:
            if ln:
                parser_cache[ln] = build_ts_parser(ln)

    ast_rows, lex_rows = [], []
    for path in file_paths:
        text = read_text(path)
        if use_ast and parser_cache:
            lang_name = guess_ts_language_from_ext(path)
            parser = parser_cache.get(lang_name) if lang_name in parser_cache else None
            ast_vec = structural_vector_from_ast(text, parser, ast_dim) if parser else np.zeros(ast_dim, np.float32)
            ast_rows.append(ast_vec)
        if use_lex:
            t = strip_py_comments_strings(text) if path.endswith(".py") else text
            lex_rows.append(char_ngram_vector(t, lex_n, lex_dim))

    if use_ast and ast_rows:
        ast_mat = np.vstack(ast_rows)
        if ast_mat.size > 0:
            ast_mat = l2_normalize_rows(ast_mat)
    if use_lex and lex_rows:
        lex_mat = np.vstack(lex_rows)
        if lex_mat.size > 0:
            lex_mat = l2_normalize_rows(lex_mat)

    return ast_mat, lex_mat


# --------------------------- FAISS helpers -------------------------------------
def build_faiss_index_np(mat: np.ndarray) -> faiss.Index:
    if faiss is None:
        raise RuntimeError("faiss-cpu is required. pip install faiss-cpu")
    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product; with L2-normalized vectors = cosine
    index.add(mat.astype('float32', copy=False))
    return index


# --------------------------- Fusion strategies ---------------------------------
def build_hybrid_vectors_concat(
    embed_np: np.ndarray,
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    w_embed: float,
    w_ast: float,
    w_lex: float,
) -> np.ndarray:
    """Early fusion; cosine on V equals weighted average over active channels."""
    parts, active_w = [], 0.0
    parts.append(np.sqrt(max(w_embed, 0.0)) * embed_np); active_w += max(w_embed, 0.0)

    ast_active = ast_np.size > 0 and np.any(ast_np) and w_ast > 0.0
    if ast_active:
        parts.append(np.sqrt(w_ast) * ast_np); active_w += w_ast

    lex_active = lex_np.size > 0 and np.any(lex_np) and w_lex > 0.0
    if lex_active:
        parts.append(np.sqrt(w_lex) * lex_np); active_w += w_lex

    V = np.concatenate(parts, axis=1)
    V = V / math.sqrt(max(active_w, 1e-12))
    return V.astype('float32', copy=False)


def compute_pairs_concat(
    file_paths: List[str],
    hybrid_np: np.ndarray,
    threshold: float,
) -> List[Tuple[str, str, float]]:
    index = build_faiss_index_np(hybrid_np)
    lims, dists, labels = index.range_search(hybrid_np, float(threshold))
    n = hybrid_np.shape[0]
    pairs: List[Tuple[str, str, float]] = []
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
    embed_np_final: np.ndarray,   # possibly centered; used for final semantic cosine
    ast_np: np.ndarray,
    lex_np: np.ndarray,
    w_embed: float,
    w_ast: float,
    w_lex: float,
    prefilter_threshold: float,
    final_threshold: float,
    min_ast_sim: float,
    min_lex_sim: float,
    prefilter_np: Optional[np.ndarray] = None,  # vectors used for FAISS prefilter (may be uncentered)
) -> List[Tuple[str, str, float]]:
    """Late fusion with structural gate; prefilter can use different vectors than final scoring."""
    base = prefilter_np if prefilter_np is not None else embed_np_final
    emb_index = build_faiss_index_np(base)
    lims, dists, labels = emb_index.range_search(base, float(prefilter_threshold))

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

            # semantic similarity from final embedding space
            sim_e = float(np.dot(embed_np_final[i], embed_np_final[j]))  # with L2-normed rows -> cosine

            num, den = w_embed * sim_e, w_embed
            sim_a = float(np.dot(a_i, ast_np[j])) if ast_active else 0.0
            sim_l = float(np.dot(l_i, lex_np[j])) if lex_active else 0.0

            # structural gate
            gate_fail = True
            if ast_active and sim_a >= min_ast_sim: gate_fail = False
            if lex_active and sim_l >= min_lex_sim: gate_fail = False
            if (not ast_active and not lex_active): gate_fail = False
            if gate_fail:
                continue

            if ast_active: num += w_ast * sim_a; den += w_ast
            if lex_active: num += w_lex * sim_l; den += w_lex

            s = num / max(den, 1e-12)
            if s >= final_threshold:
                pairs.append((file_paths[i], file_paths[j], s))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs


# --------------------------- CLI / main ----------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid semantic+structural clone detection (FAISS)")
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

    # Weights (Type-1 friendly defaults)
    ap.add_argument("--w-embed", type=float, default=1.0, help="Weight for embedding channel")
    ap.add_argument("--w-ast", type=float, default=0.7, help="Weight for AST structural channel")
    ap.add_argument("--w-lex", type=float, default=0.0, help="Weight for lexical channel (char n-grams)")

    # Structural features
    ap.add_argument("--no-ast", action="store_true", help="Disable AST structural features")
    ap.add_argument("--ast-dim", type=int, default=2048, help="Dimensionality for hashed AST features")
    ap.add_argument("--no-lex", action="store_true", help="Disable lexical n-gram features")
    ap.add_argument("--lex-dim", type=int, default=4096, help="Dimensionality for hashed lexical features")
    ap.add_argument("--lex-n", type=int, default=5, help="Character n-gram length")

    # Robustness / hubness controls
    ap.add_argument("--no-center", dest="center", action="store_false",
                    help="Disable mean-centering of embeddings (default: enabled)")
    ap.set_defaults(center=True)
    ap.add_argument("--prefilter-raw-below", type=int, default=50,
                    help="Use UNcentered embeddings for prefilter when N < this (avoids empty candidate sets on tiny corpora)")
    ap.add_argument("--min-ast-sim", type=float, default=0.05,
                    help="(late) Gate: minimum AST cosine for acceptance")
    ap.add_argument("--min-lex-sim", type=float, default=0.05,
                    help="(late) Gate: minimum lexical cosine for acceptance")

    # Output controls
    ap.add_argument("--topk", type=int, default=0, help="Print only top-K pairs (0=all)")
    ap.add_argument("--debug-components", action="store_true",
                    help="Print per-pair components (sim_embed, sim_ast, sim_lex, sim_final) for the first --topk pairs")

    args = ap.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        ap.error(f"{root} is not a valid directory")

    files = load_files(root, args.extensions)
    if not files:
        print(f"No files found in {root} matching extensions {args.extensions}", file=sys.stderr)
        return

    print(f"Found {len(files)} files. Loading model '{args.model}' …")

    # Dependency checks
    if SentenceTransformer is None or torch is None:
        print("Error: requires 'sentence_transformers' and 'torch'.", file=sys.stderr)
        sys.exit(1)
    if faiss is None:
        print("Error: requires 'faiss-cpu'.", file=sys.stderr)
        sys.exit(1)

    # Load embedding model
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)
    except Exception as exc:
        print(f"Error loading model {args.model}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Embeddings
    emb_t = embed_files(files, model, batch_size=args.batch_size)
    embed_np_raw = emb_t.detach().cpu().numpy().astype('float32')
    embed_np = embed_np_raw

    # Mean-centering (reduces hubness on large N)
    if args.center:
        mu = embed_np_raw.mean(axis=0, keepdims=True).astype(embed_np_raw.dtype)
        embed_np = l2_normalize_rows(embed_np_raw - mu)

    # Prefilter vectors: for small N, use uncentered to avoid empty candidate sets
    N = len(files)
    use_raw_for_prefilter = args.center and (N < max(args.prefilter_raw_below, 1))
    prefilter_np = embed_np_raw if use_raw_for_prefilter else embed_np

    # Structural features
    use_ast = not args.no_ast
    use_lex = not args.no_lex
    if use_ast and not (_TS_AVAILABLE and _TS_LANGPACK):
        print("Note: tree-sitter / language pack not available; AST features disabled.\n"
              "      Install: pip install 'tree-sitter>=0.25,<0.26' tree-sitter-language-pack",
              file=sys.stderr)
        use_ast = False

    ast_np, lex_np = compute_structural_features(
        files, ast_dim=args.ast_dim, use_ast=use_ast,
        lex_dim=args.lex_dim, lex_n=args.lex_n, use_lex=use_lex,
    )

    # Weighted-active channels (respect dimensions + nonzero + weight>0)
    ast_weighted_active = use_ast and (ast_np.size > 0) and np.any(ast_np) and (args.w_ast > 0.0)
    lex_weighted_active = use_lex and (lex_np.size > 0) and np.any(lex_np) and (args.w_lex > 0.0)

    # Effective weights
    w_e = float(args.w_embed)
    w_a = float(args.w_ast) if ast_weighted_active else 0.0
    w_l = float(args.w_lex) if lex_weighted_active else 0.0

    # Ensure empty matrices for inactive channels
    if not ast_weighted_active:
        ast_np = np.zeros((embed_np.shape[0], 0), dtype=np.float32)
    if not lex_weighted_active:
        lex_np = np.zeros((embed_np.shape[0], 0), dtype=np.float32)

    print(f"Channels active → embed:yes  ast:{'yes' if ast_weighted_active else 'no'}  lex:{'yes' if lex_weighted_active else 'no'}")
    print(f"Weights (effective) → w_e={w_e}  w_ast={w_a}  w_lex={w_l}")
    print(f"Features → embed:{embed_np.shape[1]}  ast:{ast_np.shape[1]}  lex:{lex_np.shape[1]}")
    print(f"Fusion: {args.fusion}, threshold={args.threshold}, prefilter_threshold={args.prefilter_threshold}"
          f"{' (prefilter uses UNcentered)' if use_raw_for_prefilter else ''}")

    # Compute pairs
    if args.fusion == "concat":
        hybrid_np = build_hybrid_vectors_concat(embed_np, ast_np, lex_np, w_e, w_a, w_l)
        pairs = compute_pairs_concat(files, hybrid_np, args.threshold)
    else:
        pairs = compute_pairs_late(
            files, embed_np, ast_np, lex_np,
            w_embed=w_e, w_ast=w_a, w_lex=w_l,
            prefilter_threshold=args.prefilter_threshold,
            final_threshold=args.threshold,
            min_ast_sim=args.min_ast_sim,
            min_lex_sim=args.min_lex_sim,
            prefilter_np=prefilter_np,
        )

    if not pairs:
        print(f"No pairs met the similarity threshold {args.threshold}")
        return

    # Optionally limit output
    out_pairs = pairs[:args.topk] if args.topk and args.topk > 0 else pairs

    print(f"\nDetected {len(pairs)} candidate clone pairs (showing {len(out_pairs)}):\n")
    if args.debug_components and args.fusion == "late":
        # Print per-pair components for first K pairs
        for p1, p2, sim in out_pairs:
            i = files.index(p1); j = files.index(p2)
            sim_e = float(np.dot(embed_np[i], embed_np[j]))
            sim_a = float(np.dot(ast_np[i], ast_np[j])) if ast_np.shape[1] > 0 else 0.0
            sim_l = float(np.dot(lex_np[i], lex_np[j])) if lex_np.shape[1] > 0 else 0.0
            print(f"{sim:.4f}\t{p1}\t{p2}\t(e={sim_e:.4f}, ast={sim_a:.4f}, lex={sim_l:.4f})")
    else:
        for p1, p2, sim in out_pairs:
            print(f"{sim:.4f}\t{p1}\t{p2}")


if __name__ == "__main__":
    main()
