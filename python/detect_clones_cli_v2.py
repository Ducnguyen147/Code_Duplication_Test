#!/usr/bin/env python3
"""
CLI utility for semantic code clone detection over a directory of source files.

This script builds on a basic pairwise comparison pipeline and extends it to
operate efficiently over large codebases by leveraging FAISS for approximate
nearest‐neighbour search.  It scans a directory recursively, embeds each
supported file using a HuggingFace/SentenceTransformers model and then uses
FAISS's range search to find all file pairs whose cosine similarity exceeds a
user‑specified threshold.  The results are printed in descending order of
similarity.

Key features:

* Supports any HuggingFace or sentence‑transformers model that produces code
  embeddings.  By default it uses ``mchochlov/codebert‑base‑cd‑ft``, a model
  fine‑tuned for clone detection.
* Recursively walks a directory to find source files.  You can restrict which
  files are considered by specifying a list of extensions (e.g. ``.py .java``).
  If no extensions are provided, all regular files are embedded.
* Embeddings are computed once per file and normalised to unit length.  A
  FAISS index is built to allow efficient similarity search.
* Uses FAISS range search to find all pairs of files whose cosine similarity
  exceeds the threshold.  This avoids the quadratic cost of computing the
  full similarity matrix for large repositories.

Example usage::

    python detect_clones_cli.py --dir path/to/project \
        --extensions .py .java --model mchochlov/codebert-base-cd-ft --threshold 0.85

Dependencies:
    pip install sentence-transformers faiss-cpu

"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Tuple

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


def load_files(directory: str, extensions: List[str] | None) -> List[str]:
    """Recursively collect file paths from ``directory``.

    If ``extensions`` is provided, only files with one of the specified
    extensions (case‑sensitive) will be included.  Extensions should include
    the leading dot, e.g. ``.py`` or ``.java``.

    Returns:
        A sorted list of absolute file paths.
    """
    collected: List[str] = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            # Skip hidden files and directories
            if fname.startswith('.'):
                continue
            if extensions:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                collected.append(path)
    return sorted(collected)


def embed_files(
    file_paths: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute embeddings for a list of files using ``model``.

    The returned tensor has shape (num_files, embed_dim).  Each row
    corresponds to the embedding of the file at the same index in
    ``file_paths``.  The embeddings are normalised to unit length.

    ``batch_size`` controls how many files are encoded at once.  Larger
    batches will generally be faster on a GPU but require more memory.
    """
    texts: List[str] = []
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            texts.append("")

    embeddings: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_emb = model.encode(
            batch_texts,
            convert_to_tensor=True,
            truncation=True,
        )
        embeddings.append(batch_emb)
    all_embeddings = torch.cat(embeddings, dim=0)
    # Normalise for cosine similarity via dot product
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    return all_embeddings


def build_faiss_index(embeddings: torch.Tensor) -> faiss.Index:
    """Build a FAISS index for inner product (cosine similarity) search.

    Args:
        embeddings: Tensor of shape (n_files, dim), normalised rows.

    Returns:
        A FAISS index loaded with the embeddings.
    """
    if faiss is None:
        raise RuntimeError(
            "faiss library is required but not installed. Please install faiss-cpu."
        )
    # Convert to numpy array of type float32
    emb_np = embeddings.detach().cpu().numpy().astype('float32')
    dim = emb_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb_np)
    return index


def compute_similar_pairs(
    embeddings: torch.Tensor,
    file_paths: List[str],
    threshold: float,
    index: faiss.Index,
) -> List[Tuple[str, str, float]]:
    """Compute similarity scores and return pairs exceeding the threshold using FAISS.

    This function performs a range search on the FAISS index to find all pairs
    of embeddings whose cosine similarity is at least ``threshold``.  It
    returns pairs (path_i, path_j, similarity) sorted by descending similarity.

    Args:
        embeddings: Tensor of shape (n_files, dim), normalised rows.
        file_paths: Corresponding list of file paths.
        threshold: Cosine similarity threshold (0–1) for reporting clones.
        index: A FAISS index containing all embeddings.

    Returns:
        A list of tuples (path_i, path_j, similarity).  Self‑pairs and
        duplicate (i,j) vs (j,i) pairs are filtered out.
    """
    n = embeddings.size(0)
    emb_np = embeddings.detach().cpu().numpy().astype('float32')
    # Perform range search on all vectors with the given threshold
    # FAISS range search returns arrays: lims, distances, labels
    # such that for vector i, neighbours are labels[lims[i]:lims[i+1]]
    # with distances[lims[i]:lims[i+1]].  Distances are inner products (cos sim).
    # We search on the index with the same vectors; this will return self‑match
    # with distance 1.0, which we need to filter out.
    # Note: range_search might be expensive for very large datasets; for
    # mid‑sized codebases it works fine.
    # Build query index for search
    # The index has been built with all embeddings; we search all embeddings.
    # threshold must be float32 for faiss API
    radius = float(threshold)
    lims, distances, labels = index.range_search(emb_np, radius)
    similar_pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        start = lims[i]
        end = lims[i + 1]
        # Iterate over neighbours of i
        for j_idx in range(start, end):
            j = labels[j_idx]
            sim = distances[j_idx]
            # Skip self match and ensure i < j to avoid duplicates
            if j == i:
                continue
            if i < j:
                similar_pairs.append((file_paths[i], file_paths[j], float(sim)))
            else:
                # When i > j, the pair will be captured when we process j
                continue
    # Sort by descending similarity
    similar_pairs.sort(key=lambda tup: tup[2], reverse=True)
    return similar_pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic code clone detection over a directory with FAISS indexing",
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Path to the root directory containing source files",
    )
    parser.add_argument(
        "--model",
        default="mchochlov/codebert-base-cd-ft",
        help="Name of the sentence-transformers model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold (0–1) for reporting clones",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="List of file extensions to include (e.g. .py .java). If omitted, all files are considered.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for embedding files",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        parser.error(f"{root} is not a valid directory")

    files = load_files(root, args.extensions)
    if not files:
        print(
            f"No files found in {root} matching extensions {args.extensions}",
            file=sys.stderr,
        )
        return
    print(f"Found {len(files)} files to analyse. Loading model {args.model}…")

    # Ensure dependencies are available
    if SentenceTransformer is None or torch is None:
        print(
            "Error: This script requires the 'sentence_transformers' and 'torch' packages. "
            "Please install them in your Python environment (e.g., via 'pip install sentence-transformers torch').",
            file=sys.stderr,
        )
        sys.exit(1)
    if faiss is None:
        print(
            "Error: This script requires the 'faiss-cpu' package. "
            "Please install it in your Python environment (e.g., via 'pip install faiss-cpu').",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load model
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(args.model, device=device)
    except Exception as exc:
        print(f"Error loading model {args.model}: {exc}", file=sys.stderr)
        sys.exit(1)

    # Embed files
    embeddings = embed_files(files, model, batch_size=args.batch_size)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Compute similarity and report pairs
    pairs = compute_similar_pairs(embeddings, files, args.threshold, index)
    if not pairs:
        print(f"No pairs exceeded the similarity threshold {args.threshold}")
        return
    print(f"\nDetected {len(pairs)} candidate clone pairs (threshold={args.threshold}):\n")
    for sim_path1, sim_path2, sim in pairs:
        print(f"{sim:.4f}\t{sim_path1}\t{sim_path2}")


if __name__ == '__main__':
    main()