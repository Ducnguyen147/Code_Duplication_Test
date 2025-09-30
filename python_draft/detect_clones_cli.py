#!/usr/bin/env python3
"""
CLI utility for semantic code clone detection across a directory of source files.

This script builds upon the simple pairwise comparison example provided by the user
and extends it to operate over an entire codebase.  Instead of comparing just two
files, it will recursively scan a given directory, embed each file's contents
using a pre-trained code model, and then compute semantic similarity scores
between all pairs of files.  Files whose cosine similarity exceeds a user-
specified threshold are reported as potential code clones.

Key features:

* Supports any HuggingFace or sentence-transformers model that produces code
  embeddings.  By default it uses the "mchochlov/codebert-base-cd-ft" model
  fine-tuned for clone detection, but you can specify any other model from
  the provided list (or your own) via the ``--model`` flag.
* Recursively walks a directory to find source files.  You can restrict which
  files are considered by specifying a list of extensions (e.g. ``.py .java``).
  If no extensions are provided, all regular files are embedded.
* Embeddings are computed once per file and normalised to unit length.  The
  similarity between two files is then just the dot product of their vectors.
* Reports all pairs of files whose similarity exceeds the threshold.  Results
  are sorted by descending similarity so the most similar pairs appear first.

This tool is suitable for small- to medium-sized codebases (hundreds of files).
For very large repositories, the quadratic number of pairwise comparisons may
become expensive.  In such cases you can integrate an approximate nearest
neighbour (ANN) index such as FAISS or Annoy instead of the brute-force matrix
computation performed here.  However, for most master thesis experiments and
small projects this straightforward approach is sufficient and easier to
understand.

Example usage::

    python detect_clones_cli.py --dir path/to/project --model mchochlov/codebert-base-cd-ft \
        --extensions .py .java .js --threshold 0.85

The above command will scan all Python, Java and JavaScript files under
``path/to/project``, compute semantic embeddings for each, and then report
pairs whose cosine similarity is at least 0.85.
"""

import argparse
import os
import sys
from typing import List, Tuple

try:
    import torch  # type: ignore
except ImportError as exc:
    # Provide a clear message if torch is not installed.  We defer raising
    # until later so that --help and basic parsing still work.
    torch = None  # type: ignore
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None  # type: ignore


def load_files(directory: str, extensions: List[str] | None) -> List[str]:
    """Recursively collect file paths from ``directory``.

    If ``extensions`` is provided, only files with one of the specified
    extensions (case-sensitive) will be included.  Extensions should include
    the leading dot, e.g. ``.py`` or ``.java``.

    Returns:
        A list of absolute file paths.
    """
    collected = []
    for root, _dirs, files in os.walk(directory):
        for fname in files:
            # Skip hidden files (e.g. .gitignore) and directories
            if fname.startswith('.'):
                continue
            if extensions:
                if not any(fname.endswith(ext) for ext in extensions):
                    continue
            path = os.path.join(root, fname)
            if os.path.isfile(path):
                collected.append(path)
    return sorted(collected)


def embed_files(file_paths: List[str], model: SentenceTransformer, batch_size: int = 32) -> torch.Tensor:
    """Compute embeddings for a list of files using ``model``.

    The returned tensor has shape (num_files, embed_dim).  Each row
    corresponds to the embedding of the file at the same index in
    ``file_paths``.  The embeddings are normalised to unit length.

    ``batch_size`` controls how many files are encoded at once.  Larger
    batches will generally be faster on a GPU but require more memory.
    """
    embeddings = []
    texts = []
    # Read file contents first.  We read separately from encoding to
    # minimise disk IO overhead during model calls.
    for path in file_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
        except Exception as exc:
            print(f"Warning: could not read {path}: {exc}", file=sys.stderr)
            texts.append("")  # fallback to empty string
    # Encode in batches
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        # Convert to tensor; truncation prevents extremely long files from
        # exceeding the model's maximum input length.
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_tensor=True,
            truncation=True,
        )
        embeddings.append(batch_embeddings)
    # Concatenate into a single tensor of shape (n_files, dim)
    all_embeddings = torch.cat(embeddings, dim=0)
    # Normalise so cosine similarity can be computed via dot product
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    return all_embeddings


def compute_similar_pairs(embeddings: torch.Tensor, file_paths: List[str], threshold: float) -> List[Tuple[str, str, float]]:
    """Compute similarity scores and return pairs exceeding the threshold.

    This function computes the cosine similarity between all unique pairs of
    embeddings (i < j) and returns those pairs whose similarity is at least
    ``threshold``.

    Args:
        embeddings: Tensor of shape (n_files, dim), normalised rows.
        file_paths: Corresponding list of file paths.
        threshold: Only pairs with similarity >= threshold are kept.

    Returns:
        A list of tuples (path_i, path_j, similarity).  The list is sorted
        by descending similarity so the most similar pairs are first.
    """
    n = embeddings.size(0)
    # Pre-compute similarity matrix using matrix multiplication.  Since
    # embeddings are normalised, the dot product is the cosine similarity.
    sim_matrix = embeddings @ embeddings.T
    # Extract upper triangular part without diagonal
    similar_pairs: List[Tuple[str, str, float]] = []
    for i in range(n):
        # Only consider j > i to avoid duplicate pairs and self-comparison
        sims = sim_matrix[i, i + 1:]
        for offset, sim in enumerate(sims):
            if float(sim) >= threshold:
                j = i + 1 + offset
                similar_pairs.append((file_paths[i], file_paths[j], float(sim)))
    # Sort results by descending similarity
    similar_pairs.sort(key=lambda tup: tup[2], reverse=True)
    return similar_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic code clone detection over a directory")
    parser.add_argument("--dir", required=True, help="Path to the root directory containing source files")
    parser.add_argument("--model", default="mchochlov/codebert-base-cd-ft", help="Name of the sentence-transformers model")
    parser.add_argument("--threshold", type=float, default=0.9, help="Cosine similarity threshold (0-1) for reporting clones")
    parser.add_argument("--extensions", nargs="*", default=None, help="List of file extensions to include (e.g. .py .java)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding files")
    args = parser.parse_args()

    # Validate directory
    root = os.path.abspath(args.dir)
    if not os.path.isdir(root):
        parser.error(f"{root} is not a valid directory")

    # Collect files
    files = load_files(root, args.extensions)
    if not files:
        print(f"No files found in {root} matching extensions {args.extensions}")
        return
    print(f"Found {len(files)} files to analyse. Loading model {args.model}…")

    # Ensure dependencies are available
    if SentenceTransformer is None or torch is None:
        print(
            "Error: This script requires the 'sentence_transformers' and 'torch' packages. "
            "Please install them in your Python environment (e.g., via 'pip install sentence_transformers').",
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

    # Compute similarity and report pairs
    pairs = compute_similar_pairs(embeddings, files, args.threshold)
    if not pairs:
        print(f"No pairs exceeded the similarity threshold {args.threshold}")
        return
    print(f"\nDetected {len(pairs)} candidate clone pairs (threshold={args.threshold}):\n")
    for sim_path1, sim_path2, sim in pairs:
        print(f"{sim:.4f}\t{sim_path1}\t{sim_path2}")


if __name__ == '__main__':
    main()