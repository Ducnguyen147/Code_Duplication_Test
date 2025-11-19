#!/usr/bin/env python3
"""
Parse Dolos *terminal* output, list pairs >= threshold, report wrong detections,
and (automatically) compute Precision/Recall/F1 using a same-prefix heuristic.

What's new (metrics, zero extra flags):
- Uses a proxy gold: pairs are POSITIVE iff both filenames share the same group
  captured by --group-regex (default: ^(C\\d+)_). Otherwise NEGATIVE.
- At your chosen threshold, computes Precision, Recall, F1 and prints TP/FP/FN.

Usage (unchanged):
  dolos run -l python -k 12 -w 17 mix.zip \
    | python dolos_stdout_filter.py -t 60% --sort

Notes:
- "Wrong detection" is counted only when BOTH files yield a group and they differ.
  If a group is missing for either file, the pair is counted as "unclassified" by default.
  Use --treat-missing-group-as-wrong to include those in the wrong-detection count.
- Metrics use the same-prefix proxy gold for *all* parsed pairs from stdin.
  Predicted positives are those with similarity >= threshold.
"""

import sys, re, argparse, os
from dataclasses import dataclass
from typing import Iterable, Optional, Pattern, Tuple, List, Set

@dataclass
class Pair:
    left: str
    right: str
    similarity: float  # 0..1
    longest: int
    overlap: int

# Match any line that ends with three numeric fields: sim (float), longest (int), overlap (int)
ROW_HEAD_RE = re.compile(
    r"^(?P<left_right>.*?)\s+(?P<sim>\d+(?:\.\d+)?)\s+(?P<long>\d+)\s+(?P<overlap>\d+)\s*$"
)
# Split the two file columns by 2+ spaces (table column gap)
COL_SPLIT_RE = re.compile(r"\s{2,}")

def parse_threshold(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    v = float(s)
    return v / 100.0 if v > 1 else v

def parse_stream(lines: Iterable[str]) -> Iterable[Pair]:
    """
    Yield Pair objects from Dolos terminal output.
    Rows arrive as a 'head' line (with numbers) plus 0..N continuation lines
    that contain only the two file columns (no numbers).
    """
    cur_left = ""
    cur_right = ""
    cur_metrics = None  # (similarity, longest, overlap)

    for raw in lines:
        line = raw.rstrip("\n")
        m = ROW_HEAD_RE.match(line)
        if m:
            # If a previous row was open, flush it before starting a new one
            if cur_metrics is not None:
                sim, longest, overlap = cur_metrics
                yield Pair(cur_left.strip(), cur_right.strip(), sim, longest, overlap)
                cur_left = cur_right = ""
                cur_metrics = None

            lr = m.group("left_right").rstrip()
            parts = COL_SPLIT_RE.split(lr.strip())
            if parts:
                if len(parts) >= 1:
                    cur_left += parts[0].strip()
                if len(parts) >= 2:
                    cur_right += parts[1].strip()
            cur_metrics = (
                float(m.group("sim")),
                int(m.group("long")),
                int(m.group("overlap")),
            )
        else:
            # Continuation of file path columns?
            if cur_metrics is None:
                continue  # still in header or noise
            if not line.strip():
                continue
            parts = COL_SPLIT_RE.split(line.strip())
            if parts:
                if len(parts) >= 1:
                    cur_left += parts[0].strip()
                if len(parts) >= 2:
                    cur_right += parts[1].strip()

    # Flush last row
    if cur_metrics is not None:
        sim, longest, overlap = cur_metrics
        yield Pair(cur_left.strip(), cur_right.strip(), sim, longest, overlap)

def compile_group_regex(s: str) -> Pattern:
    """
    Compile the user-supplied group regex. It must have at least one capturing group.
    The first capturing group is used as the "group id".
    """
    try:
        rx = re.compile(s)
    except re.error as e:
        raise SystemExit(f"Invalid --group-regex: {e}")
    # sanity check: ensure it has a capturing group
    if rx.groups < 1:
        raise SystemExit("--group-regex must contain at least one capturing group, e.g. '^(C\\d+)_'")
    return rx

def extract_group(path: str, rx: Pattern) -> Optional[str]:
    """
    Extract group from the *basename* using the first capturing group of rx.
    Returns None if not found.
    """
    name = os.path.basename(path)
    m = rx.search(name)
    return m.group(1) if m else None

def classify_pairs(
    pairs: List[Pair],
    rx: Pattern,
    treat_missing_as_wrong: bool = False,
) -> Tuple[List[Pair], List[Pair], List[Pair]]:
    """
    Returns (wrong, correct, unclassified).
    wrong: both groups present AND different, OR (missing & treat_missing_as_wrong=True)
    correct: both groups present AND equal
    unclassified: at least one group missing (unless treat_missing_as_wrong)
    """
    wrong, correct, unclassified = [], [], []
    for p in pairs:
        g1 = extract_group(p.left, rx)
        g2 = extract_group(p.right, rx)
        if g1 is None or g2 is None:
            if treat_missing_as_wrong:
                wrong.append(p)
            else:
                unclassified.append(p)
        elif g1 != g2:
            wrong.append(p)
        else:
            correct.append(p)
    return wrong, correct, unclassified

# ---- Metrics helpers (same-prefix proxy gold) ----

def _pair_key(a: str, b: str) -> Tuple[str, str]:
    """Order-independent key for a pair of paths."""
    return (a, b) if a <= b else (b, a)

def compute_prf1_from_groups(
    all_pairs: List[Pair],
    thr: float,
    group_rx: Pattern,
) -> Tuple[int, int, int, float, float, float, int]:
    """
    Build proxy gold from groups, then compute PR/F1 at threshold:
      - Gold positive = both groups present AND equal
      - Gold negative = otherwise (different or missing)
      - Predicted positive = similarity >= thr
    Returns (tp, fp, fn, precision, recall, f1, gold_pos_count)
    """
    # Gold positives across ALL parsed pairs (proxy by group)
    gold_pos: Set[Tuple[str, str]] = set()
    for p in all_pairs:
        g1 = extract_group(p.left, group_rx)
        g2 = extract_group(p.right, group_rx)
        if g1 is not None and g2 is not None and g1 == g2:
            gold_pos.add(_pair_key(p.left, p.right))

    # Predictions at threshold
    pred: Set[Tuple[str, str]] = {
        _pair_key(p.left, p.right) for p in all_pairs if p.similarity >= thr
    }

    tp = sum(1 for k in pred if k in gold_pos)
    fp = len(pred) - tp
    fn = len(gold_pos) - tp

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1, len(gold_pos)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--threshold", required=True, help="Similarity threshold (e.g. 60% or 0.6)")
    ap.add_argument("--sort", action="store_true", help="Sort by similarity desc")
    ap.add_argument("--basename", action="store_true", help="Show only base filenames")
    ap.add_argument("--group-regex", default=r"^(C\d+)_",
                    help=r"Regex with ONE capturing group that defines 'group' from basename (default: '^(C\\d+)_')")
    ap.add_argument("--list-wrong", action="store_true",
                    help="After the main table, print a section listing wrong-detection pairs")
    ap.add_argument("--treat-missing-group-as-wrong", action="store_true",
                    help="If either file lacks a detectable group, count that pair as WRONG")
    args = ap.parse_args()

    thr = parse_threshold(args.threshold)
    group_rx = compile_group_regex(args.group_regex)

    # Parse ALL pairs from stdin (needed for recall), but display only >= thr
    all_pairs = list(parse_stream(sys.stdin))
    display_pairs = [p for p in all_pairs if p.similarity >= thr]

    # Optional sorting (for display only)
    if args.sort:
        display_pairs.sort(key=lambda p: p.similarity, reverse=True)

    # Pretty print the main table
    print("Similarity\tLeft file\tRight file\tLongest\tTotal overlap")
    for p in display_pairs:
        left = os.path.basename(p.left) if args.basename else p.left
        right = os.path.basename(p.right) if args.basename else p.right
        print(f"{p.similarity*100:.2f}%\t{left}\t{right}\t{p.longest}\t{p.overlap}")

    # Summary lines for wrong/correct/unclassified among displayed pairs
    total_pairs = len(display_pairs)
    if total_pairs == 0:
        print(f"\nNo pairs \u2265 {thr*100:.2f}%.")
        print(f"Total pairs \u2265 {thr*100:.2f}%: 0")
        print(f"Wrong-detection pairs \u2265 {thr*100:.2f}%: 0")
    else:
        wrong, correct, unclassified = classify_pairs(
            display_pairs,
            group_rx,
            treat_missing_as_wrong=args.treat_missing_group_as_wrong,
        )
        print(f"\nTotal pairs \u2265 {thr*100:.2f}%: {total_pairs}")
        print(f"Wrong-detection pairs (group mismatch) \u2265 {thr*100:.2f}%: {len(wrong)}")
        print(f"Correct-detection pairs (same group) \u2265 {thr*100:.2f}%: {len(correct)}")
        if not args.treat_missing_group_as_wrong:
            print(f"Unclassified (group missing) \u2265 {thr*100:.2f}%: {len(unclassified)}")
        if args.list_wrong and wrong:
            print("\n--- Wrong-detection pairs ---")
            print("Similarity\tLeft file\tRight file\tLongest\tTotal overlap")
            for p in wrong:
                left = os.path.basename(p.left) if args.basename else p.left
                right = os.path.basename(p.right) if args.basename else p.right
                print(f"{p.similarity*100:.2f}%\t{left}\t{right}\t{p.longest}\t{p.overlap}")

    # ---- Always-on metrics (Precision/Recall/F1) using same-prefix proxy gold ----
    tp, fp, fn, prec, rec, f1, gold_n = compute_prf1_from_groups(all_pairs, thr, group_rx)
    print("\n=== Proxy evaluation (same-prefix heuristic) ===")
    print(f"Gold positives (same prefix) in input: {gold_n}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

if __name__ == "__main__":
    main()
