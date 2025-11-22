#!/usr/bin/env python3
"""
jplag_metrics_calc.py
Reads JPlag --csv-export from stdin (or a file) and prints:
- A pretty table: "Similarity  Left file  Right file  Longest  Total overlap"
- Counts for Total / Wrong / Correct / Unclassified at the threshold
- Precision, Recall, F1 using a "same-prefix" gold.

What's new:
- Optional --group-regex to extract the group from the *basename* using a capturing group.
  Example (works for both 'C06_Wordcount...' and 'C06Wordcount...'):
      --group-regex '^(C\\d+)(?:[_-]|(?=[A-Z]))'
- If --group-regex is not provided, falls back to underscore-based --group-depth (as before).
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Tuple

# ---------- Utilities ----------

def parse_threshold(thr: str) -> float:
    """Accepts '60%', '0.6', or '60' and returns a decimal 0..1."""
    s = thr.strip()
    if s.endswith("%"):
        return float(s[:-1].strip()) / 100.0
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold: {thr!r}")
    return v if 0.0 <= v <= 1.0 else v / 100.0


def normalize_hdr(h: str) -> str:
    return re.sub(r"\s+", " ", h.strip().lower())


def find_col(headers: List[str], candidates: Iterable[str]) -> Optional[str]:
    """Case-insensitive, whitespace-normalized header matching."""
    norm = {normalize_hdr(h): h for h in headers}
    for cand in candidates:
        key = normalize_hdr(cand)
        if key in norm:
            return norm[key]
    # fuzzy: contains all words in order
    for want in candidates:
        w = normalize_hdr(want)
        w_words = w.split()
        for k, orig in norm.items():
            if all(word in k for word in w_words):
                return orig
    return None


def to_percent_value(s: str) -> float:
    """Parse '89.19%' or '0.8919' or '89.19' -> decimal in [0,1]."""
    s = str(s).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    v = float(s)
    return v if v <= 1.0 else v / 100.0


def group_prefix(path: str, depth: int = 1) -> Optional[str]:
    """
    Extract group prefix from a filename using underscores.
    Example: '/.../C03_gcd_type1_a.py' -> 'C03' (depth=1)
             depth=2 -> 'C03_gcd'
    Returns None if the stem is empty or depth > number of parts.
    """
    base = os.path.basename(path or "")
    name, _ = os.path.splitext(base)
    if not name:
        return None
    parts = name.split("_")
    if len(parts) < depth:
        return None
    return "_".join(parts[:depth])


# New: compile+extract via regex on the *basename*
def compile_group_regex(s: Optional[str]) -> Optional[re.Pattern]:
    if not s:
        return None
    try:
        rx = re.compile(s)
    except re.error as e:
        raise SystemExit(f"Invalid --group-regex: {e}")
    if rx.groups < 1:
        raise SystemExit("--group-regex must contain at least one capturing group; "
                         "e.g., '^(C\\d+)(?:[_-]|(?=[A-Z]))'")
    return rx


def extract_group_via_regex(path: str, rx: re.Pattern) -> Optional[str]:
    base = os.path.basename(path or "")
    m = rx.search(base)
    return m.group(1) if m else None


# ---------- Input parsing ----------

def parse_csv_text(text: str) -> List[Dict[str, str]]:
    """
    Parse CSV text robustly (comma/semicolon/tab). Returns list of dict rows.
    Raises ValueError if it doesn't look like CSV.
    """
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
        dialect.delimiter = ","
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    rows = [dict(r) for r in reader]
    if not rows or not reader.fieldnames:
        raise ValueError("No CSV rows found.")
    return rows


def parse_pretty_table_text(text: str) -> List[Dict[str, str]]:
    """
    Fallback: parse preformatted table lines like
    '100.00% /path/left  /path/right  19  38'.
    """
    rows: List[Dict[str, str]] = []
    for line in text.splitlines():
        line = line.rstrip()
        if not line or "Total pairs" in line or "Proxy evaluation" in line:
            continue
        if line.lower().startswith(("similarity", "milarity")):
            continue

        # Grab the trailing two integers (longest, total) if present
        tail = re.search(r"(\d+)\s+(\d+)\s*$", line)
        longest = total = ""
        if tail:
            longest, total = tail.group(1), tail.group(2)
            head = line[:tail.start()].rstrip()
        else:
            head = line

        m = re.match(r"^\s*(\d+(?:\.\d+)?)%\s+(.*)$", head)
        if not m:
            continue
        sim_str, rest = m.group(1), m.group(2).strip()

        # Split left/right by at least two spaces
        parts = re.split(r"\s{2,}", rest)
        if len(parts) < 2:
            parts = rest.split(None, 1)
            if len(parts) < 2:
                continue

        left, right = parts[0].strip(), parts[1].strip()
        rows.append(
            {
                "Similarity": f"{sim_str}%",
                "Left file": left,
                "Right file": right,
                "Longest": longest,
                "Total overlap": total,
            }
        )
    if not rows:
        raise ValueError("Not a recognizable JPlag table.")
    return rows


def load_rows(stdin_text: str) -> List[Dict[str, str]]:
    """Try CSV first (JPlag --csv-export), then fallback to pretty-table parsing."""
    try:
        return parse_csv_text(stdin_text)
    except Exception:
        return parse_pretty_table_text(stdin_text)


# ---------- Core evaluation ----------

def evaluate(
    rows: List[Dict[str, str]],
    threshold: float,
    group_depth: int,
    sort_rows: bool,
    group_rx: Optional[re.Pattern],
):
    # Map headers
    headers = list(rows[0].keys())

    sim_col = find_col(
        headers,
        ["similarity", "avg similarity", "average similarity", "percentage", "%"],
    )
    left_col = find_col(headers, ["left file", "left", "file1", "submission 1"])
    right_col = find_col(headers, ["right file", "right", "file2", "submission 2"])
    long_col = find_col(
        headers,
        ["longest", "longest match", "number of tokens in the longest match"],
    )
    total_col = find_col(
        headers,
        ["total overlap", "matched tokens", "total", "total matched tokens"],
    )

    missing = [name for name, col in {
        "Similarity": sim_col, "Left file": left_col, "Right file": right_col
    }.items() if col is None]
    if missing:
        raise SystemExit(f"ERROR: Missing required columns from JPlag CSV: {', '.join(missing)}")

    # Normalize rows
    norm_rows: List[Dict[str, object]] = []
    for r in rows:
        try:
            sim = to_percent_value(r.get(sim_col, "0"))
        except Exception:
            continue
        left = str(r.get(left_col, "")).strip()
        right = str(r.get(right_col, "")).strip()
        longest = r.get(long_col, "")
        total = r.get(total_col, "")
        norm_rows.append(
            {
                "similarity": sim,  # decimal [0,1]
                "left": left,
                "right": right,
                "longest": (int(longest) if str(longest).strip().isdigit() else None),
                "total": (int(total) if str(total).strip().isdigit() else None),
            }
        )

    # Optionally sort by similarity desc
    if sort_rows:
        norm_rows.sort(key=lambda d: d["similarity"], reverse=True)

    # Compute groups and flags
    for d in norm_rows:
        if group_rx is not None:
            g1 = extract_group_via_regex(d["left"], group_rx)
            g2 = extract_group_via_regex(d["right"], group_rx)
        else:
            g1 = group_prefix(d["left"], depth=group_depth)
            g2 = group_prefix(d["right"], depth=group_depth)
        d["group1"] = g1
        d["group2"] = g2
        d["same_group"] = (g1 is not None) and (g1 == g2)
        d["unclassified"] = (g1 is None) or (g2 is None)
        d["pred_dup"] = d["similarity"] >= threshold

    # Confusion elements on "same_group" as the gold
    tp = sum(1 for d in norm_rows if d["pred_dup"] and d["same_group"])
    fp = sum(1 for d in norm_rows if d["pred_dup"] and not d["same_group"] and not d["unclassified"])
    fn = sum(1 for d in norm_rows if (not d["pred_dup"]) and d["same_group"])
    tn = sum(1 for d in norm_rows if (not d["pred_dup"]) and (not d["same_group"]) and not d["unclassified"])

    # Above-threshold breakdown
    above = [d for d in norm_rows if d["pred_dup"]]
    wrong = [d for d in above if not d["same_group"] and not d["unclassified"]]
    correct = [d for d in above if d["same_group"]]
    unclassified = [d for d in above if d["unclassified"]]

    # Metrics
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "rows": norm_rows,
        "above": above,
        "wrong": wrong,
        "correct": correct,
        "unclassified": unclassified,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fmt_percent(x: float, places: int = 2) -> str:
    return f"{x*100:.{places}f}%"


def print_table(above: List[Dict[str, object]]):
    if not above:
        print("No pairs at or above the threshold.")
        return

    # Column widths
    left_w = max(len("Left file"), *(len(str(d["left"])) for d in above))
    right_w = max(len("Right file"), *(len(str(d["right"])) for d in above))
    long_w = max(len("Longest"), *(len(str(d["longest"])) if d["longest"] is not None else 1 for d in above))
    total_w = max(len("Total overlap"), *(len(str(d["total"])) if d["total"] is not None else 1 for d in above))

    # Header
    print(f"{'Similarity':>10}  {'Left file':<{left_w}}  {'Right file':<{right_w}}  {'Longest':>{long_w}}  {'Total overlap':>{total_w}}")

    # Rows
    for d in above:
        sim = fmt_percent(d["similarity"])
        left = str(d["left"])
        right = str(d["right"])
        longest = "" if d["longest"] is None else str(d["longest"])
        total = "" if d["total"] is None else str(d["total"])
        print(f"{sim:>10}  {left:<{left_w}}  {right:<{right_w}}  {longest:>{long_w}}  {total:>{total_w}}")


def main():
    ap = argparse.ArgumentParser(
        description="Pretty-print JPlag --csv-export and compute Correct/Wrong detections + Precision/Recall/F1."
    )
    ap.add_argument(
        "-t", "--threshold", type=parse_threshold, default="60%",
        help="Similarity threshold (e.g., 60%%, 0.6, or 60). Default: 60%%"
    )
    ap.add_argument(
        "--sort", action="store_true",
        help="Sort pairs by similarity descending before printing (recommended)."
    )
    ap.add_argument(
        "-g", "--group-depth", type=int, default=1,
        help="Underscore-based prefix depth used as 'group' if --group-regex is NOT set (1 -> C03, 2 -> C03_gcd). Default: 1"
    )
    ap.add_argument(
        "--group-regex", type=str, default=None,
        help=("Regex with ONE capturing group to extract the group from the basename. "
              "Example: '^(C\\d+)(?:[_-]|(?=[A-Z]))'")
    )
    ap.add_argument(
        "-i", "--input", type=str, default=None,
        help="Optional path to CSV/table; if omitted, read from STDIN."
    )
    args = ap.parse_args()

    # Read input
    if args.input:
        with open(args.input, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = sys.stdin.read()
        if not text.strip():
            ap.error("No input on STDIN. Pipe JPlag like:  jplag ... --csv-export | python jplag_metrics_calc.py -t 60% --sort")

    # Compile optional group regex
    group_rx = compile_group_regex(args.group_regex)

    # Load and evaluate
    rows = load_rows(text)
    res = evaluate(rows, threshold=args.threshold, group_depth=args.group_depth, sort_rows=args.sort, group_rx=group_rx)

    # Output table (only ≥ threshold)
    print_table(res["above"])
    print()

    # Summary
    print(f"Total pairs ≥ {fmt_percent(args.threshold)}: {len(res['above'])}")
    print(f"Wrong-detection pairs (group mismatch) ≥ {fmt_percent(args.threshold)}: {len(res['wrong'])}")
    print(f"Correct-detection pairs (same group) ≥ {fmt_percent(args.threshold)}: {len(res['correct'])}")
    print(f"Unclassified (group missing) ≥ {fmt_percent(args.threshold)}: {len(res['unclassified'])}")
    print()

    # Gold positives across the entire input = same-prefix pairs (any similarity)
    gold_pos = sum(1 for d in res["rows"] if d["same_group"])
    print("=== Proxy evaluation (same-prefix heuristic) ===")
    print(f"Gold positives (same prefix) in input: {gold_pos}")
    print(f"TP: {res['tp']}  FP: {res['fp']}  FN: {res['fn']}")
    print(f"Precision: {res['precision']:.4f}  Recall: {res['recall']:.4f}  F1: {res['f1']:.4f}")


if __name__ == "__main__":
    main()
