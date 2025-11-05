#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute pairwise duplication metrics from a jscpd JSON report and evaluate
'correct' vs 'wrong' detections by a same-prefix (e.g., C03_*) heuristic.

Similarity per pair = average of the duplicated-line ratios on each side:
  0.5 * (dup_lines_A / total_lines_A + dup_lines_B / total_lines_B)

Usage (example):
  python jscpd_metrics_calc.py -t 60% --sort -i result-jscpd/jscpd-report.json
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

def parse_threshold(s: str) -> float:
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    try:
        v = float(s)
        # Accept either 0..1 or 0..100
        return v / 100.0 if v > 1 else v
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold: {s}")

def group_key(path: str):
    """
    Extract group 'prefix' used to decide correct/wrong detection.
    For names like 'C03_gcd_type1_a.py' -> 'C03'
    """
    base = os.path.basename(path)
    m = re.match(r'([A-Za-z]\d+)_', base)
    if m:
        return m.group(1)
    # Fallback: everything before first underscore, if any
    if '_' in base:
        return base.split('_', 1)[0]
    return None  # unclassified

def load_totals_from_statistics(stats: dict) -> dict:
    """
    jscpd JSON provides totals under statistics.formats[format].sources
    We collect all formats (Python only in your case) into a single mapping:
      { absolute_path: total_lines }
    """
    totals = {}
    formats = stats.get("formats", {})
    # Newer jscpd versions nest per-format info; handle defensively.
    for _fmt, info in formats.items():
        sources = info.get("sources") or {}
        for name, meta in sources.items():
            # 'lines' is the total number of lines in that file
            lines = meta.get("lines")
            if isinstance(lines, int) and lines > 0:
                totals[name] = lines
    # Older fallbacks (rare):
    if not totals and "sources" in stats:
        for name, meta in stats["sources"].items():
            lines = meta.get("lines")
            if isinstance(lines, int) and lines > 0:
                totals[name] = lines
    return totals

def main():
    ap = argparse.ArgumentParser(description="Compute pairwise duplication metrics from jscpd JSON")
    ap.add_argument("-i", "--input", default="report/jscpd-report.json",
                    help="Path to jscpd JSON report (default: report/jscpd-report.json)")
    ap.add_argument("-t", "--threshold", type=parse_threshold, default=0.6,
                    help="Threshold for pairwise similarity (e.g., '60%%' or '0.6'). Default: 60%%")
    ap.add_argument("--sort", action="store_true", help="Sort pairs by decreasing similarity")
    args = ap.parse_args()

    # Load jscpd JSON
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Cannot open {args.input}", file=sys.stderr)
        sys.exit(2)

    duplicates = data.get("duplicates", [])
    statistics = data.get("statistics", {})
    total_lines = load_totals_from_statistics(statistics)

    # Build list of *all* scanned files (for 'gold positives')
    all_files = sorted(total_lines.keys())

    # Accumulate per-pair sets of duplicated line numbers (de-duplicated within a file)
    # Also remember longest fragment length ('lines') per pair.
    pair_sets = defaultdict(lambda: { "A": set(), "B": set(), "longest": 0, "files": None })
    for dup in duplicates:
        f1 = dup.get("firstFile", {}).get("name")
        f2 = dup.get("secondFile", {}).get("name")
        if not f1 or not f2 or f1 == f2:
            continue

        # Normalize pair key (sorted by name so A/B are stable)
        a, b = sorted([f1, f2])
        key = (a, b)

        # Extract fragment line bounds (inclusive). jscpd gives 'start'/'end'.
        s1 = dup.get("firstFile", {}).get("start")
        e1 = dup.get("firstFile", {}).get("end")
        s2 = dup.get("secondFile", {}).get("start")
        e2 = dup.get("secondFile", {}).get("end")

        # Defensive: fall back to 'lines' if end not present
        frag_len = dup.get("lines")
        if frag_len is None and all(isinstance(x, int) for x in (s1, e1)):
            frag_len = max(0, int(e1) - int(s1) + 1)
        if isinstance(frag_len, int):
            pair_sets[key]["longest"] = max(pair_sets[key]["longest"], frag_len)

        # Prepare A/B mapping aligned with (a,b)
        # Clamp ranges to known totals if available
        def clamp_range(start, end, max_lines):
            if not (isinstance(start, int) and isinstance(end, int)):
                return set()
            lo = max(1, start)
            hi = end if max_lines is None else min(end, max_lines)
            if hi < lo:
                return set()
            return set(range(lo, hi + 1))

        maxA = total_lines.get(a)
        maxB = total_lines.get(b)

        # Map the reported fragments to the normalized (a,b)
        if f1 == a:
            linesA = clamp_range(s1, e1, maxA)
            linesB = clamp_range(s2, e2, maxB)
        else:
            linesA = clamp_range(s2, e2, maxA)
            linesB = clamp_range(s1, e1, maxB)

        pair_sets[key]["A"].update(linesA)
        pair_sets[key]["B"].update(linesB)
        pair_sets[key]["files"] = (a, b)

    # Prepare results per pair: duplicated-line counts, ratios, symmetric %,
    # longest fragment (lines), and total overlap (A+B unique duplicated lines).
    rows = []
    for (a, b), payload in pair_sets.items():
        dupA = len(payload["A"])
        dupB = len(payload["B"])
        totA = total_lines.get(a, 0)
        totB = total_lines.get(b, 0)
        if totA == 0 or totB == 0:
            continue
        rA = dupA / totA
        rB = dupB / totB
        sym = 0.5 * (rA + rB)  # our "pairwise similarity"
        rows.append({
            "pair": (a, b),
            "dupA": dupA, "dupB": dupB,
            "totA": totA, "totB": totB,
            "rA": rA, "rB": rB,
            "sym": sym,
            "longest": payload["longest"],
            "total_overlap_lines": dupA + dupB
        })

    # Filter by threshold
    thr = args.threshold
    rows_thr = [r for r in rows if r["sym"] >= thr]

    # Sort if requested
    if args.sort:
        rows_thr.sort(key=lambda r: (r["sym"], r["longest"], r["total_overlap_lines"]), reverse=True)

    # Tally correct/wrong by same-prefix heuristic
    def same_group(a, b):
        ga, gb = group_key(a), group_key(b)
        if ga is None or gb is None:
            return None
        return ga == gb

    correct = wrong = unclassified = 0
    for r in rows_thr:
        a, b = r["pair"]
        sg = same_group(a, b)
        if sg is None:
            unclassified += 1
        elif sg:
            correct += 1
        else:
            wrong += 1

    # Build "gold positives" as all pairs from the scanned set that have the same prefix.
    files_by_group = defaultdict(list)
    for f in all_files:
        g = group_key(f)
        if g is not None:
            files_by_group[g].append(f)

    gold_pairs = set()
    for g, lst in files_by_group.items():
        if len(lst) >= 2:
            for a, b in combinations(sorted(lst), 2):
                gold_pairs.add((a, b))

    predicted_pairs = set((r["pair"][0], r["pair"][1]) for r in rows_thr)
    TP = len(predicted_pairs & gold_pairs)
    FP = len(predicted_pairs - gold_pairs)
    FN = len(gold_pairs - predicted_pairs)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Pretty print (mimic your JPlag/Dolos table columns/naming)
    hdr = f"{'Similarity':<10}  {'Left file':<60}  {'Right file':<60}  {'Longest':>7}  {'Total overlap':>13}"
    print(hdr)
    for r in rows_thr:
        a, b = r["pair"]
        sim_pct = 100.0 * r["sym"]
        print(f"{sim_pct:6.2f}%  {a:<60}  {b:<60}  {r['longest']:7d}  {r['total_overlap_lines']:13d}")

    thr_pct = 100.0 * thr
    print()
    print(f"Total pairs \u2265 {thr_pct:.2f}%: {len(rows_thr)}")
    print(f"Wrong-detection pairs (group mismatch) \u2265 {thr_pct:.2f}%: {wrong}")
    print(f"Correct-detection pairs (same group) \u2265 {thr_pct:.2f}%: {correct}")
    print(f"Unclassified (group missing) \u2265 {thr_pct:.2f}%: {unclassified}")
    print()
    print("=== Proxy evaluation (same-prefix heuristic) ===")
    print(f"Gold positives (same prefix) in input: {len(gold_pairs)}")
    print(f"TP: {TP}  FP: {FP}  FN: {FN}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

if __name__ == "__main__":
    main()
