#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute pairwise duplication metrics from a jscpd JSON report and evaluate
'correct' vs 'wrong' detections by a same-prefix heuristic.

Similarity per pair = average of the duplicated-line ratios on each side:
  0.5 * (dup_lines_A / total_lines_A + dup_lines_B / total_lines_B)

Usage (examples):
  # Underscore-based grouping (fallback): first token before '_'
  python jscpd_metrics_calc.py -t 60% --sort -i report/jscpd-report.json

  # Regex grouping (one capturing group on basename; handles CamelCase or underscore)
  python jscpd_metrics_calc.py -t 60% --sort \
    --group-regex '^(C\\d+)(?:[_-]|(?=[A-Z]))' \
    -i report/jscpd-report.json
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, Optional, Tuple, List

# ------------------------- CLI helpers -------------------------

def parse_threshold(s: str) -> float:
    s = s.strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    try:
        v = float(s)
        return v / 100.0 if v > 1 else v  # accept 0..1 or 0..100
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid threshold: {s}")

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

# ------------------------- grouping -------------------------

def make_group_extractor(group_regex: Optional[re.Pattern]):
    """
    Returns a function path->group (or None). If regex is provided, it is applied
    to the *basename* and its FIRST capturing group is returned on match.
    Otherwise we fall back to the old underscore-based behavior.
    """
    def extract(path: str) -> Optional[str]:
        base = os.path.basename(path or "")
        if group_regex is not None:
            m = group_regex.search(base)
            if m:
                return m.group(1)
        # Fallbacks (underscore style)
        if '_' in base:
            return base.split('_', 1)[0]
        # Last-resort: a leading letter+digits token (e.g., "C06Wordcount..." -> "C06")
        m2 = re.match(r'([A-Za-z]\d+)', base)
        return m2.group(1) if m2 else None
    return extract

# ------------------------- JSCPD JSON parsing -------------------------

def load_totals_from_statistics(stats: dict) -> Dict[str, int]:
    """
    jscpd JSON provides totals under statistics.formats[format].sources
      -> { "path or name": { "lines": <int>, ...}, ... }
    Return a mapping { absolute_or_relative_path: total_lines }.
    """
    totals: Dict[str, int] = {}
    formats = stats.get("formats", {})
    for _fmt, info in (formats or {}).items():
        sources = (info or {}).get("sources") or {}
        for name, meta in (sources or {}).items():
            lines = (meta or {}).get("lines")
            if isinstance(lines, int) and lines > 0:
                totals[name] = int(lines)
    # Older fallback
    if not totals and "sources" in stats:
        for name, meta in (stats.get("sources") or {}).items():
            lines = (meta or {}).get("lines")
            if isinstance(lines, int) and lines > 0:
                totals[name] = int(lines)
    return totals

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute pairwise duplication metrics from jscpd JSON")
    ap.add_argument("-i", "--input", default="report/jscpd-report.json",
                    help="Path to jscpd JSON report (default: report/jscpd-report.json)")
    ap.add_argument("-t", "--threshold", type=parse_threshold, default=0.6,
                    help="Threshold for pairwise similarity (e.g., '60%%' or '0.6'). Default: 60%%")
    ap.add_argument("--sort", action="store_true", help="Sort pairs by decreasing similarity")
    ap.add_argument("--group-regex", type=str, default=None,
                    help="Regex with ONE capturing group applied to basename to extract the grouping label; "
                         "e.g., '^(C\\d+)(?:[_-]|(?=[A-Z]))'")
    args = ap.parse_args()

    # Load jscpd JSON
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Cannot open {args.input}", file=sys.stderr)
        sys.exit(2)

    duplicates = data.get("duplicates") or data.get("clones") or data.get("matches") or []
    statistics = data.get("statistics", {})
    total_lines = load_totals_from_statistics(statistics)

    # Build list of *all* files seen: union of statistics sources and duplicates.
    all_files_set = set(total_lines.keys())
    # duplicates may use keys "firstFile"/"secondFile" or "first"/"second"
    def _get(d, *ks):
        for k in ks:
            if isinstance(d, dict) and k in d and d[k] is not None:
                return d[k]
        return None

    for dup in duplicates:
        a = _get(dup, "firstFile", "first") or {}
        b = _get(dup, "secondFile", "second") or {}
        f1 = _get(a, "name", "path", "file")
        f2 = _get(b, "name", "path", "file")
        if f1: all_files_set.add(f1)
        if f2: all_files_set.add(f2)

    all_files = sorted(all_files_set)

    # Accumulate per-pair sets of duplicated line numbers (de-duplicated within a file)
    # Also remember longest fragment length per pair (in lines).
    pair_sets: Dict[Tuple[str, str], dict] = defaultdict(lambda: { "A": set(), "B": set(), "longest": 0, "files": None })

    for dup in duplicates:
        a = _get(dup, "firstFile", "first") or {}
        b = _get(dup, "secondFile", "second") or {}
        f1 = _get(a, "name", "path", "file")
        f2 = _get(b, "name", "path", "file")
        if not f1 or not f2 or f1 == f2:
            continue

        # Normalize pair key (sorted by name so A/B are stable)
        a_path, b_path = sorted([f1, f2])
        key = (a_path, b_path)

        s1 = _get(a, "start", "startLine", "startLineNumber", "startLineIndex")
        e1 = _get(a, "end", "endLine", "endLineNumber", "endLineIndex")
        s2 = _get(b, "start", "startLine", "startLineNumber", "startLineIndex")
        e2 = _get(b, "end", "endLine", "endLineNumber", "endLineIndex")

        # Defensive: compute fragment length if missing
        frag_len = dup.get("lines") or dup.get("lineCount") or dup.get("length")
        if not isinstance(frag_len, int):
            if isinstance(s1, int) and isinstance(e1, int):
                frag_len = max(1, int(e1) - int(s1) + 1)
            else:
                frag_len = 1

        # Clamp ranges to known totals if available
        def clamp_range(start, end, max_lines):
            if not (isinstance(start, int) and isinstance(end, int)):
                return set()
            lo = max(1, int(start))
            hi = int(end) if (max_lines is None) else min(int(end), int(max_lines))
            if hi < lo:
                return set()
            return set(range(lo, hi + 1))

        maxA = total_lines.get(a_path)
        maxB = total_lines.get(b_path)

        # Map the reported fragments to the normalized (a_path,b_path)
        if f1 == a_path:
            linesA = clamp_range(s1, e1, maxA)
            linesB = clamp_range(s2, e2, maxB)
        else:
            linesA = clamp_range(s2, e2, maxA)
            linesB = clamp_range(s1, e1, maxB)

        pair_sets[key]["A"].update(linesA)
        pair_sets[key]["B"].update(linesB)
        pair_sets[key]["files"] = (a_path, b_path)
        pair_sets[key]["longest"] = max(pair_sets[key]["longest"], int(frag_len))

    # Prepare results per pair
    rows = []
    for (a, b), payload in pair_sets.items():
        dupA = len(payload["A"])
        dupB = len(payload["B"])
        totA = total_lines.get(a, 0)
        totB = total_lines.get(b, 0)
        if totA == 0 or totB == 0:
            # If a file didn't appear in 'statistics', we can't compute a ratio reliably.
            # Skip such pairs from similarity output, but they still exist for gold counting.
            continue
        rA = dupA / totA
        rB = dupB / totB
        sym = 0.5 * (rA + rB)
        rows.append({
            "pair": (a, b),
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

    # Group extractor (regex first, then fallbacks)
    group_rx = compile_group_regex(args.group_regex)
    group_of = make_group_extractor(group_rx)

    # Tally correct/wrong/unclassified among ≥ threshold
    correct = wrong = unclassified = 0
    for r in rows_thr:
        a, b = r["pair"]
        ga, gb = group_of(a), group_of(b)
        if ga is None or gb is None:
            unclassified += 1
        elif ga == gb:
            correct += 1
        else:
            wrong += 1

    # Build "gold positives" as all pairs from the *complete* scanned set that share the same group
    files_by_group = defaultdict(list)
    for f in all_files:
        g = group_of(f)
        if g is not None:
            files_by_group[g].append(f)

    gold_pairs = set()
    for g, lst in files_by_group.items():
        if len(lst) >= 2:
            flist = sorted(lst)
            for i in range(len(flist)):
                for j in range(i + 1, len(flist)):
                    gold_pairs.add((flist[i], flist[j]))

    predicted_pairs = set((r["pair"][0], r["pair"][1]) for r in rows_thr)
    TP = len(predicted_pairs & gold_pairs)
    FP = len(predicted_pairs - gold_pairs)
    FN = len(gold_pairs - predicted_pairs)
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # Pretty print (mimic JPlag/Dolos table columns/naming)
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
