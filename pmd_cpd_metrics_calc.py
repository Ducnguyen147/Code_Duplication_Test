#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple


# ------------------------- helpers -------------------------

def group_prefix(path: str, depth: int = 1) -> Optional[str]:
    """
    Extract group prefix from a filename by underscore depth.
    e.g. '/.../C03_gcd_type1_a.py' -> 'C03' (depth=1) or 'C03_gcd' (depth=2)
    Returns None if the basename has fewer than `depth` parts.
    """
    base = os.path.basename(path or "")
    stem, _ = os.path.splitext(base)
    if not stem:
        return None
    parts = stem.split("_")
    return "_".join(parts[:depth]) if len(parts) >= depth else None


def compile_group_regex(s: Optional[str]) -> Optional[re.Pattern]:
    """Compile a user-supplied regex; it must contain at least one capturing group."""
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
    """Apply regex to the *basename*; return the first capturing group or None."""
    base = os.path.basename(path or "")
    m = rx.search(base)
    return m.group(1) if m else None


def parse_threshold(s: str) -> float:
    s = s.strip()
    if s.endswith("%"):
        return float(s[:-1].strip()) / 100.0
    v = float(s)
    return v if 0 <= v <= 1 else v / 100.0


def fmt_percent(x: float, places: int = 2) -> str:
    return f"{x*100:.{places}f}%"


def union_length(intervals: List[Tuple[int, int]]) -> int:
    """
    Intervals are inclusive token indices: (beginToken, endToken).
    Returns total length of their union: sum(end-begin+1) over merged intervals.
    """
    if not intervals:
        return 0
    norm = [(min(a, b), max(a, b)) for (a, b) in intervals]
    norm.sort()
    merged = []
    cur_s, cur_e = norm[0]
    for s, e in norm[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return sum(e - s + 1 for s, e in merged)


# ------------------------- XML parsing -------------------------

def _detect_ns(root: ET.Element) -> str:
    """Extract the XML namespace from a tag like '{ns}pmd-cpd' -> 'ns'."""
    if root.tag.startswith("{"):
        return root.tag.split("}")[0].strip("{")
    return ""


def load_cpd_xml(path_or_stdin: str) -> Tuple[Dict[str, int], List[dict]]:
    """
    Returns:
      file_tokens: {path -> totalNumberOfTokens}
      duplications: list of { 'tokens': int, 'files': [ {'path': str, 'begin': int, 'end': int}, ... ] }
    """
    if path_or_stdin == "-":
        xml_text = sys.stdin.read()
        root = ET.fromstring(xml_text)
    else:
        tree = ET.parse(path_or_stdin)
        root = tree.getroot()

    ns = _detect_ns(root)
    q = (lambda name: f"{{{ns}}}{name}") if ns else (lambda name: name)

    file_tokens: Dict[str, int] = {}
    for f in root.findall(q("file")):
        tnt = f.attrib.get("totalNumberOfTokens")
        if tnt is not None:
            p = f.attrib.get("path")
            if p:
                try:
                    file_tokens[p] = int(tnt)
                except ValueError:
                    pass

    duplications: List[dict] = []
    for dup in root.findall(q("duplication")):
        tok = int(dup.attrib.get("tokens", "0"))
        files = []
        for occ in dup.findall(q("file")):
            p = occ.attrib.get("path")
            bt = occ.attrib.get("begintoken")
            et = occ.attrib.get("endtoken")
            if not (p and bt and et):
                continue
            files.append({"path": p, "begin": int(bt), "end": int(et)})
        if len(files) >= 2:
            duplications.append({"tokens": tok, "files": files})

    return file_tokens, duplications


# ------------------------- pair aggregation -------------------------

def build_pair_stats(file_tokens: Dict[str, int], duplications: List[dict]):
    """
    For each pair {A,B}, collect:
      - intervals[A]: list of (begin,end) ranges in A duplicated with B (union later)
      - intervals[B]: list of (begin,end) ranges in B duplicated with A
      - longest: max duplication tokens seen between A and B
    """
    pair_intervals: Dict[Tuple[str, str], Dict[str, List[Tuple[int, int]]]] = {}
    pair_longest: Dict[Tuple[str, str], int] = defaultdict(int)

    for dup in duplications:
        files = dup["files"]
        longest_here = int(dup["tokens"])
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                a = files[i]
                b = files[j]
                a_path, b_path = a["path"], b["path"]
                if a_path == b_path:
                    continue
                left, right = sorted([a_path, b_path])
                key = (left, right)
                if key not in pair_intervals:
                    pair_intervals[key] = {left: [], right: []}
                pair_intervals[key][a_path].append((a["begin"], a["end"]))
                pair_intervals[key][b_path].append((b["begin"], b["end"]))
                pair_longest[key] = max(pair_longest[key], longest_here)

    results = []
    for key, sides in pair_intervals.items():
        left, right = key
        left_union = union_length(sides[left])
        right_union = union_length(sides[right])
        lt = max(1, file_tokens.get(left, 0))
        rt = max(1, file_tokens.get(right, 0))
        left_pct = left_union / lt if lt > 0 else 0.0
        right_pct = right_union / rt if rt > 0 else 0.0
        sim = 0.5 * (left_pct + right_pct)
        results.append({
            "left": left,
            "right": right,
            "left_union": left_union,
            "right_union": right_union,
            "left_total": lt,
            "right_total": rt,
            "similarity": sim,
            "longest": pair_longest[key],
            "total_overlap": left_union + right_union,
        })

    return results


# ------------------------- metrics & printing -------------------------

def evaluate_and_print(
    pairs: List[dict],
    all_files: List[str],
    *,
    threshold: float,
    group_depth: int,
    group_rx: Optional[re.Pattern],
    sort_rows: bool,
):

    sim_map: Dict[Tuple[str, str], float] = {}
    for rec in pairs:
        k = (rec["left"], rec["right"])
        sim_map[k] = rec["similarity"]

    at_or_above = [rec for rec in pairs if rec["similarity"] >= threshold]
    if sort_rows:
        at_or_above.sort(key=lambda d: d["similarity"], reverse=True)
    if at_or_above:
        left_w = max(len("Left file"), *(len(r["left"]) for r in at_or_above))
        right_w = max(len("Right file"), *(len(r["right"]) for r in at_or_above))
        long_w = max(len("Longest"), *(len(str(r["longest"])) for r in at_or_above))
        tot_w = max(len("Total overlap"), *(len(str(r["total_overlap"])) for r in at_or_above))
        print(f"{'Similarity':>10}  {'Left file':<{left_w}}  {'Right file':<{right_w}}  {'Longest':>{long_w}}  {'Total overlap':>{tot_w}}")
        for r in at_or_above:
            print(
                f"{fmt_percent(r['similarity']):>10}  "
                f"{r['left']:<{left_w}}  {r['right']:<{right_w}}  "
                f"{r['longest']:>{long_w}}  {r['total_overlap']:>{tot_w}}"
            )
        print()
    else:
        print("No pairs at or above the threshold.\n")

    def gp(p: str) -> Optional[str]:
        if group_rx is not None:
            return extract_group_via_regex(p, group_rx)
        return group_prefix(p, depth=group_depth)

    wrong, correct, unclassified = [], [], []
    for r in at_or_above:
        g1, g2 = gp(r["left"]), gp(r["right"])
        if g1 is None or g2 is None:
            unclassified.append(r)
        elif g1 == g2:
            correct.append(r)
        else:
            wrong.append(r)

    print(f"Total pairs \u2265 {fmt_percent(threshold)}: {len(at_or_above)}")
    print(f"Wrong-detection pairs (group mismatch) \u2265 {fmt_percent(threshold)}: {len(wrong)}")
    print(f"Correct-detection pairs (same group) \u2265 {fmt_percent(threshold)}: {len(correct)}")
    print(f"Unclassified (group missing) \u2265 {fmt_percent(threshold)}: {len(unclassified)}\n")

    files_sorted = sorted(all_files)
    tp = fp = fn = tn = 0
    gold_positives = 0
    for i in range(len(files_sorted)):
        for j in range(i + 1, len(files_sorted)):
            a, b = files_sorted[i], files_sorted[j]
            g1, g2 = gp(a), gp(b)
            same_group = (g1 is not None) and (g2 is not None) and (g1 == g2)
            uncls = (g1 is None) or (g2 is None)
            sim = sim_map.get((min(a, b), max(a, b)), 0.0)
            pred = sim >= threshold
            if same_group:
                gold_positives += 1
                if pred:
                    tp += 1
                else:
                    fn += 1
            elif not uncls:
                if pred:
                    fp += 1
                else:
                    tn += 1


    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print("=== Evaluation (same-prefix heuristic over all files) ===")
    print(f"Gold positives (same prefix) among all files: {gold_positives}")
    print(f"TP: {tp}  FP: {fp}  FN: {fn}")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")


# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pretty-print CPD pairs and compute Correct/Wrong detections + Precision/Recall/F1."
    )
    ap.add_argument("-i", "--input", type=str, default=None,
                    help="Path to CPD XML. If omitted, read from STDIN (use '-' to force stdin).")
    ap.add_argument("-t", "--threshold", type=parse_threshold, default="60%",
                    help="Similarity threshold (e.g., 60%%, 0.6, or 60). Default: 60%%")
    ap.add_argument("--sort", action="store_true",
                    help="Sort pairs by similarity descending before printing.")
    ap.add_argument("-g", "--group-depth", type=int, default=1,
                    help="Underscore-based prefix depth used as 'group' if --group-regex is not set (1 -> C03, 2 -> C03_gcd).")
    ap.add_argument("--group-regex", type=str, default=None,
                    help="Regex with ONE capturing group to extract the group from the basename; "
                         "e.g., '^(C\\d+)(?:[_-]|(?=[A-Z]))'")
    args = ap.parse_args()

    path = args.input or "-"
    file_tokens, duplications = load_cpd_xml(path)
    if not file_tokens:
        sys.exit("ERROR: No <file ... totalNumberOfTokens=.../> elements found in CPD XML.")

    pairs = build_pair_stats(file_tokens, duplications)

    group_rx = compile_group_regex(args.group_regex)

    evaluate_and_print(
        pairs,
        list(file_tokens.keys()),
        threshold=args.threshold,
        group_depth=args.group_depth,
        group_rx=group_rx,
        sort_rows=args.sort,
    )


if __name__ == "__main__":
    main()
