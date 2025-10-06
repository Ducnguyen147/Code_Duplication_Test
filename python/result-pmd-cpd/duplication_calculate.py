# command: python3 duplication_calculate.py /home/duc/Desktop/code_duplication/Code_Duplication_Test/python/result-pmd-cpd/hybrid.xml

from __future__ import annotations
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable

CPD_NS = {"cpd": "https://pmd-code.org/schema/cpd-report"}

def _union_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge inclusive [start, end] token intervals, coalescing overlaps and touching ranges."""
    sorted_ints = sorted((int(a), int(b)) for a, b in intervals if a is not None and b is not None)
    merged: List[Tuple[int, int]] = []
    for s, e in sorted_ints:
        if not merged:
            merged.append((s, e))
            continue
        ps, pe = merged[-1]
        if s <= pe + 1:  # overlap or touching -> merge
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged

def compute_per_file_duplication(xml_path: str) -> Dict[str, dict]:
    """
    Parse a PMD-CPD XML report and return:
      { path: {
           'duplicated_tokens': int,
           'total_tokens': int,
           'percent': float,
           'intervals': [(start_token, end_token), ...]  # merged, inclusive
        }, ... }
    """
    root = ET.parse(xml_path).getroot()

    # 1) total tokens per file (from top-level <file totalNumberOfTokens="...">)
    totals: Dict[str, int] = {}
    for f in root.findall("cpd:file", CPD_NS):
        p = f.attrib.get("path")
        t = f.attrib.get("totalNumberOfTokens")
        if p and t:
            totals[p] = int(t)

    # 2) collect all duplicated token intervals per file from each <duplication>
    per_file_intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for dup in root.findall("cpd:duplication", CPD_NS):
        for occ in dup.findall("cpd:file", CPD_NS):
            p = occ.attrib.get("path")
            bt = occ.attrib.get("begintoken")
            et = occ.attrib.get("endtoken")
            if p and bt and et:
                per_file_intervals[p].append((int(bt), int(et)))

    # 3) union intervals and compute duplicated tokens / total tokens / percent
    result: Dict[str, dict] = {}
    for path, intervals in per_file_intervals.items():
        merged = _union_intervals(intervals)
        dup_tokens = sum(e - s + 1 for s, e in merged)  # inclusive
        total = totals.get(path)
        if total is None:
            # Fallback: infer minimal bound if report lacks totals (older formats).
            total = max(e for _, e in merged)
        percent = (dup_tokens / total * 100.0) if total else 0.0
        result[path] = {
            "duplicated_tokens": dup_tokens,
            "total_tokens": total,
            "percent": percent,
            "intervals": merged,
        }

    # 4) include files with totals but zero duplications
    for path, total in totals.items():
        if path not in result:
            result[path] = {
                "duplicated_tokens": 0,
                "total_tokens": total,
                "percent": 0.0,
                "intervals": [],
            }

    return result

if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Compute per-file duplication% from PMD-CPD XML.")
    ap.add_argument("xml", help="Path to CPD XML report (pmd cpd --format xml ...)")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of a table")
    args = ap.parse_args()

    res = compute_per_file_duplication(args.xml)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        # pretty table sorted by highest duplication%
        for path, data in sorted(res.items(), key=lambda kv: kv[1]["percent"], reverse=True):
            print(f"{path}\n  duplicated_tokens={data['duplicated_tokens']}, "
                  f"total_tokens={data['total_tokens']}, "
                  f"percent={data['percent']:.2f}%\n  intervals={data['intervals']}\n")
