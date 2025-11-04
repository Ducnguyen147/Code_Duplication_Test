"""
Prints the duplication percentages for all pairs of files detected in a
PMD‑CPD XML report.  This script uses the pairwise duplication logic
implemented in `dup_pairwise.py` to identify which files share duplicated
tokens and calculates the percentage of each file that is duplicated with
the other file in the pair.

Usage:
    python3 dup_all_pair_percentages.py report.xml

This will output one line per pair in the form:
    file1 <-> file2: file1_percent% (of file1), file2_percent% (of file2)

Pairs are sorted by the number of duplicated tokens in descending order.

If you need a JSON output, run `dup_pairwise.py --json` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET

try:
    from dup_pairwise import compute_pairwise_duplication  # type: ignore
except ImportError:
    # Minimal fallback for compute_pairwise_duplication if module isn't available.
    from collections import defaultdict
    from typing import Dict, Iterable, List, Tuple

    CPD_NS = {"cpd": "https://pmd-code.org/schema/cpd-report"}

    def _union_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        sorted_ints = sorted((int(a), int(b)) for a, b in intervals if a is not None and b is not None)
        merged: List[Tuple[int, int]] = []
        for s, e in sorted_ints:
            if not merged:
                merged.append((s, e))
                continue
            ps, pe = merged[-1]
            if s <= pe + 1:
                merged[-1] = (ps, max(pe, e))
            else:
                merged.append((s, e))
        return merged

    def compute_per_file_totals(root: ET.Element) -> Dict[str, int]:
        totals: Dict[str, int] = {}
        for f in root.findall("cpd:file", CPD_NS):
            p = f.attrib.get("path")
            t = f.attrib.get("totalNumberOfTokens")
            if p and t:
                totals[p] = int(t)
        return totals

    def compute_pairwise_duplication(xml_path: str) -> Dict[Tuple[str, str], dict]:
        root = ET.parse(xml_path).getroot()
        totals = compute_per_file_totals(root)
        pair_intervals: Dict[Tuple[str, str], Dict[str, List[Tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        for dup in root.findall("cpd:duplication", CPD_NS):
            entries: List[Tuple[str, int, int]] = []
            for occ in dup.findall("cpd:file", CPD_NS):
                p = occ.attrib.get("path")
                bt = occ.attrib.get("begintoken")
                et = occ.attrib.get("endtoken")
                if p and bt and et:
                    entries.append((p, int(bt), int(et)))
            for i in range(len(entries)):
                for j in range(i + 1, len(entries)):
                    pa, bt_a, et_a = entries[i]
                    pb, bt_b, et_b = entries[j]
                    key = tuple(sorted((pa, pb)))
                    pair_intervals[key][pa].append((bt_a, et_a))
                    pair_intervals[key][pb].append((bt_b, et_b))
        result: Dict[Tuple[str, str], dict] = {}
        for pair, file_to_intervals in pair_intervals.items():
            a, b = pair
            merged_a = _union_intervals(file_to_intervals[a])
            merged_b = _union_intervals(file_to_intervals[b])
            dup_tokens_a = sum(e - s + 1 for s, e in merged_a)
            dup_tokens_b = sum(e - s + 1 for s, e in merged_b)
            duplicated_tokens = min(dup_tokens_a, dup_tokens_b)
            total_a = totals.get(a, max(e for _, e in merged_a))
            total_b = totals.get(b, max(e for _, e in merged_b))
            percent_a = duplicated_tokens / total_a * 100.0
            percent_b = duplicated_tokens / total_b * 100.0
            result[pair] = {
                "duplicated_tokens": duplicated_tokens,
                "percent_of_a": percent_a,
                "percent_of_b": percent_b,
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List duplication percentages for all file pairs in a CPD report."
    )
    parser.add_argument("xml", help="Path to CPD XML report (pmd cpd --format xml ...)")
    args = parser.parse_args()

    xml_path = args.xml
    if not os.path.exists(xml_path):
        print(f"Error: XML file '{xml_path}' not found.")
        sys.exit(1)

    pair_stats = compute_pairwise_duplication(xml_path)
    if not pair_stats:
        print("No duplication pairs found in the report.")
        sys.exit(0)

    # Sort pairs by descending final_percent (average of the two file percentages).
    # We compute the average of percent_of_a and percent_of_b for each pair as the
    # sorting key. This ensures that pairs with the highest overall duplication
    # ratio appear first in the output.
    sorted_pairs = sorted(
        pair_stats.items(),
        key=lambda kv: (kv[1]["percent_of_a"] + kv[1]["percent_of_b"]) / 2.0,
        reverse=True,
    )

    for (file_a, file_b), data in sorted_pairs:
        percent_a = data["percent_of_a"]
        percent_b = data["percent_of_b"]
        # Compute a single, symmetric duplication metric by averaging the two file percentages.
        # This treats each file equally and represents the average proportion of each file
        # that overlaps with the other. If desired, you could use a different formula,
        # such as duplicated_tokens * 200 / (total_a + total_b), but that requires
        # total token counts in the result data.
        final_percent = (percent_a + percent_b) / 2.0
        print(
            f"{file_a} <-> {file_b}: {final_percent:.2f}% duplication "
            f"(avg of {percent_a:.2f}% and {percent_b:.2f}%)"
        )


if __name__ == "__main__":
    main()