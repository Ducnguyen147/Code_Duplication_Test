#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from typing import Tuple, Optional, List

TP_FP_FN_RE = re.compile(r"TP:\s*(\d+)\s+FP:\s*(\d+)\s+FN:\s*(\d+)")
PREC_REC_F1_RE = re.compile(
    r"Precision:\s*([0-9]+(?:\.[0-9]+)?)\s+Recall:\s*([0-9]+(?:\.[0-9]+)?)\s+F1:\s*([0-9]+(?:\.[0-9]+)?)"
)

GRID_K = [12, 17, 23]
GRID_W = [17, 25, 40]
THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def run_dolos(language: str, zip_path: str, k: int, w: int) -> Tuple[int, str, str]:
    """
    Run `dolos run -f terminal -l <language> -k <k> -w <w> <zip_path>`
    Return (returncode, stdout, stderr) with text.
    """
    cmd = [
        "dolos", "run",
        "-f", "terminal",
        "-l", language,
        "-k", str(k),
        "-w", str(w),
        zip_path,
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def run_filter_on_stdout(
    filter_script: str,
    threshold_pct: int,
    dolos_stdout: str,
    group_regex: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], str]:
    """
    Pipe `dolos_stdout` into:
        python dolos_stdout_filter.py -t {threshold}% [--group-regex <regex>]
    Return parsed metrics (tp, fp, fn, prec, rec, f1, raw_output).
    """
    cmd = [sys.executable, filter_script, "-t", f"{threshold_pct}%"]
    if group_regex:
        cmd += ["--group-regex", group_regex]

    proc = subprocess.run(cmd, input=dolos_stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    err = proc.stderr

    m_counts = TP_FP_FN_RE.search(out)
    m_scores = PREC_REC_F1_RE.search(out)

    if m_counts and m_scores:
        tp = int(m_counts.group(1))
        fp = int(m_counts.group(2))
        fn = int(m_counts.group(3))
        precision = float(m_scores.group(1))
        recall = float(m_scores.group(2))
        f1 = float(m_scores.group(3))
        return tp, fp, fn, precision, recall, f1, out
    else:
        return None, None, None, None, None, None, out + ("\n--- stderr ---\n" + err if err else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help="Path to the input ZIP (or info.csv / file glob supported by Dolos)")
    ap.add_argument("--language", default="python", help="Language for Dolos -l (default: python)")
    ap.add_argument("--filter-script", default="./dolos_stdout_filter.py", help="Path to dolos_stdout_filter.py (the metrics-enhanced version)")
    ap.add_argument("--out", default="grid_metrics.csv", help="Output CSV path (default: grid_metrics.csv)")
    ap.add_argument("--k", nargs="*", type=int, help="Optional custom list of k values; default is 12 17 23")
    ap.add_argument("--w", nargs="*", type=int, help="Optional custom list of w values; default is 17 25 40")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Optional custom percent thresholds; default is 50 55 ... 95")
    ap.add_argument("--group-regex", help="Forwarded to the filter script (e.g. '^(C\\d+)(?:[_-]|(?=[A-Z]))')")
    args = ap.parse_args()

    if shutil.which("dolos") is None:
        sys.exit("Error: `dolos` is not on PATH. Install Dolos CLI and try again.")

    if not os.path.exists(args.filter_script):
        sys.exit(f"Error: filter script not found at {args.filter_script}")

    grid_k = args.k if args.k else GRID_K
    grid_w = args.w if args.w else GRID_W
    thresholds = args.thresholds if args.thresholds else THRESHOLDS

    rows: List[List[str]] = []
    header = ["language", "k", "w", "threshold_pct", "tp", "fp", "fn", "precision", "recall", "f1"]
    rows.append(header)

    for k in grid_k:
        for w in grid_w:
            print(f"[+] Running Dolos for k={k}, w={w} ...", file=sys.stderr)
            rc, dolos_out, dolos_err = run_dolos(args.language, args.zip, k, w)
            if rc != 0:
                print(f"[!] Dolos failed for k={k} w={w}:\n{dolos_err}", file=sys.stderr)

                for t in thresholds:
                    rows.append([args.language, k, w, t, "", "", "", "", "", ""])
                continue

            for t in thresholds:
                tp, fp, fn, precision, recall, f1, raw = run_filter_on_stdout(
                    args.filter_script, t, dolos_out, group_regex=args.group_regex
                )
                if tp is None:
                    print(f"[!] Could not parse metrics for k={k} w={w} t={t}%\n{raw}", file=sys.stderr)
                    rows.append([args.language, k, w, t, "", "", "", "", "", ""])
                else:
                    rows.append([
                        args.language, k, w, t,
                        tp, fp, fn,
                        f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}"
                    ])

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[✓] Wrote {out_path}")

if __name__ == "__main__":
    main()
