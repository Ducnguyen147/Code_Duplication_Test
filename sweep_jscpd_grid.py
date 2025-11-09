#!/usr/bin/env python3

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

TP_FP_FN_RE = re.compile(r"TP:\s*(\d+)\s+FP:\s*(\d+)\s+FN:\s*(\d+)")
PRF1_RE = re.compile(
    r"Precision:\s*([0-9]+(?:\.[0-9]+)?)\s+Recall:\s*([0-9]+(?:\.[0-9]+)?)\s+F1:\s*([0-9]+(?:\.[0-9]+)?)"
)

DEFAULT_MIN_TOKENS = [3, 5, 10, 15, 20]
DEFAULT_THRESHOLDS = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

def run_jscpd(
    jscpd_cmd: str,
    input_path: str,
    min_tokens: int,
    min_lines: int,
    mode: str,
    output_dir: str,
    extra_reporters: Optional[str] = None,
) -> Tuple[int, str, str, str]:
    """
    Run JSCPD once and write JSON report to output_dir/jscpd-report.json.
    Returns (rc, stdout, stderr, expected_json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    reporters = "json" if not extra_reporters else f"json,{extra_reporters}"
    cmd = [
        jscpd_cmd,
        input_path,
        "--min-tokens", str(min_tokens),
        "--min-lines", str(min_lines),
        "--mode", mode,
        "--reporters", reporters,
        "--output", output_dir,
        "--exitCode", "0",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    json_path = os.path.join(output_dir, "jscpd-report.json")
    return proc.returncode, proc.stdout, proc.stderr, json_path

def run_metrics(
    metrics_script: str,
    json_path: str,
    threshold_pct: int,
    group_regex: Optional[str],
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], str]:
    """
    Run your jscpd_metrics_calc.py and parse TP/FP/FN + Precision/Recall/F1.
    Returns (tp, fp, fn, precision, recall, f1, raw_output_or_debug).
    """
    if not os.path.exists(json_path):
        return None, None, None, None, None, None, f"[missing JSON] {json_path}"

    cmd = [sys.executable, metrics_script, "-i", json_path, "-t", f"{threshold_pct}%", "--sort"]
    if group_regex:
        cmd.extend(["--group-regex", group_regex])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    err = proc.stderr

    m_counts = TP_FP_FN_RE.search(out)
    m_scores = PRF1_RE.search(out)
    if m_counts and m_scores:
        tp = int(m_counts.group(1))
        fp = int(m_counts.group(2))
        fn = int(m_counts.group(3))
        precision = float(m_scores.group(1))
        recall = float(m_scores.group(2))
        f1 = float(m_scores.group(3))
        return tp, fp, fn, precision, recall, f1, out

    return None, None, None, None, None, None, out + ("\n--- stderr ---\n" + err if err else "")

def main():
    ap = argparse.ArgumentParser(description="Sweep JSCPD (--min-tokens) and thresholds; write metrics to CSV (Dolos-style).")
    ap.add_argument("--jscpd", default="jscpd", help="JSCPD executable (default: jscpd on PATH)")
    ap.add_argument("--input", required=True, help="Input path (folder or glob) to analyze (e.g., mix)")
    ap.add_argument("--metrics-script", default="./jscpd_metrics_calc.py", help="Path to jscpd_metrics_calc.py")
    ap.add_argument("--results-root", default="result-jscpd/sweep", help="Where per-run reports are written")
    ap.add_argument("--out", default="jscpd_grid_metrics.csv", help="Output CSV path")
    ap.add_argument("--language-label", default="python", help="CSV label for 'language' column (purely cosmetic)")
    ap.add_argument("--min-tokens", nargs="*", type=int, help="Custom min-tokens list (default: 3 5 10 15 20)")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Custom percent thresholds (default: 50..95 step 5)")
    ap.add_argument("--min-lines", type=int, default=1, help="JSCPD --min-lines (default: 1)")
    ap.add_argument("--mode", default="weak", choices=["strict", "mild", "weak"], help="JSCPD --mode (default: weak)")
    ap.add_argument("--extra-reporters", help="Optional extra reporters (comma-separated), e.g. 'console,html'")
    ap.add_argument("--group-regex", type=str, default=None,
                    help="Regex with ONE capturing group to extract the group label from basenames "
                         "(e.g., '^(C\\d+)(?:[_-]|(?=[A-Z]))'). Passed through to jscpd_metrics_calc.py.")
    args = ap.parse_args()

    min_tokens_list = args.min_tokens if args.min_tokens else DEFAULT_MIN_TOKENS
    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    rows: List[List[str]] = [["language", "min_tokens", "threshold_pct", "tp", "fp", "fn", "precision", "recall", "f1"]]

    for mt in min_tokens_list:
        out_dir = os.path.join(args.results_root, f"t{mt}")
        print(f"[+] JSCPD --min-tokens={mt}  ->  {out_dir}", file=sys.stderr)
        rc, jscpd_out, jscpd_err, json_path = run_jscpd(
            args.jscpd,
            args.input,
            mt,
            args.min_lines,
            args.mode,
            out_dir,
            extra_reporters=args.extra_reporters,
        )

        if not os.path.exists(json_path):
            print(f"[!] Could not find JSON report for min_tokens={mt} at {json_path}\n{jscpd_err}", file=sys.stderr)
            for th in thresholds:
                rows.append([args.language_label, mt, th, "", "", "", "", "", ""])
            continue

        for th in thresholds:
            tp, fp, fn, precision, recall, f1, raw = run_metrics(
                args.metrics_script, json_path, th, args.group_regex
            )
            if tp is None:
                print(f"[!] Parse error mt={mt} threshold={th}%\n{raw}", file=sys.stderr)
                rows.append([args.language_label, mt, th, "", "", "", "", "", ""])
            else:
                rows.append([
                    args.language_label, mt, th,
                    tp, fp, fn,
                    f"{precision:.6f}", f"{recall:.6f}", f"{f1:.6f}"
                ])

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[✓] Wrote {out_path}")

if __name__ == "__main__":
    main()
