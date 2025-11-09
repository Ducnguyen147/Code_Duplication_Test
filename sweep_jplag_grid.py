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
PRF1_RE     = re.compile(r"Precision:\s*([0-9]+(?:\.[0-9]+)?)\s+Recall:\s*([0-9]+(?:\.[0-9]+)?)\s+F1:\s*([0-9]+(?:\.[0-9]+)?)")

DEFAULT_T_VALUES     = [3, 5, 10, 15, 20]
DEFAULT_THRESHOLDS   = [50,55,60,65,70,75,80,85,90,95]

def run_jplag(jar: str, language: str, input_root: str, t_value: int, result_base: str) -> Tuple[int, str, str, str]:
    """
    Run:
      java -jar <jar> -l <language> -t <t> <input_root>
           -r <result_base> --csv-export --mode run --overwrite

    Returns (returncode, stdout, stderr, expected_csv_path)
    """
    os.makedirs(os.path.dirname(os.path.abspath(result_base)), exist_ok=True)

    cmd = [
        "java", "-jar", jar,
        "-l", language,
        "-t", str(t_value),
        input_root,
        "-r", result_base,
        "--csv-export",
        "--mode", "run",
        "--overwrite"
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    expected_csv = os.path.join(result_base, "results.csv")
    return proc.returncode, proc.stdout, proc.stderr, expected_csv

def run_metrics(
    metrics_script: str,
    threshold_pct: int,
    results_csv: str,
    sort: bool = True,
    group_regex: Optional[str] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float], Optional[float], Optional[float], str]:
    """
    Run:
      python jplag_metrics_calc.py -t <threshold>% -i <results_csv> [--sort] [--group-regex REGEX]
    Parse TP/FP/FN + Precision/Recall/F1.
    Returns (tp, fp, fn, precision, recall, f1, full_stdout_or_debug)
    """
    if not os.path.exists(results_csv):
        return None, None, None, None, None, None, f"[missing results.csv] {results_csv}"

    cmd = [sys.executable, metrics_script, "-t", f"{threshold_pct}%", "-i", results_csv]
    if sort:
        cmd.append("--sort")
    if group_regex:
        cmd += ["--group-regex", group_regex]

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
        recall    = float(m_scores.group(2))
        f1        = float(m_scores.group(3))
        return tp, fp, fn, precision, recall, f1, out
    else:
        return None, None, None, None, None, None, out + ("\n--- stderr ---\n" + err if err else "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jar", required=True, help="Path to JPlag jar (e.g., jplag-6.0.0-jar-with-dependencies.jar)")
    ap.add_argument("--input", required=True, help="Root directory of submissions (e.g., mix)")
    ap.add_argument("--language", default="python3", help="JPlag -l language (default: python3)")
    ap.add_argument("--metrics-script", default="./jplag_metrics_calc.py", help="Path to jplag_metrics_calc.py")
    ap.add_argument("--results-root", default="result-jplag", help="Folder to hold per-t result bases (default: result-jplag)")
    ap.add_argument("--t-values", nargs="*", type=int, help="Min-token values to sweep; default: 3 5 10 15 20")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Percent thresholds; default: 50 55 60 65 70 75 80 85 90 95")
    ap.add_argument("--group-regex", help="Forwarded to metrics script (e.g., '^(C\\d+)(?:[_-]|(?=[A-Z]))')")
    ap.add_argument("--out", default="jplag_grid_metrics.csv", help="Output CSV path")
    ap.add_argument("--clean", action="store_true", help="Delete any existing per-t result base folder before running JPlag")
    args = ap.parse_args()

    if shutil.which("java") is None:
        sys.exit("Error: `java` not found on PATH.")

    t_values   = args.t_values if args.t_values else DEFAULT_T_VALUES
    thresholds = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS

    dataset_name = os.path.basename(os.path.normpath(args.input))
    os.makedirs(args.results_root, exist_ok=True)

    rows: List[List[str]] = []
    header = ["language", "t", "threshold_pct", "tp", "fp", "fn", "precision", "recall", "f1", "results_csv"]
    rows.append(header)

    for t in t_values:
        result_base = os.path.join(args.results_root, f"{dataset_name}_t{t}")
        if args.clean and os.path.isdir(result_base):
            shutil.rmtree(result_base, ignore_errors=True)

        print(f"[+] JPlag run  t={t}  ->  {result_base}", file=sys.stderr)
        rc, jplag_out, jplag_err, csv_path = run_jplag(args.jar, args.language, args.input, t, result_base)
        if rc != 0:
            print(f"[!] JPlag failed for t={t}:\n{jplag_err}", file=sys.stderr)
            for th in thresholds:
                rows.append([args.language, t, th, "", "", "", "", "", "", ""])
            continue

        if jplag_err.strip():
            print(jplag_err.strip(), file=sys.stderr)

        for th in thresholds:
            tp, fp, fn, prec, rec, f1, raw = run_metrics(
                args.metrics_script, th, csv_path, sort=True, group_regex=args.group_regex
            )
            if tp is None:
                print(f"[!] Could not parse metrics for t={t}, threshold={th}%\n{raw}", file=sys.stderr)
                rows.append([args.language, t, th, "", "", "", "", "", "", csv_path])
            else:
                rows.append([
                    args.language, t, th,
                    tp, fp, fn,
                    f"{prec:.6f}", f"{rec:.6f}", f"{f1:.6f}",
                    csv_path
                ])

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[✓] Wrote {out_path}")

if __name__ == "__main__":
    main()
