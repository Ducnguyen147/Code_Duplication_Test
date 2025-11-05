#!/usr/bin/env python3
"""
Sweep PMD CPD over multiple --minimum-tokens values and thresholds, collect
TP/FP/FN/Precision/Recall/F1 into a CSV (Dolos-style).

- min_tokens: 5,10,25,40,60  (override with --min-tokens)
- thresholds: 50..95 step 5  (override with --thresholds)
- output CSV: language,min_tokens,threshold_pct,tp,fp,fn,precision,recall,f1

CPD notes:
- By default CPD exits with status 4 when duplications are found, and 5 for recoverable errors.
  We add --no-fail-on-violation and --no-fail-on-error so CPD exits 0 but still writes the report.
"""

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

DEFAULT_MIN_TOKENS = [5, 10, 25, 40, 60]
DEFAULT_THRESHOLDS = [50,55,60,65,70,75,80,85,90,95]

def run_cpd(pmd_exe: str, dir_path: str, language: str, min_tokens: int) -> Tuple[int, str, str]:
    """
    Run CPD and return (rc, stdout_xml, stderr).
    We force zero exit on violations/errors to simplify automation.
    """
    cmd = [
        pmd_exe, "cpd",
        "--minimum-tokens", str(min_tokens),
        "--dir", dir_path,
        "--language", language,
        "--format", "xml",
        "--no-fail-on-violation",
        "--no-fail-on-error",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def run_metrics(metrics_script: str, xml_path: str, threshold_pct: int):
    """Invoke pmd_cpd_metrics_calc.py and parse TP/FP/FN + PR/F1."""
    cmd = [sys.executable, metrics_script, "-i", xml_path, "-t", f"{threshold_pct}%", "--sort"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out = proc.stdout
    m_counts, m_scores = TP_FP_FN_RE.search(out), PRF1_RE.search(out)
    if m_counts and m_scores:
        return (
            int(m_counts.group(1)),
            int(m_counts.group(2)),
            int(m_counts.group(3)),
            float(m_scores.group(1)),
            float(m_scores.group(2)),
            float(m_scores.group(3)),
            out,
        )
    return None, None, None, None, None, None, out + ("\n--- stderr ---\n" + proc.stderr if proc.stderr else "")

def main():
    ap = argparse.ArgumentParser(description="Sweep PMD CPD (--minimum-tokens) and thresholds; write metrics to CSV.")
    ap.add_argument("--pmd", required=True, help="Full path to the PMD launcher, e.g. /path/to/pmd-bin-7.x/bin/pmd")
    ap.add_argument("--dir", required=True, help="Absolute path to submissions directory (CPD --dir)")
    ap.add_argument("--language", default="python", help="CPD --language (default: python)")
    ap.add_argument("--metrics-script", required=True, help="Absolute path to pmd_cpd_metrics_calc.py")
    ap.add_argument("--results-root", required=True, help="Absolute folder to store per-run XMLs")
    ap.add_argument("--out", required=True, help="Absolute path to output CSV")
    ap.add_argument("--min-tokens", nargs="*", type=int, help="Custom values (default: 5 10 25 40 60)")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Custom thresholds (default: 50 55 ... 95)")
    args = ap.parse_args()

    # Validate tools / paths
    if shutil.which(args.pmd) is None and not os.path.isfile(args.pmd):
        sys.exit(f"Error: PMD launcher not found: {args.pmd}")
    if not os.path.isdir(args.dir):
        sys.exit(f"Error: --dir is not a directory: {args.dir}")
    if not os.path.isfile(args.metrics_script):
        sys.exit(f"Error: metrics script not found: {args.metrics_script}")
    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    min_tokens_list = args.min_tokens if args.min_tokens else DEFAULT_MIN_TOKENS
    thresholds      = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS

    dataset = os.path.basename(os.path.normpath(args.dir))
    rows: List[List[str]] = [["language","min_tokens","threshold_pct","tp","fp","fn","precision","recall","f1"]]

    for mt in min_tokens_list:
        print(f"[+] CPD --minimum-tokens={mt}", file=sys.stderr)
        rc, xml_stdout, xml_stderr = run_cpd(args.pmd, args.dir, args.language, mt)
        xml_path = os.path.join(args.results_root, f"{dataset}_mt{mt}.xml")

        # Treat as success if we got a report (even if rc != 0)
        if not xml_stdout.strip():
            print(f"[!] CPD produced no XML for min_tokens={mt}. rc={rc}\n{xml_stderr}", file=sys.stderr)
            for th in thresholds:
                rows.append([args.language, mt, th, "", "", "", "", "", ""])
            continue

        try:
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xml_stdout)
        except OSError as e:
            print(f"[!] Cannot write XML {xml_path}: {e}", file=sys.stderr)
            for th in thresholds:
                rows.append([args.language, mt, th, "", "", "", "", "", ""])
            continue

        for th in thresholds:
            tp, fp, fn, p, r, f1, raw = run_metrics(args.metrics_script, xml_path, th)
            if tp is None:
                print(f"[!] Parse error mt={mt} threshold={th}%\n{raw}", file=sys.stderr)
                rows.append([args.language, mt, th, "", "", "", "", "", ""])
            else:
                rows.append([args.language, mt, th, tp, fp, fn, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[✓] Wrote {args.out}")

if __name__ == "__main__":
    main()
