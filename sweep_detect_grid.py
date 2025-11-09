#!/usr/bin/env python3
"""
Sweep detect_clone_cli_v8/8.2/9 over --min-tokens and thresholds; write TP/FP/FN/Precision/Recall/F1 to CSV.

What it does
------------
- Runs your detector once per min_tokens ∈ {5,10,15,20} (configurable).
- Forces detector output threshold to 0.0 to collect a superset of pairs;
  then evaluates at thresholds {50,55,...,95}% offline.
- Computes gold using a same-group regex on *basenames* (1st capture group).
- Ignores "unclassified" pairs (either file doesn't match the regex) in FP/TN counts,
  and computes FN against all gold-positive file pairs.
- Robustly parses detector rows whether they are tab- or space-separated, and
  also tolerates extra trailing columns (e.g., --debug-components).

CSV schema
----------
language,min_tokens,threshold_pct,tp,fp,fn,precision,recall,f1,raw_pairs_txt

Example
-------
python sweep_detect_v8_grid.py \
  --script ./detect_clone_cli_v9.py \
  --dir /abs/path/to/mix \
  --extensions .py .java .js .cpp \
  --mode hybrid \
  --results-root /abs/path/to/result-v8/sweep \
  --out /abs/path/to/final-v8/mix_metrics.csv \
  --language-label python \
  --group-regex '^(?i)(C\\d+)(?:[_-]|(?=[A-Z]))' \
  --prefilter-topM 50 --mutual-nearest
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

# ------------------------------- defaults --------------------------------------

DEFAULT_MIN_TOKENS = [5, 10, 15, 20]
DEFAULT_THRESHOLDS = [50,55,60,65,70,75,80,85,90,95]

# Accept both TSV and "aligned with spaces", and tolerate trailing columns
#   <sim> <left> <right> [optional extras...]
PAIR_RE = re.compile(
    r"""^\s*
        (?P<sim>\d+(?:\.\d+)?)          # similarity (float)
        [\t ]+                          # separator(s)
        (?P<left>\S+)                   # left path (no spaces)
        [\t ]+                          # separator(s)
        (?P<right>\S+)                  # right path (no spaces)
        (?:\s+.*)?$                     # ignore anything else
    """,
    re.VERBOSE,
)

# ------------------------------- runner ----------------------------------------

def _abspath(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p

def run_detector(
    script: str,
    root_dir: str,
    extensions: List[str],
    mode: str,
    min_tokens: int,
    *,
    model: Optional[str],
    prefilter_topM: int,
    mutual_nearest: bool,
    extra_args: List[str],
) -> Tuple[int, str, str]:
    """
    Call your detector and capture stdout.
    We explicitly pass --threshold 0.0 and --topk 0 to collect as many pairs as possible.
    """
    cmd = [sys.executable, script,
           "--dir", root_dir,
           "--extensions", *extensions,
           "--mode", mode,
           "--min-tokens", str(min_tokens),
           "--prefilter-topM", str(prefilter_topM),
           "--threshold", "0.0",
           "--topk", "0"]
    if mutual_nearest:
        cmd.append("--mutual-nearest")
    if model:
        cmd += ["--model", model]
    if extra_args:
        cmd += extra_args

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

# ------------------------------ parsing ----------------------------------------

def parse_pairs(stdout_text: str) -> List[Tuple[str, str, float]]:
    """
    Extract lines of the form:
        <similarity> <left_path> <right_path> ...
    Works with tabs or spaces and ignores extra columns.
    """
    results: List[Tuple[str, str, float]] = []
    for line in stdout_text.splitlines():
        m = PAIR_RE.match(line.rstrip())
        if not m:
            continue
        try:
            sim = float(m.group("sim"))
        except ValueError:
            continue
        left = m.group("left").strip()
        right = m.group("right").strip()
        if left and right and left != right:
            results.append((_abspath(left), _abspath(right), sim))
    return results

def list_files(root_dir: str, extensions: List[str]) -> List[str]:
    out: List[str] = []
    root_dir = _abspath(root_dir)
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.startswith("."):
                continue
            if extensions and not any(fn.endswith(ext) for ext in extensions):
                continue
            p = os.path.join(r, fn)
            if os.path.isfile(p):
                out.append(_abspath(p))
    return sorted(out)

# ------------------------------ grouping / gold --------------------------------

def compile_group_regex(rx: str) -> re.Pattern:
    try:
        pat = re.compile(rx)
    except re.error as e:
        raise SystemExit(f"Invalid --group-regex: {e}")
    if pat.groups < 1:
        raise SystemExit("--group-regex must have at least one capturing group; e.g. '^(C\\d+)(?:[_-]|(?=[A-Z]))'")
    return pat

def basename_group(path: str, pat: re.Pattern) -> Optional[str]:
    name = os.path.basename(path)
    m = pat.search(name)
    return m.group(1) if m else None

def build_gold_pairs(files: List[str], pat: re.Pattern) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    """
    Return (group -> files), and list of all unordered pairs (a,b) with same group.
    """
    groups: Dict[str, List[str]] = {}
    for f in files:
        g = basename_group(f, pat)
        if g is None:
            continue
        groups.setdefault(g, []).append(f)

    gold_pairs: List[Tuple[str, str]] = []
    for g, lst in groups.items():
        if len(lst) >= 2:
            L = sorted(_abspath(x) for x in lst)
            for a, b in combinations(L, 2):
                gold_pairs.append((a, b))
    return groups, gold_pairs

# ------------------------------ metrics ----------------------------------------

def evaluate_threshold(
    pairs: List[Tuple[str, str, float]],
    files: List[str],
    pat: re.Pattern,
    thr: float,
) -> Tuple[int, int, int, float, float, float]:
    """
    Compute TP/FP/FN at threshold 'thr' (0..1).
    - Prediction set = { (a,b) in printed pairs : sim >= thr }.
    - Gold = all file pairs with same extracted group from 'files'.
    - FP excludes pairs where either file lacks a group (we ignore "unclassified").
    """
    # gold
    _groups, gold_pairs = build_gold_pairs(files, pat)
    gold_set = set(tuple(sorted(t)) for t in gold_pairs)

    # predicted
    pred = [ (a,b) for (a,b,s) in pairs if s >= thr ]
    pred_set = set(tuple(sorted((_abspath(a), _abspath(b)))) for (a,b) in pred)

    # compute per-pair groups to decide FP eligibility
    def g(path: str) -> Optional[str]:
        return basename_group(path, pat)

    tp = 0
    fp = 0
    for a,b in pred_set:
        ga, gb = g(a), g(b)
        if ga is None or gb is None:
            # ignore unclassified predicted pairs in FP/TN accounting
            continue
        if ga == gb:
            tp += 1
        else:
            fp += 1

    fn = len(gold_set - pred_set)

    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return tp, fp, fn, precision, recall, f1

# ------------------------------ main -------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Sweep detect_clone_cli_v8/v8.2/v9 (min-tokens, thresholds) → CSV metrics.")
    ap.add_argument("--script", default="./detect_clone_cli_v8.py",
                    help="Path to detector script (v8/v8.2/v9). Example: ./detect_clone_cli_v9.py")
    ap.add_argument("--dir", required=True, help="Root directory of submissions")
    ap.add_argument("--extensions", nargs="+", required=True,
                    help="File extensions to include (e.g., .py .java .js .cpp .cc .cxx .hpp .h)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid","semantic","semantic-plus","structural"],
                    help="Detector mode")
    ap.add_argument("--model", default=None, help="Optional SentenceTransformer model to pass to detector")
    ap.add_argument("--prefilter-topM", type=int, default=50, help="Detector --prefilter-topM")
    ap.add_argument("--mutual-nearest", action="store_true", help="Detector --mutual-nearest")
    ap.add_argument("--extra-args", nargs="*", default=[],
                    help="Extra args to pass to detector (e.g. --no-strip-comments --short-token-gate 5)")
    ap.add_argument("--results-root", default="result-v8/sweep", help="Where raw detector outputs are saved")
    ap.add_argument("--out", default="v8_grid_metrics.csv", help="Output CSV path")
    ap.add_argument("--language-label", default="python", help="CSV label for the 'language' column (cosmetic)")
    ap.add_argument("--min-tokens", nargs="*", type=int, help="Values to sweep (default 5 10 15 20)")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Percent thresholds (default 50 55 ... 95)")
    ap.add_argument("--group-regex", default=r"^(C\d+)(?:[_-]|(?=[A-Z]))",
                    help=r"Regex with ONE capturing group for 'group' from basename. "
                         r"Use (?i) at the start for case-insensitive groups.")
    ap.add_argument("--clean", action="store_true", help="Delete per-run outputs before running")
    args = ap.parse_args()

    if not os.path.isfile(args.script):
        sys.exit(f"Detector not found: {args.script}")
    if not os.path.isdir(args.dir):
        sys.exit(f"--dir not found: {args.dir}")

    min_tokens_list = args.min_tokens if args.min_tokens else DEFAULT_MIN_TOKENS
    thresholds_pct  = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    thresholds_dec  = [ t/100.0 for t in thresholds_pct ]

    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Build full file list once for gold (mirror detector's extension filtering).
    files = list_files(args.dir, args.extensions)
    group_rx = compile_group_regex(args.group_regex)

    rows: List[List[str]] = [["language","min_tokens","threshold_pct","tp","fp","fn","precision","recall","f1","raw_pairs_txt"]]
    dataset_name = os.path.basename(os.path.normpath(_abspath(args.dir)))

    for mt in min_tokens_list:
        run_base = os.path.join(args.results_root, f"{dataset_name}_t{mt}")
        txt_path = f"{run_base}.txt"

        if args.clean:
            try:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            except OSError:
                pass

        print(f"[+] {os.path.basename(args.script)}  min_tokens={mt}  mode={args.mode}", file=sys.stderr)
        rc, out, err = run_detector(
            script=args.script,
            root_dir=_abspath(args.dir),
            extensions=args.extensions,
            mode=args.mode,
            min_tokens=mt,
            model=args.model,
            prefilter_topM=args.prefilter_topM,
            mutual_nearest=args.mutual_nearest,
            extra_args=args.extra_args,
        )

        # Persist raw detector output for inspection/repro
        try:
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(out)
                if err.strip():
                    fh.write("\n\n# --- stderr ---\n")
                    fh.write(err)
        except OSError as e:
            print(f"[!] Cannot write {txt_path}: {e}", file=sys.stderr)

        if rc != 0 and not out.strip():
            print(f"[!] Detector failed (rc={rc}). stderr:\n{err}", file=sys.stderr)
            for tpct in thresholds_pct:
                rows.append([args.language_label, mt, tpct, "", "", "", "", "", "", txt_path])
            continue

        pairs = parse_pairs(out)
        if not pairs:
            print(f"[!] No parseable pairs found for min_tokens={mt}. stderr:\n{err}", file=sys.stderr)
            for tpct in thresholds_pct:
                rows.append([args.language_label, mt, tpct, "", "", "", "", "", "", txt_path])
            continue

        for tpct, t in zip(thresholds_pct, thresholds_dec):
            tp, fp, fn, p, r, f1 = evaluate_threshold(pairs, files, group_rx, t)
            rows.append([
                args.language_label, mt, tpct,
                tp, fp, fn,
                f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}",
                txt_path
            ])

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[✓] Wrote {out_path}")

if __name__ == "__main__":
    main()
