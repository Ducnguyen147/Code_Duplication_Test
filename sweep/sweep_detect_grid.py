#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, os, re, subprocess, sys
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

# ------------------------------- grids & defaults -------------------------------
GRID_K = [12, 17, 23]
GRID_W = [17, 25, 40]
DEFAULT_THRESHOLDS = [50,55,60,65,70,75,80,85,90,95]

PAIR_RE = re.compile(r"""^\s*(?P<sim>\d+(?:\.\d+)?)[\t ]+(?P<left>\S+)[\t ]+(?P<right>\S+)(?:\s+.*)?$""")

# ------------------------------- runner ----------------------------------------
def _abspath(p: str) -> str:
    try: return os.path.abspath(p)
    except Exception: return p

def run_detector_kw(
    script: str,
    root_dir: str,
    extensions: List[str],
    mode: str,
    k: int, w: int,
    *,
    model: Optional[str],
    prefilter_topM: int,
    mutual_nearest: bool,
    extra_args: List[str],
) -> Tuple[int, str, str]:
    """
    Call detector and capture stdout for a specific (k,w).
    We force --threshold 0.0 / --topk 0 (collect all), and (silently) enable fp-in-score.
    """
    cmd = [sys.executable, script,
           "--dir", root_dir,
           "--extensions", *extensions,
           "--mode", mode,
           "--prefilter-topM", str(prefilter_topM),
           "--threshold", "0.0",
           "--topk", "0",
           "--fp-k", str(k), "--fp-w", str(w),
           "--use-fp-in-score", "--fp-sim-mode", "combo", "--fp-weight", "0.35"]
    if mutual_nearest: cmd.append("--mutual-nearest")
    if model: cmd += ["--model", model]
    if extra_args: cmd += extra_args
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr

# ------------------------------ parsing ----------------------------------------
def parse_pairs(stdout_text: str) -> List[Tuple[str, str, float]]:
    out: List[Tuple[str, str, float]] = []
    for line in stdout_text.splitlines():
        m = PAIR_RE.match(line.rstrip()); 
        if not m: continue
        try: sim = float(m.group("sim"))
        except ValueError: continue
        left, right = m.group("left").strip(), m.group("right").strip()
        if left and right and left != right:
            out.append((_abspath(left), _abspath(right), sim))
    return out

def list_files(root_dir: str, extensions: List[str]) -> List[str]:
    items: List[str] = []; root_dir = _abspath(root_dir)
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn.startswith("."): continue
            if extensions and not any(fn.endswith(ext) for ext in extensions): continue
            p = os.path.join(r, fn)
            if os.path.isfile(p): items.append(_abspath(p))
    return sorted(items)

# ------------------------------ grouping / gold --------------------------------
def compile_group_regex(rx: str) -> re.Pattern:
    try: pat = re.compile(rx)
    except re.error as e: raise SystemExit(f"Invalid --group-regex: {e}")
    if pat.groups < 1: raise SystemExit("--group-regex must capture one group id, e.g. '^(C\\d+)'")
    return pat

def basename_group(path: str, pat: re.Pattern) -> Optional[str]:
    m = pat.search(os.path.basename(path)); return m.group(1) if m else None

def build_gold_pairs(files: List[str], pat: re.Pattern) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    groups: Dict[str, List[str]] = {}
    for f in files:
        g = basename_group(f, pat)
        if g is None: continue
        groups.setdefault(g, []).append(f)
    gold_pairs: List[Tuple[str, str]] = []
    for g, lst in groups.items():
        if len(lst) >= 2:
            L = sorted(_abspath(x) for x in lst)
            for i in range(len(L)):
                for j in range(i+1, len(L)):
                    gold_pairs.append((L[i], L[j]))
    return groups, gold_pairs

# ------------------------------ metrics ----------------------------------------
def evaluate_threshold(
    pairs: List[Tuple[str, str, float]], files: List[str], pat: re.Pattern, thr: float,
) -> Tuple[int, int, int, float, float, float]:
    _groups, gold_pairs = build_gold_pairs(files, pat)
    gold_set = set(tuple(sorted(t)) for t in gold_pairs)
    pred = [ (a,b) for (a,b,s) in pairs if s >= thr ]
    pred_set = set(tuple(sorted((_abspath(a), _abspath(b)))) for (a,b) in pred)

    def g(p: str) -> Optional[str]: return basename_group(p, pat)

    tp = fp = 0
    for a,b in pred_set:
        ga, gb = g(a), g(b)
        if ga is None or gb is None: 
            continue
        if ga == gb: tp += 1
        else:        fp += 1

    fn = len(gold_set - pred_set)
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    return tp, fp, fn, precision, recall, f1

# ------------------------------ CSV helpers ------------------------------------
def ensure_csv_ready(path: str, header: List[str], clean: bool) -> None:
    if clean and os.path.exists(path):
        try: os.remove(path)
        except OSError: pass
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    need_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    if need_header:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass

def append_row(fh, writer, row: List[object]) -> None:
    writer.writerow(row)
    fh.flush()
    try: os.fsync(fh.fileno())
    except Exception: pass

# ------------------------------ main -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="(k,w) sweep with incremental CSV appends.")
    ap.add_argument("--script", default="./detect_clone_cli.py", help="Detector script path")
    ap.add_argument("--dir", required=True, help="Root directory of submissions")
    ap.add_argument("--extensions", nargs="+", required=True, help="Extensions to include (e.g., .java)")
    ap.add_argument("--mode", default="hybrid", choices=["hybrid","semantic","semantic-plus","structural"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--prefilter-topM", type=int, default=50)
    ap.add_argument("--mutual-nearest", action="store_true")
    ap.add_argument("--extra-args", nargs="*", default=[], help="Advanced: forwarded to detector")
    ap.add_argument("--results-root", default="result-kw/sweep", help="Where to save raw detector outputs")
    ap.add_argument("--out", default="kw_grid_metrics.csv", help="Output CSV (appended live)")
    ap.add_argument("--language-label", default="python")
    ap.add_argument("--thresholds", nargs="*", type=int, help="Percent thresholds; default 50..95")
    ap.add_argument("--group-regex", default=r"^(C\d+)(?:[_-]|(?=[A-Z]))", help="ONE capturing group for 'group'")
    ap.add_argument("--clean", action="store_true", help="Delete --out before running")
    args = ap.parse_args()

    if not os.path.isfile(args.script): sys.exit(f"Detector not found: {args.script}")
    if not os.path.isdir(args.dir):    sys.exit(f"--dir not found: {args.dir}")

    thresholds_pct = args.thresholds if args.thresholds else DEFAULT_THRESHOLDS
    thresholds_dec = [t/100.0 for t in thresholds_pct]

    files = list_files(args.dir, args.extensions)
    group_rx = compile_group_regex(args.group_regex)

    header = ["language","k","w","threshold_pct","tp","fp","fn","precision","recall","f1","raw_pairs_txt"]
    out_path = os.path.abspath(args.out)
    ensure_csv_ready(out_path, header, clean=args.clean)

    os.makedirs(args.results_root, exist_ok=True)
    dataset_name = os.path.basename(os.path.normpath(_abspath(args.dir)))

    with open(out_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for k, w in product(GRID_K, GRID_W):
            run_base = os.path.join(args.results_root, f"{dataset_name}_k{k}_w{w}")
            txt_path = f"{run_base}.txt"

            print(f"[+] {os.path.basename(args.script)}  (k={k}, w={w})  mode={args.mode}", file=sys.stderr)
            rc, out, err = run_detector_kw(
                script=args.script, root_dir=_abspath(args.dir), extensions=args.extensions, mode=args.mode,
                k=k, w=w, model=args.model, prefilter_topM=args.prefilter_topM,
                mutual_nearest=args.mutual_nearest, extra_args=args.extra_args,
            )

            try:
                with open(txt_path, "w", encoding="utf-8") as t:
                    t.write(out)
                    if err.strip():
                        t.write("\n\n# --- stderr ---\n"); t.write(err)
            except OSError as e:
                print(f"[!] Cannot write {txt_path}: {e}", file=sys.stderr)

            if rc != 0 and not out.strip():
                print(f"[!] Detector failed (rc={rc}). stderr:\n{err}", file=sys.stderr)
                for tpct in thresholds_pct:
                    append_row(fh, writer, [args.language_label, k, w, tpct, "", "", "", "", "", "", txt_path])
                continue

            pairs = parse_pairs(out)
            if not pairs:
                print(f"[!] No parseable pairs for (k={k}, w={w}). stderr:\n{err}", file=sys.stderr)
                for tpct in thresholds_pct:
                    append_row(fh, writer, [args.language_label, k, w, tpct, "", "", "", "", "", "", txt_path])
                continue

            for tpct, t in zip(thresholds_pct, thresholds_dec):
                tp, fp, fn, p, r, f1 = evaluate_threshold(pairs, files, group_rx, t)
                append_row(fh, writer, [
                    args.language_label, k, w, tpct,
                    tp, fp, fn,
                    f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}",
                    txt_path
                ])
                print(f"[→] wrote row k={k} w={w} t={tpct}%  F1={f1:.4f}", file=sys.stderr)

    print(f"[✓] Appending live to {out_path}")
    print("[i] tail -f the CSV while it runs.")

if __name__ == "__main__":
    main()
