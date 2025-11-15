#!/usr/bin/env python3
import argparse
from pathlib import Path
from math import ceil, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _to_num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _rename_if_present(df, old_names, new_name):
    for c in df.columns:
        if c.lower() in old_names and new_name not in df.columns:
            df = df.rename(columns={c: new_name})
    return df

def _normalize_threshold_pct(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().max() is not None and s.dropna().max() <= 1.0:
        return s * 100.0
    return s

def _ensure_pr_columns(df: pd.DataFrame) -> pd.DataFrame:
    # If precision/recall present, coerce numeric.
    if {"precision", "recall"}.issubset(df.columns):
        _to_num(df, "precision"); _to_num(df, "recall")
        return df
    # Else compute from tp/fp/fn when available.
    if {"tp", "fp", "fn"}.issubset(df.columns):
        for c in ("tp", "fp", "fn"):
            _to_num(df, c)
        df["precision"] = df["tp"] / (df["tp"] + df["fp"]).replace(0, np.nan)
        df["recall"] = df["tp"] / (df["tp"] + df["fn"]).replace(0, np.nan)
    return df

def read_tool_csv(path: str, tool: str) -> pd.DataFrame:
    """Load a tool CSV and standardize columns + parameter labels."""
    df = pd.read_csv(path)
    df["tool"] = tool

    # Normalize common synonyms
    df = _rename_if_present(df,
        {"threshold_pct","threshold","threshold_percent","sim_threshold",
         "similarity_threshold","thresh","t_pct"},
        "threshold_pct")
    df = _rename_if_present(df, {"f1","f1_score"}, "f1")

    for c in ("threshold_pct", "f1"):
        _to_num(df, c)

    if "threshold_pct" in df.columns:
        df["threshold_pct"] = _normalize_threshold_pct(df["threshold_pct"])

    # Tool-specific parameter labels
    if tool == "Dolos":
        required = {"k", "w"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Dolos CSV missing columns: {missing}")
        df["param_label"] = df.apply(lambda r: f"(k={int(r['k'])},w={int(r['w'])})", axis=1)

    elif tool == "JPlag":
        if "t" not in df.columns:
            raise ValueError("JPlag CSV missing column: 't'")
        df["param_label"] = df.apply(lambda r: f"-t={int(r['t'])}", axis=1)

    elif tool in ("PMD-CPD", "jscpd"):
        if "min_tokens" not in df.columns:
            raise ValueError(f"{tool} CSV missing column: 'min_tokens'")
        df["param_label"] = df.apply(lambda r: f"--min-tokens={int(r['min_tokens'])}", axis=1)

    elif tool == "Embeddings":
        if "min_tokens" in df.columns:
            df["param_label"] = df.apply(lambda r: f"min_tokens={int(r['min_tokens'])}", axis=1)
        else:
            df["param_label"] = ""
    else:
        df["param_label"] = ""

    # Optional extras (precision/recall or tp/fp/fn)
    df = _ensure_pr_columns(df)

    return df

def best_per_threshold(df_tool: pd.DataFrame) -> pd.DataFrame:
    """Return 1 row per threshold (max F1 across parameter settings)."""
    # Drop NaN F1/threshold rows to avoid idxmax errors
    df_tool = df_tool.dropna(subset=["f1", "threshold_pct"])
    idx = df_tool.groupby("threshold_pct")["f1"].idxmax()
    best = df_tool.loc[idx].sort_values("threshold_pct").copy()
    return best

def band_per_threshold(df_tool: pd.DataFrame, q_lo=0.10, q_hi=0.90) -> pd.DataFrame:
    """Aggregate F1 across parameter settings at each threshold: median & quantile band."""
    df_tool = df_tool.dropna(subset=["f1", "threshold_pct"]).copy()
    agg = (df_tool
           .groupby("threshold_pct")["f1"]
           .agg(median="median", lo=lambda s: s.quantile(q_lo), hi=lambda s: s.quantile(q_hi))
           .reset_index()
           .sort_values("threshold_pct"))
    return agg

def opt_point_from_best(best_df: pd.DataFrame):
    """Given best-per-threshold rows, return (thr, f1, row) at global max F1."""
    if best_df.empty:
        return None, None, None
    idx = best_df["f1"].idxmax()
    row = best_df.loc[idx]
    return float(row["threshold_pct"]), float(row["f1"]), row

def savefig(fig, outpath_png: Path, outpath_svg: Path, dpi: int=150):
    fig.tight_layout()
    fig.savefig(outpath_png, dpi=dpi)
    fig.savefig(outpath_svg)

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot F1 summaries from tool CSVs (with robustness & PR).")
    parser.add_argument("--dolos", type=str, default=None, help="Path to Dolos CSV")
    parser.add_argument("--jplag", type=str, default=None, help="Path to JPlag CSV")
    parser.add_argument("--pmd-cpd", dest="pmd_cpd", type=str, default=None, help="Path to PMD-CPD CSV")
    parser.add_argument("--jscpd", type=str, default=None, help="Path to jscpd CSV")
    parser.add_argument("--emb", "--embeddings", dest="emb", type=str, default=None,
                        help="Path to embeddings algorithm CSV")
    parser.add_argument("--language", type=str, default=None,
                        help="Language label to show in the charts, e.g. 'Python'")
    parser.add_argument("--out-dir", type=str, default="/out", help="Output directory (default: /out)")
    parser.add_argument("--dpi", type=int, default=150, help="Image DPI (default: 150)")
    parser.add_argument("--format", type=str, default="png", choices=["png","svg","both"],
                        help="Image format to save (default: png)")
    args = parser.parse_args()

    frames, tools = [], []

    if args.dolos:
        frames.append(read_tool_csv(args.dolos, "Dolos")); tools.append("Dolos")
    if args.jplag:
        frames.append(read_tool_csv(args.jplag, "JPlag")); tools.append("JPlag")
    if args.pmd_cpd:
        frames.append(read_tool_csv(args.pmd_cpd, "PMD-CPD")); tools.append("PMD-CPD")
    if args.jscpd:
        frames.append(read_tool_csv(args.jscpd, "jscpd")); tools.append("jscpd")
    if args.emb:
        frames.append(read_tool_csv(args.emb, "Embeddings")); tools.append("Embeddings")

    if not frames:
        raise SystemExit("No input CSVs provided. Pass at least one of --dolos/--jplag/--pmd-cpd/--jscpd/--emb.")

    all_df = pd.concat(frames, ignore_index=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    def outpaths(filename_stem: str):
        p_png = out_dir / f"{filename_stem}.png"
        p_svg = out_dir / f"{filename_stem}.svg"
        return p_png, p_svg

    lang_suffix = f" — {args.language}" if args.language else ""

    # ---------- Chart 1: Highest F1 per tool ----------
    best_rows = all_df.loc[all_df.groupby("tool")["f1"].idxmax()].copy()
    best_rows.sort_values("f1", ascending=False, inplace=True)

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    bars = ax1.bar(best_rows["tool"], best_rows["f1"])
    ax1.set_title("Highest F1 by Tool" + lang_suffix)
    ax1.set_xlabel("Tool")
    ax1.set_ylabel("F1")

    for bar, (_, row) in zip(bars, best_rows.iterrows()):
        thr = int(row["threshold_pct"]) if pd.notna(row.get("threshold_pct")) else "NA"
        txt = f"thr={thr}\n{row['param_label']}"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 txt, ha="center", va="bottom", fontsize=9)

    savefig(fig1, *outpaths("chart1_highest_f1_per_tool"), dpi=args.dpi)

    # ---------- Chart 2: F1 vs Threshold (best parameter per threshold) ----------
    # (Plus an operating-point marker per tool.)
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111)

    tools_in_order = ["Embeddings", "Dolos", "JPlag", "PMD-CPD", "jscpd"]
    tools_available = [t for t in tools_in_order if t in set(tools)]

    opt_summary = []  # collect per-tool operating points for CSV

    for tool in tools_available:
        tdf = all_df[all_df["tool"] == tool].copy()
        if "threshold_pct" not in tdf.columns:
            continue
        best = best_per_threshold(tdf)
        if best.empty:
            continue
        ax2.plot(best["threshold_pct"], best["f1"], marker="o", linewidth=1.5, label=tool)
        # Operating point
        thr_star, f1_star, row_star = opt_point_from_best(best)
        if thr_star is not None:
            ax2.plot([thr_star], [f1_star], marker="D", markersize=7)
            ax2.annotate(f"{tool}\nthr={int(thr_star)}",
                         xy=(thr_star, f1_star), xytext=(5, 6),
                         textcoords="offset points", fontsize=8)
            opt_summary.append({
                "tool": tool,
                "best_f1": f1_star,
                "best_threshold_pct": thr_star,
                "param_label_at_best": row_star.get("param_label", "")
            })

    ax2.set_title("F1 vs Threshold (best parameter per threshold)" + lang_suffix)
    ax2.set_xlabel("Threshold (%)")
    ax2.set_ylabel("F1")
    ax2.legend(loc="upper right", ncol=1, frameon=True)
    savefig(fig2, *outpaths("chart2_f1_vs_threshold_best"), dpi=args.dpi)

    # ---------- Chart 2b: Robustness bands across parameters ----------
    fig2b = plt.figure(figsize=(10, 7))
    ax2b = fig2b.add_subplot(111)
    for tool in tools_available:
        tdf = all_df[(all_df["tool"] == tool)].copy()
        if "threshold_pct" not in tdf.columns:
            continue
        band = band_per_threshold(tdf)
        if band.empty:
            continue
        ax2b.plot(band["threshold_pct"], band["median"], marker=".", label=f"{tool} (median)")
        ax2b.fill_between(band["threshold_pct"], band["lo"], band["hi"], alpha=0.15, linewidth=0)

    ax2b.set_title("F1 vs Threshold — Robustness Across Hyperparameters" + lang_suffix)
    ax2b.set_xlabel("Threshold (%)")
    ax2b.set_ylabel("F1 (median and 10–90% band)")
    ax2b.legend(loc="upper right", frameon=True)
    savefig(fig2b, *outpaths("chart2b_f1_vs_threshold_bands"), dpi=args.dpi)

    # ---------- Chart 3 (optional): Precision–Recall (best parameter per threshold) ----------
    has_pr_any = False
    for tool in tools_available:
        tdf = all_df[all_df["tool"] == tool]
        if {"precision", "recall"}.issubset(tdf.columns):
            has_pr_any = True
            break

    if has_pr_any:
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        for tool in tools_available:
            tdf = all_df[all_df["tool"] == tool]
            if not {"precision", "recall"}.issubset(tdf.columns):
                continue
            # pick best params per threshold to build a single PR envelope
            if "threshold_pct" in tdf.columns:
                best = best_per_threshold(tdf.dropna(subset=["precision","recall"]))
                pr = best.dropna(subset=["precision","recall"]).sort_values("recall")
                if pr.empty: 
                    continue
            else:
                pr = tdf.dropna(subset=["precision","recall"]).sort_values("recall")
            ax3.plot(pr["recall"], pr["precision"], marker=".", linewidth=1.5, label=tool)

        ax3.set_title("Precision–Recall (best parameter per threshold)" + lang_suffix)
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.legend(loc="lower left", frameon=True)
        savefig(fig3, *outpaths("chart3_precision_recall"), dpi=args.dpi)

    # ---------- Chart 4: Small multiples (per-tool best-per-threshold curves) ----------
    n = len(tools_available)
    if n > 1:
        cols = int(ceil(sqrt(n)))
        rows = int(ceil(n / cols))
        fig4 = plt.figure(figsize=(4*cols, 3.5*rows))
        axes = []
        for i in range(1, n+1):
            axes.append(fig4.add_subplot(rows, cols, i))
        for ax, tool in zip(axes, tools_available):
            tdf = all_df[all_df["tool"] == tool].copy()
            if "threshold_pct" not in tdf.columns:
                ax.set_title(tool + " (no thresholds)")
                continue
            best = best_per_threshold(tdf)
            if best.empty:
                ax.set_title(tool + " (no data)")
                continue
            ax.plot(best["threshold_pct"], best["f1"], marker="o", linewidth=1.5)
            thr_star, f1_star, _ = opt_point_from_best(best)
            if thr_star is not None:
                ax.plot([thr_star], [f1_star], marker="D", markersize=6)
                ax.annotate(f"thr={int(thr_star)}", xy=(thr_star, f1_star),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
            ax.set_title(tool)
            ax.set_xlabel("Threshold (%)")
            ax.set_ylabel("F1")
        savefig(fig4, *outpaths("chart4_small_multiples_f1_vs_threshold"), dpi=args.dpi)

    # ---------- CSV summary of operating points ----------
    if opt_summary:
        pd.DataFrame(opt_summary).sort_values("best_f1", ascending=False)\
          .to_csv(out_dir / "summary_best_by_tool.csv", index=False)

    # Done
    suffix = "(+SVG)" if args.format in ("both","svg") else ""
    print(f"Saved charts to: {out_dir.resolve()} {suffix}")

if __name__ == "__main__":
    main()
