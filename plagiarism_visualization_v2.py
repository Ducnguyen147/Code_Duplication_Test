#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_f1_summaries.py

Generates figures from one or more tool CSVs:
  1) chart1_highest_f1_per_tool.png
  2) chart2_f1_vs_threshold.png           <-- every point is annotated with parameters
  3) chart3_param_influence_by_tool.png   <-- single-parameter tools
  4) chart3b_dolos_kw_heatmap.png         <-- (k, w) heatmap if Dolos is present

CSV expectations per tool (columns used by this script):
- Dolos:        k, w, threshold_pct, f1
- JPlag:        t, threshold_pct, f1
- PMD-CPD:      min_tokens, threshold_pct, f1
- jscpd:        min_tokens (alias: minTokens), threshold_pct, f1
- Embeddings:   EITHER k, w, threshold_pct, f1   <-- supported (hybrid sweep)
                 OR   min_tokens, threshold_pct, f1

If a column 'param_label' is already present, it's used; otherwise we synthesize one.

Example:
  python plot_f1_summaries.py \
    --dolos results_dolos.csv \
    --jplag results_jplag.csv \
    --pmd-cpd results_pmd.csv \
    --jscpd results_jscpd.csv \
    --emb results_emb.csv \
    --language "Java" \
    --out-dir ./out
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _to_num(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _format_params_from_row(r: pd.Series) -> str:
    """
    Build a compact parameter label from the row, even if param_label is missing.
    Priority: (k,w) > t > min_tokens.
    """
    parts: List[str] = []
    if "k" in r and pd.notna(r["k"]): parts.append(f"k={int(r['k'])}")
    if "w" in r and pd.notna(r["w"]): parts.append(f"w={int(r['w'])}")
    if "t" in r and pd.notna(r["t"]): parts.append(f"t={int(r['t'])}")
    if "min_tokens" in r and pd.notna(r["min_tokens"]): parts.append(f"min_tokens={int(r['min_tokens'])}")
    return "(" + ",".join(parts) + ")" if parts else ""

def read_tool_csv(path: str, tool: str) -> pd.DataFrame:
    """Load a tool CSV and standardize columns."""
    df = pd.read_csv(path)
    df["tool"] = tool

    # Normalize possible aliases
    if "minTokens" in df.columns and "min_tokens" not in df.columns:
        df["min_tokens"] = df["minTokens"]

    # Make sure relevant numeric columns are numeric
    for c in ("threshold_pct", "f1", "k", "w", "t", "min_tokens"):
        _to_num(df, c)

    # Synthesize a param_label when needed (tool-specific)
    if tool == "Dolos":
        required = {"k", "w"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Dolos CSV missing columns: {missing}")
        if "param_label" not in df.columns or df["param_label"].isna().all():
            df["param_label"] = df.apply(lambda r: f"(k={int(r['k'])},w={int(r['w'])})", axis=1)

    elif tool == "JPlag":
        if "t" not in df.columns:
            raise ValueError("JPlag CSV missing column: 't'")
        if "param_label" not in df.columns or df["param_label"].isna().all():
            df["param_label"] = df.apply(lambda r: f"-t={int(r['t'])}", axis=1)

    elif tool in ("PMD-CPD", "jscpd"):
        if "min_tokens" not in df.columns:
            raise ValueError(f"{tool} CSV missing column: 'min_tokens'")
        if "param_label" not in df.columns or df["param_label"].isna().all():
            df["param_label"] = df.apply(lambda r: f"--min-tokens={int(r['min_tokens'])}", axis=1)

    elif tool == "Embeddings":
        # NEW: prefer (k,w) if present, else min_tokens, else blank
        if "param_label" not in df.columns or df["param_label"].isna().all():
            has_kw = {"k", "w"}.issubset(df.columns) and df[["k", "w"]].notna().all(axis=1).any()
            if has_kw:
                df["param_label"] = df.apply(
                    lambda r: f"(k={int(r['k'])},w={int(r['w'])})" if pd.notna(r.get("k")) and pd.notna(r.get("w")) else "",
                    axis=1,
                )
            elif "min_tokens" in df.columns and df["min_tokens"].notna().any():
                df["param_label"] = df.apply(
                    lambda r: f"min_tokens={int(r['min_tokens'])}" if pd.notna(r.get("min_tokens")) else "",
                    axis=1,
                )
            else:
                df["param_label"] = ""
    else:
        if "param_label" not in df.columns:
            df["param_label"] = ""

    return df


def best_per_threshold(df_tool: pd.DataFrame) -> pd.DataFrame:
    """Return one row per threshold: the row with max F1 across parameters."""
    idx = df_tool.groupby("threshold_pct")["f1"].idxmax()
    best = df_tool.loc[idx].sort_values("threshold_pct").copy()
    return best


def best_per_param(df_tool: pd.DataFrame, param_cols: List[str]) -> pd.DataFrame:
    """
    Return one row per parameter configuration (max F1 across thresholds).
    """
    if not all(col in df_tool.columns for col in param_cols):
        return pd.DataFrame(columns=list(param_cols) + ["f1", "threshold_pct", "param_label", "tool"])
    idx = df_tool.groupby(param_cols)["f1"].idxmax()
    best = df_tool.loc[idx].sort_values(param_cols).copy()
    return best


def get_param_cols(tool: str, df: pd.DataFrame) -> List[str]:
    """Which parameter columns drive the 'parameter influence' charts?"""
    mapping = {
        "Dolos": ["k", "w"],
        "JPlag": ["t"],
        "PMD-CPD": ["min_tokens"],
        "jscpd": ["min_tokens"],
        # Embeddings: show min_tokens curve *if* present; (k,w) is handled via heatmap-like view or the annotations.
        "Embeddings": ["min_tokens"] if "min_tokens" in df.columns else [],
    }
    return [c for c in mapping.get(tool, []) if c in df.columns]


PARAM_LABELS = {
    "t": "Minimum matching tokens (-t)",
    "min_tokens": "Minimum tokens (--min-tokens)",
    "k": "k-gram length (k)",
    "w": "Winnowing window (w)",
}

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot F1 summaries from tool CSVs.")
    parser.add_argument("--dolos", type=str, default=None, help="Path to Dolos CSV")
    parser.add_argument("--jplag", type=str, default=None, help="Path to JPlag CSV")
    parser.add_argument("--pmd-cpd", dest="pmd_cpd", type=str, default=None, help="Path to PMD-CPD CSV")
    parser.add_argument("--jscpd", type=str, default=None, help="Path to jscpd CSV")
    parser.add_argument("--emb", "--embeddings", dest="emb", type=str, default=None,
                        help="Path to embeddings algorithm CSV")
    parser.add_argument("--language", type=str, default=None,
                        help="Language label to show in the charts, e.g. 'Java'")
    parser.add_argument("--out-dir", type=str, default="/out", help="Output directory (default: /out)")
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
        thr = int(row["threshold_pct"]) if pd.notna(row["threshold_pct"]) else "NA"
        raw = str(row.get("param_label", "") or "").strip()
        label = raw if raw else _format_params_from_row(row)
        txt = f"thr={thr}\n{label}"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 txt, ha="center", va="bottom", fontsize=9)

    fig1.tight_layout()
    fig1.savefig(out_dir / "chart1_highest_f1_per_tool.png", dpi=150)

    # ---------- Chart 2: F1 vs Threshold (best parameter per threshold) ----------
    fig2 = plt.figure(figsize=(10, 7))
    ax2 = fig2.add_subplot(111)

    tools_in_order = ["Embeddings", "Dolos", "JPlag", "PMD-CPD", "jscpd"]
    tools_available = [t for t in tools_in_order if t in set(tools)]

    for tool in tools_available:
        tdf = all_df[all_df["tool"] == tool].copy()
        best = best_per_threshold(tdf)
        ax2.plot(best["threshold_pct"], best["f1"], marker="o", label=tool)

        # Annotate each point with the chosen parameter setting at that threshold.
        # For Embeddings, this now shows (k,w) when present in the CSV.
        for _, r in best.iterrows():
            raw = str(r.get("param_label", "") or "").strip()
            label = raw if raw else _format_params_from_row(r)
            ax2.annotate(
                label,
                xy=(r["threshold_pct"], r["f1"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax2.set_title("F1 vs Threshold (best parameter per threshold)" + lang_suffix)
    ax2.set_xlabel("Threshold (%)")
    ax2.set_ylabel("F1")
    ax2.legend(loc="upper right")

    fig2.tight_layout()
    fig2.savefig(out_dir / "chart2_f1_vs_threshold.png", dpi=150)

    # ---------- Chart 3A: Parameter influence (one-parameter tools) ----------
    single_param_tools = []
    for tool in tools_available:
        tdf = all_df[all_df["tool"] == tool].copy()
        param_cols = get_param_cols(tool, tdf)
        if len(param_cols) == 1:
            best = best_per_param(tdf, param_cols)
            if not best.empty:
                single_param_tools.append((tool, param_cols[0], best))

    if single_param_tools:
        n = len(single_param_tools)
        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))
        fig3, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 4.5 * nrows))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        for ax, (tool, pcol, best) in zip(axes, single_param_tools):
            best = best.sort_values(pcol)
            ax.plot(best[pcol], best["f1"], marker="o")
            for _, r in best.iterrows():
                thr = int(r["threshold_pct"]) if pd.notna(r["threshold_pct"]) else "NA"
                ax.annotate(f"thr={thr}", xy=(r[pcol], r["f1"]),
                            xytext=(0, 6), textcoords="offset points",
                            ha="center", fontsize=8)
            ax.set_title(f"{tool}: F1 vs {PARAM_LABELS.get(pcol, pcol)}" + lang_suffix)
            ax.set_xlabel(PARAM_LABELS.get(pcol, pcol))
            ax.set_ylabel("F1")

        # Hide any leftover empty axes
        for ax in axes[len(single_param_tools):]:
            ax.axis("off")

        fig3.tight_layout()
        fig3.savefig(out_dir / "chart3_param_influence_by_tool.png", dpi=150)

    # ---------- Chart 3B: Dolos (k, w) heatmap ----------
    if "Dolos" in tools_available:
        ddf = all_df[all_df["tool"] == "Dolos"].copy()
        if {"k", "w"}.issubset(ddf.columns):
            best_kw = best_per_param(ddf, ["k", "w"])
            if not best_kw.empty:
                pivot = best_kw.pivot(index="k", columns="w", values="f1").sort_index().sort_index(axis=1)

                figH = plt.figure(figsize=(9, 6))
                axH = figH.add_subplot(111)
                im = axH.imshow(pivot.values, origin="lower", aspect="auto")
                axH.set_xticks(np.arange(len(pivot.columns)))
                axH.set_xticklabels(pivot.columns.astype(int))
                axH.set_yticks(np.arange(len(pivot.index)))
                axH.set_yticklabels(pivot.index.astype(int))
                axH.set_xlabel(PARAM_LABELS["w"])
                axH.set_ylabel(PARAM_LABELS["k"])
                axH.set_title("Dolos: Max F1 per (k, w)" + lang_suffix)

                # Annotate cells with F1 values
                for i, k in enumerate(pivot.index):
                    for j, w in enumerate(pivot.columns):
                        val = pivot.loc[k, w]
                        if pd.notna(val):
                            axH.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

                figH.colorbar(im, ax=axH, shrink=0.85, label="F1")
                figH.tight_layout()
                figH.savefig(out_dir / "chart3b_dolos_kw_heatmap.png", dpi=150)

    print(f"Saved charts to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
