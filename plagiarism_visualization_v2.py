#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Dict

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
    Build a compact parameter label with just the values (no names) for cleaner charts.
    Priority: (k,w) > t > min_tokens.
      Examples:
        k=12,w=17        -> "(12,17)"
        -t=9 or t=9      -> "9"
        --min-tokens=10  -> "10"
    """
    has_k = ("k" in r) and pd.notna(r["k"])
    has_w = ("w" in r) and pd.notna(r["w"])
    if has_k and has_w:
        return f"({int(r['k'])},{int(r['w'])})"
    if "t" in r and pd.notna(r["t"]):
        return f"{int(r['t'])}"
    if "min_tokens" in r and pd.notna(r["min_tokens"]):
        return f"{int(r['min_tokens'])}"
    return ""

def read_tool_csv(path: str, tool: str) -> pd.DataFrame:
    """Load a tool CSV and standardize columns."""
    df = pd.read_csv(path)
    df["tool"] = tool

    # Map aliases used by some tools
    if "minTokens" in df.columns and "min_tokens" not in df.columns:
        df["min_tokens"] = df["minTokens"]

    # Parse common numeric columns if present
    for c in ("threshold_pct", "f1", "k", "w", "t", "min_tokens"):
        _to_num(df, c)

    if tool == "Dolos":
        required = {"k", "w"}
        if not required.issubset(df.columns):
            missing = required - set(df.columns)
            raise ValueError(f"Dolos CSV missing columns: {missing}")

    elif tool == "JPlag":
        if "t" not in df.columns:
            raise ValueError("JPlag CSV missing column: 't'")

    elif tool in ("PMD-CPD", "jscpd"):
        if "min_tokens" not in df.columns:
            raise ValueError(f"{tool} CSV missing column: 'min_tokens'")

    elif tool == "Embeddings":
        # Prefer (k,w) if present (hybrid sweeps), else min_tokens if present.
        pass

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
        return pd.DataFrame(columns=list(param_cols) + ["f1", "threshold_pct", "tool"])
    idx = df_tool.groupby(param_cols)["f1"].idxmax()
    best = df_tool.loc[idx].sort_values(param_cols).copy()
    return best


def get_param_cols(tool: str, df: pd.DataFrame) -> List[str]:
    """
    Which parameter columns drive summaries?
    Prefer (k,w) when present — for both Dolos and Embeddings (hybrid).
    """
    if {"k", "w"}.issubset(df.columns) and df[["k", "w"]].notna().any(axis=None):
        return ["k", "w"]
    mapping = {
        "JPlag": ["t"],
        "PMD-CPD": ["min_tokens"],
        "jscpd": ["min_tokens"],
        "Embeddings": ["min_tokens"] if "min_tokens" in df.columns else [],
        "Dolos": ["k", "w"],  # covered above
    }
    return [c for c in mapping.get(tool, []) if c in df.columns]


def spread_per_threshold(
    df_tool: pd.DataFrame,
    include_zero_f1_in_band: bool = False,
    use_iqr: bool = False,
) -> pd.DataFrame:
    """
    For each threshold, compute band stats across parameter settings.
    By default we *exclude* f1 == 0.0 from the band to avoid huge spans.

    Returns columns: threshold_pct, f1_min, f1_max, q25, q75 (with NaNs if no valid values).
    """
    def filt(s: pd.Series) -> pd.Series:
        return s if include_zero_f1_in_band else s[s > 0.0]

    g = df_tool.groupby("threshold_pct")["f1"]
    stats = g.agg(
        f1_min=lambda s: filt(s).min(),
        f1_max=lambda s: filt(s).max(),
        q25=lambda s: filt(s).quantile(0.25),
        q75=lambda s: filt(s).quantile(0.75),
    ).reset_index().sort_values("threshold_pct")

    # Ensure arrays are float and keep NaNs where groups had no positive f1.
    for c in ("f1_min", "f1_max", "q25", "q75"):
        stats[c] = pd.to_numeric(stats[c], errors="coerce")

    # Choose which pair to use for the band
    if use_iqr:
        stats["y1"] = stats["q25"]
        stats["y2"] = stats["q75"]
    else:
        stats["y1"] = stats["f1_min"]
        stats["y2"] = stats["f1_max"]

    return stats


def _draw_band(ax, x, y1, y2, color, alpha: float, outline: bool) -> None:
    """
    Draw a shaded band, only where both bounds are valid.
    Uses 'where' and 'interpolate=True' so NaN gaps don't fill across the chart.
    """
    y1 = np.asarray(y1, dtype=float)
    y2 = np.asarray(y2, dtype=float)

    # Clamp order and build a valid mask
    ylo = np.minimum(y1, y2)
    yhi = np.maximum(y1, y2)
    mask = ~(np.isnan(ylo) | np.isnan(yhi))

    poly = ax.fill_between(
        x, ylo, yhi,
        where=mask,
        interpolate=True,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=1
    )
    if outline:
        poly.set_edgecolor(color)
        poly.set_linewidth(0.8)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot F1 summaries from tool CSVs (short labels, robust bands).")
    parser.add_argument("--dolos", type=str, default=None, help="Path to Dolos CSV")
    parser.add_argument("--jplag", type=str, default=None, help="Path to JPlag CSV")
    parser.add_argument("--pmd-cpd", dest="pmd_cpd", type=str, default=None, help="Path to PMD-CPD CSV")
    parser.add_argument("--jscpd", type=str, default=None, help="Path to jscpd CSV")
    parser.add_argument("--emb", "--embeddings", dest="emb", type=str, default=None,
                        help="Path to embeddings algorithm CSV")
    parser.add_argument("--language", type=str, default=None,
                        help="Language label to show in the charts, e.g. 'Java'")
    parser.add_argument("--out-dir", type=str, default="/out", help="Output directory (default: /out)")

    # Chart 2 controls
    parser.add_argument("--spread", choices=["minmax", "iqr"], default="minmax",
                        help="Band on Chart 2: 'minmax' span or interquartile range 'iqr'.")
    parser.add_argument("--spread-alpha", type=float, default=0.18,
                        help="Alpha for sweep bands (default: 0.18).")
    parser.add_argument("--band-outline", action="store_true",
                        help="Draw a thin outline around each sweep band (helps separation).")
    parser.add_argument("--chart2-layout", choices=["overlay", "facet"], default="facet",
                        help="Overlay all tools on one axis, or facet into small multiples (default: facet).")

    # New: control how zeros are treated in the band
    parser.add_argument("--include-zero-f1-in-band", action="store_true",
                        help="Include 0.0 F1 values when computing the sweep band (default: excluded).")

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
        label = _format_params_from_row(row)  # short form only
        txt = f"thr={thr}\n{label}" if label else f"thr={thr}"
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 txt, ha="center", va="bottom", fontsize=9)

    fig1.tight_layout()
    fig1.savefig(out_dir / "chart1_highest_f1_per_tool.png", dpi=150)

    # ---------- Chart 2: F1 vs Threshold (band shows full sweep; line shows best) ----------
    tools_in_order = ["Embeddings", "Dolos", "JPlag", "PMD-CPD", "jscpd"]
    tools_available = [t for t in tools_in_order if t in set(tools)]

    use_iqr = (args.spread == "iqr")

    if args.chart2_layout == "facet":
        # Small multiples layout to avoid overlapping bands
        n = len(tools_available)
        ncols = 2 if n > 1 else 1
        if n >= 4: ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig2, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2 * ncols, 4.6 * nrows), sharex=True, sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])
        axes = axes.flatten()

        # Precompute stats + best and figure out robust global y-lims (ignore NaNs)
        stats_per_tool: Dict[str, pd.DataFrame] = {}
        best_per_tool: Dict[str, pd.DataFrame] = {}
        ymins, ymaxs = [], []
        for tool in tools_available:
            tdf = all_df[all_df["tool"] == tool].copy()
            stats = spread_per_threshold(tdf, include_zero_f1_in_band=args.include_zero_f1_in_band, use_iqr=use_iqr)
            stats_per_tool[tool] = stats
            best = best_per_threshold(tdf)
            best_per_tool[tool] = best
            ymins.append(np.nanmin(stats["y1"].values))
            ymaxs.append(np.nanmax(stats["y2"].values))

        # Fallback if everything was NaN (unlikely but safe)
        global_min = float(np.nanmin(ymins)) if np.isfinite(np.nanmin(ymins)) else 0.0
        global_max = float(np.nanmax(ymaxs)) if np.isfinite(np.nanmax(ymaxs)) else 1.0
        global_min = max(0.0, global_min - 0.02)
        global_max = min(1.0, global_max + 0.02)

        for ax, tool in zip(axes, tools_available):
            stats = stats_per_tool[tool]
            x = stats["threshold_pct"].values.astype(float)

            # get a color from the cycle by plotting a hidden line
            dummy_line, = ax.plot(x, stats["y2"].values, alpha=0.0)
            color = dummy_line.get_color()
            dummy_line.remove()

            _draw_band(ax, x, stats["y1"].values, stats["y2"].values,
                       color=color, alpha=args.spread_alpha, outline=args.band_outline)

            best = best_per_tool[tool]
            ax.plot(best["threshold_pct"], best["f1"], marker="o", label=tool, zorder=3, linewidth=2.0, color=color)

            # annotate best configuration at each threshold (short labels)
            for _, r in best.iterrows():
                label = _format_params_from_row(r)
                if label:
                    ax.annotate(label,
                                xy=(r["threshold_pct"], r["f1"]),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center",
                                fontsize=8,
                                zorder=4)

            ax.set_title(tool)
            ax.grid(False)
            ax.set_ylim(global_min, global_max)
            ax.set_xlabel("Threshold (%)")
            ax.set_ylabel("F1")

        # hide extra axes if any
        for ax in axes[len(tools_available):]:
            ax.axis("off")

        fig2.suptitle("F1 vs Threshold — per-tool panels (shaded sweep; line = best per threshold)" + lang_suffix, y=0.98)
        fig2.tight_layout()
        fig2.savefig(out_dir / "chart2_f1_vs_threshold_facet.png", dpi=150)

    else:
        # Overlay layout (all tools on a single axis)
        fig2 = plt.figure(figsize=(10, 7))
        ax2 = fig2.add_subplot(111)

        # Precompute global y-lims robustly (ignore NaNs)
        ymins, ymaxs = [], []
        pertool_stats: Dict[str, pd.DataFrame] = {}
        for tool in tools_available:
            tdf = all_df[all_df["tool"] == tool].copy()
            stats = spread_per_threshold(tdf, include_zero_f1_in_band=args.include_zero_f1_in_band, use_iqr=use_iqr)
            pertool_stats[tool] = stats
            ymins.append(np.nanmin(stats["y1"].values))
            ymaxs.append(np.nanmax(stats["y2"].values))

        global_min = float(np.nanmin(ymins)) if np.isfinite(np.nanmin(ymins)) else 0.0
        global_max = float(np.nanmax(ymaxs)) if np.isfinite(np.nanmax(ymaxs)) else 1.0
        global_min = max(0.0, global_min - 0.02)
        global_max = min(1.0, global_max + 0.02)

        for tool in tools_available:
            stats = pertool_stats[tool]
            x = stats["threshold_pct"].values.astype(float)

            dummy_line, = ax2.plot(x, stats["y2"].values, alpha=0.0)
            color = dummy_line.get_color()
            dummy_line.remove()

            _draw_band(ax2, x, stats["y1"].values, stats["y2"].values,
                       color=color, alpha=args.spread_alpha, outline=args.band_outline)

            best = best_per_threshold(all_df[all_df["tool"] == tool])
            ax2.plot(best["threshold_pct"], best["f1"], marker="o", label=tool, zorder=3, linewidth=2.0, color=color)

            for _, r in best.iterrows():
                label = _format_params_from_row(r)
                if label:
                    ax2.annotate(label,
                                 xy=(r["threshold_pct"], r["f1"]),
                                 xytext=(0, 6),
                                 textcoords="offset points",
                                 ha="center",
                                 fontsize=8,
                                 zorder=4)

        ax2.set_title("F1 vs Threshold (shaded = sweep across parameters; line = best per threshold)" + lang_suffix)
        ax2.set_xlabel("Threshold (%)")
        ax2.set_ylabel("F1")
        ax2.set_ylim(global_min, global_max)
        ax2.legend(loc="upper right")

        fig2.tight_layout()
        fig2.savefig(out_dir / "chart2_f1_vs_threshold.png", dpi=150)

    print(f"Saved charts to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
