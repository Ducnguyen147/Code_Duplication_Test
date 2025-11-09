#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------

def _to_num(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def read_tool_csv(path: str, tool: str) -> pd.DataFrame:
    """Load a tool CSV and standardize columns."""
    df = pd.read_csv(path)
    df["tool"] = tool

    for c in ("threshold_pct", "f1"):
        _to_num(df, c)

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

    return df


def best_per_threshold(df_tool: pd.DataFrame) -> pd.DataFrame:
    """Return 1 row per threshold (max F1 across parameter settings)."""
    idx = df_tool.groupby("threshold_pct")["f1"].idxmax()
    best = df_tool.loc[idx].sort_values("threshold_pct").copy()
    return best


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot F1 summaries from tool CSVs.")
    parser.add_argument("--dolos", type=str, default=None, help="Path to Dolos CSV")
    parser.add_argument("--jplag", type=str, default=None, help="Path to JPlag CSV")
    parser.add_argument("--pmd-cpd", dest="pmd_cpd", type=str, default=None, help="Path to PMD-CPD CSV")
    parser.add_argument("--jscpd", type=str, default=None, help="Path to jscpd CSV")
    parser.add_argument("--emb", "--embeddings", dest="emb", type=str, default=None,
                        help="Path to embeddings algorithm CSV")
    parser.add_argument("--out-dir", type=str, default="/out", help="Output directory (default: /out)")
    args = parser.parse_args()

    frames = []
    tools = []

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

    # ---------- Chart 1: Highest F1 per tool ----------
    best_rows = all_df.loc[all_df.groupby("tool")["f1"].idxmax()].copy()
    best_rows.sort_values("f1", ascending=False, inplace=True)

    fig1 = plt.figure(figsize=(8, 5))
    ax1 = fig1.add_subplot(111)
    bars = ax1.bar(best_rows["tool"], best_rows["f1"])
    ax1.set_title("Highest F1 by Tool")
    ax1.set_xlabel("Tool")
    ax1.set_ylabel("F1")

    for bar, (_, row) in zip(bars, best_rows.iterrows()):
        thr = int(row["threshold_pct"]) if pd.notna(row["threshold_pct"]) else "NA"
        txt = f"thr={thr}\n{row['param_label']}"
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

        for _, r in best.iterrows():
            ax2.annotate(
                r["param_label"],
                xy=(r["threshold_pct"], r["f1"]),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    ax2.set_title("F1 vs Threshold (best parameter per threshold)")
    ax2.set_xlabel("Threshold (%)")
    ax2.set_ylabel("F1")
    ax2.legend(loc="upper right")
    fig2.tight_layout()
    fig2.savefig(out_dir / "chart2_f1_vs_threshold.png", dpi=150)

    print(f"Saved charts to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
