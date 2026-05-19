"""Stage 9: Cross-dataset summary comparison plot.

Reads all_seeds_metrics.csv from every completed run and produces:
    results/summary/
        all_results.csv          flat table of all runs
        summary_f1_heatmap.png   mean rare-F1 per (dataset×rts, method)
        summary_f1_bar.png       grouped bar: mean ± std across seeds, one panel per dataset

Usage:
    python src/09_summary_compare.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Experiment manifest ───────────────────────────────────────────────────────

EXPERIMENTS = [
    ("configs/immune_dc.yaml",       "ASDC",                 ["0.01", "0.05", "0.1", "all"]),
    ("configs/immune_dc_cdc1.yaml",  "cDC1",                 ["0.01", "0.05", "0.1", "all"]),
    ("configs/pancreas_gamma.yaml",  "gamma",                ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_spleen.yaml",   "innate lymphoid cell", ["0.01", "0.05", "0.1", "all"]),
]
SEEDS = [42, 43, 44]

METHOD_ORDER  = ["baseline", "knn_k15", "lr", "scbalance", "scRareRefine"]
METHOD_LABELS = {
    "baseline":     "Baseline\n(scANVI)",
    "knn_k15":      "kNN\n(k=15)",
    "lr":           "CellTypist",
    "scbalance":    "scBalance",
    "scRareRefine": "scRareRefine",
}
METHOD_COLORS = {
    "baseline":     "#8da0cb",
    "knn_k15":      "#66c2a5",
    "lr":           "#fc8d62",
    "scbalance":    "#a6d854",
    "scRareRefine": "#e78ac3",
}

DATASET_LABELS = {
    "immune_dc_ASDC":               "DC-ASDC",
    "immune_dc_cDC1":               "DC-cDC1",
    "pancreas_gamma":               "Pancreas-γ",
    "tabula_spleen_innate lymphoid cell": "Spleen-ILC",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _rts_label(rts: str) -> str:
    try:
        v = float(rts)
        if v < 1:
            return f"{int(v*100)}%"
        return str(int(v))
    except ValueError:
        return rts


# ── Collect ───────────────────────────────────────────────────────────────────

def collect_all() -> pd.DataFrame:
    """Glob every all_seeds_metrics.csv under outputs/ and merge with dataset metadata."""
    import yaml

    # Build a lookup: (dataset_name, rare_class) → ds_key
    ds_meta: dict[tuple, tuple] = {}
    for cfg_path, rare_class, rts_list in EXPERIMENTS:
        with open(cfg_path) as f:
            config = yaml.safe_load(f)
        dataset_name = config["dataset"]["name"]
        ds_key = f"{dataset_name}_{rare_class}"
        ds_meta[(dataset_name, rare_class)] = (ds_key, rts_list)

    rows = []
    for csv in sorted(Path("outputs").glob("*/*/all_seeds_metrics.csv")):
        df = _safe_read(csv)
        if df.empty:
            continue
        # determine dataset and rare_class from the dataframe itself
        if "rare_class" not in df.columns:
            continue
        for _, row in df.iterrows():
            dataset_name = csv.parts[1]           # outputs/{dataset}/...
            rare_class   = str(row["rare_class"])
            rts          = str(row.get("rare_train_size", ""))
            ds_key, allowed_rts = ds_meta.get((dataset_name, rare_class), (None, None))
            if ds_key is None:
                continue
            if allowed_rts is not None and rts not in allowed_rts:
                continue
            rows.append({
                "dataset":         ds_key,
                "dataset_label":   DATASET_LABELS.get(ds_key, ds_key),
                "rare_class":      rare_class,
                "rare_train_size": rts,
                "rts_label":       _rts_label(rts),
                "seed":            int(row["seed"]),
                "method":          str(row["method"]),
                "rare_f1":         float(row.get("rare_f1",        np.nan)),
                "rare_recall":     float(row.get("rare_recall",    np.nan)),
                "rare_precision":  float(row.get("rare_precision", np.nan)),
                "overall_accuracy":float(row.get("overall_accuracy", np.nan)),
            })

    df_all = pd.DataFrame(rows).drop_duplicates(
        subset=["dataset", "rare_train_size", "seed", "method"]
    )
    return df_all


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_summary_bar(df: pd.DataFrame, out_path: Path) -> None:
    """One row per dataset, one panel per rts. Grouped bars = methods."""
    datasets = [d for d in df["dataset"].unique()
                if d in DATASET_LABELS or True]
    datasets = sorted(datasets)
    rts_vals  = sorted(df["rare_train_size"].unique(),
                       key=lambda x: (float(x) if x != "all" else 999))
    methods   = [m for m in METHOD_ORDER if m in df["method"].values]

    n_rows = len(datasets)
    n_cols = len(rts_vals)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.5 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    fig.suptitle("Rare-class F1 comparison (mean ± std across seeds)",
                 fontsize=12, fontweight="bold", y=1.01)

    x = np.arange(len(methods))
    width = 0.6
    rng = np.random.default_rng(0)

    for ri, ds in enumerate(datasets):
        for ci, rts in enumerate(rts_vals):
            ax = axes[ri][ci]
            sub = df[(df["dataset"] == ds) & (df["rare_train_size"] == rts)]
            means, stds = [], []
            for m in methods:
                vals = sub.loc[sub["method"] == m, "rare_f1"].dropna().tolist()
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)

            bars = ax.bar(x, means, width,
                          color=[METHOD_COLORS.get(m, "#aaa") for m in methods],
                          yerr=stds, capsize=3,
                          error_kw={"linewidth": 1, "ecolor": "#444"},
                          edgecolor="white", linewidth=0.6)

            for bar, mean, std in zip(bars, means, stds):
                if mean > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + std + 0.02,
                            f"{mean:.2f}", ha="center", va="bottom",
                            fontsize=6, fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods],
                fontsize=6, rotation=30, ha="right"
            )
            ax.set_ylim(0, 1.15)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ci == 0:
                ds_label = DATASET_LABELS.get(ds, ds)
                ax.set_ylabel(ds_label, fontsize=8, fontweight="bold")
            if ri == 0:
                rts_label = _rts_label(rts)
                ax.set_title(f"rts={rts_label}", fontsize=8, fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Heatmap: rows = (dataset, rts), cols = method, values = mean rare_f1."""
    methods = [m for m in METHOD_ORDER if m in df["method"].values]
    df["row_key"] = df["dataset_label"] + " / rts=" + df["rts_label"]
    pivot = (
        df.groupby(["row_key", "method"])["rare_f1"]
        .mean()
        .unstack("method")
        .reindex(columns=methods)
    )
    pivot = pivot.sort_index()

    fig, ax = plt.subplots(figsize=(len(methods) * 1.4 + 1, len(pivot) * 0.5 + 1.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean rare-class F1")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods],
                       fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot)):
        for j, m in enumerate(methods):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    ax.set_title("Mean rare-class F1 (all seeds)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Collecting results...")
    df = collect_all()
    if df.empty:
        print("No results found. Run experiments first.")
        return

    out_dir = Path("results/summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "all_results.csv", index=False)
    print(f"  Collected {len(df)} rows from {df['dataset'].nunique()} datasets")

    n_methods   = df["method"].nunique()
    n_datasets  = df["dataset"].nunique()
    n_rts       = df["rare_train_size"].nunique()
    n_seeds     = df["seed"].nunique()
    print(f"  methods={n_methods}  datasets={n_datasets}  rts={n_rts}  seeds={n_seeds}")

    print("\nMean rare_f1 by method:")
    summary = (
        df.groupby("method")["rare_f1"]
        .agg(["mean", "std", "count"])
        .reindex([m for m in METHOD_ORDER if m in df["method"].values])
    )
    print(summary.to_string())

    print("\nGenerating plots...")
    plot_summary_bar(df, out_dir / "summary_f1_bar.png")
    plot_heatmap(df,     out_dir / "summary_f1_heatmap.png")
    print("\nDone. Results in results/summary/")


if __name__ == "__main__":
    main()
