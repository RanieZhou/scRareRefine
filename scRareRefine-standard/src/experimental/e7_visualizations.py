"""E7: Visualizations for all experiments.

Produces:
1. fig_e1_three_seed_bars.png — grouped bar chart: 3 methods × 3 rare classes, mean±std
2. fig_e2_lambda_curve.png — λ vs rare_f1 on validation for each dataset
3. fig_e3_knn_comparison.png — bar chart comparing 3 kNN variants across datasets
4. fig_e4_smote_comparison.png — bar chart: baseline vs LR vs SMOTE-LR
5. fig_e5_gmm_comparison.png — bar chart: euclidean vs mahal-pooled vs GMM
6. fig_e6_combined_pipeline.png — before/after: current method vs new distance+marker
7. fig_summary_heatmap.png — heatmap: all methods × all datasets, rare_f1 values

Usage:
    python src/experimental/e7_visualizations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import read_table

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "_experimental" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

E1_DIR = ROOT / "outputs" / "_experimental" / "e1_three_seed_validation"
E2_DIR = ROOT / "outputs" / "_experimental" / "e2_adaptive_penalty"
E3_DIR = ROOT / "outputs" / "_experimental" / "e3_weighted_knn"
E4_DIR = ROOT / "outputs" / "_experimental" / "e4_smote_latent"
E5_DIR = ROOT / "outputs" / "_experimental" / "e5_gmm_prototype"
E6_DIR = ROOT / "outputs" / "_experimental" / "e6_combined_pipeline"

COLORS = {
    "scANVI baseline":          "#4878CF",
    "euclidean nearest-proto":  "#6ACC65",
    "mahal-pooled+posterior":   "#D65F5F",
    "standard":                 "#4878CF",
    "distance_weighted":        "#6ACC65",
    "class_balanced":           "#D65F5F",
    "scANVI":                   "#4878CF",
    "Standard LR":              "#6ACC65",
    "SMOTE-LR":                 "#D65F5F",
    "Euclidean":                "#4878CF",
    "Mahal-pool+post":          "#6ACC65",
    "GMM":                      "#D65F5F",
    "Current (Eucl+gate+marker)": "#4878CF",
    "Mahal+gate+marker":        "#D65F5F",
}

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.spines.top":  False,
    "axes.spines.right": False,
}


def fig1_e1_three_seed_bars():
    """Grouped bar chart: 3 methods × 3 rare classes, mean±std across seeds."""
    agg_path = E1_DIR / "aggregated_results.csv"
    if not agg_path.exists():
        print("  E1 aggregated results not found, skipping fig1.")
        return

    agg = read_table(agg_path)
    rare_classes = ["cDC1", "ASDC", "epsilon"]
    methods = ["scANVI baseline", "euclidean nearest-proto", "mahal-pooled+posterior"]
    method_labels = ["scANVI", "Euclidean", "Mahal-pool+post"]

    x = np.arange(len(rare_classes))
    width = 0.25

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (method, label) in enumerate(zip(methods, method_labels)):
            means, stds = [], []
            for rc in rare_classes:
                row = agg[(agg["rare_class"] == rc) & (agg["method"] == method)]
                if row.empty:
                    means.append(0.0); stds.append(0.0)
                else:
                    means.append(float(row["mean_rare_f1"].iloc[0]))
                    stds.append(float(row["std_rare_f1"].iloc[0]))
            bars = ax.bar(
                x + i * width, means, width,
                yerr=stds, capsize=4,
                label=label,
                color=list(COLORS.values())[i],
                alpha=0.85,
            )

        ax.set_xlabel("Rare class", fontsize=12)
        ax.set_ylabel("Rare-class F1 (mean ± std, 3 seeds)", fontsize=12)
        ax.set_title("E1: 3-seed validation — Mahalanobis vs Euclidean vs scANVI", fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(rare_classes, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / "fig_e1_three_seed_bars.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig2_e2_lambda_curve():
    """λ vs rare_f1 on validation for each dataset."""
    csv_files = list(E2_DIR.glob("*_lambda_curve.csv"))
    if not csv_files:
        print("  E2 lambda curves not found, skipping fig2.")
        return

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        palette = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
        for idx, f in enumerate(sorted(csv_files)):
            df = read_table(f)
            label = df["dataset"].iloc[0] if "dataset" in df.columns else f.stem
            # Shorten label
            label = label.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
            ax.plot(
                df["lambda"], df["val_rare_f1"],
                marker="o", linewidth=2,
                color=palette[idx % len(palette)],
                label=label,
            )

        ax.set_xlabel("λ (posterior penalty weight)", fontsize=12)
        ax.set_ylabel("Validation rare-class F1", fontsize=12)
        ax.set_title("E2: Adaptive posterior penalty — λ sweep on validation", fontsize=13)
        ax.set_xscale("symlog", linthresh=0.01)
        ax.legend(fontsize=9, loc="lower right")
        fig.tight_layout()
        out = FIG_DIR / "fig_e2_lambda_curve.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig3_e3_knn_comparison():
    """Bar chart comparing 3 kNN variants across datasets."""
    results_path = E3_DIR / "results.csv"
    if not results_path.exists():
        print("  E3 results not found, skipping fig3.")
        return

    df = read_table(results_path)
    rare_classes = df["rare_class"].unique()
    modes = ["standard", "distance_weighted", "class_balanced"]
    mode_labels = ["Standard kNN", "Distance-weighted", "Class-balanced (ours)"]
    k_best = 30  # show best k

    sub = df[df["k"] == k_best]

    x = np.arange(len(rare_classes))
    width = 0.25

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (mode, label) in enumerate(zip(modes, mode_labels)):
            vals = []
            for rc in rare_classes:
                row = sub[(sub["rare_class"] == rc) & (sub["mode"] == mode)]
                vals.append(float(row["rare_f1"].iloc[0]) if not row.empty else 0.0)
            ax.bar(
                x + i * width, vals, width,
                label=label,
                color=list(COLORS.values())[i],
                alpha=0.85,
            )

        ax.set_xlabel("Rare class", fontsize=12)
        ax.set_ylabel("Rare-class F1", fontsize=12)
        ax.set_title(f"E3: kNN variants (k={k_best}) — class-balanced weighting", fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(rare_classes, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / "fig_e3_knn_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig4_e4_smote_comparison():
    """Bar chart: baseline vs LR vs SMOTE-LR."""
    results_path = E4_DIR / "results.csv"
    if not results_path.exists():
        print("  E4 results not found, skipping fig4.")
        return

    df = read_table(results_path)
    rare_classes = df["rare_class"].tolist()
    methods = ["test_rare_f1_scanvi", "test_rare_f1_lr", "test_rare_f1_smote_lr"]
    method_labels = ["scANVI", "Standard LR", "SMOTE-LR (ours)"]

    x = np.arange(len(rare_classes))
    width = 0.25

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (col, label) in enumerate(zip(methods, method_labels)):
            vals = df[col].fillna(0).tolist()
            ax.bar(
                x + i * width, vals, width,
                label=label,
                color=list(COLORS.values())[i],
                alpha=0.85,
            )

        ax.set_xlabel("Rare class", fontsize=12)
        ax.set_ylabel("Rare-class F1", fontsize=12)
        ax.set_title("E4: SMOTE in latent space — oversampling rare class", fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(rare_classes, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / "fig_e4_smote_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig5_e5_gmm_comparison():
    """Bar chart: euclidean vs mahal-pooled vs GMM."""
    results_path = E5_DIR / "results.csv"
    if not results_path.exists():
        print("  E5 results not found, skipping fig5.")
        return

    df = read_table(results_path)
    labels_x = [f"{row['rare_class']}\n(n={row['n_rare_train']})" for _, row in df.iterrows()]
    methods = ["test_rare_f1_euclidean", "test_rare_f1_mahal_pool_post", "test_rare_f1_gmm"]
    method_labels = ["Euclidean", "Mahal-pool+post", "GMM (ours)"]

    x = np.arange(len(df))
    width = 0.25

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (col, label) in enumerate(zip(methods, method_labels)):
            vals = df[col].fillna(0).tolist()
            ax.bar(
                x + i * width, vals, width,
                label=label,
                color=list(COLORS.values())[i],
                alpha=0.85,
            )

        ax.set_xlabel("Dataset / rare class", fontsize=12)
        ax.set_ylabel("Rare-class F1", fontsize=12)
        ax.set_title("E5: GMM prototype vs Euclidean vs Mahalanobis", fontsize=13)
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels_x, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=10)
        fig.tight_layout()
        out = FIG_DIR / "fig_e5_gmm_comparison.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig6_e6_combined_pipeline():
    """Before/after: current method vs new distance+marker."""
    results_path = E6_DIR / "results.csv"
    if not results_path.exists():
        print("  E6 results not found, skipping fig6.")
        return

    df = read_table(results_path)
    labels_x = df["rare_class"].tolist()
    methods = [
        "test_rare_f1_scanvi",
        "test_rare_f1_euclidean_no_gate",
        "test_rare_f1_current_gate_marker",
        "test_rare_f1_mahal_gate_marker",
    ]
    method_labels = ["scANVI", "Euclidean (no gate)", "Current (Eucl+gate+marker)", "Mahal+gate+marker (new)"]
    colors = ["#4878CF", "#6ACC65", "#B47CC7", "#D65F5F"]

    x = np.arange(len(df))
    width = 0.2

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, (col, label) in enumerate(zip(methods, method_labels)):
            vals = df[col].fillna(0).tolist()
            ax.bar(
                x + i * width, vals, width,
                label=label,
                color=colors[i],
                alpha=0.85,
            )

        ax.set_xlabel("Rare class", fontsize=12)
        ax.set_ylabel("Rare-class F1", fontsize=12)
        ax.set_title("E6: Combined pipeline — Mahal distance + gate + marker", fontsize=13)
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels(labels_x, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        fig.tight_layout()
        out = FIG_DIR / "fig_e6_combined_pipeline.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def fig7_summary_heatmap():
    """Heatmap: all methods × all datasets, rare_f1 values."""
    # Collect all results
    data = {}

    # E1 per-run results
    e1_path = E1_DIR / "per_run_results.csv"
    if e1_path.exists():
        e1 = read_table(e1_path)
        for _, row in e1.iterrows():
            ds = f"{row['rare_class']} s{row['seed']}"
            data.setdefault(ds, {})
            data[ds]["scANVI"]          = row["scanvi_rare_f1"]
            data[ds]["Euclidean"]       = row["euclidean_rare_f1"]
            data[ds]["Mahal-pool+post"] = row["mahal_pool_post_rare_f1"]

    # E2 results
    e2_path = E2_DIR / "results.csv"
    if e2_path.exists():
        e2 = read_table(e2_path)
        for _, row in e2.iterrows():
            ds = row["rare_class"]
            data.setdefault(ds, {})
            data[ds]["Adaptive-λ"] = row["test_rare_f1_adaptive"]

    # E3 results (best k=30, class_balanced)
    e3_path = E3_DIR / "results.csv"
    if e3_path.exists():
        e3 = read_table(e3_path)
        sub = e3[(e3["k"] == 30) & (e3["mode"] == "class_balanced")]
        for _, row in sub.iterrows():
            ds = row["rare_class"]
            data.setdefault(ds, {})
            data[ds]["CB-kNN(k=30)"] = row["rare_f1"]

    # E4 results
    e4_path = E4_DIR / "results.csv"
    if e4_path.exists():
        e4 = read_table(e4_path)
        for _, row in e4.iterrows():
            ds = row["rare_class"]
            data.setdefault(ds, {})
            data[ds]["SMOTE-LR"] = row["test_rare_f1_smote_lr"]

    # E5 results
    e5_path = E5_DIR / "results.csv"
    if e5_path.exists():
        e5 = read_table(e5_path)
        for _, row in e5.iterrows():
            ds = f"{row['rare_class']}(n={row['n_rare_train']})"
            data.setdefault(ds, {})
            data[ds]["GMM"] = row["test_rare_f1_gmm"]

    if not data:
        print("  No data for summary heatmap, skipping.")
        return

    # Build matrix
    all_methods = sorted({m for d in data.values() for m in d})
    all_datasets = sorted(data.keys())
    matrix = np.full((len(all_datasets), len(all_methods)), np.nan)
    for i, ds in enumerate(all_datasets):
        for j, m in enumerate(all_methods):
            if m in data[ds]:
                matrix[i, j] = data[ds][m]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(max(10, len(all_methods) * 1.2), max(6, len(all_datasets) * 0.4)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Rare-class F1")

        ax.set_xticks(range(len(all_methods)))
        ax.set_xticklabels(all_methods, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(all_datasets)))
        ax.set_yticklabels(all_datasets, fontsize=8)
        ax.set_title("Summary heatmap: rare-class F1 across all methods and datasets", fontsize=12)

        # Annotate cells
        for i in range(len(all_datasets)):
            for j in range(len(all_methods)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

        fig.tight_layout()
        out = FIG_DIR / "fig_summary_heatmap.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out}")


def main():
    print("Generating E7 visualizations ...")
    fig1_e1_three_seed_bars()
    fig2_e2_lambda_curve()
    fig3_e3_knn_comparison()
    fig4_e4_smote_comparison()
    fig5_e5_gmm_comparison()
    fig6_e6_combined_pipeline()
    fig7_summary_heatmap()
    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
