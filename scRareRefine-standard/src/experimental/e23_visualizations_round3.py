"""E23: Visualizations for Round 3 experiments.

Generates:
1. fig_e16_adaptive_recalibrated.png
2. fig_e17_mahal_full_pipeline.png
3. fig_e18_epsilon_confusion.png
4. fig_e18_epsilon_pca.png
5. fig_e19_contrastive.png
6. fig_e20_platt.png
7. fig_e21_multiscale.png
8. fig_e22_final_heatmap.png
9. fig_e22_final_bars.png
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
from matplotlib.colors import LinearSegmentedColormap

from utils import read_table

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "outputs" / "_experimental" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = ROOT / "outputs" / "_experimental"

COLORS = {
    "scANVI": "#7f7f7f",
    "euclidean": "#1f77b4",
    "mahal_pooled": "#ff7f0e",
    "cb_knn": "#2ca02c",
    "smote_lr": "#d62728",
    "old_adaptive": "#9467bd",
    "new_adaptive": "#8c564b",
    "contrastive": "#e377c2",
    "platt": "#17becf",
    "multiscale": "#bcbd22",
}


def _bar_group(ax, df, x_col, methods, colors, labels=None, title="", ylabel="rare F1"):
    n_groups = len(df)
    n_methods = len(methods)
    x = np.arange(n_groups)
    width = 0.8 / n_methods

    for i, (method, col) in enumerate(methods):
        vals = df[col].to_numpy()
        offset = (i - n_methods / 2 + 0.5) * width
        color = colors.get(method, f"C{i}")
        label = labels[i] if labels else method
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].tolist(), rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="black", linewidth=0.5)


# ── Figure 1: E16 Adaptive Recalibrated ───────────────────────────────────────
def fig_e16():
    path = EXP_DIR / "e16_adaptive_recalibrated" / "results.csv"
    if not path.exists():
        print(f"  SKIP fig_e16: {path} not found")
        return

    df = read_table(path)
    df["dataset_label"] = df["rare_class"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1 comparison across datasets
    ax = axes[0]
    methods = [
        ("scANVI", "scanvi_rare_f1"),
        ("euclidean", "euclidean_rare_f1"),
        ("mahal_pooled", "mahal_pooled_rare_f1"),
        ("smote_lr", "smote_lr_rare_f1"),
        ("old_adaptive", "old_adaptive_rare_f1"),
        ("new_adaptive", "new_adaptive_rare_f1"),
    ]
    labels = ["scANVI", "Euclidean", "Mahal-pooled", "SMOTE-LR", "Old adaptive (E10)", "New adaptive (E16)"]
    _bar_group(ax, df, "rare_class", methods, COLORS, labels,
               title="E16: Recalibrated Adaptive Selector — rare F1")

    # Right: Delta (new - old adaptive)
    ax = axes[1]
    if "old_adaptive_rare_f1" in df.columns and "new_adaptive_rare_f1" in df.columns:
        delta = df["new_adaptive_rare_f1"] - df["old_adaptive_rare_f1"]
        colors_delta = ["#2ca02c" if d >= 0 else "#d62728" for d in delta]
        ax.bar(range(len(df)), delta, color=colors_delta, alpha=0.85)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["rare_class"].tolist(), rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel("Δ rare F1 (new - old)")
        ax.set_title("E16: Improvement of recalibrated vs old adaptive selector")

        # Annotate S values
        if "separability_ratio" in df.columns:
            for i, (s, d) in enumerate(zip(df["separability_ratio"], delta)):
                ax.text(i, d + 0.01 * np.sign(d), f"S={s:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    out = FIG_DIR / "fig_e16_adaptive_recalibrated.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 2: E17 Mahal Full Pipeline ─────────────────────────────────────────
def fig_e17():
    path = EXP_DIR / "e17_mahal_main_pipeline" / "results.csv"
    if not path.exists():
        print(f"  SKIP fig_e17: {path} not found")
        return

    df = read_table(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    methods = [
        ("scANVI", "scanvi_rare_f1"),
        ("euclidean", "euclidean_no_gate_f1"),
        ("mahal_pooled", "mahal_no_gate_f1"),
    ]
    labels = ["scANVI", "Euclidean (no gate)", "Mahal-pooled (no gate)"]

    # Add gate+marker columns if present
    if "euclidean_gate_marker_f1" in df.columns:
        methods.append(("old_adaptive", "euclidean_gate_marker_f1"))
        labels.append("Euclidean+gate+marker")
    if "mahal_gate_marker_f1" in df.columns:
        methods.append(("new_adaptive", "mahal_gate_marker_f1"))
        labels.append("Mahal+gate+marker")

    _bar_group(ax, df, "rare_class", methods, COLORS, labels,
               title="E17: Mahal-pooled in Full Pipeline (prototype→gate→marker)")

    plt.tight_layout()
    out = FIG_DIR / "fig_e17_mahal_full_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 3: E18 Epsilon Confusion ───────────────────────────────────────────
def fig_e18_confusion():
    path = EXP_DIR / "e18_epsilon_analysis" / "confusion_summary.csv"
    if not path.exists():
        print(f"  SKIP fig_e18_confusion: {path} not found")
        return

    df = read_table(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F1 / recall / precision per method
    ax = axes[0]
    methods_order = ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN", "SMOTE-LR"]
    df_ordered = df.set_index("method").reindex(methods_order).reset_index()

    x = np.arange(len(df_ordered))
    width = 0.25
    ax.bar(x - width, df_ordered["rare_f1"], width, label="F1", color="#1f77b4", alpha=0.85)
    ax.bar(x, df_ordered["rare_recall"], width, label="Recall", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, df_ordered["rare_precision"], width, label="Precision", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df_ordered["method"].tolist(), rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("E18: Epsilon — F1/Recall/Precision per method")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Right: What do epsilon cells get predicted as?
    ax = axes[1]
    if "epsilon_predicted_as_epsilon" in df_ordered.columns:
        n_eps = df_ordered["epsilon_predicted_as_epsilon"].fillna(0)
        n_other = df_ordered["epsilon_predicted_as_other"].fillna(0)
        ax.bar(x, n_eps, label="Predicted as epsilon", color="#2ca02c", alpha=0.85)
        ax.bar(x, n_other, bottom=n_eps, label="Predicted as other", color="#d62728", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df_ordered["method"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("# epsilon test cells")
        ax.set_title("E18: Epsilon test cells — prediction breakdown")
        ax.legend()

    plt.tight_layout()
    out = FIG_DIR / "fig_e18_epsilon_confusion.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 4: E18 Epsilon PCA ─────────────────────────────────────────────────
def fig_e18_pca():
    train_path = EXP_DIR / "e18_epsilon_analysis" / "train_pca.csv"
    test_path  = EXP_DIR / "e18_epsilon_analysis" / "test_pca.csv"
    if not train_path.exists() or not test_path.exists():
        print(f"  SKIP fig_e18_pca: PCA files not found")
        return

    train_df = read_table(train_path)
    test_df  = read_table(test_path)

    methods = ["scanvi_pred", "euclidean_pred", "mahal_pred", "knn_pred", "smote_pred"]
    method_labels = ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN", "SMOTE-LR"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # First panel: true labels
    ax = axes[0]
    unique_labels = sorted(train_df["true_label"].unique())
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    label_to_color = {l: cmap(i) for i, l in enumerate(unique_labels)}

    for lbl in unique_labels:
        mask = train_df["true_label"] == lbl
        alpha = 0.8 if lbl == "epsilon" else 0.2
        size = 20 if lbl == "epsilon" else 5
        ax.scatter(train_df.loc[mask, "pc1"], train_df.loc[mask, "pc2"],
                   c=[label_to_color[lbl]], s=size, alpha=alpha, label=lbl)
    ax.set_title("True labels (train)")
    ax.legend(fontsize=6, markerscale=2)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Remaining panels: method predictions on test
    for i, (method, label) in enumerate(zip(methods, method_labels)):
        ax = axes[i + 1]
        if method not in test_df.columns:
            ax.set_visible(False)
            continue

        for lbl in unique_labels:
            mask = test_df[method] == lbl
            is_true_eps = test_df["true_label"] == "epsilon"
            alpha = 0.8 if lbl == "epsilon" else 0.2
            size = 20 if lbl == "epsilon" else 5
            ax.scatter(test_df.loc[mask, "pc1"], test_df.loc[mask, "pc2"],
                       c=[label_to_color.get(lbl, "gray")], s=size, alpha=alpha)

        # Highlight true epsilon cells with X marker
        eps_mask = test_df["true_label"] == "epsilon"
        ax.scatter(test_df.loc[eps_mask, "pc1"], test_df.loc[eps_mask, "pc2"],
                   marker="x", c="black", s=30, linewidths=1.5, label="True epsilon")
        ax.set_title(f"{label} predictions")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6)

    plt.suptitle("E18: Epsilon cells in 2D PCA — method predictions", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e18_epsilon_pca.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 5: E19 Contrastive ─────────────────────────────────────────────────
def fig_e19():
    path = EXP_DIR / "e19_contrastive_finetune" / "results.csv"
    if not path.exists():
        print(f"  SKIP fig_e19: {path} not found")
        return

    df = read_table(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F1 comparison
    ax = axes[0]
    methods = [
        ("scANVI", "scanvi_rare_f1"),
        ("euclidean", "euclidean_orig_rare_f1"),
        ("mahal_pooled", "mahal_pooled_orig_rare_f1"),
        ("contrastive", "contrastive_proj_rare_f1"),
    ]
    labels = ["scANVI", "Euclidean (30-dim)", "Mahal-pooled (30-dim)", "Contrastive (8-dim)"]
    _bar_group(ax, df, "rare_class", methods, COLORS, labels,
               title="E19: Contrastive Fine-tuning — rare F1")

    # Right: Separability ratio before/after
    ax = axes[1]
    if "separability_ratio_original" in df.columns and "separability_ratio_projected" in df.columns:
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df["separability_ratio_original"], width,
               label="Original (30-dim)", color="#1f77b4", alpha=0.85)
        ax.bar(x + width/2, df["separability_ratio_projected"], width,
               label="Projected (8-dim)", color="#e377c2", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["rare_class"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("Separability ratio S")
        ax.set_title("E19: Separability ratio before/after contrastive projection")
        ax.legend()
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="S=1.0")
        ax.axhline(1.2, color="orange", linestyle="--", linewidth=1, label="S=1.2")

    plt.tight_layout()
    out = FIG_DIR / "fig_e19_contrastive.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 6: E20 Platt Calibration ───────────────────────────────────────────
def fig_e20():
    path = EXP_DIR / "e20_platt_calibration" / "results.csv"
    if not path.exists():
        print(f"  SKIP fig_e20: {path} not found")
        return

    df = read_table(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F1 comparison
    ax = axes[0]
    methods = [
        ("scANVI", "scanvi_rare_f1"),
        ("euclidean", "euclidean_rare_f1"),
        ("platt", "platt_rare_f1"),
    ]
    labels = ["scANVI", "Euclidean nearest-proto", "Platt-calibrated scANVI"]
    _bar_group(ax, df, "rare_class", methods, COLORS, labels,
               title="E20: Platt Calibration — rare F1")

    # Right: Calibrated probability distributions (if available)
    ax = axes[1]
    cal_files = list((EXP_DIR / "e20_platt_calibration").glob("*_calibrated_probs.csv"))
    if cal_files:
        # Show first available dataset
        cal_df = read_table(cal_files[0])
        if "raw_prob_rare" in cal_df.columns and "calibrated_prob_rare" in cal_df.columns:
            rare_mask = cal_df["true_label"] == cal_df["true_label"].unique()[0]
            # Find rare class
            for col in cal_df.columns:
                if "prob" in col.lower():
                    break

            ax.hist(cal_df["raw_prob_rare"], bins=50, alpha=0.5, label="Raw scANVI prob", color="#1f77b4")
            ax.hist(cal_df["calibrated_prob_rare"], bins=50, alpha=0.5, label="Platt calibrated", color="#17becf")
            ax.set_xlabel("P(rare class)")
            ax.set_ylabel("Count")
            ax.set_title(f"E20: Probability distribution ({cal_files[0].stem})")
            ax.legend()
    else:
        ax.text(0.5, 0.5, "No calibrated prob files found", ha="center", va="center",
                transform=ax.transAxes)

    plt.tight_layout()
    out = FIG_DIR / "fig_e20_platt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 7: E21 Multi-scale Prototype ───────────────────────────────────────
def fig_e21():
    path = EXP_DIR / "e21_multiscale_prototype" / "results.csv"
    if not path.exists():
        print(f"  SKIP fig_e21: {path} not found")
        return

    df = read_table(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: F1 comparison
    ax = axes[0]
    methods = [
        ("scANVI", "scanvi_rare_f1"),
        ("euclidean", "euclidean_rare_f1"),
        ("mahal_pooled", "mahal_pooled_rare_f1"),
        ("multiscale", "multiscale_rare_f1"),
    ]
    labels = ["scANVI", "Euclidean (centroid)", "Mahal-pooled", "Multi-scale"]
    _bar_group(ax, df, "rare_class", methods, COLORS, labels,
               title="E21: Multi-scale Prototype — rare F1")

    # Right: Best weights
    ax = axes[1]
    if all(c in df.columns for c in ["best_w1", "best_w2", "best_w3"]):
        x = np.arange(len(df))
        width = 0.25
        ax.bar(x - width, df["best_w1"], width, label="w1 (centroid)", color="#1f77b4", alpha=0.85)
        ax.bar(x, df["best_w2"], width, label="w2 (1-NN)", color="#ff7f0e", alpha=0.85)
        ax.bar(x + width, df["best_w3"], width, label="w3 (5-NN)", color="#2ca02c", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(df["rare_class"].tolist(), rotation=20, ha="right")
        ax.set_ylabel("Weight")
        ax.set_title("E21: Best multi-scale weights per dataset")
        ax.legend()
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    out = FIG_DIR / "fig_e21_multiscale.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 8: E22 Final Heatmap ───────────────────────────────────────────────
def fig_e22_heatmap():
    path = EXP_DIR / "e22_final_evaluation" / "aggregated_results.csv"
    if not path.exists():
        print(f"  SKIP fig_e22_heatmap: {path} not found")
        return

    df = read_table(path)

    methods = ["scanvi", "euclidean", "mahal_pooled", "adaptive"]
    method_labels = ["scANVI", "Euclidean", "Mahal-pooled", "Adaptive (E16)"]

    pivot = df.pivot_table(index="dataset", columns="method", values="mean_rare_f1")
    pivot = pivot.reindex(columns=methods)

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list("rg", ["#d62728", "#ffffff", "#2ca02c"])
    im = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index.tolist())

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(methods)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val > 0.3 else "white")

    plt.colorbar(im, ax=ax, label="Mean rare F1 (3 seeds)")
    ax.set_title("E22: Comprehensive 3-seed Evaluation — Mean rare F1")
    plt.tight_layout()
    out = FIG_DIR / "fig_e22_final_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


# ── Figure 9: E22 Final Bars ──────────────────────────────────────────────────
def fig_e22_bars():
    path = EXP_DIR / "e22_final_evaluation" / "aggregated_results.csv"
    if not path.exists():
        print(f"  SKIP fig_e22_bars: {path} not found")
        return

    df = read_table(path)

    methods = ["euclidean", "mahal_pooled", "adaptive"]
    method_labels = ["Euclidean", "Mahal-pooled", "Adaptive (E16)"]
    method_colors = [COLORS["euclidean"], COLORS["mahal_pooled"], COLORS["new_adaptive"]]

    datasets = df["dataset"].unique()
    n_datasets = len(datasets)
    n_methods = len(methods)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_datasets)
    width = 0.25

    for i, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        sub = df[df["method"] == method].set_index("dataset").reindex(datasets)
        means = sub["mean_rare_f1"].to_numpy()
        stds  = sub["std_rare_f1"].fillna(0).to_numpy()
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, means, width, label=label, color=color, alpha=0.85,
               yerr=stds, capsize=3, error_kw={"linewidth": 1})

    # Add scANVI as reference line per dataset
    scanvi_sub = df[df["method"] == "scanvi"].set_index("dataset").reindex(datasets)
    if not scanvi_sub.empty:
        ax.plot(x, scanvi_sub["mean_rare_f1"].to_numpy(), "k--", linewidth=1.5,
                marker="o", markersize=4, label="scANVI baseline", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylabel("Mean rare F1 (3 seeds)")
    ax.set_title("E22: Top 3 Methods × All Datasets — Mean ± Std (3 seeds)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out = FIG_DIR / "fig_e22_final_bars.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Generating Round 3 visualizations...")
    fig_e16()
    fig_e17()
    fig_e18_confusion()
    fig_e18_pca()
    fig_e19()
    fig_e20()
    fig_e21()
    fig_e22_heatmap()
    fig_e22_bars()
    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
