"""E17: Updated visualizations incorporating E8-E16 results.

Generates:
1. fig_e8_data_efficiency_mahal_vs_euclidean.png — data efficiency curves
2. fig_e9_cb_knn_efficiency.png — CB-kNN across rts
3. fig_e10_ensemble_alpha.png — alpha sweep results
4. fig_e11_soft_gate.png — soft vs hard gate comparison
5. fig_e12_temperature_scaling.png — calibration improvement
6. fig_e16_best_method_heatmap.png — best method per (dataset, rts)
7. fig_comprehensive_heatmap.png — all methods × all datasets × all rts

Usage:
    python src/experimental/e17_updated_visualizations.py
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
import matplotlib.colors as mcolors

from utils import read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "outputs" / "_experimental"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    "scANVI":       "#888888",
    "Euclidean":    "#2196F3",
    "Mahal-pooled": "#F44336",
    "CB-kNN":       "#4CAF50",
    "Ensemble":     "#FF9800",
    "Soft-gate":    "#9C27B0",
    "Dist-TempScaled": "#00BCD4",
    "Bootstrap":    "#795548",
}


def fig_e8_data_efficiency():
    """Data efficiency curves: Mahal-pooled vs Euclidean vs scANVI."""
    path = EXP_DIR / "e8_data_efficiency_sweep" / "aggregated_results.csv"
    if not path.exists():
        print("  E8 results not found, skipping fig_e8")
        return

    df = read_table(path)
    datasets = df["dataset"].unique()
    rare_classes = df["rare_class"].unique()

    # One subplot per (dataset, rare_class)
    configs = df[["dataset", "rare_class"]].drop_duplicates().values.tolist()
    n = len(configs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (dataset, rare_class) in enumerate(configs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df["dataset"] == dataset) & (df["rare_class"] == rare_class)]
        sub = sub.sort_values("rts")

        for method in ["scANVI", "Euclidean", "Mahal-pooled"]:
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue
            color = COLORS.get(method, "black")
            ax.plot(m_sub["rts"], m_sub["mean_rare_f1"],
                    marker="o", label=method, color=color, linewidth=2)
            # Error band
            if "std_rare_f1" in m_sub.columns:
                ax.fill_between(
                    m_sub["rts"],
                    m_sub["mean_rare_f1"] - m_sub["std_rare_f1"],
                    m_sub["mean_rare_f1"] + m_sub["std_rare_f1"],
                    alpha=0.15, color=color,
                )

        ax.set_title(f"{dataset}\n{rare_class}", fontsize=9)
        ax.set_xlabel("rts (rare train size)")
        ax.set_ylabel("rare_f1")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("E8: Data Efficiency — Mahal-pooled vs Euclidean vs scANVI\n(mean ± std across 3 seeds)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e8_data_efficiency_mahal_vs_euclidean.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_e9_cb_knn():
    """CB-kNN efficiency curves."""
    path = EXP_DIR / "e9_cb_knn_sweep" / "aggregated_results.csv"
    if not path.exists():
        print("  E9 results not found, skipping fig_e9")
        return

    df = read_table(path)
    configs = df[["dataset", "rare_class"]].drop_duplicates().values.tolist()
    n = len(configs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (dataset, rare_class) in enumerate(configs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df["dataset"] == dataset) & (df["rare_class"] == rare_class)]
        sub = sub.sort_values("rts")

        for method in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN"]:
            m_sub = sub[sub["method"] == method]
            if m_sub.empty:
                continue
            color = COLORS.get(method, "black")
            ls = "--" if method == "scANVI" else "-"
            ax.plot(m_sub["rts"], m_sub["mean_rare_f1"],
                    marker="o", label=method, color=color, linewidth=2, linestyle=ls)

        ax.set_title(f"{dataset}\n{rare_class}", fontsize=9)
        ax.set_xlabel("rts")
        ax.set_ylabel("rare_f1")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("E9: CB-kNN vs Euclidean vs Mahal-pooled across rts\n(mean across 3 seeds)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e9_cb_knn_efficiency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_e10_ensemble():
    """Alpha sweep results."""
    path = EXP_DIR / "e10_prototype_ensemble" / "alpha_curves.csv"
    if not path.exists():
        print("  E10 alpha curves not found, skipping fig_e10")
        return

    df = read_table(path)
    configs = df[["dataset", "rare_class", "rts"]].drop_duplicates().values.tolist()
    n = len(configs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (dataset, rare_class, rts) in enumerate(configs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df["dataset"] == dataset) & (df["rare_class"] == rare_class) & (df["rts"] == rts)]
        sub = sub.sort_values("alpha")
        ax.plot(sub["alpha"], sub["full_test_rare_f1"], marker="o", color="#FF9800", linewidth=2)
        ax.axvline(x=0.0, color="#F44336", linestyle="--", alpha=0.5, label="α=0 (Mahal)")
        ax.axvline(x=1.0, color="#2196F3", linestyle="--", alpha=0.5, label="α=1 (Eucl)")
        ax.set_title(f"{dataset}\n{rare_class} rts={rts}", fontsize=8)
        ax.set_xlabel("α (Euclidean weight)")
        ax.set_ylabel("rare_f1")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("E10: Ensemble α sweep (α=0: pure Mahal, α=1: pure Euclidean)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e10_ensemble_alpha.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_e11_soft_gate():
    """Soft vs hard gate comparison."""
    path = EXP_DIR / "e11_soft_gate" / "results.csv"
    if not path.exists():
        print("  E11 results not found, skipping fig_e11")
        return

    df = read_table(path)
    key_methods = {
        "scANVI (no gate)": "scANVI",
        "Euclidean (no gate)": "Euclidean",
        "Mahal-pooled (no gate)": "Mahal-pooled",
        "Hard gate (Mahal)": "Hard gate",
    }
    # Add soft gate
    soft = df[df["method"].str.startswith("Soft gate full-test")].copy()
    soft["method_short"] = "Soft gate"

    df_plot = df[df["method"].isin(key_methods.keys())].copy()
    df_plot["method_short"] = df_plot["method"].map(key_methods)
    df_plot = pd.concat([df_plot, soft], ignore_index=True)

    configs = df_plot[["dataset", "rare_class"]].drop_duplicates().values.tolist()
    n = len(configs)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    method_order = ["scANVI", "Euclidean", "Mahal-pooled", "Hard gate", "Soft gate"]
    colors_gate = {
        "scANVI": "#888888",
        "Euclidean": "#2196F3",
        "Mahal-pooled": "#F44336",
        "Hard gate": "#FF9800",
        "Soft gate": "#9C27B0",
    }

    for idx, (dataset, rare_class) in enumerate(configs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df_plot[(df_plot["dataset"] == dataset) & (df_plot["rare_class"] == rare_class)]
        sub = sub.sort_values("rts")

        for method in method_order:
            m_sub = sub[sub["method_short"] == method]
            if m_sub.empty:
                continue
            ax.plot(m_sub["rts"], m_sub["rare_f1"],
                    marker="o", label=method, color=colors_gate.get(method, "black"), linewidth=2)

        ax.set_title(f"{dataset}\n{rare_class}", fontsize=9)
        ax.set_xlabel("rts")
        ax.set_ylabel("rare_f1")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("E11: Soft gate vs Hard gate vs No gate", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e11_soft_gate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_e12_temperature():
    """Temperature scaling results."""
    path = EXP_DIR / "e12_temperature_scaling" / "results.csv"
    if not path.exists():
        print("  E12 results not found, skipping fig_e12")
        return

    df = read_table(path)
    # Simplify method names
    df["method_short"] = df["method"].apply(
        lambda x: "Dist-TempScaled" if x.startswith("Dist-TempScaled") else x
    )

    configs = df[["dataset", "rare_class"]].drop_duplicates().values.tolist()
    n = len(configs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, (dataset, rare_class) in enumerate(configs):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[(df["dataset"] == dataset) & (df["rare_class"] == rare_class)]
        sub = sub.sort_values("rts")

        for method in ["scANVI baseline", "Euclidean nearest-proto", "Mahal-pooled", "Dist-TempScaled"]:
            m_sub = sub[sub["method_short"] == method]
            if m_sub.empty:
                continue
            short = method.replace(" baseline", "").replace(" nearest-proto", "")
            color = COLORS.get(short, COLORS.get(method, "black"))
            ax.plot(m_sub["rts"], m_sub["rare_f1"],
                    marker="o", label=short, color=color, linewidth=2)

        ax.set_title(f"{dataset}\n{rare_class}", fontsize=9)
        ax.set_xlabel("rts")
        ax.set_ylabel("rare_f1")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("E12: Distance-based Temperature Scaling", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e12_temperature_scaling.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_comprehensive_heatmap():
    """Comprehensive heatmap: all methods × all (dataset, rare_class, rts)."""
    path = EXP_DIR / "e16_comprehensive_comparison" / "comprehensive_results.csv"
    if not path.exists():
        print("  E16 results not found, skipping comprehensive heatmap")
        return

    df = read_table(path)
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN",
                                "Ensemble", "Soft-gate", "Dist-TempScaled", "Bootstrap"]
                   if c in df.columns]

    # Create row labels
    df["config"] = df["dataset"].str.replace("tabula_", "t_") + "\n" + \
                   df["rare_class"].str[:12] + "\nrts=" + df["rts"].astype(str)

    heatmap_data = df.set_index("config")[method_cols]

    fig, ax = plt.subplots(figsize=(len(method_cols) * 1.5 + 2, len(df) * 0.5 + 2))
    im = ax.imshow(heatmap_data.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(heatmap_data.index, fontsize=7)

    # Add text annotations
    for i in range(len(df)):
        for j in range(len(method_cols)):
            val = heatmap_data.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val < 0.3 or val > 0.85 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=text_color)

    plt.colorbar(im, ax=ax, label="rare_f1")
    ax.set_title("E16: Comprehensive Method Comparison\n(all datasets × all rts)", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_comprehensive_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def fig_e16_best_method():
    """Best method per (dataset, rts) heatmap."""
    path = EXP_DIR / "e16_comprehensive_comparison" / "comprehensive_results.csv"
    if not path.exists():
        return

    df = read_table(path)
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN",
                                "Ensemble", "Soft-gate", "Dist-TempScaled", "Bootstrap"]
                   if c in df.columns]

    # Improvement of best method over scANVI
    if "scANVI" not in df.columns:
        return

    df["best_f1"] = df[method_cols].max(axis=1)
    df["improvement"] = df["best_f1"] - df["scANVI"]
    df["best_method"] = df[method_cols].idxmax(axis=1)

    # Pivot: dataset × rts
    pivot_f1 = df.pivot_table(index="dataset", columns="rts", values="best_f1", aggfunc="mean")
    pivot_imp = df.pivot_table(index="dataset", columns="rts", values="improvement", aggfunc="mean")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Best F1 heatmap
    im1 = axes[0].imshow(pivot_f1.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(pivot_f1.columns)))
    axes[0].set_xticklabels([f"rts={c}" for c in pivot_f1.columns], rotation=45)
    axes[0].set_yticks(range(len(pivot_f1.index)))
    axes[0].set_yticklabels(pivot_f1.index, fontsize=8)
    for i in range(len(pivot_f1.index)):
        for j in range(len(pivot_f1.columns)):
            val = pivot_f1.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title("Mean best rare_f1 (best method)")

    # Improvement heatmap
    im2 = axes[1].imshow(pivot_imp.values, aspect="auto", cmap="RdYlGn", vmin=-0.1, vmax=0.5)
    axes[1].set_xticks(range(len(pivot_imp.columns)))
    axes[1].set_xticklabels([f"rts={c}" for c in pivot_imp.columns], rotation=45)
    axes[1].set_yticks(range(len(pivot_imp.index)))
    axes[1].set_yticklabels(pivot_imp.index, fontsize=8)
    for i in range(len(pivot_imp.index)):
        for j in range(len(pivot_imp.columns)):
            val = pivot_imp.values[i, j]
            if not np.isnan(val):
                axes[1].text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im2, ax=axes[1])
    axes[1].set_title("Mean improvement over scANVI")

    fig.suptitle("E16: Best method performance by dataset and rts", fontsize=12)
    plt.tight_layout()
    out = FIG_DIR / "fig_e16_best_method_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def main():
    print("Generating E17 figures...")
    fig_e8_data_efficiency()
    fig_e9_cb_knn()
    fig_e10_ensemble()
    fig_e11_soft_gate()
    fig_e12_temperature()
    fig_e16_best_method()
    fig_comprehensive_heatmap()
    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
