"""E15: Visualizations for Round 2 experiments (E8-E14).

Generates:
1. fig_e8_soft_gate.png
2. fig_e9_ensemble.png
3. fig_e10_adaptive_selector.png
4. fig_e11_gmm_calibrated.png
5. fig_e12_bootstrap_uncertainty.png
6. fig_e13_label_propagation.png
7. fig_e14_mahal_sweep_scatter.png
8. fig_e14_mahal_sweep_heatmap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import read_table

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "outputs" / "_experimental"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "scANVI": "#888888",
    "Euclidean": "#4C72B0",
    "Mahal-pooled": "#DD8452",
    "Hard gate": "#55A868",
    "Soft gate": "#C44E52",
    "CB-kNN": "#8172B2",
    "Ensemble": "#937860",
    "Adaptive": "#DA8BC3",
    "GMM uncal": "#8C8C8C",
    "GMM cal": "#CCB974",
    "Bootstrap": "#64B5CD",
    "Label Prop": "#E377C2",
}


# ── E8: Soft Gate ─────────────────────────────────────────────────────────────

def fig_e8():
    df = read_table(EXP_DIR / "e8_soft_gate" / "results.csv")
    labels = [r.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
              for r in df["run"]]
    x = np.arange(len(df))
    w = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*w, df["scanvi_test_rare_f1"],    w, label="scANVI",      color=COLORS["scANVI"])
    ax.bar(x - 0.5*w, df["hard_gate_test_rare_f1"], w, label="Hard gate",   color=COLORS["Hard gate"])
    ax.bar(x + 0.5*w, df["mahal_nogate_test_rare_f1"], w, label="Mahal (no gate)", color=COLORS["Mahal-pooled"])
    ax.bar(x + 1.5*w, df["soft_gate_test_rare_f1"], w, label="Soft gate",   color=COLORS["Soft gate"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E8: Soft Gate vs Hard Gate vs Mahal (no gate)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "fig_e8_soft_gate.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E9: Ensemble ──────────────────────────────────────────────────────────────

def fig_e9():
    df = read_table(EXP_DIR / "e9_ensemble" / "results.csv")
    labels = [r.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
              for r in df["run"]]
    x = np.arange(len(df))
    w = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar comparison
    ax = axes[0]
    ax.bar(x - 1.5*w, df["scanvi_test_rare_f1"],        w, label="scANVI",      color=COLORS["scANVI"])
    ax.bar(x - 0.5*w, df["mahal_pooled_test_rare_f1"],  w, label="Mahal-pooled", color=COLORS["Mahal-pooled"])
    ax.bar(x + 0.5*w, df["cb_knn_test_rare_f1"],        w, label="CB-kNN",      color=COLORS["CB-kNN"])
    ax.bar(x + 1.5*w, df["ensemble_test_rare_f1"],      w, label="Ensemble",    color=COLORS["Ensemble"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E9: Ensemble vs Individual Methods")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    # Right: alpha sweep curves
    ax2 = axes[1]
    alpha_files = list((EXP_DIR / "e9_ensemble").glob("*_alpha_curve.csv"))
    run_colors = plt.cm.tab10(np.linspace(0, 1, len(alpha_files)))
    for i, f in enumerate(alpha_files):
        adf = pd.read_csv(f)
        run_name = adf["run"].iloc[0].replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
        ax2.plot(adf["alpha"], adf["val_rare_f1"], marker="o", label=run_name, color=run_colors[i])
    ax2.set_xlabel("α (weight on Mahal score)")
    ax2.set_ylabel("Validation rare F1")
    ax2.set_title("E9: Alpha sweep on validation")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)

    fig.tight_layout()
    out = FIG_DIR / "fig_e9_ensemble.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E10: Adaptive Selector ────────────────────────────────────────────────────

def fig_e10():
    df = read_table(EXP_DIR / "e10_adaptive_selector" / "results.csv")
    labels = [r.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
              for r in df["run"]]
    x = np.arange(len(df))
    w = 0.15

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: F1 comparison
    ax = axes[0]
    ax.bar(x - 2*w, df["scanvi_rare_f1"],       w, label="scANVI",      color=COLORS["scANVI"])
    ax.bar(x - 1*w, df["euclidean_rare_f1"],    w, label="Euclidean",   color=COLORS["Euclidean"])
    ax.bar(x + 0*w, df["mahal_pooled_rare_f1"], w, label="Mahal-pooled", color=COLORS["Mahal-pooled"])
    ax.bar(x + 1*w, df["cb_knn_rare_f1"],       w, label="CB-kNN",      color=COLORS["CB-kNN"])
    ax.bar(x + 2*w, df["adaptive_rare_f1"],     w, label="Adaptive",    color=COLORS["Adaptive"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E10: Adaptive Selector vs Individual Methods")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)

    # Right: separability ratio vs adaptive gain
    ax2 = axes[1]
    delta_adaptive_vs_best = df["adaptive_rare_f1"] - df[["euclidean_rare_f1", "mahal_pooled_rare_f1", "cb_knn_rare_f1"]].max(axis=1)
    sc = ax2.scatter(df["separability_ratio"], delta_adaptive_vs_best,
                     c=df["adaptive_rare_f1"], cmap="RdYlGn", s=80, vmin=0, vmax=1)
    for i, lbl in enumerate(labels):
        ax2.annotate(lbl, (df["separability_ratio"].iloc[i], delta_adaptive_vs_best.iloc[i]),
                     fontsize=7, ha="left", va="bottom")
    ax2.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax2.axvline(1.3, color="red", linestyle=":", alpha=0.5, label="S=1.3 threshold")
    ax2.axvline(1.0, color="orange", linestyle=":", alpha=0.5, label="S=1.0 threshold")
    ax2.set_xlabel("Separability ratio S")
    ax2.set_ylabel("Adaptive F1 − Best individual F1")
    ax2.set_title("E10: Adaptive gain vs separability")
    ax2.legend(fontsize=8)
    plt.colorbar(sc, ax=ax2, label="Adaptive F1")

    fig.tight_layout()
    out = FIG_DIR / "fig_e10_adaptive_selector.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E11: Calibrated GMM ───────────────────────────────────────────────────────

def fig_e11():
    df = read_table(EXP_DIR / "e11_gmm_calibrated" / "results.csv")
    labels = [f"{r['rare_class']} n={r['n_rare_train']}" for _, r in df.iterrows()]
    x = np.arange(len(df))
    w = 0.15

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 2*w, df["scanvi_rare_f1"],           w, label="scANVI",       color=COLORS["scANVI"])
    ax.bar(x - 1*w, df["euclidean_rare_f1"],         w, label="Euclidean",    color=COLORS["Euclidean"])
    ax.bar(x + 0*w, df["mahal_pooled_rare_f1"],      w, label="Mahal-pooled", color=COLORS["Mahal-pooled"])
    ax.bar(x + 1*w, df["gmm_uncalibrated_rare_f1"],  w, label="GMM uncal",    color=COLORS["GMM uncal"])
    ax.bar(x + 2*w, df["gmm_calibrated_rare_f1"],    w, label="GMM cal",      color=COLORS["GMM cal"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E11: Calibrated GMM vs Baselines")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.text(0.02, 0.95, "Note: GMM still fails after calibration\n(density ratio approach insufficient)",
            transform=ax.transAxes, fontsize=8, color="red", va="top")
    fig.tight_layout()
    out = FIG_DIR / "fig_e11_gmm_calibrated.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E12: Bootstrap Uncertainty ────────────────────────────────────────────────

def fig_e12():
    df = read_table(EXP_DIR / "e12_prototype_uncertainty" / "results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: F1 comparison across all runs
    ax = axes[0]
    labels = [r.replace("batch_heldout_", "").replace("seed4", "s4") for r in df["run"]]
    x = np.arange(len(df))
    w = 0.2
    ax.bar(x - 1.5*w, df["scanvi_rare_f1"],      w, label="scANVI",      color=COLORS["scANVI"])
    ax.bar(x - 0.5*w, df["euclidean_rare_f1"],   w, label="Euclidean",   color=COLORS["Euclidean"])
    ax.bar(x + 0.5*w, df["mahal_pooled_rare_f1"],w, label="Mahal-pooled",color=COLORS["Mahal-pooled"])
    ax.bar(x + 1.5*w, df["bootstrap_rare_f1"],   w, label="Bootstrap",   color=COLORS["Bootstrap"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E12: Bootstrap Uncertainty vs Baselines")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.1)

    # Right: mean threshold vs n_rare_train
    ax2 = axes[1]
    ax2.scatter(df["n_rare_train"], df["mean_rare_threshold"],
                label="Mean rare threshold (95th pct)", color=COLORS["Bootstrap"], s=80)
    ax2.scatter(df["n_rare_train"], df["mean_majority_threshold"],
                label="Mean majority threshold (5th pct)", color=COLORS["Euclidean"], s=80, marker="^")
    ax2.set_xlabel("n_rare_train")
    ax2.set_ylabel("Bootstrap distance threshold")
    ax2.set_title("E12: Bootstrap threshold vs n_rare_train")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = FIG_DIR / "fig_e12_bootstrap_uncertainty.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E13: Label Propagation ────────────────────────────────────────────────────

def fig_e13():
    df = read_table(EXP_DIR / "e13_label_propagation" / "results.csv")
    labels = [f"{r['rare_class']}\n(n={r['n_rare_train']})" for _, r in df.iterrows()]
    x = np.arange(len(df))
    w = 0.18

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5*w, df["scanvi_rare_f1"],       w, label="scANVI",       color=COLORS["scANVI"])
    ax.bar(x - 0.5*w, df["euclidean_rare_f1"],    w, label="Euclidean",    color=COLORS["Euclidean"])
    ax.bar(x + 0.5*w, df["mahal_pooled_rare_f1"], w, label="Mahal-pooled", color=COLORS["Mahal-pooled"])
    ax.bar(x + 1.5*w, df["label_prop_rare_f1"],   w, label="Label Prop\n(TRANSDUCTIVE)", color=COLORS["Label Prop"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Rare class F1")
    ax.set_title("E13: Label Propagation (Transductive) vs Inductive Methods")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)
    ax.text(0.02, 0.95, "⚠ TRANSDUCTIVE: uses test cell positions\nNot valid for deployment",
            transform=ax.transAxes, fontsize=8, color="red", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    fig.tight_layout()
    out = FIG_DIR / "fig_e13_label_propagation.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E14: Mahal Sweep Scatter ──────────────────────────────────────────────────

def fig_e14_scatter():
    df = read_table(EXP_DIR / "e14_full_mahal_sweep" / "results.csv")

    fig, ax = plt.subplots(figsize=(10, 6))

    datasets = df["rare_class"].unique()
    cmap = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    color_map = {d: cmap[i] for i, d in enumerate(datasets)}

    for rc in datasets:
        sub = df[df["rare_class"] == rc]
        ax.scatter(sub["separability_ratio"], sub["delta_mahal_minus_euc"],
                   label=rc, color=color_map[rc], s=60, alpha=0.8)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.axvline(1.0, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="S=1.0")
    ax.axvline(1.3, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="S=1.3")

    # Trend line
    from numpy.polynomial import polynomial as P
    x = df["separability_ratio"].to_numpy()
    y = df["delta_mahal_minus_euc"].to_numpy()
    c = P.polyfit(x, y, 1)
    xfit = np.linspace(x.min(), x.max(), 100)
    ax.plot(xfit, P.polyval(xfit, c), "k-", linewidth=1.5, alpha=0.5, label="Linear trend")

    ax.set_xlabel("Separability ratio S")
    ax.set_ylabel("Δ F1 (Mahal-pooled − Euclidean)")
    ax.set_title("E14: Mahal-pooled improvement vs Separability ratio\n(29 runs, seed42)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out = FIG_DIR / "fig_e14_mahal_sweep_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── E14: Mahal Sweep Heatmap ──────────────────────────────────────────────────

def fig_e14_heatmap():
    df = read_table(EXP_DIR / "e14_full_mahal_sweep" / "results.csv")

    # Pivot: rows = rare_class, cols = rare_train_size
    pivot = df.pivot_table(
        index="rare_class",
        columns="rare_train_size",
        values="delta_mahal_minus_euc",
        aggfunc="mean",
    )
    # Sort columns numerically
    try:
        pivot = pivot[sorted(pivot.columns, key=lambda x: int(x))]
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.15, vmax=0.6, aspect="auto")
    plt.colorbar(im, ax=ax, label="Δ F1 (Mahal − Euclidean)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"rts={c}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(val) < 0.3 else "white")

    ax.set_title("E14: Mahal-pooled improvement heatmap\n(Δ F1 = Mahal − Euclidean, seed42)")
    ax.set_xlabel("Rare train size")
    ax.set_ylabel("Rare class")
    fig.tight_layout()
    out = FIG_DIR / "fig_e14_mahal_sweep_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    print("Generating E15 visualizations...")
    for fn, name in [
        (fig_e8,           "E8: Soft Gate"),
        (fig_e9,           "E9: Ensemble"),
        (fig_e10,          "E10: Adaptive Selector"),
        (fig_e11,          "E11: Calibrated GMM"),
        (fig_e12,          "E12: Bootstrap Uncertainty"),
        (fig_e13,          "E13: Label Propagation"),
        (fig_e14_scatter,  "E14: Mahal Sweep Scatter"),
        (fig_e14_heatmap,  "E14: Mahal Sweep Heatmap"),
    ]:
        try:
            print(f"  Generating {name}...")
            fn()
        except Exception as exc:
            print(f"  ERROR generating {name}: {exc}")

    print(f"\nAll figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
