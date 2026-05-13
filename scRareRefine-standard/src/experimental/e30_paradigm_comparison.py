"""E30: Comprehensive paradigm comparison — Round 4 summary.

Aggregates results from E24-E29 plus previous best methods.
Compares 4 paradigms:
  - Geometric (Mahal-pooled, Euclidean) — from E1-E23
  - Probabilistic (Logit Adjustment E24, MC Dropout E29)
  - Generative (CVAE E25)
  - Statistical guarantee (Conformal Prediction E26)
  - Distribution alignment (Optimal Transport E27)
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

from utils import read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e30_paradigm_comparison"
FIG_DIR = ROOT / "outputs" / "_experimental" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Load results ──────────────────────────────────────────────────────────────

def _load(path: Path, method_col: str, f1_col: str, method_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    if f1_col not in df.columns:
        return pd.DataFrame()
    out = df[["run", "rare_class", "rts"]].copy()
    out["method"] = method_name
    out["rare_f1"] = df[f1_col]
    return out


def load_all() -> pd.DataFrame:
    base = ROOT / "outputs" / "_experimental"
    parts = []

    # E24: Logit Adjustment
    e24 = _load(base / "e24_logit_adjustment" / "results.csv",
                "method", "logit_adj_rare_f1", "Logit Adjustment")
    if not e24.empty:
        # Also add scANVI from E24
        df24 = read_table(base / "e24_logit_adjustment" / "results.csv")
        scanvi = df24[["run", "rare_class", "rts"]].copy()
        scanvi["method"] = "scANVI baseline"
        scanvi["rare_f1"] = df24["scanvi_rare_f1"]
        parts.extend([e24, scanvi])

    # E25: CVAE
    e25 = _load(base / "e25_conditional_vae" / "results.csv",
                "method", "cvae_lr_rare_f1", "CVAE-LR")
    if not e25.empty:
        parts.append(e25)
        df25 = read_table(base / "e25_conditional_vae" / "results.csv")
        smote = df25[["run", "rare_class", "rts"]].copy()
        smote["method"] = "SMOTE-LR"
        smote["rare_f1"] = df25["smote_lr_rare_f1"]
        parts.append(smote)

    # E26: Conformal
    e26_path = base / "e26_conformal_prediction" / "best_per_run.csv"
    if e26_path.exists():
        df26 = read_table(e26_path)
        conf = df26[["run", "rare_class", "rts"]].copy()
        conf["method"] = "Conformal Prediction"
        conf["rare_f1"] = df26["conformal_rare_f1"]
        parts.append(conf)

    # E27: OT
    e27 = _load(base / "e27_optimal_transport" / "results.csv",
                "method", "ot_rare_f1", "Optimal Transport")
    if not e27.empty:
        parts.append(e27)

    # E29: MC Dropout
    e29 = _load(base / "e29_mc_dropout" / "results.csv",
                "method", "mc_dropout_rare_f1", "MC Dropout")
    if not e29.empty:
        parts.append(e29)

    # Previous best: Mahal-pooled (from E14 full sweep)
    e14_path = base / "e14_full_mahal_sweep" / "results.csv"
    if e14_path.exists():
        df14 = read_table(e14_path)
        mahal = df14[["run", "rare_class"]].copy()
        mahal["rts"] = df14["rare_train_size"].astype(str)
        mahal["method"] = "Mahal-pooled"
        mahal["rare_f1"] = df14["mahal_pooled_rare_f1"]
        euc = df14[["run", "rare_class"]].copy()
        euc["rts"] = df14["rare_train_size"].astype(str)
        euc["method"] = "Euclidean"
        euc["rare_f1"] = df14["euclidean_rare_f1"]
        parts.extend([mahal, euc])

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    combined["rare_f1"] = pd.to_numeric(combined["rare_f1"], errors="coerce")
    return combined


# ── Visualizations ────────────────────────────────────────────────────────────

PARADIGM_COLORS = {
    "scANVI baseline":      "#8da0cb",
    "Euclidean":            "#66c2a5",
    "Mahal-pooled":         "#fc8d62",
    "Logit Adjustment":     "#e78ac3",
    "Conformal Prediction": "#a6d854",
    "MC Dropout":           "#ffd92f",
    "CVAE-LR":              "#e5c494",
    "SMOTE-LR":             "#b3b3b3",
    "Optimal Transport":    "#8dd3c7",
}

PARADIGM_LABELS = {
    "scANVI baseline":      "scANVI",
    "Euclidean":            "Euclidean\n(geometric)",
    "Mahal-pooled":         "Mahal-pooled\n(geometric)",
    "Logit Adjustment":     "Logit Adj\n(probabilistic)",
    "Conformal Prediction": "Conformal\n(guarantee)",
    "MC Dropout":           "MC Dropout\n(Bayesian)",
    "CVAE-LR":              "CVAE-LR\n(generative)",
    "SMOTE-LR":             "SMOTE-LR\n(generative)",
    "Optimal Transport":    "OT\n(distribution)",
}


def fig_paradigm_bars(df: pd.DataFrame) -> None:
    """Grouped bar chart: all methods × key datasets."""
    key_runs = [
        ("batch_heldout_seed42_cdc1_rare5",   "cDC1\nrts=5"),
        ("batch_heldout_seed42_cdc1_rare20",  "cDC1\nrts=20"),
        ("batch_heldout_seed42_asdc_rare5",   "ASDC\nrts=5"),
        ("batch_heldout_seed42_gamma_rare5",  "gamma\nrts=5"),
        ("batch_heldout_seed42_epsilon_rare20", "epsilon\nrts=20"),
        ("cell_stratified_seed42_non-classical_monocyte_rare20", "NCM\nrts=20"),
        ("batch_heldout_seed42_innate_lymphoid_cell_rare20", "ILC\nrts=20"),
    ]
    methods = [
        "scANVI baseline", "Euclidean", "Mahal-pooled",
        "Logit Adjustment", "Conformal Prediction", "MC Dropout",
        "CVAE-LR", "Optimal Transport",
    ]

    n_runs = len(key_runs)
    n_methods = len(methods)
    x = np.arange(n_runs)
    width = 0.9 / n_methods

    fig, ax = plt.subplots(figsize=(16, 6))
    for i, method in enumerate(methods):
        vals = []
        for run_id, _ in key_runs:
            sub = df[(df["run"] == run_id) & (df["method"] == method)]
            vals.append(float(sub["rare_f1"].iloc[0]) if not sub.empty else 0.0)
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9,
                      label=PARADIGM_LABELS.get(method, method),
                      color=PARADIGM_COLORS.get(method, "#aaa"),
                      alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in key_runs], fontsize=9)
    ax.set_ylabel("Rare-class F1", fontsize=11)
    ax.set_title("Round 4: Cross-paradigm comparison — rare-class F1 by method and dataset",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_e30_paradigm_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_paradigm_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: methods × datasets."""
    key_runs = [
        "batch_heldout_seed42_cdc1_rare5",
        "batch_heldout_seed42_cdc1_rare20",
        "batch_heldout_seed42_asdc_rare5",
        "batch_heldout_seed42_asdc_rare20",
        "batch_heldout_seed42_gamma_rare5",
        "batch_heldout_seed42_epsilon_rare20",
        "cell_stratified_seed42_non-classical_monocyte_rare20",
        "batch_heldout_seed42_innate_lymphoid_cell_rare20",
    ]
    run_labels = [r.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
                  for r in key_runs]
    methods = [
        "scANVI baseline", "Euclidean", "Mahal-pooled",
        "Logit Adjustment", "Conformal Prediction", "MC Dropout",
        "CVAE-LR", "SMOTE-LR", "Optimal Transport",
    ]

    matrix = np.full((len(methods), len(key_runs)), np.nan)
    for j, run_id in enumerate(key_runs):
        for i, method in enumerate(methods):
            sub = df[(df["run"] == run_id) & (df["method"] == method)]
            if not sub.empty:
                matrix[i, j] = float(sub["rare_f1"].iloc[0])

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Rare-class F1")
    ax.set_xticks(range(len(key_runs)))
    ax.set_xticklabels(run_labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([PARADIGM_LABELS.get(m, m).replace("\n", " ") for m in methods], fontsize=9)
    ax.set_title("Round 4: All paradigms × all datasets — rare-class F1", fontsize=12)
    for i in range(len(methods)):
        for j in range(len(key_runs)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")
    fig.tight_layout()
    out = FIG_DIR / "fig_e30_paradigm_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_logit_adj_tau(base: Path) -> None:
    """τ sweep curves for logit adjustment."""
    tau_files = list((base / "e24_logit_adjustment").glob("*_tau_curve.csv"))
    if not tau_files:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66",
               "#77BEDB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
    for idx, f in enumerate(sorted(tau_files)[:8]):
        df = read_table(f)
        label = f.stem.replace("_tau_curve", "").replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
        ax.plot(df["tau"], df["val_rare_f1"], marker="o", linewidth=2,
                color=palette[idx % len(palette)], label=label)
    ax.set_xlabel("τ (logit adjustment temperature)", fontsize=11)
    ax.set_ylabel("Validation rare-class F1", fontsize=11)
    ax.set_title("E24: Logit Adjustment — τ sweep on validation", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_e24_logit_adj_tau.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def fig_conformal_alpha(base: Path) -> None:
    """α vs rare_f1 for conformal prediction."""
    path = base / "e26_conformal_prediction" / "results.csv"
    if not path.exists():
        return
    df = read_table(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, variant in zip(axes, ["marginal", "class_conditional"]):
        sub = df[df["variant"] == variant]
        runs = sub["run"].unique()[:6]
        palette = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
        for idx, run in enumerate(runs):
            r = sub[sub["run"] == run].sort_values("alpha")
            label = run.replace("batch_heldout_seed42_", "").replace("cell_stratified_seed42_", "")
            ax.plot(r["alpha"], r["conformal_rare_f1"], marker="o", linewidth=2,
                    color=palette[idx % len(palette)], label=label)
        ax.set_xlabel("α (miscoverage rate)", fontsize=10)
        ax.set_ylabel("Rare-class F1", fontsize=10)
        ax.set_title(f"E26: Conformal ({variant})", fontsize=11)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "fig_e26_conformal_alpha.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def main() -> None:
    base = ROOT / "outputs" / "_experimental"
    df = load_all()

    if df.empty:
        print("No data loaded.")
        return

    write_table(df, OUT_DIR / "combined_results.csv")

    # Summary table
    methods = df["method"].unique()
    print("\n=== E30: Paradigm Comparison Summary ===")
    print(f"{'Method':35s}  {'Mean F1':>8s}  {'Std':>6s}  {'n_runs':>7s}")
    print("-" * 65)
    for method in sorted(methods):
        sub = df[df["method"] == method]["rare_f1"].dropna()
        print(f"{method:35s}  {sub.mean():8.3f}  {sub.std():6.3f}  {len(sub):7d}")

    # Best method per run
    best = df.loc[df.groupby("run")["rare_f1"].idxmax()]
    print("\n=== Best method per run ===")
    print(best[["run", "rare_class", "rts", "method", "rare_f1"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best method distribution ===")
    print(best["method"].value_counts().to_string())

    # Generate figures
    print("\nGenerating figures ...")
    fig_paradigm_bars(df)
    fig_paradigm_heatmap(df)
    fig_logit_adj_tau(base)
    fig_conformal_alpha(base)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
