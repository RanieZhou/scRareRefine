"""Stage 3c: Logistic regression baseline (CellTypist-equivalent).

Implements the same algorithm as CellTypist (normalize → scale → one-vs-rest
logistic regression on HVG expression). Uses sklearn directly to avoid
celltypist/sklearn version incompatibilities.

Reads:
    outputs/{dataset}/{run_id}/split_assignments.csv
    outputs/{dataset}/{run_id}/selected_hvg_genes.csv
    outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
    AnnData at config.dataset.path (for raw expression)

Writes:
    outputs/{dataset}/{run_id}/celltypist/
        test_predictions.csv
        test_metrics.csv

Usage:
    python src/03c_celltypist_baseline.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from utils import (
    classification_tables,
    load_config,
    load_adata,
    make_run_dir,
    parse_rare_train_size,
    read_table,
    write_table,
)


def _plot(metrics_df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    cols = ["rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    labels = {"baseline": "Baseline\n(scANVI)", "lr": "LR\n(CellTypist)"}
    colors = {"baseline": "#8da0cb", "lr": "#fc8d62"}
    methods = metrics_df["method"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"LR vs Baseline  |  {rare_class}", fontsize=10, fontweight="bold")
    for ax, col in zip(axes, cols):
        vals = [float(metrics_df.loc[metrics_df["method"] == m, col].iloc[0])
                if col in metrics_df.columns else 0.0 for m in methods]
        bars = ax.bar(range(len(methods)), vals,
                      color=[colors.get(m, "#aaa") for m in methods],
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


def _normalize(adata: sc.AnnData) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3c: Logistic regression baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout",
                        help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rare_class = args.rare_class or cfg["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    label_key = cfg["dataset"]["label_key"]

    run_dir = make_run_dir(cfg, args.split_mode, args.seed, rare_class, rare_train_size)
    out_dir = run_dir / "celltypist"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Read split_assignments.csv for labeled training cells ─────────────────
    assignments_path = run_dir / "split_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(
            f"Stage 2 output not found: {assignments_path}\n"
            "Run Stage 2 (02_baseline_scanvi.py) first."
        )
    assignments = read_table(assignments_path)
    train_asgn = assignments[assignments["split"] == "train"]

    labeled_rare = train_asgn[
        train_asgn["is_labeled_for_scanvi"].astype(str).isin(["True", "1", "true"]) &
        (train_asgn["scanvi_label"] == rare_class)
    ]
    if rare_train_size != "all":
        if isinstance(rare_train_size, float):
            n_rare = max(5, int(rare_train_size * len(labeled_rare)))
        else:
            n_rare = int(rare_train_size)
        if len(labeled_rare) > n_rare:
            labeled_rare = labeled_rare.sample(n_rare, random_state=args.seed)

    labeled_major = train_asgn[
        train_asgn["is_labeled_for_scanvi"].astype(str).isin(["True", "1", "true"]) &
        (train_asgn["scanvi_label"] != rare_class)
    ]
    labeled_ids = set(pd.concat([labeled_major["cell_id"], labeled_rare["cell_id"]]).astype(str))

    # ── Load raw expression ────────────────────────────────────────────────────
    print("Loading AnnData...")
    adata_full = load_adata(cfg)
    adata_full.obs_names = adata_full.obs_names.astype(str)

    # HVG list from Stage 2
    hvg_path = run_dir / "selected_hvg_genes.csv"
    if hvg_path.exists():
        hvg_df = pd.read_csv(hvg_path)
        hvgs = [g for g in hvg_df["gene"].tolist() if g in adata_full.var_names]
        adata_full = adata_full[:, hvgs].copy()
        print(f"  Using {len(hvgs)} HVGs from Stage 2")
    else:
        print("  HVG file not found, using all genes")

    # Normalize
    print("Normalizing (library-size + log1p)...")
    _normalize(adata_full)

    # ── Identify test cells ───────────────────────────────────────────────────
    test_pred_path = run_dir / "embeddings" / "test_predictions.csv"
    if not test_pred_path.exists():
        raise FileNotFoundError(f"Stage 2 test output not found: {test_pred_path}")
    test_meta = read_table(test_pred_path)
    test_ids = test_meta["cell_id"].astype(str).tolist()

    obs_ids = adata_full.obs_names.tolist()
    obs_id_set = set(obs_ids)

    train_ids_list = [i for i in labeled_ids if i in obs_id_set]
    test_ids_filtered = [i for i in test_ids if i in obs_id_set]

    adata_train = adata_full[train_ids_list].copy()
    adata_test  = adata_full[test_ids_filtered].copy()

    # Labels for training
    train_labels = (
        assignments.set_index("cell_id")["scanvi_label"]
        .reindex(train_ids_list)
        .fillna("Unknown")
        .astype(str)
    )

    print(f"  Train cells: {len(adata_train)} (rare: {(train_labels == rare_class).sum()})")
    print(f"  Test cells:  {len(adata_test)}")

    # ── Feature matrix ────────────────────────────────────────────────────────
    def to_dense(X):
        if sp.issparse(X):
            return np.asarray(X.todense())
        return np.asarray(X)

    X_train = to_dense(adata_train.X)
    X_test  = to_dense(adata_test.X)
    y_train = train_labels.values

    # Scale (same as CellTypist)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Logistic regression ───────────────────────────────────────────────────
    print("Training logistic regression (OvR, C=1.0, max_iter=200)...")
    lr = LogisticRegression(C=1.0, max_iter=200, n_jobs=1, random_state=args.seed)
    lr.fit(X_train_s, y_train)
    pred_labels = lr.predict(X_test_s).astype(str)
    print("  Training done.")

    # True labels
    true_labels = (
        test_meta.set_index("cell_id")["true_label"]
        .reindex(test_ids_filtered)
        .astype(str)
        .values
    )

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "cell_id": test_ids_filtered,
        "true_label": true_labels,
        "predicted_label": pred_labels,
    })
    write_table(pred_df, out_dir / "test_predictions.csv")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lr_metrics, _ = classification_tables(true_labels, pred_labels, rare_class=rare_class)

    # Baseline metrics from scANVI predictions for comparison
    baseline_true = test_meta.set_index("cell_id")["true_label"].reindex(test_ids_filtered).astype(str).values
    baseline_pred_vals = test_meta.set_index("cell_id")["predicted_label"].reindex(test_ids_filtered).astype(str).values
    baseline_metrics, _ = classification_tables(baseline_true, baseline_pred_vals, rare_class=rare_class)

    metrics_df = pd.DataFrame([
        {"method": "baseline", **baseline_metrics},
        {"method": "lr", **lr_metrics},
    ])
    write_table(metrics_df, out_dir / "test_metrics.csv")
    _plot(metrics_df, out_dir / "comparison.png", rare_class=rare_class)

    print(f"\n  baseline: F1={baseline_metrics['rare_f1']:.4f}")
    print(f"  lr:       F1={lr_metrics['rare_f1']:.4f}")
    print(f"  Saved → {out_dir}")


if __name__ == "__main__":
    main()
