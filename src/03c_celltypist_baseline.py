"""Stage 3c: CellTypist logistic regression baseline (best match).

Uses the actual CellTypist library (train + annotate).

Input normalization:
    CellTypist does NOT normalize raw counts internally — it only applies
    StandardScaler after receiving log1p-normalized expression.  We pre-normalize
    with scanpy (normalize_total to 10k + log1p) and pass check_expression=False
    to skip CellTypist's format validation (which would re-warn on already-
    normalized data).

sklearn compatibility:
    CellTypist 1.7.1 hardcodes multi_class='ovr' in LogisticRegression, which
    was removed in sklearn >= 1.7.  Two effects:
    (a) sklearn >= 1.7 raises TypeError → must patch celltypist/train.py to drop
        multi_class='ovr' from the LogisticRegression call.
    (b) The patch restores sklearn 1.8 default (multinomial/lbfgs for multi-class),
        which is actually BETTER for rare cell detection than OvR: with only
        O(5–20) rare training cells vs O(10k) non-rare cells, OvR binary
        classifiers fail to converge (tested: F1 drops from 0.41 → 0.03).
    Patch applied to: site-packages/celltypist/train.py, lines ~126 and ~146.

majority_voting note (rejected):
    CellTypist's majority_voting=True over-clusters test cells via leiden
    (resolution=10) and assigns each cluster its plurality label.  For rare
    cell types with O(10–100) test cells, those cells are always a minority
    in every cluster → all rare predictions are overwritten → F1=0.00.
    Majority voting is designed to smooth noisy common-cell predictions, not
    to preserve rare-cell signals.  We use mode='best match' (highest softmax
    probability) which gives CellTypist its best possible rare-class performance.

Reads:
    outputs/{dataset}/{run_id}/split_assignments.csv
    outputs/{dataset}/{run_id}/selected_hvg_genes.csv
    outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
    AnnData at config.dataset.path (for raw expression)

Writes:
    outputs/{dataset}/{run_id}/celltypist/
        test_predictions.csv
        test_metrics.csv
        comparison.png

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
    labels = {"baseline": "Baseline\n(scANVI)", "lr": "CellTypist\n(MV)"}
    colors = {"baseline": "#8da0cb", "lr": "#fc8d62"}
    methods = metrics_df["method"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"CellTypist (majority voting) vs Baseline  |  {rare_class}",
                 fontsize=10, fontweight="bold")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3c: CellTypist baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout",
                        help="batch_heldout | cell_stratified")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    try:
        import celltypist
    except ImportError:
        raise ImportError("celltypist is not installed. Run: pip install celltypist")

    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) >= Version("1.7"):
        # Verify the required patch has been applied to celltypist/train.py.
        # Without the patch, celltypist passes multi_class='ovr' to LogisticRegression,
        # which was removed in sklearn 1.7 → TypeError at training time.
        import sys, importlib
        _ct_mod = importlib.import_module("celltypist.train")
        _train_file = Path(_ct_mod.__file__).read_text()
        if "multi_class" in _train_file:
            raise RuntimeError(
                f"celltypist 1.7.1 is incompatible with scikit-learn {sklearn.__version__}. "
                "Apply patch: remove multi_class='ovr' from LogisticRegression calls "
                "in site-packages/celltypist/train.py (~lines 126 and 146)."
            )

    cfg = load_config(args.config)
    rare_class = args.rare_class or cfg["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)

    run_dir = make_run_dir(cfg, args.split_mode, args.seed, rare_class, rare_train_size)
    out_dir = run_dir / "celltypist"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Labeled training cells ─────────────────────────────────────────────────
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

    # ── Load expression and normalize ─────────────────────────────────────────
    # CellTypist expects log1p-normalized expression (normalize_total to 10k + log1p).
    # It does NOT normalize internally; it only applies StandardScaler after this step.
    print("Loading AnnData...")
    adata_full = load_adata(cfg)
    adata_full.obs_names = adata_full.obs_names.astype(str)

    hvg_path = run_dir / "selected_hvg_genes.csv"
    if hvg_path.exists():
        hvg_df = pd.read_csv(hvg_path)
        hvgs = [g for g in hvg_df["gene"].tolist() if g in adata_full.var_names]
        adata_full = adata_full[:, hvgs].copy()
        print(f"  Using {len(hvgs)} HVGs from Stage 2")
    else:
        print("  HVG file not found, using all genes")

    print("Normalizing (library-size + log1p)...")
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    # ── Subset to train / test ─────────────────────────────────────────────────
    test_pred_path = run_dir / "embeddings" / "test_predictions.csv"
    if not test_pred_path.exists():
        raise FileNotFoundError(f"Stage 2 test output not found: {test_pred_path}")
    test_meta = read_table(test_pred_path)
    test_ids = test_meta["cell_id"].astype(str).tolist()

    obs_id_set = set(adata_full.obs_names.tolist())
    train_ids_list = [i for i in labeled_ids if i in obs_id_set]
    test_ids_filtered = [i for i in test_ids if i in obs_id_set]

    adata_train = adata_full[train_ids_list].copy()
    adata_test  = adata_full[test_ids_filtered].copy()

    # CellTypist requires dense matrix for training
    if sp.issparse(adata_train.X):
        adata_train.X = np.asarray(adata_train.X.todense())
    if sp.issparse(adata_test.X):
        adata_test.X = np.asarray(adata_test.X.todense())

    train_labels = (
        assignments.set_index("cell_id")["scanvi_label"]
        .reindex(train_ids_list)
        .fillna("Unknown")
        .astype(str)
    )
    adata_train.obs["celltypist_label"] = train_labels.values

    print(f"  Train cells: {len(adata_train)} (rare: {(train_labels == rare_class).sum()})")
    print(f"  Test cells:  {len(adata_test)}")

    # ── CellTypist train ───────────────────────────────────────────────────────
    # check_expression=False: we have already done log1p normalization above;
    # CellTypist's check would reject re-normalized data whose max > 20.
    print("Training CellTypist model (OvR logistic regression, C=1.0, max_iter=200)...")
    new_model = celltypist.train(
        adata_train,
        labels="celltypist_label",
        check_expression=False,
        C=1.0,
        max_iter=200,
        n_jobs=4,
    )
    print("  Training done.")

    # ── CellTypist annotate (best match) ──────────────────────────────────────
    # mode='best match': assigns each cell the class with the highest softmax
    # probability — functionally identical to the sklearn OvR reimplementation.
    #
    # NOTE on majority_voting=True (tested, rejected for rare-cell use):
    #   CellTypist's majority voting over-clusters the test set via leiden
    #   (resolution=10 by default), then assigns each cluster its plurality label.
    #   For rare cell types (e.g. ASDC with ~10–20 test cells), those cells are
    #   a small minority in every cluster they land in, so majority voting
    #   systematically overwrites their predictions with the dominant non-rare
    #   label.  In testing this dropped rare-class F1 from 0.41 → 0.00.
    #   Majority voting is designed to smooth noisy predictions across common
    #   cell types, not to preserve rare-cell signals — using it here would
    #   artificially weaken the CellTypist baseline and reduce comparison
    #   credibility.
    print("Annotating (best match)...")
    result = celltypist.annotate(
        adata_test,
        model=new_model,
        mode="best match",
    )
    pred_labels = result.predicted_labels["predicted_labels"].astype(str).values

    true_labels = (
        test_meta.set_index("cell_id")["true_label"]
        .reindex(test_ids_filtered)
        .astype(str)
        .values
    )

    # ── Save predictions ───────────────────────────────────────────────────────
    pred_df = pd.DataFrame({
        "cell_id": test_ids_filtered,
        "true_label": true_labels,
        "predicted_label": pred_labels,
    })
    write_table(pred_df, out_dir / "test_predictions.csv")

    # ── Metrics ───────────────────────────────────────────────────────────────
    lr_metrics, _ = classification_tables(true_labels, pred_labels, rare_class=rare_class)

    baseline_true = (test_meta.set_index("cell_id")["true_label"]
                     .reindex(test_ids_filtered).astype(str).values)
    baseline_pred_vals = (test_meta.set_index("cell_id")["predicted_label"]
                          .reindex(test_ids_filtered).astype(str).values)
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
