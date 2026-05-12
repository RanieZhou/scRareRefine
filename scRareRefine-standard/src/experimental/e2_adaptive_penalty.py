"""E2: Adaptive posterior penalty (λ tuned on validation set).

Distance formula: d_c(z) = mahal(z, mu_c, Sigma_pooled) + λ * tr(Sigma_pooled^-1) / n_c

Grid search λ ∈ {0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0} on validation rare_f1.
Apply best λ to test set.

Runs on: cDC1 rare5, ASDC rare5, epsilon rare20, NCM rare20 (seed42 only).

Usage:
    python src/experimental/e2_adaptive_penalty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent,
    _class_prototypes,
    _euclidean,
    _pooled_covariance_shrunk,
    _mahalanobis,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e2_adaptive_penalty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_GRID = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
]


def mahal_adaptive(
    query: np.ndarray,
    protos: np.ndarray,
    pooled_cov: np.ndarray,
    counts: list[int],
    lam: float,
) -> np.ndarray:
    """Mahalanobis with adaptive penalty: d_c(z) = mahal(z, mu_c, Sigma_pooled) + λ * tr(Sigma^-1) / n_c"""
    pooled_covs = [pooled_cov] * len(counts)
    base = _mahalanobis(query, protos, pooled_covs)
    try:
        inv = np.linalg.inv(pooled_cov)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(pooled_cov)
    tr_inv = float(np.trace(inv))
    penalty = np.array([lam * tr_inv / max(n, 1) for n in counts])
    return base + penalty[None, :]


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"  WARNING: {emb_dir} not found, skipping.")
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        val_lat    = read_table(emb_dir / "validation_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  WARNING: missing file {e}, skipping {run_dir}")
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    val_z   = _latent(val_lat)
    test_z  = _latent(test_lat)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    counts = [counts_map[c] for c in classes]
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)

    y_val  = val_pred["true_label"].astype(str)
    y_test = test_pred["true_label"].astype(str)

    # Grid search on validation
    val_results = []
    for lam in LAMBDA_GRID:
        dists = mahal_adaptive(val_z, protos, pooled, counts, lam)
        pred  = _predict_nearest(dists, classes)
        m, _  = classification_tables(y_val, pd.Series(pred), rare_class=rare_class)
        val_results.append({"lambda": lam, "val_rare_f1": m["rare_f1"]})

    val_df = pd.DataFrame(val_results)
    best_lam = float(val_df.loc[val_df["val_rare_f1"].idxmax(), "lambda"])
    print(f"  {run_dir.name}: best λ = {best_lam} (val rare_f1 = {val_df['val_rare_f1'].max():.3f})")

    # Apply best λ to test
    test_dists = mahal_adaptive(test_z, protos, pooled, counts, best_lam)
    test_pred_labels = _predict_nearest(test_dists, classes)
    test_m, _ = classification_tables(y_test, pd.Series(test_pred_labels), rare_class=rare_class)

    # Euclidean baseline on test
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Save lambda curve
    val_df["dataset"] = run_dir.name
    val_df["rare_class"] = rare_class
    write_table(val_df, OUT_DIR / f"{run_dir.name}_lambda_curve.csv")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "best_lambda": best_lam,
        "val_rare_f1_at_best_lambda": float(val_df["val_rare_f1"].max()),
        "test_rare_f1_adaptive": test_m["rare_f1"],
        "test_rare_f1_euclidean": euc_m["rare_f1"],
        "test_rare_f1_scanvi": scanvi_m["rare_f1"],
        "test_overall_acc_adaptive": test_m["overall_accuracy"],
        "test_overall_acc_euclidean": euc_m["overall_accuracy"],
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        result = run_one(run_dir, rare_class)
        if result:
            rows.append(result)

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E2 Results (adaptive λ, seed42) ===")
    cols = ["run", "rare_class", "best_lambda", "test_rare_f1_scanvi",
            "test_rare_f1_euclidean", "test_rare_f1_adaptive"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
