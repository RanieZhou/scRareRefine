"""E8: Soft gate — probability-based rescue threshold.

Motivation: The hard gate (rank ≤ 1) is too conservative for low-sep cases.
Replace with a soft gate based on the Mahal-pooled distance ratio:

    rescue_score(i) = (d_nearest_majority(i) - d_rare(i)) / d_rare(i)

A cell is rescued if rescue_score > τ, where τ is tuned on validation.

Grid search τ ∈ {-0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0}
on validation rare_f1.

Compare:
  - scANVI baseline
  - Current hard gate (rank ≤ 1, Euclidean)
  - Soft gate (Mahal distance ratio, τ tuned on val)

Run on: cDC1 rare5, ASDC rare5, epsilon rare20, NCM rare20, gamma rare20 (seed42).

Usage:
    python src/experimental/e8_soft_gate.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e8_soft_gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_GRID = [-0.5, -0.2, 0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",                    "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",                    "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                 "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",                   "gamma"),
]


def _rescue_scores(
    query_z: np.ndarray,
    protos: np.ndarray,
    classes: list[str],
    rare_class: str,
    dists: np.ndarray,
) -> np.ndarray:
    """Compute rescue_score = (d_nearest_majority - d_rare) / d_rare for each query cell."""
    rare_idx = classes.index(rare_class)
    d_rare = dists[:, rare_idx]

    # Nearest majority distance: min distance among non-rare classes
    majority_mask = np.ones(len(classes), dtype=bool)
    majority_mask[rare_idx] = False
    d_majority = dists[:, majority_mask].min(axis=1)

    # Avoid division by zero
    d_rare_safe = np.where(d_rare < 1e-10, 1e-10, d_rare)
    return (d_majority - d_rare_safe) / d_rare_safe


def _hard_gate_predict(
    query_z: np.ndarray,
    protos: np.ndarray,
    classes: list[str],
    rare_class: str,
    euc_dists: np.ndarray,
    scanvi_pred: np.ndarray,
) -> np.ndarray:
    """Hard gate: rescue only if Euclidean rank of rare prototype is 1 (nearest)."""
    euc_pred = _predict_nearest(euc_dists, classes)
    # Hard gate: only keep rare prediction if it's the nearest prototype
    rare_idx = classes.index(rare_class)
    is_rare_nearest = euc_dists.argmin(axis=1) == rare_idx
    # Start from scANVI, rescue where rare is nearest
    result = np.array(scanvi_pred, dtype=object)
    result[is_rare_nearest] = rare_class
    return result


def _soft_gate_predict(
    rescue_scores: np.ndarray,
    classes: list[str],
    rare_class: str,
    mahal_pred: np.ndarray,
    scanvi_pred: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Soft gate: rescue if rescue_score > tau, else fall back to scANVI."""
    result = np.array(scanvi_pred, dtype=object)
    # Where Mahal predicts rare AND rescue_score > tau, keep rare
    is_rare_pred = mahal_pred == rare_class
    is_rescued = is_rare_pred & (rescue_scores > tau)
    result[is_rescued] = rare_class
    return result


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

    y_val  = val_pred["true_label"].astype(str)
    y_test = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes {classes}, skipping.")
        return None

    # Pooled covariance for Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)

    # Euclidean distances
    val_euc_dists  = _euclidean(val_z, protos)
    test_euc_dists = _euclidean(test_z, protos)

    # Mahal-pooled distances
    val_mahal_dists  = _mahalanobis(val_z, protos, pooled_covs)
    test_mahal_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Mahal predictions
    val_mahal_pred  = _predict_nearest(val_mahal_dists, classes)
    test_mahal_pred = _predict_nearest(test_mahal_dists, classes)

    # Rescue scores
    val_rescue  = _rescue_scores(val_z, protos, classes, rare_class, val_mahal_dists)
    test_rescue = _rescue_scores(test_z, protos, classes, rare_class, test_mahal_dists)

    # scANVI baseline
    scanvi_val_m,  _ = classification_tables(y_val,  val_pred["predicted_label"],  rare_class=rare_class)
    scanvi_test_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Hard gate (Euclidean rank ≤ 1)
    hard_gate_val_pred  = _hard_gate_predict(val_z,  protos, classes, rare_class, val_euc_dists,  val_pred["predicted_label"].to_numpy())
    hard_gate_test_pred = _hard_gate_predict(test_z, protos, classes, rare_class, test_euc_dists, test_pred["predicted_label"].to_numpy())
    hard_gate_val_m,  _ = classification_tables(y_val,  pd.Series(hard_gate_val_pred),  rare_class=rare_class)
    hard_gate_test_m, _ = classification_tables(y_test, pd.Series(hard_gate_test_pred), rare_class=rare_class)

    # Tau grid search on validation
    tau_rows = []
    best_tau = TAU_GRID[0]
    best_val_f1 = -1.0
    for tau in TAU_GRID:
        soft_val_pred = _soft_gate_predict(
            val_rescue, classes, rare_class,
            val_mahal_pred, val_pred["predicted_label"].to_numpy(), tau
        )
        m, _ = classification_tables(y_val, pd.Series(soft_val_pred), rare_class=rare_class)
        tau_rows.append({
            "run": run_dir.name,
            "rare_class": rare_class,
            "tau": tau,
            "val_rare_f1": m["rare_f1"],
            "val_rare_recall": m["rare_recall"],
            "val_rare_precision": m["rare_precision"],
        })
        if m["rare_f1"] > best_val_f1:
            best_val_f1 = m["rare_f1"]
            best_tau = tau

    # Apply best tau to test
    soft_test_pred = _soft_gate_predict(
        test_rescue, classes, rare_class,
        test_mahal_pred, test_pred["predicted_label"].to_numpy(), best_tau
    )
    soft_test_m, _ = classification_tables(y_test, pd.Series(soft_test_pred), rare_class=rare_class)

    # Also: pure Mahal-pooled (no gate)
    mahal_test_m, _ = classification_tables(y_test, pd.Series(test_mahal_pred), rare_class=rare_class)

    print(f"  {run_dir.name}: best_tau={best_tau:.2f}  val_f1={best_val_f1:.3f}  "
          f"test_soft={soft_test_m['rare_f1']:.3f}  test_mahal={mahal_test_m['rare_f1']:.3f}  "
          f"test_hard={hard_gate_test_m['rare_f1']:.3f}  scanvi={scanvi_test_m['rare_f1']:.3f}")

    tau_df = pd.DataFrame(tau_rows)
    tau_df.to_csv(OUT_DIR / f"{run_dir.name}_tau_curve.csv", index=False)

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "best_tau": best_tau,
        "val_f1_at_best_tau": best_val_f1,
        "scanvi_test_rare_f1": scanvi_test_m["rare_f1"],
        "hard_gate_test_rare_f1": hard_gate_test_m["rare_f1"],
        "mahal_nogate_test_rare_f1": mahal_test_m["rare_f1"],
        "soft_gate_test_rare_f1": soft_test_m["rare_f1"],
        "soft_gate_test_recall": soft_test_m["rare_recall"],
        "soft_gate_test_precision": soft_test_m["rare_precision"],
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

    print("\n=== E8 Results: Soft Gate ===")
    cols = ["run", "rare_class", "best_tau", "scanvi_test_rare_f1",
            "hard_gate_test_rare_f1", "mahal_nogate_test_rare_f1", "soft_gate_test_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
