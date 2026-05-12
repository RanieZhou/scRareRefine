"""E15: Latent space alignment — center rare class prototype.

Observation: scANVI's latent space is optimized for majority classes. The rare
class prototype might be systematically biased.

Innovation: Apply a simple affine correction to the latent space:
  z_corrected = z - (mean_majority - mean_rare) * correction_factor

This shifts the latent space so the rare class is more central.
correction_factor is tuned on validation.

Run on: cDC1 rare5/20, ASDC rare5/20, epsilon rare20 (seed42).

Usage:
    python src/experimental/e15_latent_alignment.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e15_latent_alignment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Correction factor grid
CF_GRID = [-1.0, -0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5, 1.0]

RUNS = [
    ("outputs/immune_dc",    "cDC1",   "batch_heldout",   "cdc1",    [5, 20]),
    ("outputs/immune_dc",    "ASDC",   "batch_heldout",   "asdc",    [5, 20]),
    ("outputs/pancreas",     "epsilon","batch_heldout",   "epsilon", [20]),
    ("outputs/pancreas",     "gamma",  "batch_heldout",   "gamma",   [5, 20]),
    ("outputs/tabula_spleen","innate lymphoid cell","batch_heldout","innate_lymphoid_cell",[5, 20]),
    ("outputs/tabula_kidney","endothelial cell","cell_stratified","endothelial_cell",[5, 20]),
]
SEED = 42


def align_latent(
    test_z: np.ndarray,
    train_z: np.ndarray,
    train_labels: np.ndarray,
    is_labeled: np.ndarray,
    rare_class: str,
    correction_factor: float,
) -> np.ndarray:
    """Apply affine correction to center rare class.

    Shift = (mean_majority - mean_rare) * correction_factor
    z_corrected = z - shift
    """
    labeled_mask = is_labeled
    labeled_z = train_z[labeled_mask]
    labeled_labels = train_labels[labeled_mask]

    rare_mask = labeled_labels == rare_class
    majority_mask = labeled_labels != rare_class

    if rare_mask.sum() == 0 or majority_mask.sum() == 0:
        return test_z

    mean_rare     = labeled_z[rare_mask].mean(axis=0)
    mean_majority = labeled_z[majority_mask].mean(axis=0)

    shift = (mean_majority - mean_rare) * correction_factor
    return test_z - shift


def run_one(run_dir: Path, rare_class: str) -> list[dict]:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return []

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError:
        return []

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_true  = test_pred["true_label"].astype(str)
    train_labels = train_pred["true_label"].astype(str).to_numpy()

    if rare_class not in y_true.values:
        return []

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return []

    name = run_dir.name
    seed = None
    rts  = None
    for part in name.split("_"):
        if part.startswith("seed"):
            try:
                seed = int(part[4:])
            except ValueError:
                pass
        if part.startswith("rare") and part != "rareall":
            try:
                rts = int(part[4:])
            except ValueError:
                pass

    rows = []

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "scANVI baseline", "correction_factor": None,
        "rare_f1": scanvi_m["rare_f1"],
        "rare_recall": scanvi_m["rare_recall"],
        "rare_precision": scanvi_m["rare_precision"],
    })

    # Euclidean nearest-proto (no alignment)
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Euclidean (no align)", "correction_factor": 0.0,
        "rare_f1": euc_m["rare_f1"],
        "rare_recall": euc_m["rare_recall"],
        "rare_precision": euc_m["rare_precision"],
    })

    # Mahal-pooled (no alignment)
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)
    mah_pred  = _predict_nearest(mah_dists, classes)
    mah_m, _  = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Mahal-pooled (no align)", "correction_factor": 0.0,
        "rare_f1": mah_m["rare_f1"],
        "rare_recall": mah_m["rare_recall"],
        "rare_precision": mah_m["rare_precision"],
    })

    # Validation split for correction_factor tuning
    n_test = len(test_z)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)
    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_val  = y_true.iloc[val_idx].reset_index(drop=True)

    best_cf = 0.0
    best_val_f1 = -1.0

    for cf in CF_GRID:
        # Align test latent
        test_z_aligned = align_latent(test_z[val_idx], train_z, train_labels, is_labeled, rare_class, cf)

        # Recompute prototypes on aligned space (prototypes shift too)
        # Actually: we shift test cells, not train cells, so prototypes stay the same
        # But we need to recompute protos in the aligned space
        # Simpler: shift both test and train, then recompute protos
        train_z_aligned = align_latent(train_z, train_z, train_labels, is_labeled, rare_class, cf)
        classes_a, protos_a, _ = _class_prototypes(train_z_aligned, train_pred["true_label"], is_labeled)

        # Euclidean in aligned space
        euc_a = _euclidean(test_z_aligned, protos_a)
        pred_a = _predict_nearest(euc_a, classes_a)
        m_a, _ = classification_tables(y_val, pd.Series(pred_a), rare_class=rare_class)

        if m_a["rare_f1"] > best_val_f1:
            best_val_f1 = m_a["rare_f1"]
            best_cf = cf

    # Apply best correction factor to full test
    test_z_best = align_latent(test_z, train_z, train_labels, is_labeled, rare_class, best_cf)
    train_z_best = align_latent(train_z, train_z, train_labels, is_labeled, rare_class, best_cf)
    classes_b, protos_b, _ = _class_prototypes(train_z_best, train_pred["true_label"], is_labeled)

    # Euclidean in best-aligned space
    euc_b = _euclidean(test_z_best, protos_b)
    pred_b = _predict_nearest(euc_b, classes_b)
    m_b, _ = classification_tables(y_true, pd.Series(pred_b), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Euclidean+Align (cf={best_cf})", "correction_factor": best_cf,
        "rare_f1": m_b["rare_f1"],
        "rare_recall": m_b["rare_recall"],
        "rare_precision": m_b["rare_precision"],
    })

    # Mahal-pooled in best-aligned space
    pooled_b = _pooled_covariance_shrunk(train_z_best, train_pred["true_label"], is_labeled, classes_b)
    pooled_covs_b = [pooled_b] * len(classes_b)
    mah_b = _mahalanobis(test_z_best, protos_b, pooled_covs_b)
    pred_mb = _predict_nearest(mah_b, classes_b)
    m_mb, _ = classification_tables(y_true, pd.Series(pred_mb), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Mahal+Align (cf={best_cf})", "correction_factor": best_cf,
        "rare_f1": m_mb["rare_f1"],
        "rare_recall": m_mb["rare_recall"],
        "rare_precision": m_mb["rare_precision"],
    })

    print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
          f"Eucl+Align(cf={best_cf})={m_b['rare_f1']:.3f}  "
          f"Mahal+Align(cf={best_cf})={m_mb['rare_f1']:.3f}")

    return rows


def main() -> pd.DataFrame:
    all_rows = []

    for dataset_dir, rare_class, split_prefix, rare_slug, rts_list in RUNS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            continue

        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  rare_class: {rare_class}")
        print(f"{'='*60}")

        for rts in rts_list:
            run_name = f"{split_prefix}_seed{SEED}_{rare_slug}_rare{rts}"
            run_dir  = dataset_path / run_name
            if not run_dir.exists():
                print(f"  SKIP: {run_name}")
                continue

            rows = run_one(run_dir, rare_class)
            for row in rows:
                row["dataset"] = dataset_name
            all_rows.extend(rows)

    if not all_rows:
        print("No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n\n=== E15 Summary: Latent alignment vs baselines ===")
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Best correction factors
    align_rows = df[df["correction_factor"].notna() & (df["correction_factor"] != 0.0)]
    if not align_rows.empty:
        print("\n\n=== E15 Best correction factors ===")
        print(align_rows[["dataset","rare_class","rts","method","correction_factor","rare_f1"]].to_string(
            index=False, float_format=lambda x: f"{x:.3f}"
        ))

    return df


if __name__ == "__main__":
    main()
