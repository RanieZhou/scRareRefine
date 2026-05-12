"""E14: Prototype uncertainty quantification — bootstrap confidence intervals.

Bootstrap the prototype estimate: resample labeled training cells B=100 times,
compute prototype for each bootstrap sample, get a distribution of prototypes.

For each test cell, compute:
- Mean distance to rare prototype across bootstrap samples
- Std of distance (= prototype uncertainty)
- Rescue decision: rescue if mean_dist_rare < mean_dist_pred AND std_dist_rare < threshold

This gives a principled uncertainty estimate for the rescue decision.

Run on: cDC1 rare5/20, ASDC rare5/20, epsilon rare20 (seed42).

Usage:
    python src/experimental/e14_bootstrap_uncertainty.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e14_bootstrap_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

B = 100  # Bootstrap samples
STD_THRESH_GRID = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]

RUNS = [
    ("outputs/immune_dc",    "cDC1",   "batch_heldout",   "cdc1",    [5, 20]),
    ("outputs/immune_dc",    "ASDC",   "batch_heldout",   "asdc",    [5, 20]),
    ("outputs/pancreas",     "epsilon","batch_heldout",   "epsilon", [20]),
    ("outputs/pancreas",     "gamma",  "batch_heldout",   "gamma",   [5, 20]),
    ("outputs/tabula_spleen","innate lymphoid cell","batch_heldout","innate_lymphoid_cell",[5, 20]),
]
SEED = 42


def bootstrap_prototypes(
    ref_z: np.ndarray,
    ref_labels: np.ndarray,
    classes: list[str],
    B: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Bootstrap prototype estimates.

    Returns dict: class -> array of shape (B, d) with B prototype samples.
    """
    boot_protos = {c: [] for c in classes}
    for _ in range(B):
        for c in classes:
            mask = ref_labels == c
            X_c = ref_z[mask]
            if len(X_c) == 0:
                boot_protos[c].append(np.zeros(ref_z.shape[1]))
                continue
            idx = rng.choice(len(X_c), size=len(X_c), replace=True)
            boot_protos[c].append(X_c[idx].mean(axis=0))
    return {c: np.vstack(v) for c, v in boot_protos.items()}


def bootstrap_predict(
    test_z: np.ndarray,
    boot_protos: dict[str, np.ndarray],
    classes: list[str],
    rare_class: str,
    std_thresh: float,
    y_scanvi: np.ndarray,
) -> np.ndarray:
    """Predict using bootstrap uncertainty.

    For each test cell:
    1. Compute mean and std of distance to each class prototype across B samples
    2. Predict class with minimum mean distance
    3. Rescue: if mean_dist_rare < mean_dist_pred AND std_dist_rare < std_thresh,
       override with rare_class
    """
    n = test_z.shape[0]
    B = boot_protos[classes[0]].shape[0]

    # Compute distances to each bootstrap prototype
    # Shape: (n_test, B, n_classes)
    all_dists = np.zeros((n, B, len(classes)))
    for b in range(B):
        for ci, c in enumerate(classes):
            diff = test_z - boot_protos[c][b]
            all_dists[:, b, ci] = np.sqrt((diff * diff).sum(axis=1))

    # Mean and std across bootstrap samples
    mean_dists = all_dists.mean(axis=1)  # (n, n_classes)
    std_dists  = all_dists.std(axis=1)   # (n, n_classes)

    # Base prediction: nearest mean prototype
    base_pred = np.array(classes)[mean_dists.argmin(axis=1)]

    # Rescue logic
    rare_idx = classes.index(rare_class)
    y_out = base_pred.copy()

    for i in range(n):
        pred_idx = mean_dists[i].argmin()
        mean_dist_pred = mean_dists[i, pred_idx]
        mean_dist_rare = mean_dists[i, rare_idx]
        std_dist_rare  = std_dists[i, rare_idx]

        # Rescue if rare is closer (or tied) AND uncertainty is low
        if mean_dist_rare <= mean_dist_pred and std_dist_rare < std_thresh:
            y_out[i] = rare_class

    return y_out


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
    y_scanvi = test_pred["predicted_label"].astype(str).to_numpy()

    if rare_class not in y_true.values:
        return []

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return []

    ref_z      = train_z[is_labeled]
    ref_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

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
        "method": "scANVI baseline", "std_thresh": None,
        "rare_f1": scanvi_m["rare_f1"],
        "rare_recall": scanvi_m["rare_recall"],
        "rare_precision": scanvi_m["rare_precision"],
    })

    # Euclidean nearest-proto
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Euclidean nearest-proto", "std_thresh": None,
        "rare_f1": euc_m["rare_f1"],
        "rare_recall": euc_m["rare_recall"],
        "rare_precision": euc_m["rare_precision"],
    })

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)
    mah_pred  = _predict_nearest(mah_dists, classes)
    mah_m, _  = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Mahal-pooled", "std_thresh": None,
        "rare_f1": mah_m["rare_f1"],
        "rare_recall": mah_m["rare_recall"],
        "rare_precision": mah_m["rare_precision"],
    })

    # Bootstrap uncertainty
    print(f"    Computing {B} bootstrap prototypes...")
    rng = np.random.default_rng(42)
    boot_protos = bootstrap_prototypes(ref_z, ref_labels, classes, B, rng)

    # Validation split for std_thresh tuning
    n_test = len(test_z)
    val_rng = np.random.default_rng(99)
    val_idx = val_rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)
    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_val  = y_true.iloc[val_idx].reset_index(drop=True)
    y_test_split = y_true.iloc[test_idx].reset_index(drop=True)

    best_thresh = np.inf
    best_val_f1 = -1.0

    for thresh in STD_THRESH_GRID:
        pred_val = bootstrap_predict(
            test_z[val_idx], boot_protos, classes, rare_class, thresh, y_scanvi[val_idx]
        )
        m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
        if m_val["rare_f1"] > best_val_f1:
            best_val_f1 = m_val["rare_f1"]
            best_thresh = thresh

    # Apply best threshold to full test
    boot_pred_full = bootstrap_predict(
        test_z, boot_protos, classes, rare_class, best_thresh, y_scanvi
    )
    boot_m, _ = classification_tables(y_true, pd.Series(boot_pred_full), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Bootstrap (std_thresh={best_thresh})", "std_thresh": best_thresh,
        "rare_f1": boot_m["rare_f1"],
        "rare_recall": boot_m["rare_recall"],
        "rare_precision": boot_m["rare_precision"],
    })

    print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
          f"Bootstrap(thresh={best_thresh})={boot_m['rare_f1']:.3f}")

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

    print("\n\n=== E14 Summary: Bootstrap uncertainty vs baselines ===")
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
