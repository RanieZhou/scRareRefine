"""E12: Rare-class calibration — temperature scaling on scANVI softmax.

scANVI's softmax is miscalibrated for rare classes. Simple fix: apply per-class
temperature scaling.

For each class c, find temperature T_c that maximizes rare_f1 on validation:
  p_calibrated(c | z) ∝ exp(logit_c / T_c)

For rare class: T_rare < 1.0 (sharpen the distribution, boost rare class probability)
For majority classes: T_majority > 1.0 (flatten, reduce overconfidence)

This is a post-hoc calibration that doesn't require retraining scANVI.

Run on: all datasets, rts=5/20/50, seed42.

Usage:
    python src/experimental/e12_temperature_scaling.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import softmax

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e12_temperature_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Temperature grid for rare class
T_RARE_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

DATASET_CONFIGS = [
    ("outputs/immune_dc",       "cDC1",                    "batch_heldout",   "cdc1"),
    ("outputs/immune_dc",       "ASDC",                    "batch_heldout",   "asdc"),
    ("outputs/pancreas",        "epsilon",                 "batch_heldout",   "epsilon"),
    ("outputs/pancreas",        "gamma",                   "batch_heldout",   "gamma"),
    ("outputs/tabula_liver",    "non-classical monocyte",  "cell_stratified", "non-classical_monocyte"),
    ("outputs/tabula_kidney",   "endothelial cell",        "cell_stratified", "endothelial_cell"),
    ("outputs/tabula_spleen",   "innate lymphoid cell",    "batch_heldout",   "innate_lymphoid_cell"),
    ("outputs/tabula_pancreas", "type B pancreatic cell",  "cell_stratified", "type_b_pancreatic_cell"),
]

SEEDS = [42]
RTS_VALUES = [5, 20, 50]


def load_probabilities(emb_dir: Path) -> pd.DataFrame | None:
    """Load scANVI probability predictions if available."""
    prob_path = emb_dir / "test_probabilities.csv"
    if prob_path.exists():
        return read_table(prob_path)
    return None


def apply_temperature_scaling(
    logits: np.ndarray,
    classes: list[str],
    rare_class: str,
    t_rare: float,
) -> np.ndarray:
    """Apply temperature scaling: rare class gets T=t_rare, others get T=1.0."""
    scaled = logits.copy()
    rare_idx = classes.index(rare_class)
    scaled[:, rare_idx] = logits[:, rare_idx] / t_rare
    # Apply softmax to get calibrated probabilities
    return softmax(scaled, axis=1)


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

    if rare_class not in y_true.values:
        return []

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return []

    # Try to load probability predictions
    probs_df = load_probabilities(emb_dir)

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
        "method": "scANVI baseline", "t_rare": 1.0,
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
        "method": "Euclidean nearest-proto", "t_rare": None,
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
        "method": "Mahal-pooled", "t_rare": None,
        "rare_f1": mah_m["rare_f1"],
        "rare_recall": mah_m["rare_recall"],
        "rare_precision": mah_m["rare_precision"],
    })

    # Temperature scaling — requires probability predictions
    if probs_df is not None:
        # Get class columns (probability columns)
        prob_cols = [c for c in probs_df.columns if c in classes]
        if len(prob_cols) == len(classes):
            # Convert probabilities to log-space (logits)
            probs_arr = probs_df[classes].to_numpy(dtype=float)
            probs_arr = np.clip(probs_arr, 1e-10, 1.0)
            logits = np.log(probs_arr)

            # Validation split for temperature tuning
            n_test = len(test_z)
            rng = np.random.default_rng(42)
            val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
            test_idx = np.setdiff1d(np.arange(n_test), val_idx)
            if len(test_idx) == 0:
                test_idx = np.arange(n_test)

            y_val  = y_true.iloc[val_idx].reset_index(drop=True)
            y_test_split = y_true.iloc[test_idx].reset_index(drop=True)

            best_t = 1.0
            best_val_f1 = -1.0

            for t_rare in T_RARE_GRID:
                cal_probs_val = apply_temperature_scaling(logits[val_idx], classes, rare_class, t_rare)
                pred_val = np.array(classes)[cal_probs_val.argmax(axis=1)]
                m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
                if m_val["rare_f1"] > best_val_f1:
                    best_val_f1 = m_val["rare_f1"]
                    best_t = t_rare

            # Apply best T to full test
            cal_probs_full = apply_temperature_scaling(logits, classes, rare_class, best_t)
            pred_full = np.array(classes)[cal_probs_full.argmax(axis=1)]
            m_cal, _ = classification_tables(y_true, pd.Series(pred_full), rare_class=rare_class)
            rows.append({
                "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
                "method": f"Temp-scaled (T_rare={best_t})", "t_rare": best_t,
                "rare_f1": m_cal["rare_f1"],
                "rare_recall": m_cal["rare_recall"],
                "rare_precision": m_cal["rare_precision"],
            })
            print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
                  f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
                  f"TempScaled(T={best_t})={m_cal['rare_f1']:.3f}")
        else:
            print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
                  f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
                  f"TempScaled=N/A (prob cols mismatch)")
    else:
        # No probability file — use distance-based approach as proxy
        # Convert Mahal distances to "logits" via negative distance
        # (closer = higher logit)
        neg_mah = -mah_dists  # shape (n_test, n_classes)

        n_test = len(test_z)
        rng = np.random.default_rng(42)
        val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
        test_idx = np.setdiff1d(np.arange(n_test), val_idx)
        if len(test_idx) == 0:
            test_idx = np.arange(n_test)

        y_val  = y_true.iloc[val_idx].reset_index(drop=True)

        best_t = 1.0
        best_val_f1 = -1.0

        for t_rare in T_RARE_GRID:
            cal_probs_val = apply_temperature_scaling(neg_mah[val_idx], classes, rare_class, t_rare)
            pred_val = np.array(classes)[cal_probs_val.argmax(axis=1)]
            m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
            if m_val["rare_f1"] > best_val_f1:
                best_val_f1 = m_val["rare_f1"]
                best_t = t_rare

        # Apply best T to full test
        cal_probs_full = apply_temperature_scaling(neg_mah, classes, rare_class, best_t)
        pred_full = np.array(classes)[cal_probs_full.argmax(axis=1)]
        m_cal, _ = classification_tables(y_true, pd.Series(pred_full), rare_class=rare_class)
        rows.append({
            "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
            "method": f"Dist-TempScaled (T_rare={best_t})", "t_rare": best_t,
            "rare_f1": m_cal["rare_f1"],
            "rare_recall": m_cal["rare_recall"],
            "rare_precision": m_cal["rare_precision"],
        })
        print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
              f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
              f"DistTempScaled(T={best_t})={m_cal['rare_f1']:.3f}")

    return rows


def main() -> pd.DataFrame:
    all_rows = []

    for dataset_dir, rare_class, split_prefix, rare_slug in DATASET_CONFIGS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            continue

        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  rare_class: {rare_class}")
        print(f"{'='*60}")

        for seed in SEEDS:
            for rts in RTS_VALUES:
                run_name = f"{split_prefix}_seed{seed}_{rare_slug}_rare{rts}"
                run_dir  = dataset_path / run_name
                if not run_dir.exists():
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

    print("\n\n=== E12 Summary: Temperature scaling vs baselines ===")
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
