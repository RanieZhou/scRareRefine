"""E20: Calibration-aware rescue with Platt scaling.

Algorithm:
1. Load scANVI softmax probabilities (prob_* columns in predictions.csv)
2. Fit logistic regression on validation set:
   input = log(p_rare / (1 - p_rare)), output = true_label == rare_class
3. Apply calibrated probabilities to test set
4. Rescue cells where calibrated_p_rare > threshold τ (tuned on validation)

Compare vs: scANVI baseline, Euclidean nearest-proto, Platt-calibrated scANVI.

Run on: cDC1 rare5, ASDC rare5, epsilon rare20, NCM rare20 (seed42).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent,
    _class_prototypes,
    _euclidean,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e20_platt_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",                          "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",                          "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                        "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20",   "non-classical monocyte"),
]

TAU_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _get_prob_col(df: pd.DataFrame, rare_class: str) -> str | None:
    """Find the probability column for rare_class."""
    # Try exact match first
    col = f"prob_{rare_class}"
    if col in df.columns:
        return col
    # Try case-insensitive
    for c in df.columns:
        if c.startswith("prob_") and c.lower() == f"prob_{rare_class}".lower():
            return c
    # Try partial match
    for c in df.columns:
        if c.startswith("prob_") and rare_class.lower() in c.lower():
            return c
    return None


def _platt_calibrate(
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    rare_class: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit Platt scaling on validation, apply to test.

    Returns:
        val_calibrated_probs: calibrated probabilities on validation
        test_calibrated_probs: calibrated probabilities on test
        platt_intercept: fitted intercept
    """
    prob_col = _get_prob_col(val_pred, rare_class)
    if prob_col is None:
        print(f"  WARNING: no prob column for {rare_class}, available: {[c for c in val_pred.columns if c.startswith('prob_')]}")
        return None, None, None

    # Validation
    val_p_rare = val_pred[prob_col].to_numpy().clip(1e-7, 1 - 1e-7)
    val_logit = np.log(val_p_rare / (1 - val_p_rare)).reshape(-1, 1)
    val_y = (val_pred["true_label"].astype(str) == rare_class).astype(int).to_numpy()

    # Fit Platt scaling (logistic regression on logit)
    platt = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    platt.fit(val_logit, val_y)

    val_cal = platt.predict_proba(val_logit)[:, 1]

    # Test
    test_prob_col = _get_prob_col(test_pred, rare_class)
    if test_prob_col is None:
        return val_cal, None, float(platt.intercept_[0])

    test_p_rare = test_pred[test_prob_col].to_numpy().clip(1e-7, 1 - 1e-7)
    test_logit = np.log(test_p_rare / (1 - test_p_rare)).reshape(-1, 1)
    test_cal = platt.predict_proba(test_logit)[:, 1]

    return val_cal, test_cal, float(platt.intercept_[0])


def _rescue_with_threshold(
    base_pred: np.ndarray,
    calibrated_probs: np.ndarray,
    rare_class: str,
    tau: float,
) -> np.ndarray:
    """Rescue cells where calibrated_p_rare > tau."""
    pred = base_pred.copy()
    for i, p in enumerate(calibrated_probs):
        if p > tau:
            pred[i] = rare_class
    return pred


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"  WARNING: {emb_dir} not found, skipping.")
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  WARNING: missing file {e}, skipping {run_dir}")
        return None

    # Load validation predictions
    val_pred = None
    val_lat = None
    try:
        val_pred = read_table(emb_dir / "validation_predictions.csv")
        val_lat  = read_table(emb_dir / "validation_latent.csv")
    except FileNotFoundError:
        print(f"  WARNING: no validation predictions for {run_dir.name}, using train as proxy")
        # Use labeled training cells as validation proxy
        is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
        val_pred = train_pred[is_labeled].reset_index(drop=True)
        val_lat  = train_lat[is_labeled].reset_index(drop=True)

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)
    y_val   = val_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    # ── Euclidean nearest-prototype ────────────────────────────────────────
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # ── scANVI baseline ────────────────────────────────────────────────────
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # ── Platt calibration ─────────────────────────────────────────────────
    val_cal, test_cal, platt_intercept = _platt_calibrate(val_pred, test_pred, rare_class)

    if test_cal is None:
        print(f"  WARNING: Platt calibration failed for {run_dir.name}")
        return {
            "run": run_dir.name,
            "rare_class": rare_class,
            "n_rare_train": counts_map.get(rare_class, 0),
            "scanvi_rare_f1": scanvi_m["rare_f1"],
            "euclidean_rare_f1": euc_m["rare_f1"],
            "platt_best_tau": float("nan"),
            "platt_rare_f1": float("nan"),
            "platt_recall": float("nan"),
            "platt_precision": float("nan"),
            "platt_intercept": float("nan"),
        }

    # Tune τ on validation
    val_cal_val, _, _ = _platt_calibrate(val_pred, val_pred, rare_class)
    if val_cal_val is None:
        val_cal_val = val_cal  # fallback

    best_tau = TAU_GRID[0]
    best_val_f1 = -1.0
    tau_curve = []
    for tau in TAU_GRID:
        # On validation: rescue cells where calibrated_p > tau
        val_scanvi = val_pred["predicted_label"].astype(str).to_numpy()
        val_rescued = _rescue_with_threshold(val_scanvi, val_cal_val, rare_class, tau)
        val_m, _ = classification_tables(y_val, pd.Series(val_rescued), rare_class=rare_class)
        tau_curve.append({"tau": tau, "val_rare_f1": val_m["rare_f1"]})
        if val_m["rare_f1"] > best_val_f1:
            best_val_f1 = val_m["rare_f1"]
            best_tau = tau

    # Apply best τ to test
    test_scanvi = test_pred["predicted_label"].astype(str).to_numpy()
    test_rescued = _rescue_with_threshold(test_scanvi, test_cal, rare_class, best_tau)
    platt_m, _ = classification_tables(y_test, pd.Series(test_rescued), rare_class=rare_class)

    # Save tau curve
    tau_df = pd.DataFrame(tau_curve)
    write_table(tau_df, OUT_DIR / f"{run_dir.name}_tau_curve.csv")

    # Save calibrated probabilities
    prob_col = _get_prob_col(test_pred, rare_class)
    if prob_col:
        cal_df = pd.DataFrame({
            "true_label": y_test.to_numpy(),
            "scanvi_pred": test_scanvi,
            "raw_prob_rare": test_pred[prob_col].to_numpy(),
            "calibrated_prob_rare": test_cal,
            "platt_pred": test_rescued,
        })
        write_table(cal_df, OUT_DIR / f"{run_dir.name}_calibrated_probs.csv")

    print(f"  {run_dir.name}: best_tau={best_tau:.2f}  "
          f"scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"platt={platt_m['rare_f1']:.3f}  (intercept={platt_intercept:.3f})")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "platt_best_tau": best_tau,
        "platt_rare_f1": platt_m["rare_f1"],
        "platt_recall": platt_m["rare_recall"],
        "platt_precision": platt_m["rare_precision"],
        "platt_intercept": platt_intercept,
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        try:
            result = run_one(run_dir, rare_class)
            if result:
                rows.append(result)
        except Exception as exc:
            import traceback
            print(f"  ERROR in {run_dir.name}: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E20 Results: Platt Calibration ===")
    cols = ["run", "rare_class", "n_rare_train",
            "scanvi_rare_f1", "euclidean_rare_f1",
            "platt_best_tau", "platt_rare_f1", "platt_recall", "platt_precision"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
