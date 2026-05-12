"""E11: Density ratio calibration for GMM (fixing E5).

Fix: normalize each class's log-likelihood by the expected log-likelihood
of its own training data:

    calibrated_score(i, c) = log p(z_i | GMM_c) - E[log p(z | GMM_c)]

where E[log p(z | GMM_c)] is estimated from the training cells of class c.

Compare: euclidean, mahal-pooled, GMM (uncalibrated), GMM (calibrated).

Run on: cDC1 rare5, cDC1 rare20, ASDC rare5, epsilon rare20 (seed42).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e11_gmm_calibrated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
]


def _fit_gmm(X: np.ndarray, n_components: int) -> GaussianMixture:
    """Fit GMM with fallback to fewer components if needed."""
    n_components = min(n_components, max(1, len(X) // 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-4,
            max_iter=200,
            n_init=3,
            random_state=42,
        )
        try:
            gmm.fit(X)
        except Exception:
            # Fallback to diagonal
            gmm = GaussianMixture(
                n_components=1,
                covariance_type="diag",
                reg_covar=1e-4,
                max_iter=200,
                random_state=42,
            )
            gmm.fit(X)
    return gmm


def _gmm_log_likelihoods(
    query_z: np.ndarray,
    gmms: list[GaussianMixture],
    classes: list[str],
    baselines: list[float],
    calibrate: bool,
) -> np.ndarray:
    """Return (n_query, n_classes) matrix of (calibrated) log-likelihoods."""
    n_q = len(query_z)
    n_c = len(classes)
    ll = np.zeros((n_q, n_c))
    for c, gmm in enumerate(gmms):
        ll[:, c] = gmm.score_samples(query_z)
        if calibrate:
            ll[:, c] -= baselines[c]
    return ll


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

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # Fit GMMs per class
    print(f"  Fitting GMMs for {run_dir.name}...")
    gmms = []
    baselines = []
    for c in classes:
        mask = labeled_labels == c
        X_c = labeled_z[mask]
        n_c = len(X_c)
        n_comp = 2 if n_c >= 10 else 1
        gmm = _fit_gmm(X_c, n_comp)
        gmms.append(gmm)
        # Baseline: mean log-likelihood on training cells of this class
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            baseline = float(gmm.score_samples(X_c).mean())
        baselines.append(baseline)

    # GMM uncalibrated: predict class with highest log-likelihood
    ll_uncal = _gmm_log_likelihoods(test_z, gmms, classes, baselines, calibrate=False)
    gmm_uncal_pred = np.array(classes)[ll_uncal.argmax(axis=1)]
    gmm_uncal_m, _ = classification_tables(y_test, pd.Series(gmm_uncal_pred), rare_class=rare_class)

    # GMM calibrated
    ll_cal = _gmm_log_likelihoods(test_z, gmms, classes, baselines, calibrate=True)
    gmm_cal_pred = np.array(classes)[ll_cal.argmax(axis=1)]
    gmm_cal_m, _ = classification_tables(y_test, pd.Series(gmm_cal_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  gmm_uncal={gmm_uncal_m['rare_f1']:.3f}  "
          f"gmm_cal={gmm_cal_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "gmm_uncalibrated_rare_f1": gmm_uncal_m["rare_f1"],
        "gmm_calibrated_rare_f1": gmm_cal_m["rare_f1"],
        "gmm_calibrated_recall": gmm_cal_m["rare_recall"],
        "gmm_calibrated_precision": gmm_cal_m["rare_precision"],
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
            print(f"  ERROR in {run_dir.name}: {exc}")

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E11 Results: Calibrated GMM ===")
    cols = ["run", "rare_class", "n_rare_train", "scanvi_rare_f1",
            "euclidean_rare_f1", "mahal_pooled_rare_f1",
            "gmm_uncalibrated_rare_f1", "gmm_calibrated_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
