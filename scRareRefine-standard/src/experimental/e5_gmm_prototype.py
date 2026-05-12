"""E5: Gaussian Mixture Model prototype.

Current prototype = single centroid. Innovation: model each class as a GMM
(n_components=2 for rare, auto for majority).
For rare class with n<10, use n_components=1 (falls back to single Gaussian).

Distance = negative log-likelihood under the class GMM.

Compare vs euclidean and mahalanobis-pooled.

Run on: cDC1 rare5, cDC1 rare20, ASDC rare5, epsilon rare20 (seed42).

Usage:
    python src/experimental/e5_gmm_prototype.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent,
    _class_prototypes,
    _euclidean,
    _pooled_covariance_shrunk,
    _mahalanobis_with_posterior_penalty,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e5_gmm_prototype"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20", "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
]


def fit_class_gmms(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    is_labeled: np.ndarray,
    classes: list[str],
    rare_class: str,
    *,
    n_components_rare: int = 2,
    n_components_majority: int = 2,
    min_cells_for_multi: int = 10,
) -> dict[str, GaussianMixture]:
    """Fit a GMM per class. Falls back to 1 component if too few cells."""
    gmms = {}
    for c in classes:
        mask = is_labeled & (train_labels == c)
        X_c = train_z[mask]
        n_c = X_c.shape[0]

        if c == rare_class:
            n_comp = n_components_rare if n_c >= min_cells_for_multi else 1
        else:
            n_comp = n_components_majority if n_c >= min_cells_for_multi else 1

        n_comp = min(n_comp, n_c)  # can't have more components than samples
        try:
            gmm = GaussianMixture(
                n_components=n_comp,
                covariance_type="full",
                reg_covar=1e-4,
                random_state=42,
                max_iter=200,
            )
            gmm.fit(X_c)
        except Exception:
            # Fallback: diagonal covariance
            gmm = GaussianMixture(
                n_components=1,
                covariance_type="diag",
                reg_covar=1e-4,
                random_state=42,
            )
            gmm.fit(X_c)
        gmms[c] = gmm
    return gmms


def gmm_distances(
    query: np.ndarray,
    classes: list[str],
    gmms: dict[str, GaussianMixture],
) -> np.ndarray:
    """Distance = negative log-likelihood under each class GMM.
    
    We normalize per-class by subtracting the mean log-likelihood of training
    data under that class's GMM, so distances are comparable across classes.
    """
    n_q = query.shape[0]
    n_c = len(classes)
    dists = np.zeros((n_q, n_c))
    for i, c in enumerate(classes):
        # score_samples returns log-likelihood; negate for distance
        log_lik = gmms[c].score_samples(query)
        dists[:, i] = -log_lik
    # Normalize each class column to zero mean, unit std across test cells
    # This makes distances comparable across classes
    col_means = dists.mean(axis=0, keepdims=True)
    col_stds  = dists.std(axis=0, keepdims=True)
    col_stds[col_stds < 1e-10] = 1.0
    dists = (dists - col_means) / col_stds
    return dists


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
    train_labels = train_pred["true_label"].astype(str).to_numpy()

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    counts = [counts_map[c] for c in classes]

    # Euclidean baseline
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled + posterior
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mpp_dists = _mahalanobis_with_posterior_penalty(test_z, protos, pooled_covs, counts)
    mpp_pred  = _predict_nearest(mpp_dists, classes)
    mpp_m, _  = classification_tables(y_test, pd.Series(mpp_pred), rare_class=rare_class)

    # GMM prototype
    print(f"    Fitting GMMs ...")
    gmms = fit_class_gmms(train_z, train_labels, is_labeled, classes, rare_class)
    gmm_dists = gmm_distances(test_z, classes, gmms)
    gmm_pred  = _predict_nearest(gmm_dists, classes)
    gmm_m, _  = classification_tables(y_test, pd.Series(gmm_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"    scANVI:       rare_f1={scanvi_m['rare_f1']:.3f}")
    print(f"    Euclidean:    rare_f1={euc_m['rare_f1']:.3f}")
    print(f"    Mahal-pool+p: rare_f1={mpp_m['rare_f1']:.3f}")
    print(f"    GMM:          rare_f1={gmm_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "test_rare_f1_scanvi": scanvi_m["rare_f1"],
        "test_rare_f1_euclidean": euc_m["rare_f1"],
        "test_rare_f1_mahal_pool_post": mpp_m["rare_f1"],
        "test_rare_f1_gmm": gmm_m["rare_f1"],
        "test_overall_acc_euclidean": euc_m["overall_accuracy"],
        "test_overall_acc_gmm": gmm_m["overall_accuracy"],
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

    print("\n=== E5 Results (GMM prototype, seed42) ===")
    cols = ["run", "rare_class", "n_rare_train",
            "test_rare_f1_scanvi", "test_rare_f1_euclidean",
            "test_rare_f1_mahal_pool_post", "test_rare_f1_gmm"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
