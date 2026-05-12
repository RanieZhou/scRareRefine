"""E13: Density-ratio normalized GMM (fix for E5).

E5 showed GMM fails due to calibration issues. Fix: normalize log-likelihoods
using density ratio estimation.

For each class c, compute:
  score_c(z) = log p_c(z) - log p_background(z)

where p_background is a GMM fitted on ALL labeled training cells (regardless of class).

This is the density ratio trick: it measures how much more likely z is under
class c than under the background distribution.

Run on: cDC1 rare5/20, ASDC rare5/20, epsilon rare20 (seed42).

Usage:
    python src/experimental/e13_gmm_density_ratio.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e13_gmm_density_ratio"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc",    "cDC1",   "batch_heldout",   "cdc1",    [5, 20]),
    ("outputs/immune_dc",    "ASDC",   "batch_heldout",   "asdc",    [5, 20]),
    ("outputs/pancreas",     "epsilon","batch_heldout",   "epsilon", [20]),
    ("outputs/pancreas",     "gamma",  "batch_heldout",   "gamma",   [5, 20]),
]
SEED = 42


def fit_gmm_per_class(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    is_labeled: np.ndarray,
    classes: list[str],
    n_components_rare: int = 1,
    n_components_majority: int = 2,
    rare_class: str = "",
    min_cells: int = 3,
) -> dict[str, GaussianMixture]:
    """Fit a GMM per class."""
    gmms = {}
    for c in classes:
        mask = is_labeled & (train_labels == c)
        n_c = int(mask.sum())
        if n_c < min_cells:
            continue
        X_c = train_z[mask]
        n_comp = n_components_rare if c == rare_class else n_components_majority
        n_comp = min(n_comp, n_c)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(
                    n_components=n_comp,
                    covariance_type="full",
                    reg_covar=1e-4,
                    random_state=42,
                    max_iter=200,
                )
                gmm.fit(X_c)
            gmms[c] = gmm
        except Exception as e:
            print(f"    GMM fit failed for class {c}: {e}")
    return gmms


def gmm_density_ratio_predict(
    test_z: np.ndarray,
    gmms: dict[str, GaussianMixture],
    background_gmm: GaussianMixture,
    classes: list[str],
) -> np.ndarray:
    """Predict using density ratio: score_c = log p_c(z) - log p_background(z)."""
    n = test_z.shape[0]
    n_c = len(classes)
    scores = np.full((n, n_c), -np.inf)

    log_bg = background_gmm.score_samples(test_z)  # shape (n,)

    for i, c in enumerate(classes):
        if c not in gmms:
            continue
        log_pc = gmms[c].score_samples(test_z)  # shape (n,)
        scores[:, i] = log_pc - log_bg

    # Predict class with highest density ratio score
    return np.array(classes)[scores.argmax(axis=1)]


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
        "method": "scANVI baseline",
        "rare_f1": scanvi_m["rare_f1"],
        "rare_recall": scanvi_m["rare_recall"],
        "rare_precision": scanvi_m["rare_precision"],
    })

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Euclidean nearest-proto",
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
        "method": "Mahal-pooled",
        "rare_f1": mah_m["rare_f1"],
        "rare_recall": mah_m["rare_recall"],
        "rare_precision": mah_m["rare_precision"],
    })

    # GMM density ratio
    print(f"    Fitting per-class GMMs...")
    gmms = fit_gmm_per_class(
        train_z, train_labels, is_labeled, classes,
        n_components_rare=1,
        n_components_majority=2,
        rare_class=rare_class,
    )

    # Background GMM on all labeled cells
    print(f"    Fitting background GMM...")
    labeled_z = train_z[is_labeled]
    n_bg_comp = min(5, len(labeled_z) // 10)
    n_bg_comp = max(1, n_bg_comp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bg_gmm = GaussianMixture(
                n_components=n_bg_comp,
                covariance_type="full",
                reg_covar=1e-4,
                random_state=42,
                max_iter=200,
            )
            bg_gmm.fit(labeled_z)
    except Exception as e:
        print(f"    Background GMM failed: {e}")
        return rows

    if len(gmms) < len(classes):
        print(f"    WARNING: Only {len(gmms)}/{len(classes)} class GMMs fitted")

    # Predict using density ratio
    gmm_pred = gmm_density_ratio_predict(test_z, gmms, bg_gmm, classes)
    gmm_m, _ = classification_tables(y_true, pd.Series(gmm_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "GMM density ratio",
        "rare_f1": gmm_m["rare_f1"],
        "rare_recall": gmm_m["rare_recall"],
        "rare_precision": gmm_m["rare_precision"],
    })

    # Also try: GMM density ratio with rare class n_components=2 (if enough cells)
    if counts_map.get(rare_class, 0) >= 10:
        gmms2 = fit_gmm_per_class(
            train_z, train_labels, is_labeled, classes,
            n_components_rare=2,
            n_components_majority=2,
            rare_class=rare_class,
        )
        gmm2_pred = gmm_density_ratio_predict(test_z, gmms2, bg_gmm, classes)
        gmm2_m, _ = classification_tables(y_true, pd.Series(gmm2_pred), rare_class=rare_class)
        rows.append({
            "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
            "method": "GMM density ratio (2-comp rare)",
            "rare_f1": gmm2_m["rare_f1"],
            "rare_recall": gmm2_m["rare_recall"],
            "rare_precision": gmm2_m["rare_precision"],
        })

    print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
          f"GMM-DR={gmm_m['rare_f1']:.3f}")

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

    print("\n\n=== E13 Summary: GMM density ratio vs baselines ===")
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
