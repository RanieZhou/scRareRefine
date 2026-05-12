"""E9: Ensemble — Mahal-pooled + Class-balanced kNN voting.

Algorithm:
    ensemble_score(i, rare) = α * mahal_score(i) + (1-α) * cb_knn_score(i)

where:
    mahal_score(i)   = 1 - d_rare / sum_all_distances  (normalized inverse Mahal)
    cb_knn_score(i)  = fraction of CB-kNN votes for rare_class (k=30, class-balanced)
    α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} tuned on validation

A cell is rescued if ensemble_score > 0.5.

Run on: cDC1 rare5, ASDC rare5, epsilon rare20, NCM rare20 (seed42).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent,
    _class_prototypes,
    _pooled_covariance_shrunk,
    _mahalanobis,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e9_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
K_KNN = 30

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",                         "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",                         "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                      "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
]


def _cb_knn_scores(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    query_z: np.ndarray,
    rare_class: str,
    k: int = 30,
) -> np.ndarray:
    """Class-balanced kNN score = fraction of CB-weighted votes for rare_class."""
    classes, counts = np.unique(train_labels, return_counts=True)
    class_weight = {c: 1.0 / max(cnt, 1) for c, cnt in zip(classes, counts)}

    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_z)), metric="euclidean")
    nbrs.fit(train_z)
    distances, indices = nbrs.kneighbors(query_z)

    scores = np.zeros(len(query_z))
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        total_weight = 0.0
        rare_weight = 0.0
        for j, idx in enumerate(idxs):
            lbl = train_labels[idx]
            d = max(dists[j], 1e-10)
            w = class_weight.get(lbl, 1.0) / (d * d)
            total_weight += w
            if lbl == rare_class:
                rare_weight += w
        scores[i] = rare_weight / max(total_weight, 1e-10)
    return scores


def _mahal_scores(dists: np.ndarray, rare_idx: int) -> np.ndarray:
    """Normalized inverse Mahal: 1 - d_rare / sum_all_distances."""
    d_rare = dists[:, rare_idx]
    d_sum = dists.sum(axis=1)
    d_sum_safe = np.where(d_sum < 1e-10, 1e-10, d_sum)
    return 1.0 - d_rare / d_sum_safe


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
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    rare_idx = classes.index(rare_class)

    # Pooled Mahal distances
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)

    val_mahal_dists  = _mahalanobis(val_z, protos, pooled_covs)
    test_mahal_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Mahal scores (normalized inverse)
    val_mahal_scores  = _mahal_scores(val_mahal_dists, rare_idx)
    test_mahal_scores = _mahal_scores(test_mahal_dists, rare_idx)

    # CB-kNN scores
    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    print(f"  Computing CB-kNN scores for {run_dir.name}...")
    val_knn_scores  = _cb_knn_scores(labeled_z, labeled_labels, val_z,  rare_class, K_KNN)
    test_knn_scores = _cb_knn_scores(labeled_z, labeled_labels, test_z, rare_class, K_KNN)

    # scANVI baseline
    scanvi_test_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Pure Mahal-pooled (α=1.0)
    mahal_pred = _predict_nearest(test_mahal_dists, classes)
    mahal_test_m, _ = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # Alpha grid search on validation
    alpha_rows = []
    best_alpha = ALPHA_GRID[0]
    best_val_f1 = -1.0

    for alpha in ALPHA_GRID:
        ensemble_val = alpha * val_mahal_scores + (1 - alpha) * val_knn_scores
        # Rescue: ensemble_score > 0.5 → predict rare_class, else scANVI
        val_pred_arr = val_pred["predicted_label"].to_numpy().astype(str)
        rescued = ensemble_val > 0.5
        pred = val_pred_arr.copy()
        pred[rescued] = rare_class
        m, _ = classification_tables(y_val, pd.Series(pred), rare_class=rare_class)
        alpha_rows.append({
            "run": run_dir.name,
            "rare_class": rare_class,
            "alpha": alpha,
            "val_rare_f1": m["rare_f1"],
        })
        if m["rare_f1"] > best_val_f1:
            best_val_f1 = m["rare_f1"]
            best_alpha = alpha

    # Apply best alpha to test
    ensemble_test = best_alpha * test_mahal_scores + (1 - best_alpha) * test_knn_scores
    test_pred_arr = test_pred["predicted_label"].to_numpy().astype(str)
    rescued_test = ensemble_test > 0.5
    final_pred = test_pred_arr.copy()
    final_pred[rescued_test] = rare_class
    ensemble_test_m, _ = classification_tables(y_test, pd.Series(final_pred), rare_class=rare_class)

    # Pure CB-kNN (α=0.0)
    knn_rescued = test_knn_scores > 0.5
    knn_pred = test_pred_arr.copy()
    knn_pred[knn_rescued] = rare_class
    knn_test_m, _ = classification_tables(y_test, pd.Series(knn_pred), rare_class=rare_class)

    print(f"  {run_dir.name}: best_alpha={best_alpha:.1f}  val_f1={best_val_f1:.3f}  "
          f"ensemble={ensemble_test_m['rare_f1']:.3f}  mahal={mahal_test_m['rare_f1']:.3f}  "
          f"knn={knn_test_m['rare_f1']:.3f}  scanvi={scanvi_test_m['rare_f1']:.3f}")

    alpha_df = pd.DataFrame(alpha_rows)
    alpha_df.to_csv(OUT_DIR / f"{run_dir.name}_alpha_curve.csv", index=False)

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "best_alpha": best_alpha,
        "val_f1_at_best_alpha": best_val_f1,
        "scanvi_test_rare_f1": scanvi_test_m["rare_f1"],
        "mahal_pooled_test_rare_f1": mahal_test_m["rare_f1"],
        "cb_knn_test_rare_f1": knn_test_m["rare_f1"],
        "ensemble_test_rare_f1": ensemble_test_m["rare_f1"],
        "ensemble_test_recall": ensemble_test_m["rare_recall"],
        "ensemble_test_precision": ensemble_test_m["rare_precision"],
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

    print("\n=== E9 Results: Ensemble ===")
    cols = ["run", "rare_class", "best_alpha", "scanvi_test_rare_f1",
            "mahal_pooled_test_rare_f1", "cb_knn_test_rare_f1", "ensemble_test_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
