"""E21: Multi-scale prototype (hierarchical distance).

Algorithm:
    multiscale_score(cell_i, class_c) = w1 * d_centroid(i,c) + w2 * d_1nn(i,c) + w3 * d_5nn(i,c)

where:
  d_centroid = Euclidean distance to class centroid (current method)
  d_1nn = distance to nearest within-class training cell
  d_5nn = distance to 5th nearest within-class training cell

w1, w2, w3 tuned on validation (grid search over simplex).

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
    _euclidean,
    _pooled_covariance_shrunk,
    _mahalanobis,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e21_multiscale_prototype"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",                          "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",                          "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                        "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20",   "non-classical monocyte"),
]

# Grid search over simplex: w1 + w2 + w3 = 1
W_GRID_STEP = 0.2


def _simplex_grid(step: float = 0.2) -> list[tuple[float, float, float]]:
    """Generate grid points on the 3D simplex (w1+w2+w3=1)."""
    points = []
    vals = np.arange(0.0, 1.0 + step / 2, step)
    for w1 in vals:
        for w2 in vals:
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-9:
                w3 = max(w3, 0.0)
                points.append((round(w1, 4), round(w2, 4), round(w3, 4)))
    return points


def _multiscale_distances(
    query_z: np.ndarray,
    classes: list[str],
    labeled_z: np.ndarray,
    labeled_labels: np.ndarray,
    protos: np.ndarray,
    k_nn: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute 3 distance scales for each (query, class) pair.

    Returns:
        d_centroid: (n_query, n_classes) Euclidean to centroid
        d_1nn: (n_query, n_classes) distance to nearest within-class cell
        d_knn: (n_query, n_classes) distance to k-th nearest within-class cell
    """
    n_q = len(query_z)
    n_c = len(classes)
    d_centroid = _euclidean(query_z, protos)
    d_1nn  = np.zeros((n_q, n_c))
    d_knn  = np.zeros((n_q, n_c))

    for ci, cls in enumerate(classes):
        cls_mask = labeled_labels == cls
        cls_cells = labeled_z[cls_mask]
        if len(cls_cells) == 0:
            d_1nn[:, ci] = 1e6
            d_knn[:, ci] = 1e6
            continue

        k_actual = min(k_nn, len(cls_cells))
        nbrs = NearestNeighbors(n_neighbors=k_actual, metric="euclidean")
        nbrs.fit(cls_cells)
        dists, _ = nbrs.kneighbors(query_z)

        d_1nn[:, ci] = dists[:, 0]
        d_knn[:, ci] = dists[:, -1]  # k-th nearest

    return d_centroid, d_1nn, d_knn


def _multiscale_predict(
    d_centroid: np.ndarray,
    d_1nn: np.ndarray,
    d_knn: np.ndarray,
    classes: list[str],
    w1: float,
    w2: float,
    w3: float,
) -> np.ndarray:
    combined = w1 * d_centroid + w2 * d_1nn + w3 * d_knn
    return _predict_nearest(combined, classes)


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

    # Load validation
    val_pred = None
    val_lat  = None
    try:
        val_pred = read_table(emb_dir / "validation_predictions.csv")
        val_lat  = read_table(emb_dir / "validation_latent.csv")
    except FileNotFoundError:
        # Use labeled training as validation proxy
        is_labeled_mask = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
        val_pred = train_pred[is_labeled_mask].reset_index(drop=True)
        val_lat  = train_lat[is_labeled_mask].reset_index(drop=True)

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    val_z   = _latent(val_lat)
    y_test  = test_pred["true_label"].astype(str)
    y_val   = val_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    # ── Baseline: Euclidean centroid ───────────────────────────────────────
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # ── Baseline: Mahal-pooled ─────────────────────────────────────────────
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # ── Compute multi-scale distances ─────────────────────────────────────
    print(f"  Computing multi-scale distances for {run_dir.name}...")
    val_d_c, val_d_1nn, val_d_knn = _multiscale_distances(
        val_z, classes, labeled_z, labeled_labels, protos
    )
    test_d_c, test_d_1nn, test_d_knn = _multiscale_distances(
        test_z, classes, labeled_z, labeled_labels, protos
    )

    # ── Grid search on validation ──────────────────────────────────────────
    simplex = _simplex_grid(W_GRID_STEP)
    best_w = (1.0, 0.0, 0.0)
    best_val_f1 = -1.0
    grid_rows = []

    for w1, w2, w3 in simplex:
        val_pred_ms = _multiscale_predict(val_d_c, val_d_1nn, val_d_knn, classes, w1, w2, w3)
        val_m, _ = classification_tables(y_val, pd.Series(val_pred_ms), rare_class=rare_class)
        grid_rows.append({"w1": w1, "w2": w2, "w3": w3, "val_rare_f1": val_m["rare_f1"]})
        if val_m["rare_f1"] > best_val_f1:
            best_val_f1 = val_m["rare_f1"]
            best_w = (w1, w2, w3)

    grid_df = pd.DataFrame(grid_rows).sort_values("val_rare_f1", ascending=False)
    write_table(grid_df, OUT_DIR / f"{run_dir.name}_weight_grid.csv")

    # ── Apply best weights to test ─────────────────────────────────────────
    w1, w2, w3 = best_w
    ms_pred = _multiscale_predict(test_d_c, test_d_1nn, test_d_knn, classes, w1, w2, w3)
    ms_m, _ = classification_tables(y_test, pd.Series(ms_pred), rare_class=rare_class)

    # ── scANVI baseline ────────────────────────────────────────────────────
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: best_w=({w1:.1f},{w2:.1f},{w3:.1f})  "
          f"scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  multiscale={ms_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "best_w1": w1,
        "best_w2": w2,
        "best_w3": w3,
        "best_val_f1": best_val_f1,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "multiscale_rare_f1": ms_m["rare_f1"],
        "multiscale_recall": ms_m["rare_recall"],
        "multiscale_precision": ms_m["rare_precision"],
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

    print("\n=== E21 Results: Multi-scale Prototype ===")
    cols = ["run", "rare_class", "n_rare_train",
            "best_w1", "best_w2", "best_w3",
            "scanvi_rare_f1", "euclidean_rare_f1",
            "mahal_pooled_rare_f1", "multiscale_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
