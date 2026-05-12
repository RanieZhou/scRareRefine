"""E10: Separability-adaptive method selection.

Algorithm:
    Compute separability ratio S on training data:
        S = d_inter_prototype / d_intra_rare
    where d_inter = min Euclidean distance from rare prototype to any majority prototype,
          d_intra = mean pairwise distance among rare training cells.

    Then:
        S >= 1.3  → Euclidean nearest-prototype
        1.0 <= S < 1.3 → Mahal-pooled (λ=0)
        S < 1.0  → Class-balanced kNN (k=30)

Run on ALL available datasets/runs (seed42, rts=20 where available).
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e10_adaptive_selector"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_KNN = 30

# All seed42 runs with rts=20 (or best available)
RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",                         "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",                         "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                       "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",                         "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20",  "non-classical monocyte"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare20",       "endothelial cell"),
    ("outputs/tabula_pancreas/cell_stratified_seed42_type_b_pancreatic_cell_rare20", "type B pancreatic cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",     "innate lymphoid cell"),
]


def _separability_ratio(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    rare_class: str,
    classes: list[str],
    protos: np.ndarray,
) -> float:
    """Compute separability ratio S = d_inter / d_intra."""
    rare_idx = classes.index(rare_class)
    rare_mask = train_labels == rare_class
    rare_cells = train_z[rare_mask]

    # d_intra: mean pairwise distance among rare training cells
    if len(rare_cells) < 2:
        d_intra = 1e-6
    else:
        diffs = rare_cells[:, None, :] - rare_cells[None, :, :]
        pairwise = np.sqrt((diffs * diffs).sum(axis=2))
        # upper triangle only
        n = len(rare_cells)
        idx = np.triu_indices(n, k=1)
        d_intra = float(pairwise[idx].mean()) if len(idx[0]) > 0 else 1e-6

    # d_inter: min Euclidean distance from rare prototype to any majority prototype
    rare_proto = protos[rare_idx]
    majority_protos = np.delete(protos, rare_idx, axis=0)
    diffs_inter = majority_protos - rare_proto[None, :]
    d_inter = float(np.sqrt((diffs_inter * diffs_inter).sum(axis=1)).min())

    return d_inter / max(d_intra, 1e-10)


def _cb_knn_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    query_z: np.ndarray,
    rare_class: str,
    classes: list[str],
    k: int = 30,
) -> np.ndarray:
    """Class-balanced kNN: predict rare_class if CB-weighted score > 0.5, else nearest majority."""
    class_counts = {c: int((train_labels == c).sum()) for c in classes}
    class_weight = {c: 1.0 / max(cnt, 1) for c, cnt in class_counts.items()}

    nbrs = NearestNeighbors(n_neighbors=min(k, len(train_z)), metric="euclidean")
    nbrs.fit(train_z)
    distances, indices = nbrs.kneighbors(query_z)

    preds = []
    for dists, idxs in zip(distances, indices):
        vote_weights: dict[str, float] = {c: 0.0 for c in classes}
        for j, idx in enumerate(idxs):
            lbl = train_labels[idx]
            d = max(dists[j], 1e-10)
            w = class_weight.get(lbl, 1.0) / (d * d)
            vote_weights[lbl] = vote_weights.get(lbl, 0.0) + w
        preds.append(max(vote_weights, key=vote_weights.get))
    return np.array(preds)


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

    # Separability ratio
    S = _separability_ratio(labeled_z, labeled_labels, rare_class, classes, protos)

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

    # CB-kNN
    print(f"  Computing CB-kNN for {run_dir.name} (S={S:.3f})...")
    knn_pred = _cb_knn_predict(labeled_z, labeled_labels, test_z, rare_class, classes, K_KNN)
    knn_m, _ = classification_tables(y_test, pd.Series(knn_pred), rare_class=rare_class)

    # Adaptive selector
    if S >= 1.3:
        selected_method = "euclidean"
        adaptive_pred = euc_pred
    elif S >= 1.0:
        selected_method = "mahal_pooled"
        adaptive_pred = mahal_pred
    else:
        selected_method = "cb_knn"
        adaptive_pred = knn_pred

    adaptive_m, _ = classification_tables(y_test, pd.Series(adaptive_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: S={S:.3f}  selected={selected_method}  "
          f"adaptive={adaptive_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  knn={knn_m['rare_f1']:.3f}  "
          f"scanvi={scanvi_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "separability_ratio": S,
        "selected_method": selected_method,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "cb_knn_rare_f1": knn_m["rare_f1"],
        "adaptive_rare_f1": adaptive_m["rare_f1"],
        "adaptive_recall": adaptive_m["rare_recall"],
        "adaptive_precision": adaptive_m["rare_precision"],
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

    print("\n=== E10 Results: Adaptive Selector ===")
    cols = ["run", "rare_class", "separability_ratio", "selected_method",
            "scanvi_rare_f1", "euclidean_rare_f1", "mahal_pooled_rare_f1",
            "cb_knn_rare_f1", "adaptive_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
