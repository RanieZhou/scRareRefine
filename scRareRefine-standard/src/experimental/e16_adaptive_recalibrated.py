"""E16: Recalibrated adaptive selector (S thresholds: 1.2 / 0.8).

Fix the E10 failure on epsilon by adjusting thresholds:
  S >= 1.2  → Euclidean nearest-prototype
  0.8 <= S < 1.2 → Mahal-pooled (λ=0)
  S < 0.8  → SMOTE-LR (fallback for very low sep)

Run on ALL 8 datasets (seed42, rts=20).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e16_adaptive_recalibrated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_KNN = 30

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",                           "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",                           "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                         "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",                           "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20",    "non-classical monocyte"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare20",         "endothelial cell"),
    ("outputs/tabula_pancreas/cell_stratified_seed42_type_b_pancreatic_cell_rare20", "type B pancreatic cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",       "innate lymphoid cell"),
]

# Old E10 thresholds for comparison
OLD_HIGH = 1.3
OLD_LOW  = 1.0
# New recalibrated thresholds
NEW_HIGH = 1.2
NEW_LOW  = 0.8


def _separability_ratio(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    rare_class: str,
    classes: list[str],
    protos: np.ndarray,
) -> float:
    rare_idx = classes.index(rare_class)
    rare_mask = train_labels == rare_class
    rare_cells = train_z[rare_mask]

    if len(rare_cells) < 2:
        d_intra = 1e-6
    else:
        diffs = rare_cells[:, None, :] - rare_cells[None, :, :]
        pairwise = np.sqrt((diffs * diffs).sum(axis=2))
        n = len(rare_cells)
        idx = np.triu_indices(n, k=1)
        d_intra = float(pairwise[idx].mean()) if len(idx[0]) > 0 else 1e-6

    rare_proto = protos[rare_idx]
    majority_protos = np.delete(protos, rare_idx, axis=0)
    diffs_inter = majority_protos - rare_proto[None, :]
    d_inter = float(np.sqrt((diffs_inter * diffs_inter).sum(axis=1)).min())

    return d_inter / max(d_intra, 1e-10)


def _smote_lr_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    query_z: np.ndarray,
    rare_class: str,
    classes: list[str],
) -> np.ndarray:
    """SMOTE-LR: oversample rare class, train LR, predict."""
    le = LabelEncoder()
    le.fit(train_labels)
    y_enc = le.transform(train_labels)

    try:
        from collections import Counter
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        min_class_count = min(Counter(y_enc).values())
        k_neighbors = min(5, min_class_count - 2)
        if k_neighbors < 1:
            sampler = RandomOverSampler(random_state=42)
        else:
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = sampler.fit_resample(train_z, y_enc)
    except Exception:
        X_res, y_res = train_z, y_enc

    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_res, y_res)
    return le.inverse_transform(lr.predict(query_z))


def _old_adaptive_select(S: float, euc_pred, mahal_pred, knn_pred):
    if S >= OLD_HIGH:
        return "euclidean", euc_pred
    elif S >= OLD_LOW:
        return "mahal_pooled", mahal_pred
    else:
        return "cb_knn", knn_pred


def _new_adaptive_select(S: float, euc_pred, mahal_pred, smote_pred):
    if S >= NEW_HIGH:
        return "euclidean", euc_pred
    elif S >= NEW_LOW:
        return "mahal_pooled", mahal_pred
    else:
        return "smote_lr", smote_pred


def _cb_knn_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    query_z: np.ndarray,
    rare_class: str,
    classes: list[str],
    k: int = 30,
) -> np.ndarray:
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

    # CB-kNN (for old adaptive comparison)
    print(f"  Computing CB-kNN for {run_dir.name} (S={S:.3f})...")
    knn_pred = _cb_knn_predict(labeled_z, labeled_labels, test_z, rare_class, classes, K_KNN)
    knn_m, _ = classification_tables(y_test, pd.Series(knn_pred), rare_class=rare_class)

    # SMOTE-LR (for new adaptive)
    print(f"  Computing SMOTE-LR for {run_dir.name}...")
    smote_pred = _smote_lr_predict(labeled_z, labeled_labels, test_z, rare_class, classes)
    smote_m, _ = classification_tables(y_test, pd.Series(smote_pred), rare_class=rare_class)

    # Old adaptive (E10 thresholds)
    old_method, old_pred = _old_adaptive_select(S, euc_pred, mahal_pred, knn_pred)
    old_m, _ = classification_tables(y_test, pd.Series(old_pred), rare_class=rare_class)

    # New adaptive (E16 thresholds)
    new_method, new_pred = _new_adaptive_select(S, euc_pred, mahal_pred, smote_pred)
    new_m, _ = classification_tables(y_test, pd.Series(new_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: S={S:.3f}  old={old_method}({old_m['rare_f1']:.3f})  "
          f"new={new_method}({new_m['rare_f1']:.3f})  "
          f"euc={euc_m['rare_f1']:.3f}  mahal={mahal_m['rare_f1']:.3f}  "
          f"smote={smote_m['rare_f1']:.3f}  scanvi={scanvi_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "separability_ratio": S,
        "old_selected_method": old_method,
        "new_selected_method": new_method,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "cb_knn_rare_f1": knn_m["rare_f1"],
        "smote_lr_rare_f1": smote_m["rare_f1"],
        "old_adaptive_rare_f1": old_m["rare_f1"],
        "new_adaptive_rare_f1": new_m["rare_f1"],
        "new_adaptive_recall": new_m["rare_recall"],
        "new_adaptive_precision": new_m["rare_precision"],
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

    print("\n=== E16 Results: Recalibrated Adaptive Selector ===")
    cols = ["run", "rare_class", "separability_ratio",
            "old_selected_method", "old_adaptive_rare_f1",
            "new_selected_method", "new_adaptive_rare_f1",
            "scanvi_rare_f1", "euclidean_rare_f1", "mahal_pooled_rare_f1", "smote_lr_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Summary: how often does new adaptive match or beat best individual method?
    if len(df) > 0:
        df["best_individual"] = df[["euclidean_rare_f1", "mahal_pooled_rare_f1", "smote_lr_rare_f1"]].max(axis=1)
        df["new_vs_best"] = df["new_adaptive_rare_f1"] - df["best_individual"]
        df["old_vs_best"] = df["old_adaptive_rare_f1"] - df["best_individual"]
        print(f"\nNew adaptive vs best individual: mean Δ = {df['new_vs_best'].mean():.3f}")
        print(f"Old adaptive vs best individual: mean Δ = {df['old_vs_best'].mean():.3f}")
        write_table(df, OUT_DIR / "results_with_delta.csv")

    return df


if __name__ == "__main__":
    main()
