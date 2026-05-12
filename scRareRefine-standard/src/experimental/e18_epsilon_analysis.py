"""E18: Epsilon deep-dive — why does CB-kNN fail?

Investigate:
1. Confusion matrix analysis: What does each method predict for epsilon cells?
2. Prototype distance analysis: How far is epsilon prototype from nearest majority?
3. Neighbor analysis: For epsilon test cells, what fraction are epsilon vs majority?
4. Why SMOTE-LR works: Visualize decision boundary in 2D PCA

Run on: epsilon rare20, seed42.
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
from sklearn.decomposition import PCA

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e18_epsilon_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = ROOT / "outputs" / "pancreas" / "batch_heldout_seed42_epsilon_rare20"
RARE_CLASS = "epsilon"
K_KNN = 30


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


def _smote_lr_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    query_z: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression, LabelEncoder]:
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
    return le.inverse_transform(lr.predict(query_z)), lr, le


def main() -> None:
    emb_dir = RUN_DIR / "embeddings"
    train_pred = read_table(emb_dir / "train_predictions.csv")
    train_lat  = read_table(emb_dir / "train_latent.csv")
    test_pred  = read_table(emb_dir / "test_predictions.csv")
    test_lat   = read_table(emb_dir / "test_latent.csv")

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    print(f"Classes: {classes}")
    print(f"Counts: {counts_map}")
    print(f"Test epsilon cells: {(y_test == RARE_CLASS).sum()}")

    # ── 1. Compute all method predictions ─────────────────────────────────
    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)

    # CB-kNN
    print("Computing CB-kNN...")
    knn_pred = _cb_knn_predict(labeled_z, labeled_labels, test_z, RARE_CLASS, classes, K_KNN)

    # SMOTE-LR
    print("Computing SMOTE-LR...")
    smote_pred, smote_lr, smote_le = _smote_lr_predict(labeled_z, labeled_labels, test_z)

    # scANVI
    scanvi_pred = test_pred["predicted_label"].astype(str).to_numpy()

    # ── 2. Confusion matrix analysis ──────────────────────────────────────
    methods = {
        "scANVI": scanvi_pred,
        "Euclidean": euc_pred,
        "Mahal-pooled": mahal_pred,
        "CB-kNN": knn_pred,
        "SMOTE-LR": smote_pred,
    }

    confusion_rows = []
    for method_name, preds in methods.items():
        m, per_class = classification_tables(y_test, pd.Series(preds), rare_class=RARE_CLASS)
        # For epsilon test cells specifically, what do they get predicted as?
        eps_mask = y_test.to_numpy() == RARE_CLASS
        eps_preds = preds[eps_mask] if isinstance(preds, np.ndarray) else np.array(preds)[eps_mask]
        pred_counts = pd.Series(eps_preds).value_counts().to_dict()
        confusion_rows.append({
            "method": method_name,
            "rare_f1": m["rare_f1"],
            "rare_recall": m["rare_recall"],
            "rare_precision": m["rare_precision"],
            "epsilon_predicted_as_epsilon": pred_counts.get(RARE_CLASS, 0),
            "epsilon_predicted_as_other": sum(v for k, v in pred_counts.items() if k != RARE_CLASS),
            "top_confusion_class": max(
                {k: v for k, v in pred_counts.items() if k != RARE_CLASS}.items(),
                key=lambda x: x[1], default=(None, 0)
            )[0],
        })
        print(f"  {method_name}: f1={m['rare_f1']:.3f}  epsilon_preds={pred_counts}")

    confusion_df = pd.DataFrame(confusion_rows)
    write_table(confusion_df, OUT_DIR / "confusion_summary.csv")

    # ── 3. Prototype distance analysis ────────────────────────────────────
    rare_idx = classes.index(RARE_CLASS)
    rare_proto = protos[rare_idx]
    majority_protos = np.delete(protos, rare_idx, axis=0)
    majority_classes = [c for c in classes if c != RARE_CLASS]

    dist_rows = []
    for i, (cls, proto) in enumerate(zip(majority_classes, majority_protos)):
        euc_d = float(np.linalg.norm(rare_proto - proto))
        diff = rare_proto - proto
        try:
            inv = np.linalg.inv(pooled)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(pooled)
        mahal_d = float(np.sqrt(max(diff @ inv @ diff, 0.0)))
        dist_rows.append({
            "rare_class": RARE_CLASS,
            "majority_class": cls,
            "n_majority_train": counts_map.get(cls, 0),
            "euclidean_dist": euc_d,
            "mahal_dist": mahal_d,
        })

    dist_df = pd.DataFrame(dist_rows).sort_values("euclidean_dist")
    write_table(dist_df, OUT_DIR / "prototype_distances.csv")
    print("\nPrototype distances (epsilon vs majority):")
    print(dist_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── 4. Neighbor analysis for epsilon test cells ────────────────────────
    eps_test_mask = y_test.to_numpy() == RARE_CLASS
    eps_test_z = test_z[eps_test_mask]

    nbrs = NearestNeighbors(n_neighbors=min(K_KNN, len(labeled_z)), metric="euclidean")
    nbrs.fit(labeled_z)
    distances, indices = nbrs.kneighbors(eps_test_z)

    neighbor_rows = []
    for i in range(len(eps_test_z)):
        neighbor_labels = labeled_labels[indices[i]]
        frac_rare = float((neighbor_labels == RARE_CLASS).mean())
        neighbor_rows.append({
            "epsilon_test_cell_idx": i,
            "frac_epsilon_neighbors": frac_rare,
            "n_epsilon_neighbors": int((neighbor_labels == RARE_CLASS).sum()),
            "mean_dist_to_neighbors": float(distances[i].mean()),
        })

    neighbor_df = pd.DataFrame(neighbor_rows)
    write_table(neighbor_df, OUT_DIR / "epsilon_neighbor_analysis.csv")
    print(f"\nEpsilon test cells neighbor analysis (k={K_KNN}):")
    print(f"  Mean fraction epsilon neighbors: {neighbor_df['frac_epsilon_neighbors'].mean():.3f}")
    print(f"  Mean n_epsilon_neighbors: {neighbor_df['n_epsilon_neighbors'].mean():.1f}")

    # ── 5. 2D PCA analysis ────────────────────────────────────────────────
    # Fit PCA on all labeled training cells
    pca = PCA(n_components=2, random_state=42)
    all_z = np.vstack([labeled_z, test_z])
    pca.fit(all_z)

    train_pca = pca.transform(labeled_z)
    test_pca  = pca.transform(test_z)

    # Save PCA coordinates for visualization
    train_pca_df = pd.DataFrame({
        "pc1": train_pca[:, 0],
        "pc2": train_pca[:, 1],
        "true_label": labeled_labels,
        "split": "train",
    })
    test_pca_df = pd.DataFrame({
        "pc1": test_pca[:, 0],
        "pc2": test_pca[:, 1],
        "true_label": y_test.to_numpy(),
        "scanvi_pred": scanvi_pred,
        "euclidean_pred": euc_pred,
        "mahal_pred": mahal_pred,
        "knn_pred": knn_pred,
        "smote_pred": smote_pred,
        "split": "test",
    })
    write_table(train_pca_df, OUT_DIR / "train_pca.csv")
    write_table(test_pca_df, OUT_DIR / "test_pca.csv")

    # Standard LR (no SMOTE) for comparison
    le_std = LabelEncoder()
    le_std.fit(labeled_labels)
    lr_std = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_std.fit(labeled_z, le_std.transform(labeled_labels))
    std_lr_pred = le_std.inverse_transform(lr_std.predict(test_z))
    std_lr_m, _ = classification_tables(y_test, pd.Series(std_lr_pred), rare_class=RARE_CLASS)
    print(f"\nStandard LR (no SMOTE): f1={std_lr_m['rare_f1']:.3f}")

    # Save summary
    summary = {
        "rare_class": RARE_CLASS,
        "n_epsilon_train": counts_map.get(RARE_CLASS, 0),
        "n_epsilon_test": int(eps_test_mask.sum()),
        "mean_frac_epsilon_neighbors": float(neighbor_df["frac_epsilon_neighbors"].mean()),
        "nearest_majority_euclidean": float(dist_df["euclidean_dist"].min()),
        "nearest_majority_mahal": float(dist_df["mahal_dist"].min()),
        "scanvi_f1": confusion_df[confusion_df["method"] == "scANVI"]["rare_f1"].iloc[0],
        "euclidean_f1": confusion_df[confusion_df["method"] == "Euclidean"]["rare_f1"].iloc[0],
        "mahal_f1": confusion_df[confusion_df["method"] == "Mahal-pooled"]["rare_f1"].iloc[0],
        "cb_knn_f1": confusion_df[confusion_df["method"] == "CB-kNN"]["rare_f1"].iloc[0],
        "smote_lr_f1": confusion_df[confusion_df["method"] == "SMOTE-LR"]["rare_f1"].iloc[0],
        "std_lr_f1": std_lr_m["rare_f1"],
    }
    write_table(pd.DataFrame([summary]), OUT_DIR / "summary.csv")

    print("\n=== E18 Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
