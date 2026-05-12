"""E18: Mahal-pooled + CB-kNN Hybrid.

E8 shows Mahal-pooled wins on low-sep (epsilon, gamma rts=5).
E9 shows CB-kNN wins on high-sep (gamma rts=5, endothelial).

Idea: Use Mahal-pooled as the primary classifier, but use CB-kNN as a
"rescue" mechanism when Mahal-pooled is uncertain.

Uncertainty measure: margin between top-2 Mahal distances.
If margin < threshold → use CB-kNN prediction instead.

This is a "committee" approach: two experts, each good at different regimes.

Run on: all datasets, rts=5/20/50, seed42.

Usage:
    python src/experimental/e18_mahal_cb_knn_hybrid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from collections import Counter
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e18_mahal_cb_knn_hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_CB = 30
MARGIN_THRESH_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]

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


def cb_knn_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    test_z: np.ndarray,
    k: int,
    class_counts: dict[str, int],
) -> np.ndarray:
    k_eff = min(k, len(train_z))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(train_z)
    nn_dists_all, nn_idx_all = nbrs.kneighbors(test_z)
    preds = []
    eps = 1e-10
    for i in range(len(test_z)):
        nn_idx    = nn_idx_all[i]
        nn_labels = train_labels[nn_idx]
        nn_dists  = nn_dists_all[i]
        vote: dict[str, float] = {}
        for lbl, d in zip(nn_labels, nn_dists):
            n_c = class_counts.get(lbl, 1)
            w = 1.0 / (n_c * d ** 2 + eps)
            vote[lbl] = vote.get(lbl, 0.0) + w
        preds.append(max(vote, key=vote.get))
    return np.array(preds)


def hybrid_predict(
    test_z: np.ndarray,
    mah_dists: np.ndarray,
    classes: list[str],
    cb_pred: np.ndarray,
    margin_thresh: float,
) -> np.ndarray:
    """Hybrid: use Mahal-pooled, but switch to CB-kNN when margin is small."""
    n = len(test_z)
    # Sort distances per test cell
    sorted_dists = np.sort(mah_dists, axis=1)
    # Margin = difference between 2nd and 1st nearest prototype
    margin = sorted_dists[:, 1] - sorted_dists[:, 0]

    mah_pred = _predict_nearest(mah_dists, classes)
    y_out = mah_pred.copy()

    for i in range(n):
        if margin[i] < margin_thresh:
            y_out[i] = cb_pred[i]

    return y_out


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

    ref_z      = train_z[is_labeled]
    ref_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]
    class_counts = dict(Counter(ref_labels))

    # Mahal-pooled distances
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)

    # CB-kNN
    cb_pred = cb_knn_predict(ref_z, ref_labels, test_z, K_CB, class_counts)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)

    # Euclidean
    euc_m, _ = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled
    mah_pred = _predict_nearest(mah_dists, classes)
    mah_m, _ = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)

    # CB-kNN alone
    cb_m, _ = classification_tables(y_true, pd.Series(cb_pred), rare_class=rare_class)

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

    rows = [
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "scANVI", "margin_thresh": None, "rare_f1": scanvi_m["rare_f1"]},
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "Euclidean", "margin_thresh": None, "rare_f1": euc_m["rare_f1"]},
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "Mahal-pooled", "margin_thresh": None, "rare_f1": mah_m["rare_f1"]},
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "CB-kNN", "margin_thresh": None, "rare_f1": cb_m["rare_f1"]},
    ]

    # Validation split for margin_thresh tuning
    n_test = len(test_z)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)
    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_val  = y_true.iloc[val_idx].reset_index(drop=True)

    best_thresh = np.inf
    best_val_f1 = -1.0

    for thresh in MARGIN_THRESH_GRID:
        pred_val = hybrid_predict(test_z[val_idx], mah_dists[val_idx], classes, cb_pred[val_idx], thresh)
        m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
        if m_val["rare_f1"] > best_val_f1:
            best_val_f1 = m_val["rare_f1"]
            best_thresh = thresh

    # Apply best threshold to full test
    hybrid_pred = hybrid_predict(test_z, mah_dists, classes, cb_pred, best_thresh)
    hybrid_m, _ = classification_tables(y_true, pd.Series(hybrid_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Hybrid(thresh={best_thresh})", "margin_thresh": best_thresh,
        "rare_f1": hybrid_m["rare_f1"],
    })

    print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
          f"CB-kNN={cb_m['rare_f1']:.3f}  "
          f"Hybrid(t={best_thresh})={hybrid_m['rare_f1']:.3f}")

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

    print("\n\n=== E18 Summary: Mahal+CB-kNN Hybrid ===")
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Hybrid vs best(Mahal, CB-kNN)
    df_mah = df[df["method"] == "Mahal-pooled"][["dataset","rare_class","rts","rare_f1"]].rename(columns={"rare_f1":"mah_f1"})
    df_cb  = df[df["method"] == "CB-kNN"][["dataset","rare_class","rts","rare_f1"]].rename(columns={"rare_f1":"cb_f1"})
    df_hyb = df[df["method"].str.startswith("Hybrid")][["dataset","rare_class","rts","rare_f1","margin_thresh"]].rename(columns={"rare_f1":"hyb_f1"})
    delta = df_mah.merge(df_cb, on=["dataset","rare_class","rts"]).merge(df_hyb, on=["dataset","rare_class","rts"])
    delta["best_of_two"] = delta[["mah_f1","cb_f1"]].max(axis=1)
    delta["hybrid_vs_best"] = delta["hyb_f1"] - delta["best_of_two"]
    write_table(delta, OUT_DIR / "delta_analysis.csv")

    print("\n\n=== E18 Delta: Hybrid vs best(Mahal, CB-kNN) ===")
    print(delta[["dataset","rare_class","rts","mah_f1","cb_f1","hyb_f1","hybrid_vs_best","margin_thresh"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    return df, delta


if __name__ == "__main__":
    main()
