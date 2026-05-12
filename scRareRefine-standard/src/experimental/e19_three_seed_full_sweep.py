"""E19: Three-seed full sweep — Mahal-pooled vs CB-kNN vs Euclidean.

E8 and E9 already ran 3 seeds for Mahal-pooled and CB-kNN separately.
This experiment runs a focused 3-seed comparison on the most interesting
configurations: rts=5, rts=20, rts=50 for all datasets.

Key question: Which method is most consistently best across seeds?

Usage:
    python src/experimental/e19_three_seed_full_sweep.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e19_three_seed_full_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_CB = 30

DATASET_CONFIGS = [
    ("outputs/immune_dc",       "cDC1",                    "batch_heldout",   "cdc1"),
    ("outputs/immune_dc",       "ASDC",                    "batch_heldout",   "asdc"),
    ("outputs/pancreas",        "epsilon",                 "batch_heldout",   "epsilon"),
    ("outputs/pancreas",        "gamma",                   "batch_heldout",   "gamma"),
    ("outputs/tabula_liver",    "non-classical monocyte",  "cell_stratified", "non-classical_monocyte"),
    ("outputs/tabula_kidney",   "endothelial cell",        "cell_stratified", "endothelial_cell"),
    ("outputs/tabula_spleen",   "innate lymphoid cell",    "batch_heldout",   "innate_lymphoid_cell"),
]

SEEDS = [42, 43, 44]
RTS_VALUES = [5, 20, 50]


def cb_knn_predict(train_z, train_labels, test_z, k, class_counts):
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


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError:
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_true  = test_pred["true_label"].astype(str)

    if rare_class not in y_true.values:
        return None

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return None

    ref_z      = train_z[is_labeled]
    ref_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]
    class_counts = dict(Counter(ref_labels))

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)
    mah_pred  = _predict_nearest(mah_dists, classes)
    mah_m, _  = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)

    # CB-kNN
    cb_pred = cb_knn_predict(ref_z, ref_labels, test_z, K_CB, class_counts)
    cb_m, _ = classification_tables(y_true, pd.Series(cb_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)

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

    return {
        "run": name,
        "rare_class": rare_class,
        "seed": seed,
        "rts": rts,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_f1":    scanvi_m["rare_f1"],
        "euclidean_f1": euc_m["rare_f1"],
        "mahal_f1":     mah_m["rare_f1"],
        "cb_knn_f1":    cb_m["rare_f1"],
    }


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

                result = run_one(run_dir, rare_class)
                if result is None:
                    continue

                result["dataset"] = dataset_name
                all_rows.append(result)
                print(f"  seed={seed} rts={rts:3d}  "
                      f"scANVI={result['scanvi_f1']:.3f}  "
                      f"Eucl={result['euclidean_f1']:.3f}  "
                      f"Mahal={result['mahal_f1']:.3f}  "
                      f"CB-kNN={result['cb_knn_f1']:.3f}")

    if not all_rows:
        print("No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate mean±std per (dataset, rare_class, rts) across seeds
    agg_rows = []
    for (dataset, rare_class, rts), grp in df.groupby(["dataset", "rare_class", "rts"]):
        for method, col in [
            ("scANVI",       "scanvi_f1"),
            ("Euclidean",    "euclidean_f1"),
            ("Mahal-pooled", "mahal_f1"),
            ("CB-kNN",       "cb_knn_f1"),
        ]:
            vals = grp[col].dropna().values
            agg_rows.append({
                "dataset": dataset,
                "rare_class": rare_class,
                "rts": rts,
                "method": method,
                "mean_f1": float(np.mean(vals)),
                "std_f1":  float(np.std(vals)),
                "n_seeds": len(vals),
            })

    agg = pd.DataFrame(agg_rows)
    write_table(agg, OUT_DIR / "aggregated_results.csv")

    print("\n\n=== E19 Summary: 3-seed mean±std ===")
    pivot = agg.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="mean_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Best method per config
    pivot_reset = pivot.reset_index()
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN"] if c in pivot_reset.columns]
    pivot_reset["best_method"] = pivot_reset[method_cols].idxmax(axis=1)
    pivot_reset["best_f1"] = pivot_reset[method_cols].max(axis=1)

    print("\n\n=== E19 Best method per config ===")
    print(pivot_reset[["dataset","rare_class","rts","best_method","best_f1"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== E19 Best method distribution ===")
    print(pivot_reset["best_method"].value_counts().to_string())

    # Mahal vs Euclidean win rate
    if "Mahal-pooled" in pivot_reset.columns and "Euclidean" in pivot_reset.columns:
        mahal_wins = (pivot_reset["Mahal-pooled"] > pivot_reset["Euclidean"]).sum()
        total = len(pivot_reset)
        print(f"\nMahal-pooled wins vs Euclidean: {mahal_wins}/{total} ({100*mahal_wins/total:.1f}%)")
        print(f"Mean delta: {(pivot_reset['Mahal-pooled'] - pivot_reset['Euclidean']).mean():.3f}")

    write_table(pivot_reset, OUT_DIR / "best_method_per_config.csv")

    return df, agg


if __name__ == "__main__":
    main()
