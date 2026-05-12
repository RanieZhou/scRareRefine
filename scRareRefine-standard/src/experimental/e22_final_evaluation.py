"""E22: Final evaluation — best methods, 3 seeds, all rts.

Consolidate findings from E8-E21. Run the top 3 methods:
1. Mahal-pooled (λ=0) — best for low-sep
2. CB-kNN (k=30) — best for high-sep, low rts
3. Euclidean — strong baseline for high-sep

Run on: all datasets, rts=5/20/50, 3 seeds.
Report: mean±std, win rates, regime analysis.

Usage:
    python src/experimental/e22_final_evaluation.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e22_final_evaluation"
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
    ("outputs/tabula_pancreas", "type B pancreatic cell",  "cell_stratified", "type_b_pancreatic_cell"),
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
        "scanvi_recall":    scanvi_m["rare_recall"],
        "euclidean_recall": euc_m["rare_recall"],
        "mahal_recall":     mah_m["rare_recall"],
        "cb_knn_recall":    cb_m["rare_recall"],
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

    # Pivot for display
    pivot = agg.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="mean_f1",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN"] if c in pivot.columns]
    pivot["best_method"] = pivot[method_cols].idxmax(axis=1)
    pivot["best_f1"] = pivot[method_cols].max(axis=1)

    write_table(pivot, OUT_DIR / "best_method_summary.csv")

    print("\n\n=== E22 Final Evaluation: 3-seed mean rare_f1 ===")
    print(pivot[["dataset","rare_class","rts"] + method_cols + ["best_method","best_f1"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n\n=== E22 Best method distribution ===")
    print(pivot["best_method"].value_counts().to_string())

    # Win rates
    if "Mahal-pooled" in pivot.columns and "Euclidean" in pivot.columns:
        mahal_wins = (pivot["Mahal-pooled"] > pivot["Euclidean"]).sum()
        total = len(pivot)
        print(f"\nMahal-pooled wins vs Euclidean: {mahal_wins}/{total} ({100*mahal_wins/total:.1f}%)")
        print(f"Mean delta: {(pivot['Mahal-pooled'] - pivot['Euclidean']).mean():.3f}")

    if "CB-kNN" in pivot.columns and "Euclidean" in pivot.columns:
        cb_wins = (pivot["CB-kNN"] > pivot["Euclidean"]).sum()
        print(f"\nCB-kNN wins vs Euclidean: {cb_wins}/{total} ({100*cb_wins/total:.1f}%)")
        print(f"Mean delta: {(pivot['CB-kNN'] - pivot['Euclidean']).mean():.3f}")

    # Regime analysis: high-sep vs low-sep
    # High-sep: cDC1, ASDC, gamma, ILC (Euclidean works well)
    # Low-sep: epsilon, NCM, endothelial (Euclidean fails)
    high_sep = ["cDC1", "ASDC", "gamma", "innate lymphoid cell"]
    low_sep  = ["epsilon", "non-classical monocyte", "endothelial cell", "type B pancreatic cell"]

    pivot_hs = pivot[pivot["rare_class"].isin(high_sep)]
    pivot_ls = pivot[pivot["rare_class"].isin(low_sep)]

    print("\n\n=== E22 Regime Analysis ===")
    print(f"\nHigh-sep cases ({len(pivot_hs)} configs):")
    for method in method_cols:
        if method in pivot_hs.columns:
            print(f"  {method:20s}: mean={pivot_hs[method].mean():.3f}  std={pivot_hs[method].std():.3f}")

    print(f"\nLow-sep cases ({len(pivot_ls)} configs):")
    for method in method_cols:
        if method in pivot_ls.columns:
            print(f"  {method:20s}: mean={pivot_ls[method].mean():.3f}  std={pivot_ls[method].std():.3f}")

    return df, agg, pivot


if __name__ == "__main__":
    main()
