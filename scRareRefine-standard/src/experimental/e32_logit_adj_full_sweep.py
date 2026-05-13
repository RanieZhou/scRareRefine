"""E32: Logit Adjustment full sweep — all datasets, all rts, 3 seeds.

E24 showed Logit Adjustment is the best new method from Round 4.
This experiment validates it across the full dataset × rts × seed grid,
the same coverage as E22 (final evaluation).

Compare: scANVI, Euclidean, Mahal-pooled, Logit Adjustment.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent, _class_prototypes, _pooled_covariance_shrunk, _mahalanobis, _euclidean, _predict_nearest
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e32_logit_adj_full_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

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


def _prob_cols(df):
    return [c for c in df.columns if c.startswith("prob_")]


def _logit_adj_predict(probs_df, log_pi, tau):
    classes = [c[len("prob_"):] for c in probs_df.columns]
    log_probs = np.log(probs_df.to_numpy(dtype=float) + 1e-12)
    adj = np.array([tau * log_pi.get(c, 0.0) for c in classes])
    adjusted = log_probs - adj[None, :]
    return np.array(classes)[adjusted.argmax(axis=1)]


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError:
        return None

    prob_cols = _prob_cols(train_pred)
    if not prob_cols:
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool)
    labeled = train_pred[is_labeled]
    class_counts = labeled["true_label"].value_counts()
    total = class_counts.sum()
    log_pi = {c: float(np.log(n / total)) for c, n in class_counts.items()}

    val_labels = val_pred["true_label"].astype(str).to_numpy()
    y_test = test_pred["true_label"].astype(str)

    if rare_class not in y_test.values:
        return None

    # Tune τ on validation
    best_tau = 1.0
    best_val_f1 = -1.0
    for tau in TAU_GRID:
        val_adj = _logit_adj_predict(val_pred[prob_cols], log_pi, tau)
        m, _ = classification_tables(pd.Series(val_labels), pd.Series(val_adj), rare_class=rare_class)
        if m["rare_f1"] > best_val_f1:
            best_val_f1 = m["rare_f1"]
            best_tau = tau

    # Apply to test
    test_adj = _logit_adj_predict(test_pred[prob_cols], log_pi, best_tau)
    la_m, _ = classification_tables(y_test, pd.Series(test_adj), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Euclidean and Mahal-pooled
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    classes_geo, protos, _ = _class_prototypes(train_z, train_pred["true_label"], is_labeled.to_numpy())
    if rare_class not in classes_geo:
        return None

    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes_geo)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled.to_numpy(), classes_geo)
    pooled_covs = [pooled] * len(classes_geo)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes_geo)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    name = run_dir.name
    seed = None
    rts  = None
    for part in name.split("_"):
        if part.startswith("seed"):
            try: seed = int(part[4:])
            except: pass
        if part.startswith("rare") and part != "rareall":
            try: rts = int(part[4:])
            except: pass

    return {
        "run": name,
        "rare_class": rare_class,
        "seed": seed,
        "rts": rts,
        "n_rare_train": int(class_counts.get(rare_class, 0)),
        "best_tau": best_tau,
        "scanvi_f1": scanvi_m["rare_f1"],
        "euclidean_f1": euc_m["rare_f1"],
        "mahal_f1": mahal_m["rare_f1"],
        "logit_adj_f1": la_m["rare_f1"],
        "logit_adj_recall": la_m["rare_recall"],
        "logit_adj_precision": la_m["rare_precision"],
    }


def main() -> pd.DataFrame:
    all_rows = []
    for dataset_dir, rare_class, split_prefix, rare_slug in DATASET_CONFIGS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            continue
        dataset_name = dataset_path.name
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}  rare_class: {rare_class}")
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
                      f"LA={result['logit_adj_f1']:.3f}")

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate
    agg_rows = []
    for (dataset, rare_class, rts), grp in df.groupby(["dataset", "rare_class", "rts"]):
        for method, col in [
            ("scANVI",       "scanvi_f1"),
            ("Euclidean",    "euclidean_f1"),
            ("Mahal-pooled", "mahal_f1"),
            ("Logit Adj",    "logit_adj_f1"),
        ]:
            vals = grp[col].dropna().values
            agg_rows.append({
                "dataset": dataset, "rare_class": rare_class, "rts": rts,
                "method": method,
                "mean_f1": float(np.mean(vals)),
                "std_f1":  float(np.std(vals)),
                "n_seeds": len(vals),
            })
    agg = pd.DataFrame(agg_rows)
    write_table(agg, OUT_DIR / "aggregated_results.csv")

    pivot = agg.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method", values="mean_f1", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "Logit Adj"] if c in pivot.columns]
    pivot["best_method"] = pivot[method_cols].idxmax(axis=1)
    write_table(pivot, OUT_DIR / "best_method_summary.csv")

    print("\n\n=== E32: Logit Adjustment Full Sweep ===")
    print(pivot[["dataset","rare_class","rts"] + method_cols + ["best_method"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best method distribution ===")
    print(pivot["best_method"].value_counts().to_string())

    if "Logit Adj" in pivot.columns and "scANVI" in pivot.columns:
        la_wins = (pivot["Logit Adj"] > pivot["scANVI"]).sum()
        total = len(pivot)
        print(f"\nLogit Adj wins vs scANVI: {la_wins}/{total} ({100*la_wins/total:.1f}%)")
        print(f"Mean delta LA vs scANVI: {(pivot['Logit Adj'] - pivot['scANVI']).mean():.3f}")

    if "Logit Adj" in pivot.columns and "Mahal-pooled" in pivot.columns:
        la_vs_mahal = (pivot["Logit Adj"] > pivot["Mahal-pooled"]).sum()
        print(f"\nLogit Adj wins vs Mahal-pooled: {la_vs_mahal}/{total} ({100*la_vs_mahal/total:.1f}%)")
        print(f"Mean delta LA vs Mahal: {(pivot['Logit Adj'] - pivot['Mahal-pooled']).mean():.3f}")

    # Regime analysis
    high_sep = ["cDC1", "ASDC", "gamma", "innate lymphoid cell"]
    low_sep  = ["epsilon", "non-classical monocyte", "endothelial cell"]
    pivot_hs = pivot[pivot["rare_class"].isin(high_sep)]
    pivot_ls = pivot[pivot["rare_class"].isin(low_sep)]
    print(f"\nHigh-sep ({len(pivot_hs)} configs):")
    for m in method_cols:
        if m in pivot_hs.columns:
            print(f"  {m:20s}: {pivot_hs[m].mean():.3f} ± {pivot_hs[m].std():.3f}")
    print(f"\nLow-sep ({len(pivot_ls)} configs):")
    for m in method_cols:
        if m in pivot_ls.columns:
            print(f"  {m:20s}: {pivot_ls[m].mean():.3f} ± {pivot_ls[m].std():.3f}")

    return df, agg, pivot


if __name__ == "__main__":
    main()
