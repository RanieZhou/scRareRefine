"""E8: Full data efficiency sweep — Mahal-pooled vs Euclidean.

For EVERY available run directory (all datasets, all rts, all seeds):
- Compare: scANVI baseline, Euclidean nearest-proto, Mahal-pooled (λ=0)
- Report rare_f1 for each
- Aggregate: mean±std per (dataset, rare_class, rts) across seeds

This is the most important experiment — it tells us whether Mahal-pooled
consistently helps across the full data efficiency curve.

Usage:
    python src/experimental/e8_data_efficiency_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e8_data_efficiency_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All run configurations: (dataset_dir, rare_class, split_prefix)
# We'll auto-discover rts and seeds from directory names
DATASET_CONFIGS = [
    ("outputs/immune_dc",       "cDC1",                    "batch_heldout",       "cdc1"),
    ("outputs/immune_dc",       "ASDC",                    "batch_heldout",       "asdc"),
    ("outputs/pancreas",        "epsilon",                 "batch_heldout",       "epsilon"),
    ("outputs/pancreas",        "gamma",                   "batch_heldout",       "gamma"),
    ("outputs/tabula_liver",    "non-classical monocyte",  "cell_stratified",     "non-classical_monocyte"),
    ("outputs/tabula_kidney",   "endothelial cell",        "cell_stratified",     "endothelial_cell"),
    ("outputs/tabula_spleen",   "innate lymphoid cell",    "batch_heldout",       "innate_lymphoid_cell"),
    ("outputs/tabula_pancreas", "type B pancreatic cell",  "cell_stratified",     "type_b_pancreatic_cell"),
]

SEEDS = [42, 43, 44]
RTS_VALUES = [5, 10, 20, 50]


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    """Run Euclidean and Mahal-pooled on a single run directory."""
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

    # Check rare class exists in test
    if rare_class not in y_true.values:
        return None

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return None

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled (λ=0, no posterior penalty)
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)
    mah_pred  = _predict_nearest(mah_dists, classes)
    mah_m, _  = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)

    # Parse seed and rts from directory name
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
        "scanvi_rare_f1":    scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mah_m["rare_f1"],
        "scanvi_overall_acc":    scanvi_m["overall_accuracy"],
        "euclidean_overall_acc": euc_m["overall_accuracy"],
        "mahal_pooled_overall_acc": mah_m["overall_accuracy"],
        "scanvi_rare_recall":    scanvi_m["rare_recall"],
        "euclidean_rare_recall": euc_m["rare_recall"],
        "mahal_pooled_rare_recall": mah_m["rare_recall"],
    }


def main() -> pd.DataFrame:
    all_rows = []

    for dataset_dir, rare_class, split_prefix, rare_slug in DATASET_CONFIGS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            print(f"  SKIP: {dataset_path} not found")
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
                    print(f"  SKIP: {run_name} (missing embeddings or rare class)")
                    continue

                result["dataset"] = dataset_name
                all_rows.append(result)
                print(f"  seed={seed} rts={rts:3d}  "
                      f"scANVI={result['scanvi_rare_f1']:.3f}  "
                      f"Eucl={result['euclidean_rare_f1']:.3f}  "
                      f"Mahal={result['mahal_pooled_rare_f1']:.3f}")

    if not all_rows:
        print("No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate mean±std per (dataset, rare_class, rts) across seeds
    agg_rows = []
    for (dataset, rare_class, rts), grp in df.groupby(["dataset", "rare_class", "rts"]):
        for method, col in [
            ("scANVI",       "scanvi_rare_f1"),
            ("Euclidean",    "euclidean_rare_f1"),
            ("Mahal-pooled", "mahal_pooled_rare_f1"),
        ]:
            vals = grp[col].dropna().values
            agg_rows.append({
                "dataset": dataset,
                "rare_class": rare_class,
                "rts": rts,
                "method": method,
                "mean_rare_f1": float(np.mean(vals)),
                "std_rare_f1":  float(np.std(vals)),
                "n_seeds": len(vals),
            })

    agg = pd.DataFrame(agg_rows)
    write_table(agg, OUT_DIR / "aggregated_results.csv")

    # Print summary table
    print("\n\n=== E8 Summary: mean rare_f1 across seeds ===")
    pivot = agg.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="mean_rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Compute Mahal vs Euclidean delta
    df_euc  = agg[agg["method"] == "Euclidean"][["dataset","rare_class","rts","mean_rare_f1"]].rename(columns={"mean_rare_f1":"euc_f1"})
    df_mah  = agg[agg["method"] == "Mahal-pooled"][["dataset","rare_class","rts","mean_rare_f1"]].rename(columns={"mean_rare_f1":"mah_f1"})
    df_scan = agg[agg["method"] == "scANVI"][["dataset","rare_class","rts","mean_rare_f1"]].rename(columns={"mean_rare_f1":"scan_f1"})
    delta = df_euc.merge(df_mah, on=["dataset","rare_class","rts"]).merge(df_scan, on=["dataset","rare_class","rts"])
    delta["mahal_vs_eucl"] = delta["mah_f1"] - delta["euc_f1"]
    delta["eucl_vs_scanvi"] = delta["euc_f1"] - delta["scan_f1"]
    write_table(delta, OUT_DIR / "delta_analysis.csv")

    print("\n\n=== E8 Delta: Mahal-pooled vs Euclidean (positive = Mahal wins) ===")
    print(delta[["dataset","rare_class","rts","scan_f1","euc_f1","mah_f1","mahal_vs_eucl"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    return df, agg, delta


if __name__ == "__main__":
    main()
