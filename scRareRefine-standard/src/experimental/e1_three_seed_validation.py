"""E1: 3-seed validation of Mahalanobis + posterior penalty.

For cDC1 (rare5), ASDC (rare5), epsilon (rare20) — run all 3 seeds (42, 43, 44).
Compare:
  - scANVI baseline
  - euclidean nearest-prototype
  - mahalanobis pooled + posterior penalty (best variant from PoC)

Reports mean ± std across seeds.

Usage:
    python src/experimental/e1_three_seed_validation.py
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
    _mahalanobis_with_posterior_penalty,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e1_three_seed_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    # (run_dir, rare_class)
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed43_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed44_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",  "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed43_asdc_rare5",  "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed44_asdc_rare5",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed43_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed44_epsilon_rare20", "epsilon"),
]


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
    y_true  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    counts = [counts_map[c] for c in classes]

    # Pooled covariance for Mahal-pooled+posterior
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled + posterior
    mpp_dists = _mahalanobis_with_posterior_penalty(test_z, protos, pooled_covs, counts)
    mpp_pred  = _predict_nearest(mpp_dists, classes)
    mpp_m, _  = classification_tables(y_true, pd.Series(mpp_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)

    # Extract seed from run_dir name
    name = run_dir.name
    seed = None
    for part in name.split("_"):
        if part.startswith("seed"):
            seed = int(part[4:])
            break

    return {
        "run": name,
        "rare_class": rare_class,
        "seed": seed,
        "scanvi_rare_f1":   scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pool_post_rare_f1": mpp_m["rare_f1"],
        "scanvi_overall_acc":   scanvi_m["overall_accuracy"],
        "euclidean_overall_acc": euc_m["overall_accuracy"],
        "mahal_pool_post_overall_acc": mpp_m["overall_accuracy"],
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        result = run_one(run_dir, rare_class)
        if result:
            rows.append(result)

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate mean ± std per (rare_class, method)
    agg_rows = []
    for rare_class in df["rare_class"].unique():
        sub = df[df["rare_class"] == rare_class]
        for method, col in [
            ("scANVI baseline",          "scanvi_rare_f1"),
            ("euclidean nearest-proto",  "euclidean_rare_f1"),
            ("mahal-pooled+posterior",   "mahal_pool_post_rare_f1"),
        ]:
            vals = sub[col].dropna().values
            agg_rows.append({
                "rare_class": rare_class,
                "method": method,
                "mean_rare_f1": float(np.mean(vals)),
                "std_rare_f1":  float(np.std(vals)),
                "n_seeds": len(vals),
            })

    agg = pd.DataFrame(agg_rows)
    write_table(agg, OUT_DIR / "aggregated_results.csv")

    print("\n=== E1 Results (mean ± std across 3 seeds) ===")
    for _, row in agg.iterrows():
        print(f"  {row['rare_class']:12s}  {row['method']:30s}  "
              f"rare_f1 = {row['mean_rare_f1']:.3f} ± {row['std_rare_f1']:.3f}")

    return df, agg


if __name__ == "__main__":
    main()
