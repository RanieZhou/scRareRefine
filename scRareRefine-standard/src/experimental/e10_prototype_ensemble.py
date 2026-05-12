"""E10: Prototype ensemble — combine Euclidean + Mahal-pooled.

Idea: Use a weighted combination of Euclidean and Mahal-pooled distances.
  d_ensemble(z, c) = α * d_euclidean(z, c) + (1-α) * d_mahal(z, c)

Grid search α ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0} on validation.
Apply best α to test.

Hypothesis: The ensemble might be more robust than either alone —
Euclidean is better for high-sep, Mahal for low-sep, and the ensemble
might find a middle ground.

Run on: all datasets, rts=5/20/50, seed42.

Usage:
    python src/experimental/e10_prototype_ensemble.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e10_prototype_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

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

SEEDS = [42]  # seed42 for this experiment
RTS_VALUES = [5, 20, 50]


def normalize_distances(dists: np.ndarray) -> np.ndarray:
    """Normalize distances to [0,1] range per test cell."""
    row_max = dists.max(axis=1, keepdims=True)
    row_min = dists.min(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom < 1e-10] = 1.0
    return (dists - row_min) / denom


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

    # Compute base distances
    euc_dists = _euclidean(test_z, protos)
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Normalize both distance matrices
    euc_norm = normalize_distances(euc_dists)
    mah_norm = normalize_distances(mah_dists)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_true, test_pred["predicted_label"], rare_class=rare_class)

    # Grid search over alpha
    # Use a simple 80/20 split of test set as "validation" for alpha selection
    n_test = len(test_z)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)

    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_val  = y_true.iloc[val_idx].reset_index(drop=True)
    y_test_split = y_true.iloc[test_idx].reset_index(drop=True)

    best_alpha = 0.5
    best_val_f1 = -1.0

    alpha_results = []
    for alpha in ALPHA_GRID:
        # Ensemble on validation
        ens_val = alpha * euc_norm[val_idx] + (1 - alpha) * mah_norm[val_idx]
        pred_val = _predict_nearest(ens_val, classes)
        m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
        alpha_results.append({"alpha": alpha, "val_rare_f1": m_val["rare_f1"]})
        if m_val["rare_f1"] > best_val_f1:
            best_val_f1 = m_val["rare_f1"]
            best_alpha = alpha

    # Apply best alpha to test
    ens_test = best_alpha * euc_norm[test_idx] + (1 - best_alpha) * mah_norm[test_idx]
    pred_test = _predict_nearest(ens_test, classes)
    m_test, _ = classification_tables(y_test_split, pd.Series(pred_test), rare_class=rare_class)

    # Also compute full-test metrics for each alpha (for reporting)
    full_alpha_rows = []
    for alpha in ALPHA_GRID:
        ens_full = alpha * euc_norm + (1 - alpha) * mah_norm
        pred_full = _predict_nearest(ens_full, classes)
        m_full, _ = classification_tables(y_true, pd.Series(pred_full), rare_class=rare_class)
        full_alpha_rows.append({
            "run": run_dir.name,
            "rare_class": rare_class,
            "alpha": alpha,
            "full_test_rare_f1": m_full["rare_f1"],
        })

    # Euclidean and Mahal baselines on full test
    euc_pred = _predict_nearest(euc_dists, classes)
    euc_m, _ = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)
    mah_pred = _predict_nearest(mah_dists, classes)
    mah_m, _ = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)

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
        "best_alpha": best_alpha,
        "best_val_f1": best_val_f1,
        "ensemble_test_rare_f1": m_test["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mah_m["rare_f1"],
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "alpha_curve": full_alpha_rows,
    }


def main() -> pd.DataFrame:
    all_rows = []
    all_alpha_rows = []

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

                alpha_curve = result.pop("alpha_curve")
                for row in alpha_curve:
                    row["dataset"] = dataset_name
                    row["rts"] = result["rts"]
                    all_alpha_rows.append(row)

                result["dataset"] = dataset_name
                all_rows.append(result)
                print(f"  seed={seed} rts={rts:3d}  "
                      f"best_α={result['best_alpha']:.1f}  "
                      f"Eucl={result['euclidean_rare_f1']:.3f}  "
                      f"Mahal={result['mahal_pooled_rare_f1']:.3f}  "
                      f"Ensemble={result['ensemble_test_rare_f1']:.3f}  "
                      f"scANVI={result['scanvi_rare_f1']:.3f}")

    if not all_rows:
        print("No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    df_alpha = pd.DataFrame(all_alpha_rows)
    write_table(df_alpha, OUT_DIR / "alpha_curves.csv")

    print("\n\n=== E10 Summary: Ensemble vs Euclidean vs Mahal-pooled ===")
    cols = ["dataset","rare_class","rts","best_alpha","euclidean_rare_f1","mahal_pooled_rare_f1","ensemble_test_rare_f1","scanvi_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Delta analysis
    df["ensemble_vs_eucl"] = df["ensemble_test_rare_f1"] - df["euclidean_rare_f1"]
    df["ensemble_vs_mahal"] = df["ensemble_test_rare_f1"] - df["mahal_pooled_rare_f1"]
    df["best_of_two"] = df[["euclidean_rare_f1","mahal_pooled_rare_f1"]].max(axis=1)
    df["ensemble_vs_best"] = df["ensemble_test_rare_f1"] - df["best_of_two"]

    print("\n\n=== E10 Delta: Ensemble vs best(Eucl, Mahal) ===")
    print(df[["dataset","rare_class","rts","best_alpha","best_of_two","ensemble_test_rare_f1","ensemble_vs_best"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    write_table(df, OUT_DIR / "delta_analysis.csv")

    # Alpha distribution
    print("\n\n=== E10 Best alpha distribution ===")
    print(df["best_alpha"].value_counts().sort_index().to_string())

    return df, df_alpha


if __name__ == "__main__":
    main()
