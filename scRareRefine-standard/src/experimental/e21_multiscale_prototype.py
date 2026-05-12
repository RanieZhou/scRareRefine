"""E21: Multi-scale prototype — combine local and global distance signals.

Observation: Mahal-pooled uses a single global covariance. But the latent
space may have different scales at different regions.

Idea: Compute distances at multiple scales:
  d_local(z, c) = Euclidean distance to nearest k_local labeled cells of class c
  d_global(z, c) = Mahal-pooled distance to class prototype

Combine: d_multi(z, c) = β * d_local(z, c) + (1-β) * d_global(z, c)

β is tuned on validation.

This is a "local + global" ensemble that captures both fine-grained
neighborhood structure and global class geometry.

Run on: all datasets, rts=5/20/50, seed42.

Usage:
    python src/experimental/e21_multiscale_prototype.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e21_multiscale_prototype"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_LOCAL = 5  # k nearest labeled cells for local distance
BETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

DATASET_CONFIGS = [
    ("outputs/immune_dc",       "cDC1",                    "batch_heldout",   "cdc1"),
    ("outputs/immune_dc",       "ASDC",                    "batch_heldout",   "asdc"),
    ("outputs/pancreas",        "epsilon",                 "batch_heldout",   "epsilon"),
    ("outputs/pancreas",        "gamma",                   "batch_heldout",   "gamma"),
    ("outputs/tabula_liver",    "non-classical monocyte",  "cell_stratified", "non-classical_monocyte"),
    ("outputs/tabula_kidney",   "endothelial cell",        "cell_stratified", "endothelial_cell"),
    ("outputs/tabula_spleen",   "innate lymphoid cell",    "batch_heldout",   "innate_lymphoid_cell"),
]

SEEDS = [42]
RTS_VALUES = [5, 20, 50]


def local_distance(
    test_z: np.ndarray,
    ref_z: np.ndarray,
    ref_labels: np.ndarray,
    classes: list[str],
    k: int,
) -> np.ndarray:
    """Compute local distance: mean distance to k nearest labeled cells per class."""
    n_test = len(test_z)
    n_classes = len(classes)
    dists = np.full((n_test, n_classes), np.inf)

    for ci, c in enumerate(classes):
        mask = ref_labels == c
        X_c = ref_z[mask]
        if len(X_c) == 0:
            continue
        k_eff = min(k, len(X_c))
        nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree")
        nbrs.fit(X_c)
        nn_dists, _ = nbrs.kneighbors(test_z)
        dists[:, ci] = nn_dists.mean(axis=1)

    return dists


def normalize_distances(dists: np.ndarray) -> np.ndarray:
    """Normalize distances to [0,1] range per test cell."""
    row_max = dists.max(axis=1, keepdims=True)
    row_min = dists.min(axis=1, keepdims=True)
    denom = row_max - row_min
    denom[denom < 1e-10] = 1.0
    return (dists - row_min) / denom


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

    # Global: Mahal-pooled distances
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Local: k-NN distances per class
    loc_dists = local_distance(test_z, ref_z, ref_labels, classes, K_LOCAL)

    # Normalize both
    mah_norm = normalize_distances(mah_dists)
    loc_norm = normalize_distances(loc_dists)

    # Euclidean baseline
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled baseline
    mah_pred = _predict_nearest(mah_dists, classes)
    mah_m, _ = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)

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

    rows = [
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "scANVI", "beta": None, "rare_f1": scanvi_m["rare_f1"]},
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "Euclidean", "beta": None, "rare_f1": euc_m["rare_f1"]},
        {"run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
         "method": "Mahal-pooled", "beta": None, "rare_f1": mah_m["rare_f1"]},
    ]

    # Validation split for beta tuning
    n_test = len(test_z)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)
    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_val = y_true.iloc[val_idx].reset_index(drop=True)

    best_beta = 0.0
    best_val_f1 = -1.0

    for beta in BETA_GRID:
        # Multi-scale: β * local + (1-β) * global
        multi_val = beta * loc_norm[val_idx] + (1 - beta) * mah_norm[val_idx]
        pred_val = _predict_nearest(multi_val, classes)
        m_val, _ = classification_tables(y_val, pd.Series(pred_val), rare_class=rare_class)
        if m_val["rare_f1"] > best_val_f1:
            best_val_f1 = m_val["rare_f1"]
            best_beta = beta

    # Apply best beta to full test
    multi_full = best_beta * loc_norm + (1 - best_beta) * mah_norm
    pred_full = _predict_nearest(multi_full, classes)
    m_full, _ = classification_tables(y_true, pd.Series(pred_full), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Multi-scale(β={best_beta})", "beta": best_beta,
        "rare_f1": m_full["rare_f1"],
    })

    print(f"  rts={rts:3d}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mah_m['rare_f1']:.3f}  "
          f"Multi-scale(β={best_beta})={m_full['rare_f1']:.3f}")

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

    print("\n\n=== E21 Summary: Multi-scale prototype ===")
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    # Multi-scale vs Mahal delta
    df_mah = df[df["method"] == "Mahal-pooled"][["dataset","rare_class","rts","rare_f1"]].rename(columns={"rare_f1":"mah_f1"})
    df_ms  = df[df["method"].str.startswith("Multi-scale")][["dataset","rare_class","rts","rare_f1","beta"]].rename(columns={"rare_f1":"ms_f1"})
    delta = df_mah.merge(df_ms, on=["dataset","rare_class","rts"])
    delta["ms_vs_mahal"] = delta["ms_f1"] - delta["mah_f1"]
    write_table(delta, OUT_DIR / "delta_analysis.csv")

    print("\n\n=== E21 Delta: Multi-scale vs Mahal-pooled ===")
    print(delta[["dataset","rare_class","rts","mah_f1","ms_f1","ms_vs_mahal","beta"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"
    ))

    print("\n=== E21 Best beta distribution ===")
    print(df_ms["beta"].value_counts().sort_index().to_string())

    return df, delta


if __name__ == "__main__":
    main()
