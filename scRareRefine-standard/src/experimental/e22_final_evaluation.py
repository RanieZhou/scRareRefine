"""E22: Comprehensive 3-seed evaluation of best methods.

Top 3 methods:
1. Mahal-pooled (λ=0) — best for low-sep
2. Euclidean nearest-proto — best for high-sep (current method)
3. Recalibrated adaptive selector (from E16) — meta-method

Datasets: cDC1, ASDC, epsilon, gamma, NCM, endothelial, beta cell, ILC
Seeds: 42, 43, 44
rts: 20

Report: mean ± std across 3 seeds for each (dataset, method) pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e22_final_evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 43, 44]
RTS = 20

# (dataset_dir_template, rare_class, run_name_label)
# Template uses {seed} placeholder
DATASETS = [
    ("outputs/immune_dc/batch_heldout_seed{seed}_cdc1_rare20",                           "cDC1",                    "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed{seed}_asdc_rare20",                           "ASDC",                    "ASDC"),
    ("outputs/pancreas/batch_heldout_seed{seed}_epsilon_rare20",                         "epsilon",                 "epsilon"),
    ("outputs/pancreas/batch_heldout_seed{seed}_gamma_rare20",                           "gamma",                   "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed{seed}_non-classical_monocyte_rare20",    "non-classical monocyte",  "NCM"),
    ("outputs/tabula_kidney/cell_stratified_seed{seed}_endothelial_cell_rare20",         "endothelial cell",        "endothelial"),
    ("outputs/tabula_pancreas/cell_stratified_seed{seed}_type_b_pancreatic_cell_rare20", "type B pancreatic cell",  "beta_cell"),
    ("outputs/tabula_spleen/batch_heldout_seed{seed}_innate_lymphoid_cell_rare20",       "innate lymphoid cell",    "ILC"),
]

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
) -> np.ndarray:
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


def run_one(run_dir: Path, rare_class: str, seed: int, dataset_label: str) -> dict | None:
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

    # ── Method 1: Euclidean nearest-prototype ─────────────────────────────
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # ── Method 2: Mahal-pooled ─────────────────────────────────────────────
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # ── Method 3: Recalibrated adaptive selector ───────────────────────────
    if S >= NEW_HIGH:
        adaptive_pred = euc_pred
        adaptive_method = "euclidean"
    elif S >= NEW_LOW:
        adaptive_pred = mahal_pred
        adaptive_method = "mahal_pooled"
    else:
        adaptive_pred = _smote_lr_predict(labeled_z, labeled_labels, test_z)
        adaptive_method = "smote_lr"

    adaptive_m, _ = classification_tables(y_test, pd.Series(adaptive_pred), rare_class=rare_class)

    # ── scANVI baseline ────────────────────────────────────────────────────
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    return {
        "dataset": dataset_label,
        "rare_class": rare_class,
        "seed": seed,
        "separability_ratio": S,
        "adaptive_selected": adaptive_method,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "adaptive_rare_f1": adaptive_m["rare_f1"],
        "euclidean_recall": euc_m["rare_recall"],
        "mahal_recall": mahal_m["rare_recall"],
        "adaptive_recall": adaptive_m["rare_recall"],
    }


def main() -> pd.DataFrame:
    rows = []
    for tmpl, rare_class, dataset_label in DATASETS:
        for seed in SEEDS:
            run_path = tmpl.format(seed=seed)
            run_dir = ROOT / run_path
            print(f"Processing {run_dir.name} (seed={seed})...")
            try:
                result = run_one(run_dir, rare_class, seed, dataset_label)
                if result:
                    rows.append(result)
            except Exception as exc:
                import traceback
                print(f"  ERROR in {run_dir.name}: {exc}")
                traceback.print_exc()

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate: mean ± std across seeds
    agg_rows = []
    for (dataset, rare_class), grp in df.groupby(["dataset", "rare_class"]):
        for method in ["scanvi", "euclidean", "mahal_pooled", "adaptive"]:
            col = f"{method}_rare_f1"
            if col in grp.columns:
                agg_rows.append({
                    "dataset": dataset,
                    "rare_class": rare_class,
                    "method": method,
                    "mean_rare_f1": grp[col].mean(),
                    "std_rare_f1": grp[col].std(),
                    "n_seeds": len(grp),
                })

    agg_df = pd.DataFrame(agg_rows)
    write_table(agg_df, OUT_DIR / "aggregated_results.csv")

    print("\n=== E22 Results: 3-seed Comprehensive Evaluation ===")
    pivot = agg_df.pivot_table(
        index=["dataset", "rare_class"],
        columns="method",
        values="mean_rare_f1",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n=== Mean ± Std per method ===")
    for method in ["scanvi", "euclidean", "mahal_pooled", "adaptive"]:
        sub = agg_df[agg_df["method"] == method]
        if len(sub) > 0:
            print(f"  {method}: {sub['mean_rare_f1'].mean():.3f} ± {sub['mean_rare_f1'].std():.3f}")

    return df, agg_df


if __name__ == "__main__":
    main()
