"""E14: Full sweep — Mahal-pooled across ALL datasets and rare_train_sizes.

Run Mahal-pooled (λ=0) on ALL available runs (all datasets, all rts, seed42 only).
Compare vs: scANVI baseline, euclidean nearest-proto.

Compute for each run:
- separability_ratio
- scANVI rare_f1
- euclidean rare_f1
- mahal_pooled rare_f1
- delta = mahal_pooled - euclidean

Save comprehensive results table.
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e14_full_mahal_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# All seed42 runs across all datasets and rts
ALL_RUNS = [
    # immune_dc cDC1
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare10",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare50",  "cDC1"),
    # immune_dc ASDC
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare10",  "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",  "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare50",  "ASDC"),
    # pancreas epsilon
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare5",  "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare10", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare50", "epsilon"),
    # pancreas gamma
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare10",  "gamma"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",  "gamma"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare50",  "gamma"),
    # tabula_liver NCM
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare5",  "non-classical monocyte"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare10", "non-classical monocyte"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare50", "non-classical monocyte"),
    # tabula_kidney endothelial
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare5",  "endothelial cell"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare10", "endothelial cell"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare20", "endothelial cell"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare50", "endothelial cell"),
    # tabula_pancreas beta cell
    ("outputs/tabula_pancreas/cell_stratified_seed42_type_b_pancreatic_cell_rare20", "type B pancreatic cell"),
    # tabula_spleen ILC
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare5",  "innate lymphoid cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare10", "innate lymphoid cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20", "innate lymphoid cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare50", "innate lymphoid cell"),
]


def _separability_ratio(
    labeled_z: np.ndarray,
    labeled_labels: np.ndarray,
    rare_class: str,
    classes: list[str],
    protos: np.ndarray,
) -> float:
    rare_idx = classes.index(rare_class)
    rare_cells = labeled_z[labeled_labels == rare_class]
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
    y_test  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        return None

    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    S = _separability_ratio(labeled_z, labeled_labels, rare_class, classes, protos)

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Parse rts from run name
    name = run_dir.name
    rts = "unknown"
    for part in name.split("_"):
        if part.startswith("rare"):
            rts = part.replace("rare", "")

    # Dataset from path
    dataset = run_dir.parts[-3] if len(run_dir.parts) >= 3 else "unknown"

    return {
        "dataset": dataset,
        "run": name,
        "rare_class": rare_class,
        "rare_train_size": rts,
        "n_rare_train": counts_map.get(rare_class, 0),
        "separability_ratio": S,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "delta_mahal_minus_euc": mahal_m["rare_f1"] - euc_m["rare_f1"],
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in ALL_RUNS:
        run_dir = ROOT / rel_path
        try:
            result = run_one(run_dir, rare_class)
            if result:
                rows.append(result)
                print(f"  {result['run']}: S={result['separability_ratio']:.3f}  "
                      f"euc={result['euclidean_rare_f1']:.3f}  "
                      f"mahal={result['mahal_pooled_rare_f1']:.3f}  "
                      f"delta={result['delta_mahal_minus_euc']:+.3f}")
            else:
                print(f"  SKIPPED: {run_dir.name}")
        except Exception as exc:
            print(f"  ERROR in {run_dir.name}: {exc}")

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print(f"\n=== E14 Results: Full Mahal Sweep ({len(df)} runs) ===")
    print(df[["dataset", "rare_class", "rare_train_size", "separability_ratio",
              "scanvi_rare_f1", "euclidean_rare_f1", "mahal_pooled_rare_f1",
              "delta_mahal_minus_euc"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Summary by dataset
    print("\n--- Summary by dataset ---")
    summary = df.groupby("dataset").agg(
        mean_S=("separability_ratio", "mean"),
        mean_euc=("euclidean_rare_f1", "mean"),
        mean_mahal=("mahal_pooled_rare_f1", "mean"),
        mean_delta=("delta_mahal_minus_euc", "mean"),
        n_runs=("run", "count"),
    ).reset_index()
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    write_table(summary, OUT_DIR / "summary_by_dataset.csv")

    return df


if __name__ == "__main__":
    main()
