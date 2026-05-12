"""E6: Combined pipeline — best distance → existing gate+marker logic.

Take the best distance metric from E1-E5 (Mahal-pooled + adaptive-λ) and plug
it into the existing gate+marker logic:
1. Use Mahal-pooled distances to identify rank-1 candidates (instead of Euclidean)
2. Apply the existing marker verification from src/05_prototype_gate_marker.py
3. Compare: current method vs new distance + same marker gate

This tests whether the distance improvement compounds with the marker gate.

Run on: cDC1 rare5, epsilon rare20, NCM rare20 (seed42).

Usage:
    python src/experimental/e6_combined_pipeline.py
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e6_combined_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
]


def compute_mahal_ranks(
    query_z: np.ndarray,
    train_z: np.ndarray,
    train_labels: pd.Series,
    is_labeled: np.ndarray,
    rare_class: str,
) -> np.ndarray:
    """Compute rank of rare class under Mahal-pooled distance for each query cell."""
    classes, protos, counts_map = _class_prototypes(train_z, train_labels, is_labeled)
    pooled = _pooled_covariance_shrunk(train_z, train_labels, is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    dists = _mahalanobis(query_z, protos, pooled_covs)
    # Rank of rare class (1 = nearest)
    rare_idx = classes.index(rare_class)
    ranks = np.argsort(np.argsort(dists, axis=1), axis=1)[:, rare_idx] + 1
    return ranks, classes, protos, counts_map


def apply_gate_marker_logic(
    predictions: pd.DataFrame,
    ranks: np.ndarray,
    rare_class: str,
    *,
    gate_marker_dir: Path,
) -> pd.Series:
    """Apply the existing gate+marker logic using new distance-based ranks.
    
    Reads the existing gate_marker outputs (marker scores) and applies them
    with the new rank-1 candidates from Mahal distance.
    """
    # Load existing marker scores from gate_marker directory
    test_scored_path = gate_marker_dir / "test_scored.csv"
    threshold_path   = gate_marker_dir / "selected_thresholds.csv"

    if not test_scored_path.exists() or not threshold_path.exists():
        return None

    test_scored = read_table(test_scored_path)
    thresholds  = read_table(threshold_path)
    selected_threshold = float(thresholds["selected_marker_threshold"].iloc[0])

    # Start from scANVI predictions
    y_pred = predictions["predicted_label"].astype(str).copy()

    # New rank-1 candidates: cells not predicted as rare_class AND rank <= 1
    new_candidates_mask = (y_pred != rare_class) & (ranks <= 1)
    new_candidate_ids = set(predictions.loc[new_candidates_mask, "cell_id"].astype(str))

    # Apply marker verification: only rescue if cell_id is in test_scored AND marker_margin >= threshold
    if "cell_id" in test_scored.columns and "marker_margin" in test_scored.columns:
        verified_ids = set(
            test_scored.loc[
                test_scored["cell_id"].astype(str).isin(new_candidate_ids) &
                (pd.to_numeric(test_scored["marker_margin"], errors="coerce") >= selected_threshold),
                "cell_id"
            ].astype(str)
        )
        y_pred.loc[predictions["cell_id"].astype(str).isin(verified_ids)] = rare_class

    return y_pred


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir      = run_dir / "embeddings"
    gate_marker_dir = run_dir / "gate_marker"

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

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Euclidean nearest-prototype (current method, no gate)
    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled nearest-prototype (new distance, no gate)
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # Mahal ranks for gate+marker
    rare_idx = classes.index(rare_class) if rare_class in classes else -1
    if rare_idx >= 0:
        mahal_ranks = np.argsort(np.argsort(mahal_dists, axis=1), axis=1)[:, rare_idx] + 1
    else:
        mahal_ranks = np.ones(len(test_z), dtype=int) * 999

    # Current method: scANVI + existing gate+marker
    current_gate_marker_f1 = float("nan")
    if gate_marker_dir.exists():
        try:
            test_scored_path = gate_marker_dir / "test_scored.csv"
            threshold_path   = gate_marker_dir / "selected_thresholds.csv"
            if test_scored_path.exists() and threshold_path.exists():
                test_scored = read_table(test_scored_path)
                thresholds  = read_table(threshold_path)
                selected_threshold = float(thresholds["selected_marker_threshold"].iloc[0])

                # Load prototype scores for current method
                proto_dir = run_dir / "prototype"
                if (proto_dir / "test_scores.csv").exists():
                    proto_scores = read_table(proto_dir / "test_scores.csv")
                    rank_col = f"prototype_rank_{rare_class}"
                    if rank_col in proto_scores.columns:
                        current_ranks = proto_scores[rank_col].to_numpy()
                        current_candidates_mask = (
                            test_pred["predicted_label"].astype(str) != rare_class
                        ) & (current_ranks <= 1)
                        current_candidate_ids = set(
                            test_pred.loc[current_candidates_mask, "cell_id"].astype(str)
                        )
                        if "cell_id" in test_scored.columns and "marker_margin" in test_scored.columns:
                            verified_ids = set(
                                test_scored.loc[
                                    test_scored["cell_id"].astype(str).isin(current_candidate_ids) &
                                    (pd.to_numeric(test_scored["marker_margin"], errors="coerce") >= selected_threshold),
                                    "cell_id"
                                ].astype(str)
                            )
                            y_current = test_pred["predicted_label"].astype(str).copy()
                            y_current.loc[test_pred["cell_id"].astype(str).isin(verified_ids)] = rare_class
                            current_m, _ = classification_tables(y_test, y_current, rare_class=rare_class)
                            current_gate_marker_f1 = current_m["rare_f1"]
        except Exception as ex:
            print(f"    WARNING: could not load gate_marker results: {ex}")

    # New method: Mahal distance + existing marker gate
    new_gate_marker_f1 = float("nan")
    if gate_marker_dir.exists():
        try:
            new_pred = apply_gate_marker_logic(
                test_pred, mahal_ranks, rare_class, gate_marker_dir=gate_marker_dir
            )
            if new_pred is not None:
                new_m, _ = classification_tables(y_test, new_pred, rare_class=rare_class)
                new_gate_marker_f1 = new_m["rare_f1"]
        except Exception as ex:
            print(f"    WARNING: could not apply new gate+marker: {ex}")

    print(f"    scANVI:                  rare_f1={scanvi_m['rare_f1']:.3f}")
    print(f"    Euclidean (no gate):     rare_f1={euc_m['rare_f1']:.3f}")
    print(f"    Mahal-pool (no gate):    rare_f1={mahal_m['rare_f1']:.3f}")
    print(f"    Current gate+marker:     rare_f1={current_gate_marker_f1:.3f}")
    print(f"    Mahal+gate+marker (new): rare_f1={new_gate_marker_f1:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "test_rare_f1_scanvi": scanvi_m["rare_f1"],
        "test_rare_f1_euclidean_no_gate": euc_m["rare_f1"],
        "test_rare_f1_mahal_no_gate": mahal_m["rare_f1"],
        "test_rare_f1_current_gate_marker": current_gate_marker_f1,
        "test_rare_f1_mahal_gate_marker": new_gate_marker_f1,
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
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E6 Results (combined pipeline, seed42) ===")
    cols = ["run", "rare_class",
            "test_rare_f1_scanvi",
            "test_rare_f1_euclidean_no_gate",
            "test_rare_f1_mahal_no_gate",
            "test_rare_f1_current_gate_marker",
            "test_rare_f1_mahal_gate_marker"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
