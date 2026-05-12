"""E17: Mahal-pooled integrated into main pipeline (prototype stage replacement).

Replace Euclidean with Mahal-pooled in the FULL pipeline (prototype → gate → marker).

Steps:
1. Load existing scANVI embeddings
2. Compute Mahal-pooled distances (instead of Euclidean)
3. Use Mahal ranks to identify rank-1 candidates
4. Apply existing marker verification (load from gate_marker/ directory)
5. Compare: scANVI, current full pipeline (Euclidean+gate+marker), Mahal full pipeline

Run on: cDC1 rare5, ASDC rare5, epsilon rare20, NCM rare20, ILC rare20 (seed42).
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e17_mahal_main_pipeline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",                          "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",                          "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20",                        "epsilon"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20",   "non-classical monocyte"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",      "innate lymphoid cell"),
]


def _apply_marker_gate(
    test_pred: pd.DataFrame,
    test_scored: pd.DataFrame,
    selected_thresholds: pd.DataFrame,
    prototype_pred: np.ndarray,
    rare_class: str,
    classes: list[str],
) -> np.ndarray:
    """Apply marker verification gate to prototype predictions.

    Cells predicted as rare_class by prototype are kept only if marker_score >= threshold.
    Others fall back to scANVI predicted_label.
    """
    threshold = float(selected_thresholds["selected_marker_threshold"].iloc[0])

    # Align test_scored with test_pred by index
    # test_scored has marker_score_<rare_class> column
    rare_col = f"marker_score_{rare_class}"
    if rare_col not in test_scored.columns:
        # Try to find any marker_score column
        marker_cols = [c for c in test_scored.columns if c.startswith("marker_score_")]
        if marker_cols:
            rare_col = marker_cols[0]
        else:
            print(f"  WARNING: no marker_score column found, skipping gate")
            return prototype_pred

    marker_scores = test_scored[rare_col].to_numpy()

    # Apply gate: if prototype says rare_class but marker_score < threshold → revert to scANVI
    final_pred = prototype_pred.copy()
    scanvi_pred = test_pred["predicted_label"].astype(str).to_numpy()

    for i, pred in enumerate(prototype_pred):
        if pred == rare_class:
            if i < len(marker_scores) and marker_scores[i] < threshold:
                final_pred[i] = scanvi_pred[i]

    return final_pred


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    gate_marker_dir = run_dir / "gate_marker"
    prototype_dir = run_dir / "prototype"

    if not emb_dir.exists():
        print(f"  WARNING: {emb_dir} not found, skipping.")
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  WARNING: missing embeddings {e}, skipping {run_dir}")
        return None

    # Load gate_marker files if available
    has_gate_marker = False
    test_scored = None
    selected_thresholds = None
    if gate_marker_dir.exists():
        try:
            test_scored = read_table(gate_marker_dir / "test_scored.csv")
            selected_thresholds = read_table(gate_marker_dir / "selected_thresholds.csv")
            has_gate_marker = True
        except FileNotFoundError:
            print(f"  WARNING: gate_marker files missing for {run_dir.name}")

    # Load existing Euclidean prototype scores for comparison
    euc_proto_pred = None
    if prototype_dir.exists():
        try:
            proto_scores = read_table(prototype_dir / "test_scores.csv")
            # Euclidean prototype prediction: rank-1 = prototype_rank_{rare_class} == 1
            rare_rank_col = f"prototype_rank_{rare_class}"
            if rare_rank_col in proto_scores.columns:
                euc_proto_pred = np.where(
                    proto_scores[rare_rank_col].to_numpy() == 1,
                    rare_class,
                    test_pred["predicted_label"].astype(str).to_numpy()
                )
        except Exception:
            pass

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    # ── Euclidean nearest-prototype (no gate) ──────────────────────────────
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # ── Mahal-pooled nearest-prototype (no gate) ───────────────────────────
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # ── Euclidean + gate + marker ──────────────────────────────────────────
    euc_gate_m = None
    if has_gate_marker:
        euc_gate_pred = _apply_marker_gate(
            test_pred, test_scored, selected_thresholds, euc_pred, rare_class, classes
        )
        euc_gate_m, _ = classification_tables(y_test, pd.Series(euc_gate_pred), rare_class=rare_class)

    # ── Mahal + gate + marker ──────────────────────────────────────────────
    mahal_gate_m = None
    if has_gate_marker:
        mahal_gate_pred = _apply_marker_gate(
            test_pred, test_scored, selected_thresholds, mahal_pred, rare_class, classes
        )
        mahal_gate_m, _ = classification_tables(y_test, pd.Series(mahal_gate_pred), rare_class=rare_class)

    # ── scANVI baseline ────────────────────────────────────────────────────
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    result = {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_no_gate_f1": euc_m["rare_f1"],
        "mahal_no_gate_f1": mahal_m["rare_f1"],
        "euclidean_gate_marker_f1": euc_gate_m["rare_f1"] if euc_gate_m else float("nan"),
        "mahal_gate_marker_f1": mahal_gate_m["rare_f1"] if mahal_gate_m else float("nan"),
        "has_gate_marker": has_gate_marker,
    }

    print(f"  {run_dir.name}: scanvi={scanvi_m['rare_f1']:.3f}  "
          f"euc_no_gate={euc_m['rare_f1']:.3f}  mahal_no_gate={mahal_m['rare_f1']:.3f}  "
          f"euc+gate={result['euclidean_gate_marker_f1']:.3f}  "
          f"mahal+gate={result['mahal_gate_marker_f1']:.3f}")

    return result


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        try:
            result = run_one(run_dir, rare_class)
            if result:
                rows.append(result)
        except Exception as exc:
            import traceback
            print(f"  ERROR in {run_dir.name}: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E17 Results: Mahal Full Pipeline ===")
    cols = ["run", "rare_class", "n_rare_train",
            "scanvi_rare_f1", "euclidean_no_gate_f1", "mahal_no_gate_f1",
            "euclidean_gate_marker_f1", "mahal_gate_marker_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
