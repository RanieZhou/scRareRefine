"""Stage 7: Compile final evaluation table comparing all methods on test set.

Reads all stage outputs for one run and produces a single comparison table.

Writes:
    outputs/{dataset}/{run_id}/metrics/
        final_metrics.csv   one row per method: baseline, prototype_gate, gate_marker, fusion

Usage:
    python src/07_evaluate.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import classification_tables, load_config, make_run_dir, parse_rare_train_size, read_table, write_table


def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _baseline_row(pred: pd.DataFrame, *, rare_class: str) -> dict:
    overall, _ = classification_tables(pred["true_label"], pred["predicted_label"], rare_class=rare_class)
    return {**overall, "n_candidates": 0, "n_marker_verified": 0, "rescued_rare_errors": 0,
            "false_rescues": 0, "modification_rate": 0.0, "major_to_rare_false_rescue_rate": 0.0}


def _gate_row(pred: pd.DataFrame, gate_results: pd.DataFrame, *, rare_class: str, gate_name: str = "rank1") -> dict:
    if gate_results.empty:
        return _baseline_row(pred, rare_class=rare_class)
    row = gate_results[gate_results["gate_name"].eq(gate_name)]
    return row.iloc[0].to_dict() if not row.empty else _baseline_row(pred, rare_class=rare_class)


def _marker_row(pred: pd.DataFrame, scored: pd.DataFrame, *, rare_class: str, threshold: float) -> dict:
    if scored.empty or "marker_margin" not in scored.columns or "cell_id" not in scored.columns:
        return _baseline_row(pred, rare_class=rare_class)
    margins = pd.to_numeric(scored["marker_margin"], errors="coerce")
    verified_ids = set(scored.loc[margins.ge(threshold).fillna(False), "cell_id"].astype(str))
    y_true = pred["true_label"].astype(str)
    baseline_pred = pred["predicted_label"].astype(str)
    relabeled = baseline_pred.copy()
    if "cell_id" in pred.columns:
        relabeled.loc[pred["cell_id"].astype(str).isin(verified_ids)] = rare_class
    overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    mask = pred["cell_id"].astype(str).isin(verified_ids) if "cell_id" in pred.columns else pd.Series(False, index=pred.index)
    n_verified = int(mask.sum())
    rescued = int((mask & rare_errors).sum())
    false_rescues = int((mask & non_rare).sum())
    return {**overall, "n_candidates": len(scored), "n_marker_verified": n_verified,
            "rescued_rare_errors": rescued, "false_rescues": false_rescues,
            "modification_rate": n_verified / len(pred) if len(pred) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0}


def _prototype_row(pred: pd.DataFrame, proto_scores: pd.DataFrame, *, rare_class: str) -> dict:
    if proto_scores.empty or "prototype_rescue_candidate" not in proto_scores.columns:
        return _baseline_row(pred, rare_class=rare_class)
    mask = proto_scores["prototype_rescue_candidate"].fillna(False).astype(bool)
    y_true = pred["true_label"].astype(str)
    baseline_pred = pred["predicted_label"].astype(str)
    relabeled = baseline_pred.copy()
    relabeled.loc[mask] = rare_class
    overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    n_candidates = int(mask.sum())
    rescued = int((mask & rare_errors).sum())
    false_rescues = int((mask & non_rare).sum())
    return {**overall, "n_candidates": n_candidates, "n_marker_verified": 0,
            "rescued_rare_errors": rescued, "false_rescues": false_rescues,
            "modification_rate": n_candidates / len(pred) if len(pred) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: compile final evaluation table")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", choices=["batch_heldout", "cell_stratified"])
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)

    test_pred = read_table(run_dir / "embeddings" / "test_predictions.csv")
    proto_scores = _safe_read(run_dir / "prototype" / "test_scores.csv")
    gate_results = _safe_read(run_dir / "gate" / "test_results.csv")
    gate_marker_scored = _safe_read(run_dir / "gate_marker" / "test_scored.csv")
    threshold_df = _safe_read(run_dir / "gate_marker" / "selected_thresholds.csv")
    fusion_results = _safe_read(run_dir / "fusion" / "test_results.csv")

    selected_threshold = (
        float(threshold_df["selected_marker_threshold"].iloc[0])
        if not threshold_df.empty else float("inf")
    )
    common = {
        "seed": args.seed,
        "rare_class": rare_class,
        "rare_train_size": str(rare_train_size),
        "split_mode": args.split_mode,
    }

    rows = [
        {"method": "baseline",               **common, **_baseline_row(test_pred, rare_class=rare_class)},
        {"method": "prototype",               **common, **_prototype_row(test_pred, proto_scores, rare_class=rare_class)},
        {"method": "prototype_gate",          **common, **_gate_row(test_pred, gate_results, rare_class=rare_class)},
        {"method": "prototype_gate_marker",   **common, **_marker_row(test_pred, gate_marker_scored, rare_class=rare_class, threshold=selected_threshold)},
    ]
    if not fusion_results.empty:
        rows.append({"method": "fusion", **common, **fusion_results.iloc[0].to_dict()})

    final = pd.DataFrame(rows)
    out_dir = run_dir / "metrics"
    write_table(final, out_dir / "final_metrics.csv")

    display_cols = ["method", "rare_f1", "rare_precision", "rare_recall", "overall_accuracy", "major_to_rare_false_rescue_rate"]
    display_cols = [c for c in display_cols if c in final.columns]
    print(final[display_cols].to_string(index=False))
    print(f"\nDone. Final metrics in {out_dir / 'final_metrics.csv'}")


if __name__ == "__main__":
    main()
