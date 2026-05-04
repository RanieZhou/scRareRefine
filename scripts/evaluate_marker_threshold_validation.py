from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scrare_refine.config import load_config, output_dir
from scrare_refine.io import read_table, write_table
from scrare_refine.marker_verifier import (
    choose_marker_threshold,
    default_marker_thresholds,
    evaluate_threshold_rescue,
    marker_threshold_curve,
    stratified_validation_mask,
)
from scrare_refine.metrics import classification_tables
from scrare_refine.output_layout import existing_table_path, stage_table_path


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


def _parse_alphas(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    return out


def _align_candidates_to_predictions(candidates: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    if "cell_id" not in candidates.columns or "cell_id" not in predictions.columns:
        raise ValueError("Both candidates and predictions must contain cell_id for alignment")
    cell_to_index = pd.Series(predictions.index, index=predictions["cell_id"].astype(str))
    aligned = candidates[candidates["cell_id"].astype(str).isin(cell_to_index.index)].copy()
    aligned.index = aligned["cell_id"].astype(str).map(cell_to_index).astype(int)
    return aligned.sort_index()


def _baseline_metrics(predictions: pd.DataFrame, rare_class: str) -> dict[str, float]:
    overall, _ = classification_tables(
        predictions["true_label"].astype(str),
        predictions["predicted_label"].astype(str),
        rare_class=rare_class,
    )
    overall.update(
        {
            "marker_threshold": np.nan,
            "n_candidates": 0,
            "n_marker_verified": 0,
            "rescued_rare_errors": 0,
            "false_rescues": 0,
            "candidate_precision_for_rare_error": 0.0,
            "rare_error_recall": 0.0,
            "modification_rate": 0.0,
            "major_to_rare_false_rescue_rate": 0.0,
        }
    )
    return overall


def _summarize(effect_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_accuracy",
        "macro_f1",
        "rare_precision",
        "rare_recall",
        "rare_f1",
        "n_candidates",
        "n_marker_verified",
        "rescued_rare_errors",
        "false_rescues",
        "candidate_precision_for_rare_error",
        "rare_error_recall",
        "modification_rate",
        "major_to_rare_false_rescue_rate",
    ]
    return (
        effect_runs.groupby(["rare_train_size", "method", "max_false_rescue_rate"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    flat = _flatten_summary(summary)
    plot = flat[flat["max_false_rescue_rate"].isna() | flat["max_false_rescue_rate"].eq(0.001)].copy()
    order = ["20", "50", "100", "all"]
    methods = [
        "scANVI baseline test",
        "prototype rank1 gate test",
        "validation-tuned marker test",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["rare_f1_mean", "rare_recall_mean", "major_to_rare_false_rescue_rate_mean"],
        ["ASDC F1", "ASDC recall", "False rescue rate"],
    ):
        for method in methods:
            sub = plot[plot["method"].eq(method)].copy()
            sub["rare_train_size"] = pd.Categorical(sub["rare_train_size"].astype(str), categories=order, ordered=True)
            sub = sub.sort_values("rare_train_size")
            ax.plot(sub["rare_train_size"].astype(str), sub[metric], marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("ASDC train size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[-1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate validation-tuned marker thresholds on held-out cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", default="rank1")
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument("--alphas", default="0.0005,0.001,0.002,0.005")
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_dir(config)
    rare_class = config["experiment"]["rare_class"]
    stage = "marker_threshold_validation"
    alphas = _parse_alphas(args.alphas)

    candidate_path = stage_table_path(root, "marker_verifier", "marker_candidate_scores.csv")
    candidate_scores = read_table(candidate_path)
    candidate_scores["rare_train_size"] = candidate_scores["rare_train_size"].astype(str)

    effect_rows = []
    threshold_rows = []
    selected_rows = []

    for run_dir in _run_dirs(root):
        pred = read_table(existing_table_path(run_dir, "scanvi_predictions.parquet")).reset_index(drop=True)
        seed = int(pred["seed"].iloc[0])
        rare_train_size = str(pred["rare_train_size"].iloc[0])
        run_name = run_dir.name

        scored = candidate_scores[
            candidate_scores["run"].astype(str).eq(run_name)
            & candidate_scores["gate_name"].astype(str).eq(args.gate)
        ].copy()
        if scored.empty:
            continue
        scored = _align_candidates_to_predictions(scored, pred)

        validation_mask = stratified_validation_mask(
            pred["true_label"],
            seed=seed,
            validation_fraction=args.validation_fraction,
        )
        test_mask = ~validation_mask
        val_pred = pred.loc[validation_mask]
        test_pred = pred.loc[test_mask]
        val_candidates = scored.loc[scored.index.intersection(val_pred.index)]
        test_candidates = scored.loc[scored.index.intersection(test_pred.index)]
        if val_candidates.empty or test_candidates.empty:
            continue

        base = _baseline_metrics(test_pred, rare_class)
        base.update(
            {
                "seed": seed,
                "rare_train_size": rare_train_size,
                "run": run_name,
                "gate_name": args.gate,
                "method": "scANVI baseline test",
                "max_false_rescue_rate": np.nan,
                "n_validation_cells": int(validation_mask.sum()),
                "n_test_cells": int(test_mask.sum()),
            }
        )
        effect_rows.append(base)

        gate = evaluate_threshold_rescue(test_pred, test_candidates, rare_class=rare_class, marker_threshold=None)
        gate.update(
            {
                "seed": seed,
                "rare_train_size": rare_train_size,
                "run": run_name,
                "gate_name": args.gate,
                "method": f"prototype {args.gate} gate test",
                "max_false_rescue_rate": np.nan,
                "n_validation_cells": int(validation_mask.sum()),
                "n_test_cells": int(test_mask.sum()),
            }
        )
        effect_rows.append(gate)

        thresholds = default_marker_thresholds(val_candidates)
        val_curve = marker_threshold_curve(val_pred, val_candidates, rare_class=rare_class, thresholds=thresholds)
        val_curve.insert(0, "seed", seed)
        val_curve.insert(1, "rare_train_size", rare_train_size)
        val_curve.insert(2, "run", run_name)
        val_curve.insert(3, "gate_name", args.gate)
        val_curve.insert(4, "split", "validation")
        threshold_rows.append(val_curve)

        for alpha in alphas:
            threshold = choose_marker_threshold(val_curve, max_false_rescue_rate=alpha)
            selected_rows.append(
                {
                    "seed": seed,
                    "rare_train_size": rare_train_size,
                    "run": run_name,
                    "gate_name": args.gate,
                    "max_false_rescue_rate": alpha,
                    "selected_marker_threshold": threshold,
                }
            )
            tuned = evaluate_threshold_rescue(
                test_pred,
                test_candidates,
                rare_class=rare_class,
                marker_threshold=threshold,
            )
            tuned.update(
                {
                    "seed": seed,
                    "rare_train_size": rare_train_size,
                    "run": run_name,
                    "gate_name": args.gate,
                    "method": "validation-tuned marker test",
                    "max_false_rescue_rate": alpha,
                    "n_validation_cells": int(validation_mask.sum()),
                    "n_test_cells": int(test_mask.sum()),
                }
            )
            effect_rows.append(tuned)

    effect_runs = pd.DataFrame(effect_rows)
    threshold_curve = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    selected = pd.DataFrame(selected_rows)
    summary = _summarize(effect_runs) if not effect_runs.empty else pd.DataFrame()

    write_table(effect_runs, stage_table_path(root, stage, "validation_tuned_effect_runs.csv"))
    write_table(_flatten_summary(summary), stage_table_path(root, stage, "validation_tuned_effect_summary.csv"))
    write_table(threshold_curve, stage_table_path(root, stage, "validation_threshold_curve_runs.csv"))
    write_table(selected, stage_table_path(root, stage, "selected_validation_thresholds.csv"))
    if not summary.empty:
        _plot_summary(summary, stage_table_path(root, stage, "validation_tuned_marker_comparison.png"))

    print(f"Wrote validation-tuned marker outputs to {root / 'stages' / stage}")
    print(f"Gate evaluated: {args.gate}")
    print(f"Alpha constraints: {', '.join(str(alpha) for alpha in alphas)}")


if __name__ == "__main__":
    main()
