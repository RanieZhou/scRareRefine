"""Evaluate confidence-aware probability fusion across all P0 runs.

For each run:
1. Extract scANVI probabilities and latent embeddings.
2. Split cells into validation / test via stratified mask.
3. Grid-search (temperature, alpha_min) on validation set.
4. Select best params subject to accuracy & false-rescue constraints.
5. Report test-set performance with selected params.

Outputs comparison tables and a summary plot to
  outputs/<experiment>/stages/fusion/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scrare_refine.config import load_config, output_dir
from scrare_refine.fusion import (
    fuse_and_evaluate,
    prototype_probabilities,
    select_best_params,
)
from scrare_refine.io import read_table, write_table
from scrare_refine.marker_verifier import stratified_validation_mask
from scrare_refine.metrics import classification_tables
from scrare_refine.output_layout import existing_table_path, stage_table_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.exists() else []


def _extract_scanvi_probs(pred: pd.DataFrame) -> pd.DataFrame:
    """Extract probability columns and strip the 'prob_' prefix."""
    prob_cols = [c for c in pred.columns if c.startswith("prob_")]
    return pred[prob_cols].rename(columns=lambda c: c.removeprefix("prob_"))


def _baseline_metrics(pred: pd.DataFrame, rare_class: str) -> dict[str, float]:
    overall, _ = classification_tables(
        pred["true_label"].astype(str),
        pred["predicted_label"].astype(str),
        rare_class=rare_class,
    )
    overall.update({
        "n_changed": 0, "modification_rate": 0.0,
        "rescued_rare_errors": 0, "false_rescues": 0, "damaged_correct": 0,
        "rare_error_recall": 0.0, "major_to_rare_false_rescue_rate": 0.0,
    })
    return overall


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in out.columns
    ]
    return out


# ------------------------------------------------------------------
# Default parameter grids
# ------------------------------------------------------------------

DEFAULT_TEMPERATURES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_ALPHA_MINS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_BETAS = [0.0, 0.1, 0.3, 0.5, 1.0]


# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

def _plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    flat = _flatten_summary(summary)
    order = ["20", "50", "100", "all"]
    methods = sorted(flat["method"].unique())

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
    metric_info = [
        ("rare_f1_mean", "Rare F1"),
        ("rare_recall_mean", "Rare recall"),
        ("rare_precision_mean", "Rare precision"),
        ("major_to_rare_false_rescue_rate_mean", "False rescue rate"),
    ]
    for ax, (metric, title) in zip(axes, metric_info):
        if metric not in flat.columns:
            continue
        for method in methods:
            sub = flat[flat["method"].eq(method)].copy()
            sub["rare_train_size"] = pd.Categorical(
                sub["rare_train_size"].astype(str), categories=order, ordered=True,
            )
            sub = sub.sort_values("rare_train_size")
            ax.plot(sub["rare_train_size"].astype(str), sub[metric], marker="o", label=method)
        ax.set_title(title)
        ax.set_xlabel("Rare train size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score")
    axes[-1].legend(loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate confidence-aware probability fusion on P0 runs.",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.5)
    parser.add_argument(
        "--max-accuracy-drop", type=float, default=0.01,
        help="Maximum allowed accuracy drop vs baseline when selecting params.",
    )
    parser.add_argument(
        "--max-false-rescue-rate", type=float, default=0.01,
        help="Maximum allowed false rescue rate when selecting params.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_dir(config)
    rare_class = config["experiment"]["rare_class"]
    stage = "fusion"

    temperatures = DEFAULT_TEMPERATURES
    alpha_mins = DEFAULT_ALPHA_MINS
    betas = DEFAULT_BETAS

    effect_rows: list[dict] = []
    grid_rows: list[dict] = []
    selected_rows: list[dict] = []

    for run_dir in _run_dirs(root):
        pred = read_table(existing_table_path(run_dir, "scanvi_predictions.parquet"))
        latent_df = read_table(existing_table_path(run_dir, "scanvi_latent.csv"))
        pred = pred.reset_index(drop=True)
        latent_df = latent_df.reset_index(drop=True)

        seed = int(pred["seed"].iloc[0])
        rare_train_size = str(pred["rare_train_size"].iloc[0])
        run_name = run_dir.name

        # Extract probability columns from predictions
        p_scanvi = _extract_scanvi_probs(pred)

        # Latent columns
        latent_cols = [c for c in latent_df.columns if c.startswith("latent_")]
        latent = latent_df[latent_cols].to_numpy()

        # Validation / test split
        val_mask = stratified_validation_mask(
            pred["true_label"], seed=seed,
            validation_fraction=args.validation_fraction,
        )
        test_mask = ~val_mask

        # Baseline metrics on test set
        test_pred = pred.loc[test_mask].reset_index(drop=True)
        baseline_test = _baseline_metrics(test_pred, rare_class)
        baseline_test.update({
            "seed": seed, "rare_train_size": rare_train_size,
            "run": run_name, "method": "scANVI baseline",
            "temperature": np.nan, "alpha_min": np.nan,
        })
        effect_rows.append(baseline_test)
        baseline_accuracy = baseline_test["overall_accuracy"]

        # Grid search over (temperature, alpha_min)
        val_results_for_run: list[dict] = []
        for temperature in temperatures:
            # Compute prototype probabilities once per temperature
            p_proto = prototype_probabilities(
                latent, labels=pred["true_label"],
                is_labeled=pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
                temperature=temperature,
            )

            for alpha_min in alpha_mins:
              for beta in betas:
                # --- Validation ---
                val_idx = val_mask[val_mask].index
                val_result = fuse_and_evaluate(
                    p_scanvi.loc[val_idx].reset_index(drop=True),
                    p_proto.loc[val_idx].reset_index(drop=True),
                    margin=pred.loc[val_idx, "margin"].to_numpy(),
                    y_true=pred.loc[val_idx, "true_label"].reset_index(drop=True),
                    baseline_pred=pred.loc[val_idx, "predicted_label"].reset_index(drop=True),
                    rare_class=rare_class,
                    temperature=temperature,
                    alpha_min=alpha_min,
                    beta=beta,
                )
                val_result.update({
                    "seed": seed, "rare_train_size": rare_train_size,
                    "run": run_name, "split": "validation",
                })
                val_results_for_run.append(val_result)
                grid_rows.append(val_result)

                # --- Test ---
                test_idx = test_mask[test_mask].index
                test_result = fuse_and_evaluate(
                    p_scanvi.loc[test_idx].reset_index(drop=True),
                    p_proto.loc[test_idx].reset_index(drop=True),
                    margin=pred.loc[test_idx, "margin"].to_numpy(),
                    y_true=pred.loc[test_idx, "true_label"].reset_index(drop=True),
                    baseline_pred=pred.loc[test_idx, "predicted_label"].reset_index(drop=True),
                    rare_class=rare_class,
                    temperature=temperature,
                    alpha_min=alpha_min,
                    beta=beta,
                )
                test_result.update({
                    "seed": seed, "rare_train_size": rare_train_size,
                    "run": run_name, "split": "test",
                })
                grid_rows.append(test_result)

        # Select best params from validation
        val_df = pd.DataFrame(val_results_for_run)
        best_temp, best_alpha, best_beta = select_best_params(
            val_df,
            baseline_accuracy=baseline_accuracy,
            max_accuracy_drop=args.max_accuracy_drop,
            max_false_rescue_rate=args.max_false_rescue_rate,
        )
        selected_rows.append({
            "seed": seed, "rare_train_size": rare_train_size,
            "run": run_name, "selected_temperature": best_temp,
            "selected_alpha_min": best_alpha,
            "selected_beta": best_beta,
        })

        # Report test performance with selected params
        p_proto_best = prototype_probabilities(
            latent, labels=pred["true_label"],
            is_labeled=pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            temperature=best_temp,
        )
        test_idx = test_mask[test_mask].index
        test_best = fuse_and_evaluate(
            p_scanvi.loc[test_idx].reset_index(drop=True),
            p_proto_best.loc[test_idx].reset_index(drop=True),
            margin=pred.loc[test_idx, "margin"].to_numpy(),
            y_true=pred.loc[test_idx, "true_label"].reset_index(drop=True),
            baseline_pred=pred.loc[test_idx, "predicted_label"].reset_index(drop=True),
            rare_class=rare_class,
            temperature=best_temp,
            alpha_min=best_alpha,
            beta=best_beta,
        )
        test_best.update({
            "seed": seed, "rare_train_size": rare_train_size,
            "run": run_name, "method": "fusion (validation-tuned)",
        })
        effect_rows.append(test_best)

        print(
            f"  {run_name}: best τ={best_temp}, α_min={best_alpha}, β={best_beta}  |  "
            f"baseline rare_f1={baseline_test['rare_f1']:.3f}  "
            f"fusion rare_f1={test_best['rare_f1']:.3f}"
        )

    if not effect_rows:
        raise FileNotFoundError(f"No runs found under {root / 'runs'}")

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    effect_df = pd.DataFrame(effect_rows)
    grid_df = pd.DataFrame(grid_rows)
    selected_df = pd.DataFrame(selected_rows)

    metrics_for_summary = [
        "overall_accuracy", "macro_f1",
        "rare_precision", "rare_recall", "rare_f1",
        "n_changed", "modification_rate",
        "rescued_rare_errors", "false_rescues", "damaged_correct",
        "rare_error_recall", "major_to_rare_false_rescue_rate",
    ]
    summary = (
        effect_df.groupby(["rare_train_size", "method"], dropna=False)[metrics_for_summary]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    write_table(effect_df, stage_table_path(root, stage, "fusion_effect_runs.csv"))
    write_table(_flatten_summary(summary), stage_table_path(root, stage, "fusion_effect_summary.csv"))
    write_table(grid_df, stage_table_path(root, stage, "fusion_grid_search.csv"))
    write_table(selected_df, stage_table_path(root, stage, "selected_fusion_params.csv"))

    _plot_comparison(
        summary,
        stage_table_path(root, stage, "fusion_vs_baseline.png"),
    )

    print(f"\nWrote fusion outputs to {root / 'stages' / stage}")


if __name__ == "__main__":
    main()
