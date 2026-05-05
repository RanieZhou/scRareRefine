from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from scrare.data.loading import adata_from_config
from scrare.data.preprocess import ensure_unique_names
from scrare.evaluation.posthoc import evaluate_four_stage_methods
from scrare.infra.io import read_table, write_table
from scrare.infra.paths import artifact_path, stage_table_path


def _csv_values(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_class_name(name: str) -> str:
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def _output_root(config: dict[str, Any], *, rare_class: str, split_mode: str) -> Path:
    dataset_name = config["dataset"].get("name", "dataset")
    split_name = "inductive_cell" if split_mode == "cell_stratified" else "inductive_batch"
    return Path("outputs") / dataset_name / split_name / _safe_class_name(rare_class)


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


def run_posthoc_workflow(config: dict[str, Any], args: argparse.Namespace) -> None:
    adata = adata_from_config(config)
    ensure_unique_names(adata)

    default_rare = [config["experiment"]["rare_class"]]
    rare_classes = _csv_values(getattr(args, "rare_class", None), default_rare)
    split_modes = _csv_values(getattr(args, "split_mode", None), ["batch_heldout"])

    for rare_class in rare_classes:
        for split_mode in split_modes:
            root = _output_root(config, rare_class=rare_class, split_mode=split_mode)
            effect_runs_parts: list[pd.DataFrame] = []
            effect_summary_parts: list[pd.DataFrame] = []
            threshold_parts: list[pd.DataFrame] = []
            selected_parts: list[pd.DataFrame] = []
            candidate_parts: list[pd.DataFrame] = []
            verified_parts: list[pd.DataFrame] = []
            fusion_grid_parts: list[pd.DataFrame] = []

            for run_dir in _run_dirs(root):
                train_pred = read_table(artifact_path(run_dir, "train_predictions.csv")).reset_index(drop=True)
                val_pred = read_table(artifact_path(run_dir, "validation_predictions.csv")).reset_index(drop=True)
                test_pred = read_table(artifact_path(run_dir, "test_predictions.csv")).reset_index(drop=True)
                train_latent = read_table(artifact_path(run_dir, "train_latent.csv")).reset_index(drop=True)
                val_latent = read_table(artifact_path(run_dir, "validation_latent.csv")).reset_index(drop=True)
                test_latent = read_table(artifact_path(run_dir, "test_latent.csv")).reset_index(drop=True)
                genes = read_table(run_dir / "selected_hvg_genes.csv")["gene"].astype(str).tolist()

                seed = int(test_pred["seed"].iloc[0])
                rare_train_size = str(test_pred["rare_train_size"].iloc[0])
                outputs = evaluate_four_stage_methods(
                    adata,
                    train_pred=train_pred,
                    val_pred=val_pred,
                    test_pred=test_pred,
                    train_latent=train_latent,
                    val_latent=val_latent,
                    test_latent=test_latent,
                    genes=genes,
                    rare_class=rare_class,
                    split_mode=split_mode,
                    seed=seed,
                    rare_train_size=rare_train_size,
                    run=run_dir.name,
                    max_false_rescue_rate=args.max_false_rescue_rate,
                    top_n=args.top_n,
                    min_cells=args.min_cells,
                )
                effect_runs_parts.append(outputs["effect_runs"])
                effect_summary_parts.append(outputs["effect_summary"])
                threshold_parts.append(outputs["threshold_curve"])
                selected_parts.append(outputs["selected_thresholds"])
                candidate_parts.append(outputs["prototype_candidates"])
                verified_parts.append(outputs["marker_verified_candidates"])
                fusion_grid_parts.append(outputs["fusion_grid"])

            if not effect_runs_parts:
                raise FileNotFoundError(f"No inductive runs found under {root / 'runs'}")

            stage = "posthoc"
            effect_runs = pd.concat(effect_runs_parts, ignore_index=True)
            effect_summary = pd.concat(effect_summary_parts, ignore_index=True)
            write_table(effect_runs, stage_table_path(root, stage, "four_stage_effect_runs.csv"))
            write_table(effect_summary, stage_table_path(root, stage, "four_stage_effect_summary.csv"))
            write_table(
                pd.concat(threshold_parts, ignore_index=True) if any(not df.empty for df in threshold_parts) else pd.DataFrame(),
                stage_table_path(root, stage, "validation_marker_threshold_curve.csv"),
            )
            write_table(
                pd.concat(selected_parts, ignore_index=True) if any(not df.empty for df in selected_parts) else pd.DataFrame(),
                stage_table_path(root, stage, "selected_marker_thresholds.csv"),
            )
            write_table(
                pd.concat(candidate_parts, ignore_index=True) if any(not df.empty for df in candidate_parts) else pd.DataFrame(),
                stage_table_path(root, stage, "prototype_rank1_test_candidates.csv"),
            )
            write_table(
                pd.concat(verified_parts, ignore_index=True) if any(not df.empty for df in verified_parts) else pd.DataFrame(),
                stage_table_path(root, stage, "marker_verified_test_candidates.csv"),
            )
            write_table(
                pd.concat(fusion_grid_parts, ignore_index=True) if any(not df.empty for df in fusion_grid_parts) else pd.DataFrame(),
                stage_table_path(root, stage, "fusion_validation_grid.csv"),
            )
