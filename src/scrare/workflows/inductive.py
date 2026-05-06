from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import anndata as ad
import pandas as pd

from scrare.data.loading import adata_from_config
from scrare.data.preprocess import ensure_unique_names, select_train_hvg_var_names, subset_cells
from scrare.data.splits import batch_heldout_split, cell_stratified_split, make_inductive_scanvi_labels, parse_rare_train_size
from scrare.evaluation.posthoc import evaluate_four_stage_methods
from scrare.infra.io import read_table, write_table
from scrare.infra.paths import artifact_path, stage_table_path
from scrare.infra.resources import ResourceMonitor
from scrare.models.scanvi import load_query_model, prediction_outputs, seed_everything, train_reference_scanvi
from scrare.visualization.inductive import rebuild_inductive_plots

ALL_METHODS = [
    "baseline",
    "baseline_plus_prototype",
    "baseline_plus_prototype_gate",
    "baseline_plus_prototype_gate_plus_marker",
    "baseline_plus_fusion",
]
BASELINE_ARTIFACTS = [
    "train_predictions.csv",
    "validation_predictions.csv",
    "test_predictions.csv",
    "train_latent.csv",
    "validation_latent.csv",
    "test_latent.csv",
]
BASELINE_RUN_FILES = [
    "selected_hvg_genes.csv",
    "split_assignments.csv",
]
METHOD_OUTPUT_FILES = {
    "effect_runs": "five_method_effect_runs.csv",
    "effect_summary": "five_method_effect_summary.csv",
    "threshold_curve": "validation_marker_threshold_curve.csv",
    "selected_thresholds": "selected_marker_thresholds.csv",
    "prototype_candidates": "prototype_test_candidates.csv",
    "marker_verified_candidates": "marker_verified_test_candidates.csv",
    "fusion_grid": "fusion_validation_grid.csv",
}
RESOURCE_SUMMARY_FILENAME = "resource_summary.csv"


def _csv_values(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_class_name(name: str) -> str:
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def _output_root(config: dict[str, Any], *, rare_class: str, split_mode: str, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    dataset_name = config["dataset"].get("name", "dataset")
    split_name = "inductive_cell" if split_mode == "cell_stratified" else "inductive_batch"
    return Path("outputs") / dataset_name / split_name / _safe_class_name(rare_class)


def _run_name(seed: int, rare_train_size: str | int, split_mode: str) -> str:
    return f"{split_mode}_seed_{seed}_rare_{rare_train_size}"


def _run_values(config: dict[str, Any], seed: int | None, rare_train_size: str | int | None) -> tuple[list[int], list[str | int]]:
    experiment = config["experiment"]
    seeds = [seed] if seed is not None else [int(value) for value in experiment["seeds"]]
    sizes = [rare_train_size] if rare_train_size is not None else list(experiment["rare_train_sizes"])
    return seeds, [parse_rare_train_size(value) for value in sizes]


def _normalize_methods(value: str | None) -> list[str]:
    methods = _csv_values(value, ALL_METHODS)
    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")
    return methods


def _missing_baseline_artifacts(run_dir: Path) -> list[Path]:
    missing = [artifact_path(run_dir, name) for name in BASELINE_ARTIFACTS if not artifact_path(run_dir, name).exists()]
    missing.extend(run_dir / name for name in BASELINE_RUN_FILES if not (run_dir / name).exists())
    return missing


def _require_existing_baseline(run_dir: Path) -> None:
    missing = _missing_baseline_artifacts(run_dir)
    if not missing:
        return
    missing_text = ", ".join(path.name for path in missing)
    raise FileNotFoundError(f"Missing baseline artifacts under {run_dir}: {missing_text}")


def _iter_slices(config: dict[str, Any], args: argparse.Namespace):
    rare_classes = _csv_values(getattr(args, "rare_class", None), [config["experiment"]["rare_class"]])
    split_modes = _csv_values(getattr(args, "split_mode", None), ["batch_heldout"])
    for rare_class in rare_classes:
        seeds, sizes = _run_values(config, getattr(args, "seed", None), getattr(args, "rare_train_size", None))
        for split_mode in split_modes:
            root = _output_root(config, rare_class=rare_class, split_mode=split_mode, output_dir=getattr(args, "output_dir", None))
            for seed in seeds:
                for rare_train_size in sizes:
                    yield rare_class, split_mode, seed, rare_train_size, root


def _split_series(
    adata: ad.AnnData,
    *,
    config: dict[str, Any],
    args: argparse.Namespace,
    split_mode: str,
    seed: int,
) -> pd.Series:
    dataset = config["dataset"]
    label_key = dataset.get("label_key", "label")
    fractions = {
        "train_fraction": float(getattr(args, "train_fraction", 0.70)),
        "validation_fraction": float(getattr(args, "validation_fraction", 0.15)),
        "test_fraction": float(getattr(args, "test_fraction", 0.15)),
    }
    if split_mode == "cell_stratified":
        return cell_stratified_split(adata.obs, label_key=label_key, seed=seed, **fractions)
    if split_mode == "batch_heldout":
        batch_key = dataset.get("batch_key", "batch")
        return batch_heldout_split(adata.obs, label_key=label_key, batch_key=batch_key, seed=seed, **fractions)
    raise ValueError(f"Unknown split_mode: {split_mode}")


def _metadata_columns(
    *,
    split_name: str,
    seed: int,
    rare_train_size: str | int,
    rare_class: str,
    split_mode: str,
    run_name: str,
) -> dict[str, Any]:
    return {
        "split": split_name,
        "seed": seed,
        "rare_train_size": str(rare_train_size),
        "rare_class": rare_class,
        "split_mode": split_mode,
        "run": run_name,
    }


def _add_metadata(frame: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    for key, value in metadata.items():
        out[key] = value
    return out


def _write_baseline_outputs(
    run_dir: Path,
    *,
    split_name: str,
    predictions: pd.DataFrame,
    latent: pd.DataFrame,
) -> None:
    write_table(predictions, artifact_path(run_dir, f"{split_name}_predictions.csv"))
    write_table(latent, artifact_path(run_dir, f"{split_name}_latent.csv"))


def _run_baseline_slice(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    rare_class: str,
    split_mode: str,
    seed: int,
    rare_train_size: str | int,
    root: Path,
    run_dir: Path,
) -> None:
    del root
    dataset = config["dataset"]
    experiment = config["experiment"]
    model = config.get("model", {})
    label_key = dataset.get("label_key", "label")
    batch_key = dataset.get("batch_key", "batch")
    unlabeled_category = experiment.get("unlabeled_category", "Unknown")

    seed_everything(seed)
    adata = adata_from_config(config)
    adata = subset_cells(adata, max_cells=getattr(args, "max_cells", None), seed=seed).copy()
    ensure_unique_names(adata)

    split = _split_series(adata, config=config, args=args, split_mode=split_mode, seed=seed)
    scanvi_label, is_labeled = make_inductive_scanvi_labels(
        adata.obs,
        split,
        label_key=label_key,
        rare_class=rare_class,
        rare_train_size=rare_train_size,
        seed=seed,
        unlabeled_category=unlabeled_category,
    )
    label_categories = pd.Index(pd.unique(adata.obs[label_key].astype(str)))
    if unlabeled_category not in label_categories:
        label_categories = label_categories.append(pd.Index([unlabeled_category]))
    adata.obs["scanvi_label"] = pd.Categorical(scanvi_label.astype(str), categories=label_categories.astype(str).tolist())
    adata.obs["is_labeled_for_scanvi"] = is_labeled

    train_adata = adata[split.eq("train")].copy()
    genes = select_train_hvg_var_names(train_adata, n_top_genes=model.get("n_top_hvg"))
    adata = adata[:, genes].copy()
    train_adata = adata[split.eq("train")].copy()
    validation_adata = adata[split.eq("validation")].copy()
    test_adata = adata[split.eq("test")].copy()

    scanvi_model = train_reference_scanvi(
        train_adata,
        batch_key=batch_key,
        unlabeled_category=unlabeled_category,
        n_latent=int(model.get("n_latent", 30)),
        batch_size=int(model.get("batch_size", 256)),
        scvi_epochs=int(getattr(args, "scvi_epochs", None) or model.get("scvi_max_epochs", 200)),
        scanvi_epochs=int(getattr(args, "scanvi_epochs", None) or model.get("scanvi_max_epochs", 100)),
    )

    run_name = run_dir.name
    train_pred, train_latent = prediction_outputs(scanvi_model, train_adata, label_key, rare_class)
    train_meta = _metadata_columns(
        split_name="train",
        seed=seed,
        rare_train_size=rare_train_size,
        rare_class=rare_class,
        split_mode=split_mode,
        run_name=run_name,
    )
    _write_baseline_outputs(
        run_dir,
        split_name="train",
        predictions=_add_metadata(train_pred, train_meta),
        latent=_add_metadata(train_latent, train_meta),
    )

    query_label_categories = train_adata.obs["scanvi_label"].cat.categories.astype(str).tolist()
    for split_name, subset in [("validation", validation_adata), ("test", test_adata)]:
        query_model = load_query_model(
            subset,
            scanvi_model,
            unlabeled_category=unlabeled_category,
            label_categories=query_label_categories,
        )
        pred, latent = prediction_outputs(query_model, subset, label_key, rare_class)
        metadata = _metadata_columns(
            split_name=split_name,
            seed=seed,
            rare_train_size=rare_train_size,
            rare_class=rare_class,
            split_mode=split_mode,
            run_name=run_name,
        )
        _write_baseline_outputs(
            run_dir,
            split_name=split_name,
            predictions=_add_metadata(pred, metadata),
            latent=_add_metadata(latent, metadata),
        )

    split_assignments = adata.obs[[label_key, "scanvi_label", "is_labeled_for_scanvi"]].copy()
    split_assignments.insert(0, "cell_id", adata.obs_names.astype(str))
    split_assignments["split"] = split.astype(str).to_numpy()
    write_table(split_assignments, run_dir / "split_assignments.csv")
    write_table(pd.DataFrame({"gene": genes}), run_dir / "selected_hvg_genes.csv")


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


def _run_stage_table_path(run_dir: Path, stage: str, filename: str) -> Path:
    return run_dir / "stages" / stage / filename


def _run_resource_summary_path(run_dir: Path) -> Path:
    return run_dir / RESOURCE_SUMMARY_FILENAME


def _resource_summary_row(
    *,
    run_name: str,
    split_mode: str,
    rare_class: str,
    seed: int,
    rare_train_size: str | int,
    resource_summary: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run": run_name,
                "split_mode": split_mode,
                "rare_class": rare_class,
                "seed": seed,
                "rare_train_size": str(rare_train_size),
                "wall_time_seconds": resource_summary["wall_time_seconds"],
                "peak_memory_mb": resource_summary["peak_rss_mb"],
            }
        ]
    )


def _load_baseline_bundle(run_dir: Path) -> dict[str, Any]:
    train_pred = read_table(artifact_path(run_dir, "train_predictions.csv")).reset_index(drop=True)
    val_pred = read_table(artifact_path(run_dir, "validation_predictions.csv")).reset_index(drop=True)
    test_pred = read_table(artifact_path(run_dir, "test_predictions.csv")).reset_index(drop=True)
    train_latent = read_table(artifact_path(run_dir, "train_latent.csv")).reset_index(drop=True)
    val_latent = read_table(artifact_path(run_dir, "validation_latent.csv")).reset_index(drop=True)
    test_latent = read_table(artifact_path(run_dir, "test_latent.csv")).reset_index(drop=True)
    genes = read_table(run_dir / "selected_hvg_genes.csv")["gene"].astype(str).tolist()
    return {
        "train_pred": train_pred,
        "val_pred": val_pred,
        "test_pred": test_pred,
        "train_latent": train_latent,
        "val_latent": val_latent,
        "test_latent": test_latent,
        "genes": genes,
    }


def _select_method_rows(outputs: dict[str, pd.DataFrame], methods: list[str]) -> dict[str, pd.DataFrame]:
    selected = outputs.copy()
    if "effect_runs" in selected and not selected["effect_runs"].empty:
        effect_runs = selected["effect_runs"][selected["effect_runs"]["method_key"].isin(methods)].reset_index(drop=True)
        selected["effect_runs"] = effect_runs
        if "effect_summary" in selected and not selected["effect_summary"].empty and "method" in selected["effect_summary"].columns:
            selected["effect_summary"] = selected["effect_summary"][selected["effect_summary"]["method"].isin(effect_runs["method"].unique())].reset_index(drop=True)
    return selected


def _evaluate_method_outputs(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    rare_class: str,
    split_mode: str,
    seed: int,
    rare_train_size: str | int,
    run_dir: Path,
    baseline_bundle: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    adata = adata_from_config(config)
    ensure_unique_names(adata)
    return evaluate_four_stage_methods(
        adata,
        train_pred=baseline_bundle["train_pred"],
        val_pred=baseline_bundle["val_pred"],
        test_pred=baseline_bundle["test_pred"],
        train_latent=baseline_bundle["train_latent"],
        val_latent=baseline_bundle["val_latent"],
        test_latent=baseline_bundle["test_latent"],
        genes=baseline_bundle["genes"],
        rare_class=rare_class,
        split_mode=split_mode,
        seed=seed,
        rare_train_size=str(rare_train_size),
        run=run_dir.name,
        max_false_rescue_rate=float(getattr(args, "max_false_rescue_rate", 0.001)),
        top_n=int(getattr(args, "top_n", 25)),
        min_cells=int(getattr(args, "min_cells", 5)),
    )


def _write_run_method_outputs(outputs: dict[str, pd.DataFrame], *, run_dir: Path) -> None:
    stage = "inductive_methods"
    for key, filename in METHOD_OUTPUT_FILES.items():
        frame = outputs.get(key)
        if frame is None:
            continue
        write_table(frame, _run_stage_table_path(run_dir, stage, filename))


def _rebuild_stage_outputs(root: Path) -> None:
    stage = "inductive_methods"
    for key, filename in METHOD_OUTPUT_FILES.items():
        parts: list[pd.DataFrame] = []
        for run_dir in _run_dirs(root):
            path = _run_stage_table_path(run_dir, stage, filename)
            if not path.exists():
                continue
            parts.append(read_table(path))
        write_table(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), stage_table_path(root, stage, filename))


def _rebuild_resource_summary(root: Path) -> None:
    stage = "inductive_methods"
    parts: list[pd.DataFrame] = []
    for run_dir in _run_dirs(root):
        path = _run_resource_summary_path(run_dir)
        if not path.exists():
            continue
        parts.append(read_table(path))
    write_table(
        pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(),
        stage_table_path(root, stage, RESOURCE_SUMMARY_FILENAME),
    )


def _rebuild_plot_outputs(root: Path) -> None:
    rebuild_inductive_plots(root)


def run_inductive_workflow(config: dict[str, Any], args: argparse.Namespace) -> None:
    methods = _normalize_methods(getattr(args, "methods", None))
    reuse_baseline_only = bool(getattr(args, "reuse_baseline_only", False))

    for rare_class, split_mode, seed, rare_train_size, root in _iter_slices(config, args):
        run_dir = root / "runs" / _run_name(seed, rare_train_size, split_mode)
        with ResourceMonitor() as monitor:
            baseline_missing = _missing_baseline_artifacts(run_dir)
            if baseline_missing:
                if reuse_baseline_only:
                    _require_existing_baseline(run_dir)
                _run_baseline_slice(
                    config,
                    args,
                    rare_class=rare_class,
                    split_mode=split_mode,
                    seed=seed,
                    rare_train_size=rare_train_size,
                    root=root,
                    run_dir=run_dir,
                )
            baseline_bundle = _load_baseline_bundle(run_dir)
            outputs = _evaluate_method_outputs(
                config,
                args,
                rare_class=rare_class,
                split_mode=split_mode,
                seed=seed,
                rare_train_size=rare_train_size,
                run_dir=run_dir,
                baseline_bundle=baseline_bundle,
            )
            selected_outputs = _select_method_rows(outputs, methods)
            _write_run_method_outputs(selected_outputs, run_dir=run_dir)
        write_table(
            _resource_summary_row(
                run_name=run_dir.name,
                split_mode=split_mode,
                rare_class=rare_class,
                seed=seed,
                rare_train_size=rare_train_size,
                resource_summary=monitor.summary(),
            ),
            _run_resource_summary_path(run_dir),
        )
        _rebuild_stage_outputs(root)
        _rebuild_resource_summary(root)
        _rebuild_plot_outputs(root)
