from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scvi
import torch

from scrare_refine.anndata_utils import adata_from_config, ensure_unique_names, subset_cells
from scrare_refine.config import load_config
from scrare_refine.fusion import (
    fuse_and_evaluate,
    prototype_probabilities_from_reference,
    select_best_params,
)
from scrare_refine.inductive import (
    batch_heldout_split,
    cell_stratified_split,
    make_inductive_scanvi_labels,
    select_train_hvg_var_names,
)
from scrare_refine.io import write_table
from scrare_refine.metrics import classification_tables, compute_uncertainty
from scrare_refine.output_layout import artifact_path, stage_table_path
from scrare_refine.resources import ResourceMonitor
from scrare_refine.splits import parse_rare_train_size


DEFAULT_TEMPERATURES = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_ALPHA_MINS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_BETAS = [0.0, 0.1, 0.3, 0.5, 1.0]


def _csv_values(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _run_values(config: dict[str, Any], seed: int | None, rare_train_size: str | int | None):
    exp = config["experiment"]
    seeds = [seed] if seed is not None else list(exp["seeds"])
    sizes = [rare_train_size] if rare_train_size is not None else list(exp["rare_train_sizes"])
    return seeds, [parse_rare_train_size(size) for size in sizes]


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


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    return out


def _extract_scanvi_probs(pred: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [col for col in pred.columns if col.startswith("prob_")]
    return pred[prob_cols].rename(columns=lambda col: col.removeprefix("prob_"))


def _prediction_outputs(model: scvi.model.SCANVI, adata, label_key: str, rare_class: str):
    pred = model.predict(adata)
    soft = model.predict(adata, soft=True)
    if isinstance(soft, tuple):
        soft = soft[0]
    probabilities = soft if isinstance(soft, pd.DataFrame) else pd.DataFrame(soft)
    probabilities.index = adata.obs_names
    uncertainty = compute_uncertainty(probabilities, rare_class=rare_class)
    latent = model.get_latent_representation(adata)

    predictions = adata.obs.copy()
    predictions["cell_id"] = adata.obs_names
    predictions["true_label"] = adata.obs[label_key].astype(str).to_numpy()
    predictions["predicted_label"] = np.asarray(pred).astype(str)
    predictions = predictions.reset_index(drop=True)
    predictions = pd.concat(
        [
            predictions,
            uncertainty.reset_index(drop=True),
            probabilities.reset_index(drop=True).add_prefix("prob_"),
        ],
        axis=1,
    )
    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "cell_id", adata.obs_names.to_numpy())
    return predictions, latent_df


def _baseline_metrics(pred: pd.DataFrame, rare_class: str) -> tuple[dict[str, float], pd.DataFrame]:
    overall, per_class = classification_tables(
        pred["true_label"].astype(str),
        pred["predicted_label"].astype(str),
        rare_class=rare_class,
    )
    overall.update(
        {
            "n_changed": 0,
            "modification_rate": 0.0,
            "rescued_rare_errors": 0,
            "false_rescues": 0,
            "damaged_correct": 0,
            "rare_error_recall": 0.0,
            "major_to_rare_false_rescue_rate": 0.0,
        }
    )
    return overall, per_class


def _fusion_grid(
    pred: pd.DataFrame,
    latent: np.ndarray,
    train_pred: pd.DataFrame,
    train_latent: np.ndarray,
    *,
    rare_class: str,
    split_name: str,
) -> pd.DataFrame:
    p_scanvi = _extract_scanvi_probs(pred)
    rows = []
    for temperature in DEFAULT_TEMPERATURES:
        p_proto = prototype_probabilities_from_reference(
            latent,
            reference_latent=train_latent,
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            temperature=temperature,
        )
        for alpha_min in DEFAULT_ALPHA_MINS:
            for beta in DEFAULT_BETAS:
                result = fuse_and_evaluate(
                    p_scanvi,
                    p_proto,
                    margin=pred["margin"].to_numpy(),
                    y_true=pred["true_label"].reset_index(drop=True),
                    baseline_pred=pred["predicted_label"].reset_index(drop=True),
                    rare_class=rare_class,
                    temperature=temperature,
                    alpha_min=alpha_min,
                    beta=beta,
                )
                result["split"] = split_name
                rows.append(result)
    return pd.DataFrame(rows)


def _fusion_with_params(
    pred: pd.DataFrame,
    latent: np.ndarray,
    train_pred: pd.DataFrame,
    train_latent: np.ndarray,
    *,
    rare_class: str,
    temperature: float,
    alpha_min: float,
    beta: float,
) -> dict[str, float]:
    p_scanvi = _extract_scanvi_probs(pred)
    p_proto = prototype_probabilities_from_reference(
        latent,
        reference_latent=train_latent,
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        temperature=temperature,
    )
    return fuse_and_evaluate(
        p_scanvi,
        p_proto,
        margin=pred["margin"].to_numpy(),
        y_true=pred["true_label"].reset_index(drop=True),
        baseline_pred=pred["predicted_label"].reset_index(drop=True),
        rare_class=rare_class,
        temperature=temperature,
        alpha_min=alpha_min,
        beta=beta,
    )


def _split_series(config: dict[str, Any], adata, *, split_mode: str, seed: int, train_fraction: float, validation_fraction: float, test_fraction: float) -> pd.Series:
    dataset = config["dataset"]
    if split_mode == "cell_stratified":
        return cell_stratified_split(
            adata.obs,
            label_key=dataset["label_key"],
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    if split_mode == "batch_heldout":
        return batch_heldout_split(
            adata.obs,
            label_key=dataset["label_key"],
            batch_key=dataset["batch_key"],
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
    raise ValueError(f"Unknown split mode: {split_mode}")


def _write_stage_outputs(root: Path, effect_rows: list[dict[str, Any]], grid_rows: list[pd.DataFrame], selected_rows: list[dict[str, Any]]) -> None:
    effect_df = pd.DataFrame(effect_rows)
    grid_df = pd.concat(grid_rows, ignore_index=True) if grid_rows else pd.DataFrame()
    selected_df = pd.DataFrame(selected_rows)
    metrics = [
        "overall_accuracy",
        "macro_f1",
        "rare_precision",
        "rare_recall",
        "rare_f1",
        "n_changed",
        "modification_rate",
        "rescued_rare_errors",
        "false_rescues",
        "damaged_correct",
        "rare_error_recall",
        "major_to_rare_false_rescue_rate",
    ]
    summary = (
        effect_df.groupby(["split_mode", "rare_class", "rare_train_size", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    write_table(effect_df, stage_table_path(root, "fusion", "fusion_effect_runs.csv"))
    write_table(_flatten_summary(summary), stage_table_path(root, "fusion", "fusion_effect_summary.csv"))
    write_table(grid_df, stage_table_path(root, "fusion", "fusion_grid_search.csv"))
    write_table(selected_df, stage_table_path(root, "fusion", "selected_fusion_params.csv"))
    _plot_comparison(summary, stage_table_path(root, "fusion", "fusion_vs_baseline.png"))


def _plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    flat = _flatten_summary(summary)
    if flat.empty:
        return
    order = ["20", "50", "100", "all"]
    methods = sorted(flat["method"].unique())
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True)
    for ax, (metric, title) in zip(
        axes,
        [
            ("rare_f1_mean", "Rare F1"),
            ("rare_recall_mean", "Rare recall"),
            ("rare_precision_mean", "Rare precision"),
            ("major_to_rare_false_rescue_rate_mean", "False rescue rate"),
        ],
    ):
        for method in methods:
            sub = flat[flat["method"].eq(method)].copy()
            sub["rare_train_size"] = pd.Categorical(sub["rare_train_size"].astype(str), categories=order, ordered=True)
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


def run_one(
    config: dict[str, Any],
    *,
    rare_class: str,
    split_mode: str,
    seed: int,
    rare_train_size: str | int,
    output_root: Path,
    max_cells: int | None,
    scvi_epochs: int | None,
    scanvi_epochs: int | None,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    max_accuracy_drop: float,
    max_false_rescue_rate: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    with ResourceMonitor(sample_interval_seconds=1.0) as monitor:
        cfg = copy.deepcopy(config)
        cfg["experiment"]["rare_class"] = rare_class
        dataset = cfg["dataset"]
        exp = cfg["experiment"]
        model_cfg = cfg["model"]
        label_key = dataset["label_key"]
        batch_key = dataset["batch_key"]
        unlabeled = exp["unlabeled_category"]

        scvi.settings.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        adata = adata_from_config(cfg)
        ensure_unique_names(adata)
        adata = subset_cells(adata, max_cells=max_cells, seed=seed)
        split = _split_series(
            cfg,
            adata,
            split_mode=split_mode,
            seed=seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )
        scanvi_labels, is_labeled = make_inductive_scanvi_labels(
            adata.obs,
            split,
            label_key=label_key,
            rare_class=rare_class,
            rare_train_size=rare_train_size,
            seed=seed,
            unlabeled_category=unlabeled,
        )
        adata.obs["split"] = split.to_numpy()
        label_categories = sorted(set(adata.obs[label_key].astype(str)) | {unlabeled})
        adata.obs["scanvi_label"] = pd.Categorical(scanvi_labels.astype(str), categories=label_categories)
        adata.obs["is_labeled_for_scanvi"] = is_labeled

        train_raw = adata[adata.obs["split"].eq("train")].copy()
        hvg_names = select_train_hvg_var_names(train_raw, n_top_genes=int(model_cfg.get("n_top_hvg", 3000)))
        adata = adata[:, hvg_names].copy()
        train = adata[adata.obs["split"].eq("train")].copy()
        validation = adata[adata.obs["split"].eq("validation")].copy()
        test = adata[adata.obs["split"].eq("test")].copy()

        run_dir = output_root / "runs" / _run_name(seed, rare_train_size, split_mode)
        run_dir.mkdir(parents=True, exist_ok=True)

        scvi.model.SCVI.setup_anndata(train, batch_key=batch_key, labels_key="scanvi_label")
        vae = scvi.model.SCVI(train, n_latent=int(model_cfg.get("n_latent", 30)))
        vae.train(
            max_epochs=int(scvi_epochs or model_cfg.get("scvi_max_epochs", 200)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            enable_progress_bar=False,
            log_every_n_steps=10,
        )
        scanvi = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category=unlabeled, labels_key="scanvi_label")
        scanvi.train(
            max_epochs=int(scanvi_epochs or model_cfg.get("scanvi_max_epochs", 100)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            enable_progress_bar=False,
            log_every_n_steps=10,
        )
        scanvi.save(run_dir / "model", overwrite=True)

        query = adata[adata.obs["split"].isin(["validation", "test"])].copy()
        query.obs["scanvi_label"] = pd.Categorical([unlabeled] * query.n_obs, categories=label_categories)
        query.obs["is_labeled_for_scanvi"] = False
        query_model = scvi.model.SCANVI.load_query_data(query, scanvi)
        # We use scArches query loading for category transfer only; no query fine-tuning is performed.
        # scvi-tools marks the loaded query model as untrained, but the reference weights are already loaded.
        query_model.is_trained_ = True

        train_pred, train_latent_df = _prediction_outputs(scanvi, train, label_key, rare_class)
        query_pred, query_latent_df = _prediction_outputs(query_model, query, label_key, rare_class)
        validation_ids = set(validation.obs_names.astype(str))
        test_ids = set(test.obs_names.astype(str))
        validation_pred = query_pred[query_pred["cell_id"].astype(str).isin(validation_ids)].reset_index(drop=True)
        test_pred = query_pred[query_pred["cell_id"].astype(str).isin(test_ids)].reset_index(drop=True)
        validation_latent_df = query_latent_df[query_latent_df["cell_id"].astype(str).isin(validation_ids)].reset_index(drop=True)
        test_latent_df = query_latent_df[query_latent_df["cell_id"].astype(str).isin(test_ids)].reset_index(drop=True)

        for frame in [train_pred, validation_pred, test_pred]:
            frame["seed"] = seed
            frame["rare_train_size"] = str(rare_train_size)
            frame["rare_class"] = rare_class
            frame["split_mode"] = split_mode
        for frame in [train_latent_df, validation_latent_df, test_latent_df]:
            frame["seed"] = seed
            frame["rare_train_size"] = str(rare_train_size)
            frame["rare_class"] = rare_class
            frame["split_mode"] = split_mode

        write_table(train_pred, artifact_path(run_dir, "train_predictions.csv"))
        write_table(validation_pred, artifact_path(run_dir, "validation_predictions.csv"))
        write_table(test_pred, artifact_path(run_dir, "test_predictions.csv"))
        write_table(train_latent_df, artifact_path(run_dir, "train_latent.csv"))
        write_table(validation_latent_df, artifact_path(run_dir, "validation_latent.csv"))
        write_table(test_latent_df, artifact_path(run_dir, "test_latent.csv"))
        write_table(pd.DataFrame({"gene": hvg_names}), run_dir / "selected_hvg_genes.csv")
        write_table(adata.obs[["split", label_key, batch_key, "scanvi_label", "is_labeled_for_scanvi"]].reset_index(names="cell_id"), run_dir / "split_assignments.csv")

        train_latent = train_latent_df[[c for c in train_latent_df.columns if c.startswith("latent_")]].to_numpy()
        validation_latent = validation_latent_df[[c for c in validation_latent_df.columns if c.startswith("latent_")]].to_numpy()
        test_latent = test_latent_df[[c for c in test_latent_df.columns if c.startswith("latent_")]].to_numpy()

        validation_baseline, _ = _baseline_metrics(validation_pred, rare_class)
        test_baseline, baseline_per_class = _baseline_metrics(test_pred, rare_class)
        val_grid = _fusion_grid(
            validation_pred,
            validation_latent,
            train_pred,
            train_latent,
            rare_class=rare_class,
            split_name="validation",
        )
        best_temp, best_alpha, best_beta = select_best_params(
            val_grid,
            baseline_accuracy=validation_baseline["overall_accuracy"],
            max_accuracy_drop=max_accuracy_drop,
            max_false_rescue_rate=max_false_rescue_rate,
        )
        test_fusion = _fusion_with_params(
            test_pred,
            test_latent,
            train_pred,
            train_latent,
            rare_class=rare_class,
            temperature=best_temp,
            alpha_min=best_alpha,
            beta=best_beta,
        )
        _, fusion_per_class = classification_tables(
            test_pred["true_label"],
            _fused_labels_for_params(
                test_pred,
                test_latent,
                train_pred,
                train_latent,
                rare_class=rare_class,
                temperature=best_temp,
                alpha_min=best_alpha,
                beta=best_beta,
            ),
            rare_class=rare_class,
        )

    resource_summary = monitor.summary()
    common = {
        "seed": seed,
        "rare_train_size": str(rare_train_size),
        "rare_class": rare_class,
        "split_mode": split_mode,
        "run": run_dir.name,
        "n_train_cells": int(train.n_obs),
        "n_validation_cells": int(validation.n_obs),
        "n_test_cells": int(test.n_obs),
        "n_labeled_rare": int(((train_pred["true_label"] == rare_class) & train_pred["is_labeled_for_scanvi"].astype(bool)).sum()),
        **resource_summary,
    }
    baseline_row = {**test_baseline, **common, "method": "scANVI baseline", "temperature": np.nan, "alpha_min": np.nan, "beta": np.nan}
    fusion_row = {**test_fusion, **common, "method": "fusion", "temperature": best_temp, "alpha_min": best_alpha, "beta": best_beta}
    selected = {**common, "selected_temperature": best_temp, "selected_alpha_min": best_alpha, "selected_beta": best_beta}

    write_table(pd.DataFrame([baseline_row, fusion_row]), run_dir / "inductive_metrics.csv")
    baseline_per_class.insert(0, "method", "scANVI baseline")
    fusion_per_class.insert(0, "method", "fusion")
    write_table(pd.concat([baseline_per_class, fusion_per_class], ignore_index=True), run_dir / "per_class_metrics.csv")
    rare_rows = test_pred[test_pred["true_label"].astype(str).eq(rare_class)]
    confusion = rare_rows["predicted_label"].value_counts().rename_axis("predicted_label").reset_index(name="n_cells")
    write_table(confusion, run_dir / "rare_confusion.csv")
    write_table(pd.DataFrame([selected]), run_dir / "selected_fusion_params.csv")
    write_table(val_grid.assign(**common), run_dir / "fusion_validation_grid.csv")
    write_table(pd.DataFrame([{**{"seed": seed, "rare_train_size": str(rare_train_size)}, **resource_summary}]), run_dir / "run_resources.csv")

    print(
        f"Finished {run_dir}: baseline rare_f1={test_baseline['rare_f1']:.3f}, "
        f"fusion rare_f1={test_fusion['rare_f1']:.3f}"
    )
    return [baseline_row, fusion_row], val_grid.assign(**common), selected


def _fused_labels_for_params(
    pred: pd.DataFrame,
    latent: np.ndarray,
    train_pred: pd.DataFrame,
    train_latent: np.ndarray,
    *,
    rare_class: str,
    temperature: float,
    alpha_min: float,
    beta: float,
) -> pd.Series:
    from scrare_refine.fusion import disagreement_aware_weight, fuse_predictions

    p_scanvi = _extract_scanvi_probs(pred)
    p_proto = prototype_probabilities_from_reference(
        latent,
        reference_latent=train_latent,
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        temperature=temperature,
    )
    alpha = disagreement_aware_weight(p_scanvi, p_proto, margin=pred["margin"].to_numpy(), alpha_min=alpha_min, beta=beta)
    labels, _ = fuse_predictions(p_scanvi, p_proto, alpha=alpha)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inductive train/validation/test scANVI + fusion validation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare-class", help="Comma-separated rare classes; defaults to config experiment rare_class")
    parser.add_argument("--split-mode", default="cell_stratified", help="Comma-separated: cell_stratified,batch_heldout")
    parser.add_argument("--output-dir", help="Override output root; only valid for one rare class and one split mode")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rare-train-size")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--scvi-epochs", type=int)
    parser.add_argument("--scanvi-epochs", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.01)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_classes = _csv_values(args.rare_class, [config["experiment"]["rare_class"]])
    split_modes = _csv_values(args.split_mode, ["cell_stratified"])
    if args.output_dir and (len(rare_classes) > 1 or len(split_modes) > 1):
        raise ValueError("--output-dir can only be used with one rare class and one split mode")

    for rare_class in rare_classes:
        for split_mode in split_modes:
            cfg = copy.deepcopy(config)
            cfg["experiment"]["rare_class"] = rare_class
            seeds, sizes = _run_values(cfg, args.seed, args.rare_train_size)
            root = _output_root(cfg, rare_class=rare_class, split_mode=split_mode, output_dir=args.output_dir)
            all_effect_rows: list[dict[str, Any]] = []
            all_grid_rows: list[pd.DataFrame] = []
            selected_rows: list[dict[str, Any]] = []
            for seed in seeds:
                for rare_train_size in sizes:
                    effect_rows, grid_rows, selected = run_one(
                        cfg,
                        rare_class=rare_class,
                        split_mode=split_mode,
                        seed=int(seed),
                        rare_train_size=rare_train_size,
                        output_root=root,
                        max_cells=args.max_cells,
                        scvi_epochs=args.scvi_epochs,
                        scanvi_epochs=args.scanvi_epochs,
                        train_fraction=args.train_fraction,
                        validation_fraction=args.validation_fraction,
                        test_fraction=args.test_fraction,
                        max_accuracy_drop=args.max_accuracy_drop,
                        max_false_rescue_rate=args.max_false_rescue_rate,
                    )
                    all_effect_rows.extend(effect_rows)
                    all_grid_rows.append(grid_rows)
                    selected_rows.append(selected)
            _write_stage_outputs(root, all_effect_rows, all_grid_rows, selected_rows)
            print(f"Wrote inductive outputs to {root}")


if __name__ == "__main__":
    main()
