from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scvi
import torch

from scrare_refine.anndata_utils import (
    adata_from_config,
    ensure_unique_names,
    select_top_variable_genes,
    subset_cells,
)
from scrare_refine.config import load_config, output_dir
from scrare_refine.io import write_table
from scrare_refine.metrics import classification_tables, compute_uncertainty
from scrare_refine.output_layout import artifact_path
from scrare_refine.resources import ResourceMonitor
from scrare_refine.splits import make_scanvi_labels, parse_rare_train_size


def _run_values(config: dict[str, Any], seed: int | None, rare_train_size: str | int | None):
    exp = config["experiment"]
    seeds = [seed] if seed is not None else list(exp["seeds"])
    sizes = [rare_train_size] if rare_train_size is not None else list(exp["rare_train_sizes"])
    return seeds, [parse_rare_train_size(size) for size in sizes]


def _run_name(seed: int, rare_train_size: str | int) -> str:
    return f"seed_{seed}_rare_{rare_train_size}"


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
    uncertainty = uncertainty.reset_index(drop=True)
    probabilities = probabilities.reset_index(drop=True).add_prefix("prob_")
    predictions = pd.concat([predictions, uncertainty, probabilities], axis=1)
    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "cell_id", adata.obs_names.to_numpy())
    return predictions, latent_df


def run_one(
    config: dict[str, Any],
    *,
    seed: int,
    rare_train_size: str | int,
    max_cells: int | None,
    scvi_epochs: int | None,
    scanvi_epochs: int | None,
) -> Path:
    with ResourceMonitor(sample_interval_seconds=1.0) as monitor:
        dataset = config["dataset"]
        exp = config["experiment"]
        model_cfg = config["model"]
        rare_class = exp["rare_class"]
        unlabeled = exp["unlabeled_category"]
        label_key = dataset["label_key"]
        batch_key = dataset["batch_key"]

        scvi.settings.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        adata = adata_from_config(config)
        ensure_unique_names(adata)
        adata = subset_cells(adata, max_cells=max_cells, seed=seed)
        adata = select_top_variable_genes(adata, n_top_genes=int(model_cfg.get("n_top_hvg", 3000)))

        scanvi_labels, is_labeled = make_scanvi_labels(
            adata.obs,
            label_key=label_key,
            rare_class=rare_class,
            rare_train_size=rare_train_size,
            seed=seed,
            unlabeled_category=unlabeled,
        )
        adata.obs["scanvi_label"] = scanvi_labels.to_numpy()
        adata.obs["is_labeled_for_scanvi"] = is_labeled

        run_dir = output_dir(config) / "runs" / _run_name(seed, rare_train_size)
        run_dir.mkdir(parents=True, exist_ok=True)

        scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key, labels_key="scanvi_label")
        vae = scvi.model.SCVI(adata, n_latent=int(model_cfg.get("n_latent", 30)))
        vae.train(
            max_epochs=int(scvi_epochs or model_cfg.get("scvi_max_epochs", 200)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            enable_progress_bar=False,
            log_every_n_steps=10,
        )

        scanvi = scvi.model.SCANVI.from_scvi_model(
            vae,
            unlabeled_category=unlabeled,
            labels_key="scanvi_label",
        )
        scanvi.train(
            max_epochs=int(scanvi_epochs or model_cfg.get("scanvi_max_epochs", 100)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            enable_progress_bar=False,
            log_every_n_steps=10,
        )

        predictions, latent = _prediction_outputs(scanvi, adata, label_key, rare_class)
        predictions["seed"] = seed
        predictions["rare_train_size"] = str(rare_train_size)
        latent["seed"] = seed
        latent["rare_train_size"] = str(rare_train_size)

        write_table(predictions, artifact_path(run_dir, "scanvi_predictions.parquet"))
        write_table(latent, artifact_path(run_dir, "scanvi_latent.csv"))

        overall, per_class = classification_tables(
            predictions["true_label"], predictions["predicted_label"], rare_class=rare_class
        )
        overall.update(
            {
                "seed": seed,
                "rare_train_size": str(rare_train_size),
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "n_labeled_rare": int(
                    ((predictions["true_label"] == rare_class) & predictions["is_labeled_for_scanvi"]).sum()
                ),
            }
        )
        per_class.insert(0, "seed", seed)
        per_class.insert(1, "rare_train_size", str(rare_train_size))
        scanvi.save(run_dir / "model", overwrite=True)

    resource_summary = monitor.summary()
    overall.update(resource_summary)
    write_table(pd.DataFrame([overall]), run_dir / "scanvi_metrics.csv")
    write_table(pd.DataFrame([{**{"seed": seed, "rare_train_size": str(rare_train_size)}, **resource_summary}]), run_dir / "run_resources.csv")
    write_table(per_class, run_dir / "per_class_metrics.csv")

    print(f"Finished {run_dir}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train scVI/scANVI for the P0 Immune DC experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rare-train-size")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--scvi-epochs", type=int)
    parser.add_argument("--scanvi-epochs", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    seeds, sizes = _run_values(config, args.seed, args.rare_train_size)
    for seed in seeds:
        for rare_train_size in sizes:
            run_one(
                config,
                seed=seed,
                rare_train_size=rare_train_size,
                max_cells=args.max_cells,
                scvi_epochs=args.scvi_epochs,
                scanvi_epochs=args.scanvi_epochs,
            )


if __name__ == "__main__":
    main()
