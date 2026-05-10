"""Stage 2: Train scANVI, save embeddings and predictions.

Reads:
    data/splits/{dataset}/{split_mode}_seed{seed}/split.csv

Writes to outputs/{dataset}/{run_id}/:
    split_assignments.csv       cell_id, split, original_label, scanvi_label, is_labeled
    selected_hvg_genes.csv
    resource_summary.csv
    embeddings/
        train_predictions.csv / train_latent.csv
        validation_predictions.csv / validation_latent.csv
        test_predictions.csv / test_latent.csv

Usage:
    python src/02_baseline_scanvi.py \\
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
import scvi
import torch
from scipy import sparse
from scvi import REGISTRY_KEYS

from utils import (
    ResourceMonitor,
    compute_uncertainty,
    load_config,
    load_adata,
    make_run_dir,
    make_split_path,
    parse_rare_train_size,
    read_table,
    seed_everything,
    write_table,
)


def select_hvg_genes(train_adata, *, n_top_genes: int | None) -> list[str]:
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= train_adata.n_vars:
        return train_adata.var_names.astype(str).tolist()
    x = train_adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
    variance = mean_sq - mean * mean
    top_idx = np.argsort(-variance)[:n_top_genes]
    return train_adata.var_names[np.sort(top_idx)].astype(str).tolist()


def make_scanvi_labels(
    obs: pd.DataFrame,
    split: pd.Series,
    *,
    label_key: str,
    rare_class: str,
    rare_train_size: int | str,
    seed: int,
    unlabeled_category: str,
) -> tuple[pd.Series, np.ndarray]:
    true_labels = obs[label_key].astype(str)
    labels = pd.Series(unlabeled_category, index=obs.index, dtype=object)
    is_labeled = np.zeros(len(obs), dtype=bool)

    train_mask = split.eq("train")
    train_major = train_mask & true_labels.ne(rare_class)
    labels.loc[train_major] = true_labels.loc[train_major]
    is_labeled[train_major.to_numpy()] = True

    rare_train = train_mask & true_labels.eq(rare_class)
    rare_indices = np.flatnonzero(rare_train.to_numpy())
    if rare_train_size == "all":
        selected = rare_indices
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(rare_indices, size=min(int(rare_train_size), len(rare_indices)), replace=False)
    labels.iloc[selected] = rare_class
    is_labeled[selected] = True
    return labels.astype(str), is_labeled


def _train_device_kwargs() -> dict[str, int | str]:
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {}


def train_scanvi(
    train_adata,
    *,
    batch_key: str,
    unlabeled_category: str,
    n_latent: int,
    batch_size: int,
    scvi_epochs: int,
    scanvi_epochs: int,
) -> scvi.model.SCANVI:
    device_kwargs = _train_device_kwargs()
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key, labels_key="scanvi_label")
    vae = scvi.model.SCVI(train_adata, n_latent=n_latent)
    vae.train(max_epochs=scvi_epochs, batch_size=batch_size, enable_progress_bar=False, log_every_n_steps=10, **device_kwargs)
    model = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category=unlabeled_category, labels_key="scanvi_label")
    model.train(max_epochs=scanvi_epochs, batch_size=batch_size, enable_progress_bar=False, log_every_n_steps=10, **device_kwargs)
    return model


def _label_categories(model: scvi.model.SCANVI) -> list[str] | None:
    manager = getattr(model, "adata_manager", None)
    if manager is None:
        return None
    state_registry = manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
    categories = getattr(state_registry, "categorical_mapping", None)
    return [str(c) for c in categories] if categories is not None else None


def prediction_outputs(
    model: scvi.model.SCANVI,
    adata,
    *,
    label_key: str,
    rare_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = model.predict(adata)
    soft = model.predict(adata, soft=True)
    if isinstance(soft, tuple):
        soft = soft[0]
    categories = _label_categories(model)
    probabilities = soft.copy() if isinstance(soft, pd.DataFrame) else pd.DataFrame(soft, columns=categories)
    probabilities.index = adata.obs_names
    uncertainty = compute_uncertainty(probabilities, rare_class=rare_class)
    latent = model.get_latent_representation(adata)

    predictions = adata.obs.copy()
    predictions["cell_id"] = adata.obs_names
    predictions["true_label"] = adata.obs[label_key].astype(str).to_numpy()
    predictions["predicted_label"] = np.asarray(pred).astype(str)
    predictions = predictions.reset_index(drop=True)
    predictions = pd.concat(
        [predictions, uncertainty.reset_index(drop=True), probabilities.reset_index(drop=True).add_prefix("prob_")],
        axis=1,
    )
    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "cell_id", adata.obs_names.to_numpy())
    return predictions, latent_df


def load_query_model(
    query_adata,
    model: scvi.model.SCANVI,
    *,
    unlabeled_category: str,
    label_categories: list[str],
) -> scvi.model.SCANVI:
    query = query_adata.copy()
    categories = list(dict.fromkeys([*label_categories, unlabeled_category]))
    query.obs["scanvi_label"] = pd.Categorical([unlabeled_category] * query.n_obs, categories=categories)
    query.obs["is_labeled_for_scanvi"] = False
    query_model = scvi.model.SCANVI.load_query_data(query, model)
    query_model.is_trained_ = True
    return query_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: train scANVI baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", choices=["batch_heldout", "cell_stratified"])
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--scvi_epochs", type=int, default=None)
    parser.add_argument("--scanvi_epochs", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="忽略已有 embedding，强制重新训练")
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    dataset = config["dataset"]
    experiment = config["experiment"]
    model_cfg = config.get("model", {})
    label_key = dataset.get("label_key", "label")
    batch_key = dataset.get("batch_key", "batch")
    unlabeled_category = experiment.get("unlabeled_category", "Unknown")
    rare_train_size = parse_rare_train_size(args.rare_train_size)

    run_dir_check = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    sentinel = run_dir_check / "embeddings" / "train_predictions.csv"
    if not args.force and sentinel.exists():
        print(f"Embeddings already exist at {sentinel.parent}. Skipping training. Use --force to retrain.")
        return

    split_path = make_split_path(config, args.split_mode, args.seed)
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found at {split_path}. Run 01_split.py first.")
    split_df = read_table(split_path).set_index("cell_id")

    seed_everything(args.seed)
    print(f"Loading data from {dataset['path']} ...")
    adata = load_adata(config)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    common_cells = adata.obs_names.intersection(split_df.index)
    adata = adata[common_cells].copy()
    split = split_df.loc[adata.obs_names, "split"]

    scanvi_label, is_labeled = make_scanvi_labels(
        adata.obs, split,
        label_key=label_key,
        rare_class=rare_class,
        rare_train_size=rare_train_size,
        seed=args.seed,
        unlabeled_category=unlabeled_category,
    )
    label_cats = sorted(pd.unique(adata.obs[label_key].astype(str)).tolist())
    if unlabeled_category not in label_cats:
        label_cats = label_cats + [unlabeled_category]
    adata.obs["scanvi_label"] = pd.Categorical(scanvi_label.astype(str), categories=[str(c) for c in label_cats])
    adata.obs["is_labeled_for_scanvi"] = is_labeled

    train_adata = adata[split.eq("train")].copy()
    genes = select_hvg_genes(train_adata, n_top_genes=model_cfg.get("n_top_hvg"))
    adata = adata[:, genes].copy()
    train_adata = adata[split.eq("train")].copy()
    val_adata = adata[split.eq("validation")].copy()
    test_adata = adata[split.eq("test")].copy()

    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"

    print(f"Training scANVI (rare_class={rare_class}, rare_train_size={rare_train_size}) ...")
    with ResourceMonitor() as monitor:
        scanvi_model = train_scanvi(
            train_adata,
            batch_key=batch_key,
            unlabeled_category=unlabeled_category,
            n_latent=int(model_cfg.get("n_latent", 30)),
            batch_size=int(model_cfg.get("batch_size", 256)),
            scvi_epochs=int(args.scvi_epochs or model_cfg.get("scvi_max_epochs", 200)),
            scanvi_epochs=int(args.scanvi_epochs or model_cfg.get("scanvi_max_epochs", 100)),
        )
        model_label_cats = _label_categories(scanvi_model) or [str(c) for c in label_cats]

        train_pred, train_latent = prediction_outputs(scanvi_model, train_adata, label_key=label_key, rare_class=rare_class)
        write_table(train_pred, emb_dir / "train_predictions.csv")
        write_table(train_latent, emb_dir / "train_latent.csv")

        for split_name, subset in [("validation", val_adata), ("test", test_adata)]:
            query_model = load_query_model(subset, scanvi_model, unlabeled_category=unlabeled_category, label_categories=model_label_cats)
            pred, latent = prediction_outputs(query_model, subset, label_key=label_key, rare_class=rare_class)
            write_table(pred, emb_dir / f"{split_name}_predictions.csv")
            write_table(latent, emb_dir / f"{split_name}_latent.csv")

    assignments = adata.obs[[label_key, "scanvi_label", "is_labeled_for_scanvi"]].copy()
    assignments.insert(0, "cell_id", adata.obs_names.astype(str))
    assignments["split"] = split.to_numpy()
    write_table(assignments, run_dir / "split_assignments.csv")
    write_table(pd.DataFrame({"gene": genes}), run_dir / "selected_hvg_genes.csv")
    write_table(
        pd.DataFrame([{**monitor.summary(), "seed": args.seed, "rare_class": rare_class,
                       "rare_train_size": str(rare_train_size), "split_mode": args.split_mode}]),
        run_dir / "resource_summary.csv",
    )
    print(f"Done. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
