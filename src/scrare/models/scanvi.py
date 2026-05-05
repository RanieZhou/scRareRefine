from __future__ import annotations

import numpy as np
import pandas as pd
import scvi
import torch
from scvi import REGISTRY_KEYS


def compute_uncertainty(probabilities: pd.DataFrame, *, rare_class: str) -> pd.DataFrame:
    probs = probabilities.astype(float)
    arr = probs.to_numpy()
    classes = probs.columns.to_numpy()
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if arr.shape[1] > 1 else order[:, 0]
    top1 = arr[np.arange(arr.shape[0]), top1_idx]
    top2 = arr[np.arange(arr.shape[0]), top2_idx]
    entropy = -(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum(axis=1)

    return pd.DataFrame(
        {
            "max_prob": top1,
            "entropy": entropy,
            "margin": top1 - top2,
            "top1_label": classes[top1_idx],
            "top2_label": classes[top2_idx],
            "top2_is_" + rare_class: classes[top2_idx] == rare_class,
        },
        index=probabilities.index,
    )


def _label_categories(model: scvi.model.SCANVI) -> list[str] | None:
    manager = getattr(model, "adata_manager", None)
    if manager is None:
        return None
    state_registry = manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
    categories = getattr(state_registry, "categorical_mapping", None)
    if categories is None:
        return None
    return [str(category) for category in categories]


def _probability_frame(model: scvi.model.SCANVI, adata, soft) -> pd.DataFrame:
    if isinstance(soft, pd.DataFrame):
        probabilities = soft.copy()
    else:
        categories = _label_categories(model)
        probabilities = pd.DataFrame(soft, columns=categories)
    probabilities.index = adata.obs_names
    return probabilities


def prediction_outputs(model: scvi.model.SCANVI, adata, label_key: str, rare_class: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = model.predict(adata)
    soft = model.predict(adata, soft=True)
    if isinstance(soft, tuple):
        soft = soft[0]
    probabilities = _probability_frame(model, adata, soft)
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


def seed_everything(seed: int) -> None:
    scvi.settings.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_reference_scanvi(
    train_adata,
    *,
    batch_key: str,
    unlabeled_category: str,
    n_latent: int,
    batch_size: int,
    scvi_epochs: int,
    scanvi_epochs: int,
) -> scvi.model.SCANVI:
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key, labels_key="scanvi_label")
    vae = scvi.model.SCVI(train_adata, n_latent=n_latent)
    vae.train(
        max_epochs=scvi_epochs,
        batch_size=batch_size,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    scanvi = scvi.model.SCANVI.from_scvi_model(
        vae,
        unlabeled_category=unlabeled_category,
        labels_key="scanvi_label",
    )
    scanvi.train(
        max_epochs=scanvi_epochs,
        batch_size=batch_size,
        enable_progress_bar=False,
        log_every_n_steps=10,
    )
    return scanvi


def load_query_model(query_adata, scanvi_model: scvi.model.SCANVI, *, unlabeled_category: str, label_categories: list[str]) -> scvi.model.SCANVI:
    query = query_adata.copy()
    query.obs["scanvi_label"] = pd.Categorical([unlabeled_category] * query.n_obs, categories=label_categories)
    query.obs["is_labeled_for_scanvi"] = False
    query_model = scvi.model.SCANVI.load_query_data(query, scanvi_model)
    query_model.is_trained_ = True
    return query_model
