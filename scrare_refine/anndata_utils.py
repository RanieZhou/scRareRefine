from __future__ import annotations

from typing import Any

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def adata_from_config(config: dict[str, Any]) -> ad.AnnData:
    dataset = config["dataset"]
    adata = ad.read_h5ad(dataset["path"])
    use_layer = dataset.get("use_layer", None)
    if use_layer:
        # Use a specific layer (e.g. 'counts') as the expression matrix
        if use_layer not in adata.layers:
            raise ValueError(f"Config requested layer '{use_layer}', but available layers are: {list(adata.layers.keys())}")
        adata = ad.AnnData(
            X=adata.layers[use_layer].copy(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
    elif dataset.get("use_raw", False):
        if adata.raw is None:
            raise ValueError("Config requested raw.X, but adata.raw is missing")
        adata = ad.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            uns=adata.uns.copy(),
        )
    return adata


def subset_cells(adata: ad.AnnData, *, max_cells: int | None, seed: int) -> ad.AnnData:
    if max_cells is None or max_cells >= adata.n_obs:
        return adata
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False))
    return adata[indices].copy()


def select_top_variable_genes(adata: ad.AnnData, *, n_top_genes: int | None) -> ad.AnnData:
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= adata.n_vars:
        return adata
    for column in ("n_cells_by_counts", "total_counts", "mean_counts"):
        if column in adata.var:
            values = np.asarray(adata.var[column], dtype=float)
            top_idx = np.argsort(-values)[:n_top_genes]
            return adata[:, np.sort(top_idx)].copy()
    x = adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
    variance = mean_sq - mean * mean
    top_idx = np.argsort(-variance)[:n_top_genes]
    return adata[:, np.sort(top_idx)].copy()


def ensure_unique_names(adata: ad.AnnData) -> None:
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
