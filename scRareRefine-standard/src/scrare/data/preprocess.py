from __future__ import annotations

import anndata as ad
import numpy as np
from scipy import sparse


def subset_cells(adata: ad.AnnData, *, max_cells: int | None, seed: int) -> ad.AnnData:
    if max_cells is None or max_cells >= adata.n_obs:
        return adata
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(np.arange(adata.n_obs), size=max_cells, replace=False))
    return adata[indices].copy()


def ensure_unique_names(adata: ad.AnnData) -> None:
    adata.obs_names_make_unique()
    adata.var_names_make_unique()


def select_train_hvg_var_names(train_adata: ad.AnnData, *, n_top_genes: int | None) -> list[str]:
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
