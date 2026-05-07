from __future__ import annotations

from typing import Any

import anndata as ad


def adata_from_config(config: dict[str, Any]) -> ad.AnnData:
    dataset = config["dataset"]
    adata = ad.read_h5ad(dataset["path"])
    use_layer = dataset.get("use_layer")
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(f"Config requested layer '{use_layer}', but available layers are: {list(adata.layers.keys())}")
        return ad.AnnData(
            X=adata.layers[use_layer].copy(),
            obs=adata.obs.copy(),
            var=adata.var.copy(),
        )
    if dataset.get("use_raw", False):
        if adata.raw is None:
            raise ValueError("Config requested raw.X, but adata.raw is missing")
        return ad.AnnData(
            X=adata.raw.X.copy(),
            obs=adata.obs.copy(),
            var=adata.raw.var.copy(),
            uns=adata.uns.copy(),
        )
    return adata
