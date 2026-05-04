from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _to_dense_sample(matrix: Any, n_rows: int = 200, n_cols: int = 1000) -> np.ndarray:
    sample = matrix[:n_rows, :n_cols]
    if hasattr(sample, "to_memory"):
        sample = sample.to_memory()
    if hasattr(sample, "toarray"):
        sample = sample.toarray()
    return np.asarray(sample)


def matrix_is_integer_like(matrix: Any, n_rows: int = 200, n_cols: int = 1000) -> bool:
    arr = _to_dense_sample(matrix, n_rows=n_rows, n_cols=n_cols)
    nonzero = arr[arr != 0]
    if nonzero.size == 0:
        return True
    return bool(np.allclose(nonzero, np.round(nonzero)))


def audit_anndata(
    adata: Any,
    *,
    dataset_name: str,
    label_key: str,
    batch_key: str,
    rare_threshold: float = 0.05,
    rare_max_cells: int = 200,
    use_raw: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if label_key not in adata.obs:
        raise KeyError(f"Missing label column in adata.obs: {label_key}")
    if batch_key not in adata.obs:
        raise KeyError(f"Missing batch column in adata.obs: {batch_key}")

    obs = adata.obs[[label_key, batch_key]].copy()
    n_cells, n_genes = adata.shape
    matrix = adata.raw.X if use_raw and adata.raw is not None else adata.X
    class_counts = obs[label_key].value_counts(dropna=False)
    batch_counts = obs[batch_key].value_counts(dropna=False)

    class_dist = (
        class_counts.rename_axis("label")
        .reset_index(name="n_cells")
        .assign(
            fraction=lambda df: df["n_cells"] / n_cells,
            is_rare_candidate=lambda df: (df["fraction"] < rare_threshold)
            | (df["n_cells"] <= rare_max_cells),
        )
        .sort_values(["n_cells", "label"], ascending=[True, True])
        .reset_index(drop=True)
    )
    batch_dist = (
        batch_counts.rename_axis("batch")
        .reset_index(name="n_cells")
        .assign(fraction=lambda df: df["n_cells"] / n_cells)
        .sort_values(["n_cells", "batch"], ascending=[False, True])
        .reset_index(drop=True)
    )

    summary = {
        "dataset": dataset_name,
        "n_cells": int(n_cells),
        "n_genes": int(n_genes),
        "label_key": label_key,
        "batch_key": batch_key,
        "n_classes": int(class_counts.shape[0]),
        "n_batches": int(batch_counts.shape[0]),
        "use_raw": bool(use_raw and adata.raw is not None),
        "x_integer_like": matrix_is_integer_like(matrix),
    }
    return summary, class_dist, batch_dist

