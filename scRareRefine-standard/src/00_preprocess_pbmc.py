"""Preprocess PBMC COVID-19 Blood Atlas for pDC sep ratio screening.

Steps:
1. Filter to cells with non-nan minor_subset
2. Promote 'pDC' from minor_subset into a merged label column
3. Subsample non-pDC cells to ~50k total (keep all pDC)
4. Save to data/raw/pbmc/pbmc_pdc_50k.h5ad

Usage:
    python src/00_preprocess_pbmc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

SRC_PATH = "data/raw/pbmc/pbmc_pdc.h5ad"
DST_PATH = "data/raw/pbmc/pbmc_pdc_50k.h5ad"
TARGET_TOTAL = 50_000
SEED = 0


def main() -> None:
    print(f"Reading {SRC_PATH}...")
    adata = ad.read_h5ad(SRC_PATH)
    print(f"  Shape: {adata.shape}")

    # 1. Filter nan minor_subset (stored as string "nan", not real NaN)
    mask = adata.obs["minor_subset"].notna() & (adata.obs["minor_subset"] != "nan")
    adata = adata[mask].copy()
    print(f"  After dropping nan minor_subset: {adata.shape}")

    # 2. Build label column: use minor_subset directly
    #    minor_subset 包含 pDC、cMono、CD4.NAIVE 等 ~40 个亚型
    adata.obs["label"] = adata.obs["minor_subset"].astype(str)

    # 3. Subsample: keep all pDC, sample the rest
    pdc_mask = adata.obs["label"] == "pDC"
    pdc_cells = adata.obs.index[pdc_mask]
    other_cells = adata.obs.index[~pdc_mask]

    n_pdc = len(pdc_cells)
    n_other_target = TARGET_TOTAL - n_pdc
    rng = np.random.default_rng(SEED)

    if len(other_cells) > n_other_target:
        sampled_other = rng.choice(other_cells, size=n_other_target, replace=False)
    else:
        sampled_other = other_cells

    keep = np.concatenate([pdc_cells, sampled_other])
    adata = adata[keep].copy()
    print(f"  After subsample: {adata.shape}")
    print(f"  pDC cells: {pdc_mask.sum()} → kept {n_pdc}")
    print(f"  non-pDC cells: {len(other_cells)} → sampled {len(sampled_other)}")

    # 4. Print label distribution
    print("\nLabel distribution (top 20):")
    print(adata.obs["label"].value_counts().head(20).to_string())

    # 5. Replace X with raw integer counts (adata.X is log-normalized; adata.raw.X is raw)
    if adata.raw is not None and adata.raw.n_vars == adata.n_vars:
        raw_X = adata.raw.X
        if sp.issparse(raw_X):
            adata.X = raw_X.tocsr()
        else:
            adata.X = sp.csr_matrix(raw_X)
        print("  Replaced X with raw.X (integer counts)")
    else:
        # Fallback: ensure X is CSR
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)
        elif not isinstance(adata.X, sp.csr_matrix):
            adata.X = adata.X.tocsr()
        print("  WARNING: raw.X not available; using X as-is (may not be raw counts)")

    # 6. Keep only essential obs columns
    keep_cols = ["label", "donor_id", "sex", "disease", "assay"]
    keep_cols = [c for c in keep_cols if c in adata.obs.columns]
    adata.obs = adata.obs[keep_cols].copy()

    # 7. Save
    out = Path(DST_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
