"""Export raw-count mouse Tabula Muris Senis datasets from CELLxGENE Census.

The CELLxGENE website "Download" button can return the hosted/local H5AD
matrix. For these two TMS mouse datasets that matrix is log-normalized-like
data, while scRareRefine expects raw counts before its own preprocessing.

Run from the repository root, preferably in a Linux/WSL environment with
cellxgene-census installed:

    python tools/extract/export_mouse_tms_raw_from_census.py
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import anndata as ad


os.environ.setdefault("AWS_RESPONSE_CHECKSUM_VALIDATION", "when_required")
os.environ.setdefault("AWS_REQUEST_CHECKSUM_CALCULATION", "when_required")

import cellxgene_census  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "mouse"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    dataset_id: str
    output_name: str
    expected_n_obs: int
    fallback_filter: str | None = None

    @property
    def primary_filter(self) -> str:
        return f"dataset_id == '{self.dataset_id}'"


TMS_ALL_10X_DATASET_ID = "48b37086-25f7-4ecd-be66-f5bb378e3aea"

DATASETS = {
    "lung": DatasetSpec(
        key="lung",
        title="Tabula Muris Senis mouse lung 10x",
        dataset_id="e67a08e2-96fc-4244-b8a3-fca092b22f77",
        output_name="mouse_lung_tms_10x_raw_counts.h5ad",
        expected_n_obs=24540,
        fallback_filter=(
            f"dataset_id == '{TMS_ALL_10X_DATASET_ID}' and tissue == 'lung'"
        ),
    ),
    "pancreas": DatasetSpec(
        key="pancreas",
        title="Tabula Muris Senis mouse pancreas 10x",
        dataset_id="b257ae76-c030-4d61-9a73-775b5d195a9a",
        output_name="mouse_pancreas_tms_10x_raw_counts.h5ad",
        expected_n_obs=6201,
        fallback_filter=(
            f"dataset_id == '{TMS_ALL_10X_DATASET_ID}' and tissue == 'pancreas'"
        ),
    ),
}

OBS_COLUMNS = [
    "soma_joinid",
    "dataset_id",
    "assay",
    "cell_type",
    "tissue",
    "disease",
    "donor_id",
    "sex",
    "development_stage",
    "suspension_type",
    "is_primary_data",
]
VAR_COLUMNS = ["soma_joinid", "feature_id", "feature_name", "feature_length"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export raw-count mouse TMS H5ADs from CELLxGENE Census."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
        help="Dataset keys to export.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for raw-count H5AD files.",
    )
    parser.add_argument(
        "--census_version",
        default="2025-11-17",
        help="CELLxGENE Census version to open.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--allow_size_mismatch",
        action="store_true",
        help="Do not fail if the exported cell count differs from the website row.",
    )
    parser.add_argument(
        "--no_int32",
        action="store_true",
        help="Keep Census raw X dtype instead of casting verified counts to int32.",
    )
    return parser.parse_args()


def nonzero_values(x) -> np.ndarray:
    if sp.issparse(x):
        vals = np.asarray(x.data)
    else:
        vals = np.asarray(x).ravel()
        vals = vals[vals != 0]
    return vals[np.isfinite(vals)]


def verify_raw_counts(adata: ad.AnnData, *, label: str) -> None:
    vals = nonzero_values(adata.X)
    if vals.size == 0:
        raise ValueError(f"{label}: X has no nonzero values")

    rounded = np.rint(vals)
    is_integer = bool(np.allclose(vals, rounded, atol=1e-6))
    min_val = float(vals.min())
    max_val = float(vals.max())
    p99 = float(np.percentile(vals, 99))
    median_sum = float(np.median(np.asarray(adata.X.sum(axis=1)).ravel()))

    print(
        f"  raw check: min={min_val:.3f} max={max_val:.1f} "
        f"p99={p99:.1f} integer={is_integer} "
        f"median_cell_sum={median_sum:.1f}"
    )
    if min_val < 0:
        raise ValueError(f"{label}: raw counts contain negative values")
    if not is_integer:
        raise ValueError(f"{label}: X is not integer raw counts")


def cast_x_to_int32(adata: ad.AnnData) -> None:
    vals = nonzero_values(adata.X)
    if vals.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Counts exceed int32 range")

    if sp.issparse(adata.X):
        x = adata.X.tocsr(copy=True)
        x.data = np.rint(x.data).astype(np.int32, copy=False)
        x.eliminate_zeros()
        adata.X = x
    else:
        adata.X = np.rint(np.asarray(adata.X)).astype(np.int32, copy=False)


def drop_all_zero_genes(adata: ad.AnnData) -> ad.AnnData:
    if sp.issparse(adata.X):
        keep = np.asarray(adata.X.getnnz(axis=0)).ravel() > 0
    else:
        keep = np.asarray((adata.X != 0).sum(axis=0)).ravel() > 0

    removed = int((~keep).sum())
    if removed:
        print(f"  dropping all-zero genes: {removed}")
        adata = adata[:, keep].copy()
    return adata


def clean_anndata(adata: ad.AnnData, *, spec: DatasetSpec, cast_int32: bool) -> ad.AnnData:
    adata = drop_all_zero_genes(adata)

    if "feature_name" in adata.var:
        adata.var_names = adata.var["feature_name"].astype(str).to_numpy()
        adata.var_names_make_unique()
    if "soma_joinid" in adata.obs:
        adata.obs_names = [f"{spec.key}_{x}" for x in adata.obs["soma_joinid"].astype(str)]
        adata.obs_names_make_unique()

    for col in ["dataset_id", "assay", "cell_type", "tissue", "disease", "donor_id"]:
        if col in adata.obs:
            adata.obs[col] = adata.obs[col].astype(str)

    verify_raw_counts(adata, label=spec.key)
    if cast_int32:
        cast_x_to_int32(adata)

    return adata


def read_from_census(census, *, spec: DatasetSpec, obs_filter: str) -> ad.AnnData:
    print(f"  filter: {obs_filter}")
    return cellxgene_census.get_anndata(
        census=census,
        organism="Mus musculus",
        measurement_name="RNA",
        X_name="raw",
        obs_value_filter=obs_filter,
        obs_column_names=OBS_COLUMNS,
        var_column_names=VAR_COLUMNS,
    )


def export_one(census, *, spec: DatasetSpec, out_dir: Path, force: bool, cast_int32: bool, allow_size_mismatch: bool) -> None:
    out_path = out_dir / spec.output_name
    if out_path.exists() and not force:
        print(f"[skip] {out_path} exists; use --force to overwrite")
        return

    print(f"\n[export] {spec.title}")
    adata = read_from_census(census, spec=spec, obs_filter=spec.primary_filter)
    if adata.n_obs == 0 and spec.fallback_filter is not None:
        print("  no cells for tissue-specific dataset id; trying primary TMS all-10x filter")
        adata = read_from_census(census, spec=spec, obs_filter=spec.fallback_filter)

    if adata.n_obs == 0:
        raise RuntimeError(f"{spec.key}: Census query returned zero cells")
    if adata.n_obs != spec.expected_n_obs:
        msg = (
            f"{spec.key}: expected {spec.expected_n_obs} cells from the CELLxGENE row, "
            f"got {adata.n_obs}"
        )
        if allow_size_mismatch:
            print(f"  warning: {msg}")
        else:
            raise RuntimeError(msg)

    print(f"  loaded shape: {adata.shape}")
    adata = clean_anndata(adata, spec=spec, cast_int32=cast_int32)
    print(f"  final shape: {adata.shape}, dtype={adata.X.dtype}")

    out_dir.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"[saved] {out_path}")


def main() -> None:
    args = parse_args()
    specs: Iterable[DatasetSpec] = [DATASETS[key] for key in args.datasets]
    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        for spec in specs:
            export_one(
                census,
                spec=spec,
                out_dir=args.out_dir,
                force=args.force,
                cast_int32=not args.no_int32,
                allow_size_mismatch=args.allow_size_mismatch,
            )


if __name__ == "__main__":
    main()
