from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import anndata as ad
import pandas as pd

from scrare.evaluation.audit import audit_anndata
from scrare.infra.config import load_config, output_dir
from scrare.infra.io import write_table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the scRare dataset.")
    parser.add_argument("--config", required=True, help="Path to the workflow config file")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dataset = config["dataset"]
    analysis = config.get("analysis", {})
    audit_dir = output_dir(config)

    adata = ad.read_h5ad(dataset["path"], backed="r")
    try:
        summary, class_dist, batch_dist = audit_anndata(
            adata,
            dataset_name=dataset["name"],
            label_key=dataset["label_key"],
            batch_key=dataset["batch_key"],
            rare_threshold=float(analysis.get("rare_threshold", 0.05)),
            rare_max_cells=int(analysis.get("rare_max_cells", 200)),
            use_raw=bool(dataset.get("use_raw", False)),
        )

        summary_df = pd.DataFrame([summary])
        write_table(summary_df, audit_dir / "dataset_summary.csv")
        write_table(class_dist, audit_dir / "class_distribution.csv")
        write_table(batch_dist, audit_dir / "batch_distribution.csv")
    finally:
        if hasattr(adata, "file") and adata.file is not None:
            adata.file.close()

    print(f"Wrote audit outputs to {Path(audit_dir)}")


if __name__ == "__main__":
    main()
