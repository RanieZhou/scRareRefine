from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import pandas as pd

from scrare_refine.audit import audit_anndata
from scrare_refine.config import load_config, output_dir
from scrare_refine.io import write_table
from scrare_refine.output_layout import root_table_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the P0 scRareRefine dataset.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    analysis = config.get("analysis", {})
    out_dir = output_dir(config) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(dataset["path"], backed="r")
    summary, class_dist, batch_dist = audit_anndata(
        adata,
        dataset_name=dataset["name"],
        label_key=dataset["label_key"],
        batch_key=dataset["batch_key"],
        rare_threshold=float(analysis.get("rare_threshold", 0.05)),
        rare_max_cells=int(analysis.get("rare_max_cells", 200)),
        use_raw=bool(dataset.get("use_raw", False)),
    )

    write_table(pd.DataFrame([summary]), out_dir / "dataset_summary.csv")
    write_table(class_dist, out_dir / "class_distribution.csv")
    write_table(batch_dist, out_dir / "batch_distribution.csv")

    root_out = output_dir(config)
    write_table(pd.DataFrame([summary]), root_table_path(root_out, "dataset_summary.csv"))
    write_table(class_dist, root_table_path(root_out, "class_distribution.csv"))
    write_table(batch_dist, root_table_path(root_out, "batch_distribution.csv"))

    if hasattr(adata, "file") and adata.file is not None:
        adata.file.close()

    print(f"Wrote audit outputs to {Path(out_dir)}")


if __name__ == "__main__":
    main()
