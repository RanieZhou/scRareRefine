from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from scrare.infra.config import load_config
from scrare.workflows.inductive import run_inductive_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the scrare inductive workflow.")
    parser.add_argument("--config", required=True, help="Path to the workflow config file")
    parser.add_argument("--rare-class", help="Comma-separated rare classes; defaults to config experiment rare_class")
    parser.add_argument("--split-mode", default="batch_heldout", help="Comma-separated: cell_stratified,batch_heldout")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rare-train-size")
    parser.add_argument("--methods", help="Comma-separated methods to run")
    parser.add_argument("--reuse-baseline-only", action="store_true", help="Reuse existing baseline artifacts without retraining")
    parser.add_argument("--output-dir", help="Override output root; only valid for one rare class and one split mode")
    parser.add_argument("--max-cells", type=int)
    parser.add_argument("--scvi-epochs", type=int)
    parser.add_argument("--scanvi-epochs", type=int)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.01)
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_inductive_workflow(config, args)


if __name__ == "__main__":
    main()
