from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from scrare.infra.config import load_config
from scrare.workflows.inductive import run_inductive_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the scrare inductive workflow.")
    parser.add_argument("--config", required=True, help="Path to the workflow config file")
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_inductive_workflow(config, args)
