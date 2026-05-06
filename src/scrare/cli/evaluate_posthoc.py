from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from scrare.infra.config import load_config
from scrare.workflows.posthoc import run_posthoc_workflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate four-stage posthoc methods on held-out test cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare-class", default="ASDC,cDC1")
    parser.add_argument("--split-mode", default="batch_heldout")
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-cells", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> Any:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    return run_posthoc_workflow(config, args)


if __name__ == "__main__":
    main()
