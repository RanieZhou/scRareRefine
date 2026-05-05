from __future__ import annotations

import argparse
from typing import Any


def run_inductive_workflow(config: dict[str, Any], args: argparse.Namespace) -> None:
    del config, args
    raise SystemExit(
        "The inductive workflow is not implemented yet. "
        "Task 7 wires the CLI to this workflow entry point for later migration."
    )
