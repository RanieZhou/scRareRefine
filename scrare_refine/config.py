from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def output_dir(config: dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    return Path(experiment.get("output_dir", config.get("output_dir", "outputs")))

