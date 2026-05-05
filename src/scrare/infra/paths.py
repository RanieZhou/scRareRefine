from __future__ import annotations

from pathlib import Path


def artifact_path(run_dir: str | Path, filename: str) -> Path:
    return Path(run_dir) / "artifacts" / filename


def root_table_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "tables" / filename


def stage_table_path(root: str | Path, stage: str, filename: str) -> Path:
    return Path(root) / "stages" / stage / filename
