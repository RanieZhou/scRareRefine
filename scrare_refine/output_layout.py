from __future__ import annotations

from pathlib import Path


ARTIFACT_FILES = {
    "scanvi_predictions.csv",
    "scanvi_predictions.parquet",
    "scanvi_latent.csv",
    "prototype_scores.csv",
}


def artifact_path(run_dir: str | Path, filename: str) -> Path:
    return Path(run_dir) / "artifacts" / filename


def legacy_or_artifact_path(run_dir: str | Path, filename: str) -> Path:
    run_dir = Path(run_dir)
    new_path = artifact_path(run_dir, filename)
    if new_path.exists():
        return new_path
    return run_dir / filename


def existing_table_path(run_dir: str | Path, filename: str) -> Path:
    run_dir = Path(run_dir)
    candidates = [artifact_path(run_dir, filename), run_dir / filename]
    if filename.endswith(".parquet"):
        csv_name = filename.removesuffix(".parquet") + ".csv"
        candidates = [
            artifact_path(run_dir, filename),
            artifact_path(run_dir, csv_name),
            run_dir / filename,
            run_dir / csv_name,
        ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def root_table_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "tables" / filename


def root_figure_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "figures" / filename


def root_log_path(root: str | Path, filename: str) -> Path:
    return Path(root) / "logs" / filename


def stage_table_path(root: str | Path, stage: str, filename: str) -> Path:
    return Path(root) / "stages" / stage / filename


def classify_root_file(filename: str) -> str:
    if filename.endswith(".png"):
        return "figures"
    if filename.endswith(".log"):
        return "logs"
    if filename.endswith(".csv"):
        return "tables"
    return "misc"
