from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
            return path
        except ImportError:
            csv_path = path.with_suffix(".csv")
            df.to_csv(csv_path, index=index)
            return csv_path
    df.to_csv(path, index=index)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)

