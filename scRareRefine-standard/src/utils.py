from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def load_adata(config: dict[str, Any]) -> ad.AnnData:
    dataset = config["dataset"]
    adata = ad.read_h5ad(dataset["path"])
    use_layer = dataset.get("use_layer")
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(f"Layer '{use_layer}' not found. Available: {list(adata.layers.keys())}")
        return ad.AnnData(X=adata.layers[use_layer].copy(), obs=adata.obs.copy(), var=adata.var.copy())
    if dataset.get("use_raw", False):
        if adata.raw is None:
            raise ValueError("Config requested raw.X but adata.raw is missing")
        return ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=adata.raw.var.copy(), uns=adata.uns.copy())
    return adata


# ── IO ────────────────────────────────────────────────────────────────────────

def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ── Paths ─────────────────────────────────────────────────────────────────────

def safe_class_name(name: str) -> str:
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def parse_rare_train_size(value: str | int | float) -> int | float | str:
    """Parse rare_train_size: int → absolute count, float in (0,1] → proportion, "all" → "all".

    Accepts: int, float, "all", "5pct", "0.05", "20".
    """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s == "all":
        return "all"
    if s.endswith("pct"):
        return int(s[:-3]) / 100.0
    try:
        f = float(s)
        if 0 < f <= 1 and "." in s:
            return f
        return int(float(s))
    except ValueError:
        raise ValueError(f"Cannot parse rare_train_size: {value!r}")


def _rts_label(rare_train_size: float | int | str) -> str:
    if isinstance(rare_train_size, float):
        return f"{round(rare_train_size * 100)}pct"
    return str(rare_train_size)


def make_run_id(split_mode: str, seed: int, rare_class: str, rare_train_size: float | int | str) -> str:
    return f"{split_mode}_seed{seed}_{safe_class_name(rare_class)}_rare{_rts_label(rare_train_size)}"


def make_run_dir(config: dict[str, Any], split_mode: str, seed: int, rare_class: str, rare_train_size: float | int | str) -> Path:
    dataset_name = config["dataset"]["name"]
    run_id = make_run_id(split_mode, seed, rare_class, rare_train_size)
    return Path("outputs") / dataset_name / run_id


def make_split_path(config: dict[str, Any], split_mode: str, seed: int) -> Path:
    dataset_name = config["dataset"]["name"]
    return Path("data") / "splits" / dataset_name / f"{split_mode}_seed{seed}" / "split.csv"


# ── Metrics ───────────────────────────────────────────────────────────────────

def classification_tables(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    rare_class: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = pd.DataFrame(
        {"label": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    rare_row = per_class[per_class["label"] == rare_class]
    if rare_row.empty:
        rare_precision = rare_recall = rare_f1 = 0.0
    else:
        rare_precision = float(rare_row["precision"].iloc[0])
        rare_recall = float(rare_row["recall"].iloc[0])
        rare_f1 = float(rare_row["f1"].iloc[0])
    return {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rare_precision": rare_precision,
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
    }, per_class


def compute_uncertainty(probabilities: pd.DataFrame, *, rare_class: str) -> pd.DataFrame:
    probs = probabilities.astype(float)
    arr = probs.to_numpy()
    classes = probs.columns.to_numpy()
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if arr.shape[1] > 1 else order[:, 0]
    top1 = arr[np.arange(arr.shape[0]), top1_idx]
    top2 = arr[np.arange(arr.shape[0]), top2_idx]
    entropy = -(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum(axis=1)
    return pd.DataFrame(
        {
            "max_prob": top1,
            "entropy": entropy,
            "margin": top1 - top2,
            "top1_label": classes[top1_idx],
            "top2_label": classes[top2_idx],
            f"top2_is_{rare_class}": classes[top2_idx] == rare_class,
        },
        index=probabilities.index,
    )


# ── Expression ────────────────────────────────────────────────────────────────

def log1p_cpm(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        normalized = x.multiply(10000.0 / row_sum[:, None])
        return np.log1p(normalized.toarray()).astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    row_sum = arr.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(arr * (10000.0 / row_sum[:, None])).astype(np.float32)


# ── Seed ──────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    import torch
    import scvi
    scvi.settings.seed = seed
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Resource Monitor ──────────────────────────────────────────────────────────

@dataclass
class ResourceMonitor:
    sample_interval_seconds: float = 1.0
    _start_time: float = field(init=False, default=0.0)
    _end_time: float = field(init=False, default=0.0)
    _peak_rss_bytes: int = field(init=False, default=0)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)

    def __enter__(self) -> "ResourceMonitor":
        self._start_time = time.perf_counter()
        self._end_time = 0.0
        self._peak_rss_bytes = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sample_once()
        self._end_time = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.sample_interval_seconds * 2))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.sample_interval_seconds)

    def _sample_once(self) -> None:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(rss))

    def summary(self) -> dict[str, float]:
        end = self._end_time or time.perf_counter()
        return {
            "wall_time_seconds": float(end - self._start_time),
            "peak_rss_mb": float(self._peak_rss_bytes / (1024 * 1024)),
        }
