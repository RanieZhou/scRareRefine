from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scrare.infra.io import read_table
from scrare.infra.paths import stage_table_path

SOURCE_STAGE = "inductive_methods"
PLOT_STAGE = "inductive_plots"

EFFECT_SUMMARY_CSV = "five_method_effect_summary.csv"
MARKER_CURVE_CSV = "validation_marker_threshold_curve.csv"
SELECTED_MARKER_THRESHOLDS_CSV = "selected_marker_thresholds.csv"
FUSION_GRID_CSV = "fusion_validation_grid.csv"
RESOURCE_SUMMARY_CSV = "resource_summary.csv"

METRIC_SUMMARY_PNG = "five_method_metric_summary.png"
MARKER_CURVE_PNG = "marker_threshold_curve.png"
FUSION_HEATMAP_PNG = "fusion_validation_heatmap.png"
RUNTIME_SUMMARY_PNG = "runtime_summary.png"
MEMORY_SUMMARY_PNG = "memory_summary.png"

CORE_METRIC_COLUMNS = [
    "overall_accuracy_mean",
    "macro_f1_mean",
    "rare_precision_mean",
    "rare_recall_mean",
    "rare_f1_mean",
]



def _plot_dir(root: str | Path) -> Path:
    path = Path(root) / "stages" / PLOT_STAGE
    path.mkdir(parents=True, exist_ok=True)
    return path



def _source_csv(root: str | Path, filename: str) -> Path:
    return stage_table_path(root, SOURCE_STAGE, filename)



def _read_plot_table(path: str | Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()



def _save_empty_figure(path: str | Path, title: str, message: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, transform=ax.transAxes)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path



def _require_columns(frame: pd.DataFrame, required: Iterable[str], csv_name: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_name} is missing required columns: {', '.join(missing)}")



def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")



def _resource_size_sort_key(value: object) -> tuple[int, float | str]:
    text = str(value)
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return (0, float(numeric))
    return (1, text)



def _build_metric_summary(root: str | Path) -> Path:
    csv_name = EFFECT_SUMMARY_CSV
    out_path = _plot_dir(root) / METRIC_SUMMARY_PNG
    frame = _read_plot_table(_source_csv(root, csv_name))
    if frame.empty:
        return _save_empty_figure(out_path, "Five-method metric summary", f"No rows in {csv_name}.")

    metric_columns = CORE_METRIC_COLUMNS
    _require_columns(frame, ["method", *metric_columns], csv_name)

    plot_frame = frame[["method", *metric_columns]].copy()
    for column in metric_columns:
        plot_frame[column] = _numeric(plot_frame, column)
    grouped = plot_frame.groupby("method", sort=False)[metric_columns].mean(numeric_only=True)
    if grouped.empty:
        return _save_empty_figure(out_path, "Five-method metric summary", f"No plottable metric rows in {csv_name}.")

    ax = grouped.rename(
        columns={
            "overall_accuracy_mean": "Overall accuracy",
            "macro_f1_mean": "Macro F1",
            "rare_precision_mean": "Rare precision",
            "rare_recall_mean": "Rare recall",
            "rare_f1_mean": "Rare F1",
        }
    ).plot(kind="bar", figsize=(12, 5), ylim=(0, 1))
    ax.set_title("Five-method metric summary")
    ax.set_xlabel("Method")
    ax.set_ylabel("Mean score")
    ax.legend(title="Metric", loc="best")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)
    return out_path



def _selected_marker_thresholds(root: str | Path) -> pd.DataFrame:
    csv_name = SELECTED_MARKER_THRESHOLDS_CSV
    frame = _read_plot_table(_source_csv(root, csv_name))
    if frame.empty:
        return frame
    _require_columns(frame, ["selected_marker_threshold"], csv_name)
    plot_frame = frame.copy()
    plot_frame["selected_marker_threshold"] = _numeric(plot_frame, "selected_marker_threshold")
    return plot_frame.dropna(subset=["selected_marker_threshold"])



def _build_marker_threshold_curve(root: str | Path) -> Path:
    csv_name = MARKER_CURVE_CSV
    out_path = _plot_dir(root) / MARKER_CURVE_PNG
    frame = _read_plot_table(_source_csv(root, csv_name))
    if frame.empty:
        return _save_empty_figure(out_path, "Marker threshold curve", f"No rows in {csv_name}.")

    metric_columns = ["rare_f1", "rare_recall", "major_to_rare_false_rescue_rate"]
    _require_columns(frame, ["marker_threshold", *metric_columns], csv_name)

    plot_frame = frame.copy()
    plot_frame["marker_threshold"] = _numeric(plot_frame, "marker_threshold")
    for column in metric_columns:
        plot_frame[column] = _numeric(plot_frame, column)
    plot_frame = plot_frame.dropna(subset=["marker_threshold"])
    if plot_frame.empty:
        return _save_empty_figure(out_path, "Marker threshold curve", f"No numeric thresholds in {csv_name}.")

    grouped = plot_frame.groupby("marker_threshold", sort=True)[metric_columns].mean(numeric_only=True)
    ax = grouped.rename(
        columns={
            "rare_f1": "Rare F1",
            "rare_recall": "Rare recall",
            "major_to_rare_false_rescue_rate": "False rescue rate",
        }
    ).plot(marker="o", figsize=(8, 5))

    selected = _selected_marker_thresholds(root)
    for index, threshold in enumerate(sorted(selected["selected_marker_threshold"].unique()) if not selected.empty else []):
        label = "Selected threshold" if index == 0 else None
        ax.axvline(float(threshold), color="black", linestyle="--", linewidth=1.2, alpha=0.75, label=label)

    ax.set_title("Validation marker threshold curve")
    ax.set_xlabel("Marker threshold")
    ax.set_ylabel("Validation metric")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    if not selected.empty:
        ax.legend(title="Metric", loc="best")
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)
    return out_path



def _build_fusion_heatmap(root: str | Path) -> Path:
    csv_name = FUSION_GRID_CSV
    out_path = _plot_dir(root) / FUSION_HEATMAP_PNG
    frame = _read_plot_table(_source_csv(root, csv_name))
    if frame.empty:
        return _save_empty_figure(out_path, "Fusion validation heatmap", f"No rows in {csv_name}.")

    _require_columns(frame, ["temperature", "alpha_min", "beta", "rare_f1"], csv_name)
    plot_frame = frame.copy()
    for column in ["temperature", "alpha_min", "beta", "rare_f1"]:
        plot_frame[column] = _numeric(plot_frame, column)
    plot_frame = plot_frame.dropna(subset=["temperature", "alpha_min", "beta", "rare_f1"])
    if plot_frame.empty:
        return _save_empty_figure(out_path, "Fusion validation heatmap", f"No numeric fusion grid rows in {csv_name}.")

    plot_frame["setting"] = plot_frame.apply(lambda row: f"T={row['temperature']:g}, β={row['beta']:g}", axis=1)
    heatmap = plot_frame.pivot_table(index="setting", columns="alpha_min", values="rare_f1", aggfunc="mean")
    if heatmap.empty:
        return _save_empty_figure(out_path, "Fusion validation heatmap", f"No plottable fusion grid rows in {csv_name}.")

    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(heatmap))))
    image = ax.imshow(heatmap.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_title("Fusion validation rare F1")
    ax.set_xlabel("alpha_min")
    ax.set_ylabel("temperature, beta")
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{value:g}" for value in heatmap.columns])
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)
    fig.colorbar(image, ax=ax, label="Rare F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path



def _build_resource_summary(root: str | Path, *, value_column: str, ylabel: str, title: str, filename: str, csv_name: str = RESOURCE_SUMMARY_CSV) -> Path:
    out_path = _plot_dir(root) / filename
    frame = _read_plot_table(_source_csv(root, csv_name))
    if frame.empty:
        return _save_empty_figure(out_path, title, f"No rows in {csv_name}.")

    _require_columns(frame, ["rare_train_size", "split_mode", value_column], csv_name)
    plot_frame = frame.copy()
    plot_frame[value_column] = _numeric(plot_frame, value_column)
    plot_frame = plot_frame.dropna(subset=["rare_train_size", "split_mode", value_column])
    if plot_frame.empty:
        return _save_empty_figure(out_path, title, f"No numeric {value_column} values in {csv_name}.")

    plot_frame["rare_train_size"] = plot_frame["rare_train_size"].astype(str)
    grouped = plot_frame.groupby(["rare_train_size", "split_mode"], sort=False)[value_column].mean().reset_index()
    sizes = sorted(grouped["rare_train_size"].unique(), key=_resource_size_sort_key)
    pivot = grouped.pivot(index="rare_train_size", columns="split_mode", values=value_column).reindex(sizes)
    if pivot.empty:
        return _save_empty_figure(out_path, title, f"No plottable {value_column} values in {csv_name}.")

    ax = pivot.plot(kind="bar", figsize=(max(7, 0.8 * len(pivot.index)), 4.5))
    ax.set_title(title)
    ax.set_xlabel("Rare train size")
    ax.set_ylabel(ylabel)
    ax.legend(title="Split mode", loc="best")
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(out_path, dpi=150)
    plt.close(ax.figure)
    return out_path



def _build_runtime_summary(root: str | Path) -> Path:
    return _build_resource_summary(
        root,
        value_column="wall_time_seconds",
        ylabel="Wall time (seconds)",
        title="Runtime summary",
        filename=RUNTIME_SUMMARY_PNG,
    )



def _build_memory_summary(root: str | Path) -> Path:
    return _build_resource_summary(
        root,
        value_column="peak_memory_mb",
        ylabel="Peak memory (MB)",
        title="Memory summary",
        filename=MEMORY_SUMMARY_PNG,
    )



def rebuild_inductive_plots(root: str | Path) -> list[Path]:
    """Rebuild all inductive plot PNGs from root-level stage CSV tables."""
    return [
        _build_metric_summary(root),
        _build_marker_threshold_curve(root),
        _build_fusion_heatmap(root),
        _build_runtime_summary(root),
        _build_memory_summary(root),
    ]


__all__ = ["rebuild_inductive_plots"]
