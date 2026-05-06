from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from scrare.infra.io import write_table
from scrare.infra.paths import stage_table_path
from scrare.visualization.inductive import rebuild_inductive_plots

PLOT_FILENAMES = [
    "five_method_metric_summary.png",
    "marker_threshold_curve.png",
    "fusion_validation_heatmap.png",
    "runtime_summary.png",
    "memory_summary.png",
]

CORE_METRIC_COLUMNS = [
    "overall_accuracy_mean",
    "macro_f1_mean",
    "rare_precision_mean",
    "rare_recall_mean",
    "rare_f1_mean",
]



def _write_minimal_inputs(root: Path) -> None:
    stage = "inductive_methods"
    write_table(
        pd.DataFrame(
            {
                "split_mode": ["batch_heldout", "batch_heldout"],
                "rare_class": ["rare", "rare"],
                "rare_train_size": ["1", "1"],
                "method": ["scANVI baseline", "fusion"],
                "overall_accuracy_mean": [0.80, 0.84],
                "macro_f1_mean": [0.70, 0.76],
                "rare_precision_mean": [0.42, 0.54],
                "rare_recall_mean": [0.38, 0.47],
                "rare_f1_mean": [0.40, 0.50],
            }
        ),
        stage_table_path(root, stage, "five_method_effect_summary.csv"),
    )
    write_table(
        pd.DataFrame(
            {
                "marker_threshold": [-0.5, 0.0, 0.5],
                "rare_f1": [0.40, 0.48, 0.44],
                "rare_recall": [0.60, 0.50, 0.35],
                "major_to_rare_false_rescue_rate": [0.03, 0.01, 0.0],
            }
        ),
        stage_table_path(root, stage, "validation_marker_threshold_curve.csv"),
    )
    write_table(
        pd.DataFrame(
            {
                "run": ["run_a"],
                "split_mode": ["batch_heldout"],
                "rare_class": ["rare"],
                "rare_train_size": ["1"],
                "gate_name": ["rank1"],
                "selected_marker_threshold": [0.0],
                "max_false_rescue_rate": [0.001],
            }
        ),
        stage_table_path(root, stage, "selected_marker_thresholds.csv"),
    )
    write_table(
        pd.DataFrame(
            {
                "temperature": [0.5, 0.5, 1.0, 1.0],
                "alpha_min": [0.3, 0.5, 0.3, 0.5],
                "beta": [0.5, 0.5, 1.0, 1.0],
                "rare_f1": [0.45, 0.50, 0.42, 0.49],
            }
        ),
        stage_table_path(root, stage, "fusion_validation_grid.csv"),
    )
    write_table(
        pd.DataFrame(
            {
                "run": ["run_a", "run_b", "run_c"],
                "split_mode": ["batch_heldout", "batch_heldout", "cell_stratified"],
                "rare_train_size": ["1", "5", "1"],
                "wall_time_seconds": [10.0, 12.0, 8.0],
                "peak_memory_mb": [512.0, 768.0, 400.0],
            }
        ),
        stage_table_path(root, stage, "resource_summary.csv"),
    )



def test_rebuild_inductive_plots_creates_expected_pngs_for_minimal_inputs(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)

    paths = rebuild_inductive_plots(tmp_path)

    assert {path.name for path in paths} == set(PLOT_FILENAMES)
    for filename in PLOT_FILENAMES:
        path = tmp_path / "stages" / "inductive_plots" / filename
        assert path.exists()
        assert path.stat().st_size > 0



def test_rebuild_inductive_plots_raises_clear_error_for_missing_required_columns(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    write_table(
        pd.DataFrame(
            {
                "method": ["scANVI baseline"],
                "overall_accuracy_mean": [0.80],
                "macro_f1_mean": [0.70],
            }
        ),
        stage_table_path(tmp_path, "inductive_methods", "five_method_effect_summary.csv"),
    )

    with pytest.raises(
        ValueError,
        match="five_method_effect_summary.csv is missing required columns: rare_precision_mean, rare_recall_mean, rare_f1_mean",
    ):
        rebuild_inductive_plots(tmp_path)



def test_metric_summary_requires_core_metric_columns(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    frame = pd.DataFrame(
        {
            "method": ["scANVI baseline"],
            "overall_accuracy_mean": [0.80],
            "macro_f1_mean": [0.70],
            "rare_precision_mean": [0.42],
            "rare_recall_mean": [0.38],
            "rare_f1_mean": [0.40],
        }
    )
    write_table(frame, stage_table_path(tmp_path, "inductive_methods", "five_method_effect_summary.csv"))

    rebuild_inductive_plots(tmp_path)

    metric_path = tmp_path / "stages" / "inductive_plots" / "five_method_metric_summary.png"
    assert metric_path.exists()
    assert metric_path.stat().st_size > 0
    assert list(frame.columns[1:]) == CORE_METRIC_COLUMNS



def test_marker_threshold_curve_marks_selected_marker_thresholds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    write_table(
        pd.DataFrame({"run": ["run_a", "run_b"], "selected_marker_threshold": [0.0, 0.5]}),
        stage_table_path(tmp_path, "inductive_methods", "selected_marker_thresholds.csv"),
    )
    marked_thresholds: list[float] = []
    original_axvline = plt.Axes.axvline

    def record_axvline(self, x=0, *args, **kwargs):  # type: ignore[no-untyped-def]
        marked_thresholds.append(float(x))
        return original_axvline(self, x=x, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "axvline", record_axvline)

    rebuild_inductive_plots(tmp_path)

    assert marked_thresholds == [0.0, 0.5]



def test_resource_plots_require_rare_train_size_and_split_mode(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    write_table(
        pd.DataFrame(
            {
                "run": ["run_a"],
                "wall_time_seconds": [10.0],
                "peak_memory_mb": [512.0],
            }
        ),
        stage_table_path(tmp_path, "inductive_methods", "resource_summary.csv"),
    )

    with pytest.raises(ValueError, match="resource_summary.csv is missing required columns: rare_train_size, split_mode"):
        rebuild_inductive_plots(tmp_path)



def test_rebuild_inductive_plots_creates_empty_png_for_empty_csv_with_headers(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    write_table(
        pd.DataFrame(columns=["marker_threshold", "rare_f1", "rare_recall", "major_to_rare_false_rescue_rate"]),
        stage_table_path(tmp_path, "inductive_methods", "validation_marker_threshold_curve.csv"),
    )

    rebuild_inductive_plots(tmp_path)

    path = tmp_path / "stages" / "inductive_plots" / "marker_threshold_curve.png"
    assert path.exists()
    assert path.stat().st_size > 0



def test_rebuild_inductive_plots_creates_empty_png_for_empty_csv_without_headers(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    stage_path = stage_table_path(tmp_path, "inductive_methods", "validation_marker_threshold_curve.csv")
    stage_path.write_text("")

    rebuild_inductive_plots(tmp_path)

    path = tmp_path / "stages" / "inductive_plots" / "marker_threshold_curve.png"
    assert path.exists()
    assert path.stat().st_size > 0
