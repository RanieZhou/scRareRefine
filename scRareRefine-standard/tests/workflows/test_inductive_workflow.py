from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from scrare.cli import run_inductive
from scrare.workflows import inductive


@pytest.mark.parametrize(
    ("argv", "expected_methods"),
    [
        (["--config", "cfg.yaml"], None),
        (["--config", "cfg.yaml", "--methods", "baseline_plus_fusion"], "baseline_plus_fusion"),
    ],
)
def test_run_inductive_parser_accepts_methods_argument(argv: list[str], expected_methods: str | None) -> None:
    parser = run_inductive.build_parser()

    args = parser.parse_args(argv)

    assert args.config == "cfg.yaml"
    assert getattr(args, "methods") == expected_methods


def test_run_inductive_parser_accepts_reuse_flag() -> None:
    parser = run_inductive.build_parser()

    args = parser.parse_args(["--config", "cfg.yaml", "--reuse-baseline-only"])

    assert args.reuse_baseline_only is True


def _config() -> dict:
    return {
        "dataset": {"name": "demo"},
        "experiment": {
            "rare_class": "ASDC",
            "rare_train_sizes": [20],
            "seeds": [42],
        },
    }


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    values = {
        "rare_class": None,
        "split_mode": "batch_heldout",
        "seed": None,
        "rare_train_size": None,
        "methods": "baseline_plus_fusion",
        "reuse_baseline_only": True,
        "output_dir": str(tmp_path / "results"),
        "max_false_rescue_rate": 0.001,
        "top_n": 25,
        "min_cells": 5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _method_effect_row(run: str, rare_f1: float = 0.5) -> dict:
    return {
        "run": run,
        "method_key": "baseline_plus_fusion",
        "method": "fusion",
        "split_mode": "batch_heldout",
        "rare_class": "ASDC",
        "rare_train_size": "20",
        "overall_accuracy": 0.9,
        "macro_f1": 0.8,
        "rare_precision": 0.7,
        "rare_recall": 0.6,
        "rare_f1": rare_f1,
        "n_candidates": 1,
        "n_marker_verified": 1,
        "rescued_rare_errors": 1,
        "false_rescues": 0,
        "candidate_precision_for_rare_error": 1.0,
        "rare_error_recall": 0.5,
        "modification_rate": 0.1,
        "major_to_rare_false_rescue_rate": 0.0,
    }


def test_inductive_default_output_root_uses_results() -> None:
    root = inductive._output_root(
        _config(),
        rare_class="ASDC",
        split_mode="batch_heldout",
        output_dir=None,
    )

    assert root == Path("results/demo/inductive_batch/asdc")


def test_run_inductive_workflow_fails_when_reuse_requested_but_baseline_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing baseline artifacts"):
        inductive.run_inductive_workflow(_config(), _args(tmp_path))


def test_run_inductive_workflow_reuses_existing_baseline_for_nonbaseline_methods(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_run_baseline_slice", lambda *args, **kwargs: calls.append("baseline"))
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda *args, **kwargs: calls.append("load") or {})
    monkeypatch.setattr(inductive, "_evaluate_method_outputs", lambda *args, **kwargs: calls.append("evaluate") or {})
    monkeypatch.setattr(inductive, "_write_stage_outputs", lambda *args, **kwargs: calls.append("write"))
    monkeypatch.setattr(inductive, "_rebuild_resource_summary", lambda *args, **kwargs: calls.append("rebuild_resource"))
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda *args, **kwargs: calls.append("rebuild_plot"))

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline_plus_fusion", reuse_baseline_only=False),
    )

    assert calls == ["load", "evaluate", "write", "rebuild_resource", "rebuild_plot"]


def test_run_inductive_workflow_trains_baseline_when_requested(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [run_dir / "artifacts" / "train_predictions.csv"])
    monkeypatch.setattr(inductive, "_run_baseline_slice", lambda *args, **kwargs: calls.append("baseline"))
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda *args, **kwargs: calls.append("load") or {})
    monkeypatch.setattr(inductive, "_evaluate_method_outputs", lambda *args, **kwargs: calls.append("evaluate") or {})
    monkeypatch.setattr(inductive, "_write_stage_outputs", lambda *args, **kwargs: calls.append("write"))
    monkeypatch.setattr(inductive, "_rebuild_resource_summary", lambda *args, **kwargs: calls.append("rebuild_resource"))
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda *args, **kwargs: calls.append("rebuild_plot"))

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline", reuse_baseline_only=False),
    )

    assert calls == ["baseline", "load", "evaluate", "write", "rebuild_resource", "rebuild_plot"]


def test_run_inductive_workflow_rebuilds_plots_after_stage_and_resource_outputs(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda *args, **kwargs: {})
    monkeypatch.setattr(inductive, "_evaluate_method_outputs", lambda *args, **kwargs: {})
    monkeypatch.setattr(inductive, "_write_stage_outputs", lambda *args, **kwargs: calls.append("write"))
    monkeypatch.setattr(inductive, "_rebuild_resource_summary", lambda *args, **kwargs: calls.append("resource"))
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda *args, **kwargs: calls.append("plot"))

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline_plus_fusion", reuse_baseline_only=False),
    )

    assert calls == ["write", "resource", "plot"]


def test_write_stage_outputs_replaces_existing_run_rows_without_run_stage_directory(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages" / "inductive_methods"
    stage_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            _method_effect_row("batch_heldout_seed_42_rare_20", rare_f1=0.2),
            _method_effect_row("batch_heldout_seed_43_rare_20", rare_f1=0.4),
        ]
    ).to_csv(stage_dir / "five_method_effect_runs.csv", index=False)

    outputs = {
        "effect_runs": pd.DataFrame([_method_effect_row("batch_heldout_seed_42_rare_20", rare_f1=0.9)]),
        "threshold_curve": pd.DataFrame({"run": ["batch_heldout_seed_42_rare_20"], "marker_threshold": [0.1]}),
        "selected_thresholds": pd.DataFrame({"run": ["batch_heldout_seed_42_rare_20"], "selected_marker_threshold": [0.1]}),
        "prototype_candidates": pd.DataFrame({"run": ["batch_heldout_seed_42_rare_20"], "cell_id": ["candidate"]}),
        "marker_verified_candidates": pd.DataFrame(columns=["run", "cell_id"]),
        "fusion_grid": pd.DataFrame({"run": ["batch_heldout_seed_42_rare_20"], "temperature": [1.0]}),
    }

    inductive._write_stage_outputs(
        outputs,
        root=tmp_path,
        run_name="batch_heldout_seed_42_rare_20",
    )

    effect_runs = pd.read_csv(stage_dir / "five_method_effect_runs.csv")
    summary = pd.read_csv(stage_dir / "five_method_effect_summary.csv")

    assert set(effect_runs["run"]) == {"batch_heldout_seed_42_rare_20", "batch_heldout_seed_43_rare_20"}
    assert effect_runs.loc[effect_runs["run"].eq("batch_heldout_seed_42_rare_20"), "rare_f1"].item() == 0.9
    assert "rare_f1_mean" in summary.columns
    assert not (tmp_path / "runs" / "batch_heldout_seed_42_rare_20" / "stages").exists()


def test_rebuild_plot_outputs_uses_stage_plot_directory(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages" / "inductive_methods"
    stage_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "method": ["baseline"],
            "overall_accuracy_mean": [0.9],
            "macro_f1_mean": [0.8],
            "rare_precision_mean": [0.7],
            "rare_recall_mean": [0.6],
            "rare_f1_mean": [0.65],
        }
    ).to_csv(stage_dir / "five_method_effect_summary.csv", index=False)
    pd.DataFrame(
        {
            "marker_threshold": [0.2, 0.4],
            "rare_f1": [0.5, 0.6],
            "rare_recall": [0.7, 0.8],
            "major_to_rare_false_rescue_rate": [0.01, 0.02],
        }
    ).to_csv(stage_dir / "validation_marker_threshold_curve.csv", index=False)
    pd.DataFrame({"selected_marker_threshold": [0.4]}).to_csv(stage_dir / "selected_marker_thresholds.csv", index=False)
    pd.DataFrame(
        {
            "temperature": [1.0],
            "alpha_min": [0.2],
            "beta": [0.5],
            "rare_f1": [0.6],
        }
    ).to_csv(stage_dir / "fusion_validation_grid.csv", index=False)
    pd.DataFrame(
        {
            "rare_train_size": [20],
            "split_mode": ["batch_heldout"],
            "wall_time_seconds": [12.5],
            "peak_memory_mb": [256.0],
        }
    ).to_csv(stage_dir / "resource_summary.csv", index=False)

    inductive._rebuild_plot_outputs(tmp_path)

    plot_dir = tmp_path / "stages" / "inductive_plots"
    expected_pngs = {
        "five_method_metric_summary.png",
        "marker_threshold_curve.png",
        "fusion_validation_heatmap.png",
        "runtime_summary.png",
        "memory_summary.png",
    }
    assert {path.name for path in plot_dir.glob("*.png")} == expected_pngs
    assert not list((tmp_path / "runs").glob("**/*.png"))


def test_run_baseline_slice_writes_standard_artifacts(monkeypatch, tmp_path: Path) -> None:
    obs = pd.DataFrame(
        {
            "label": ["ASDC", "pDC", "ASDC", "pDC", "ASDC", "pDC"],
            "batch": ["b0", "b0", "b1", "b1", "b2", "b2"],
        },
        index=[f"cell{i}" for i in range(6)],
    )
    adata = ad.AnnData(
        X=np.array(
            [
                [5.0, 0.0],
                [0.0, 5.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [3.0, 0.0],
                [0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        obs=obs,
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    config = {
        "dataset": {"name": "demo", "label_key": "label", "batch_key": "batch"},
        "experiment": {"unlabeled_category": "Unknown"},
        "model": {"n_top_hvg": 2},
    }
    args = _args(tmp_path, max_cells=None)
    root = Path(args.output_dir)
    run_dir = root / "runs" / "batch_heldout_seed_42_rare_1"

    monkeypatch.setattr(inductive, "adata_from_config", lambda cfg: adata.copy())
    monkeypatch.setattr(inductive, "_split_series", lambda *a, **k: pd.Series(["train", "train", "validation", "validation", "test", "test"], index=adata.obs_names))

    def fake_train_reference_scanvi(train_adata, **kwargs):
        return object()

    def fake_load_query_model(query_adata, scanvi_model, *, unlabeled_category, label_categories):
        del scanvi_model, unlabeled_category, label_categories
        return object()

    def fake_prediction_outputs(model, subset, label_key, rare_class):
        del model, label_key, rare_class
        labels = subset.obs["label"].astype(str).tolist()
        probs = []
        preds = []
        for label in labels:
            if label == "ASDC":
                probs.append((0.8, 0.2))
                preds.append("ASDC")
            else:
                probs.append((0.2, 0.8))
                preds.append("pDC")
        predictions = subset.obs.copy()
        predictions["cell_id"] = subset.obs_names
        predictions["true_label"] = labels
        predictions["predicted_label"] = preds
        predictions["max_prob"] = [max(pair) for pair in probs]
        predictions["entropy"] = [0.5] * len(labels)
        predictions["margin"] = [0.6] * len(labels)
        predictions["top1_label"] = preds
        predictions["top2_label"] = ["pDC" if pred == "ASDC" else "ASDC" for pred in preds]
        predictions["top2_is_ASDC"] = [pred != "ASDC" for pred in preds]
        predictions["prob_ASDC"] = [pair[0] for pair in probs]
        predictions["prob_pDC"] = [pair[1] for pair in probs]
        latent = pd.DataFrame({"cell_id": subset.obs_names, "latent_0": np.arange(len(labels), dtype=float), "latent_1": np.arange(len(labels), dtype=float)})
        return predictions.reset_index(drop=True), latent

    monkeypatch.setattr(inductive, "train_reference_scanvi", fake_train_reference_scanvi)
    monkeypatch.setattr(inductive, "load_query_model", fake_load_query_model)
    monkeypatch.setattr(inductive, "prediction_outputs", fake_prediction_outputs)
    monkeypatch.setattr(inductive, "seed_everything", lambda seed: None)

    inductive._run_baseline_slice(
        config,
        args,
        rare_class="ASDC",
        split_mode="batch_heldout",
        seed=42,
        rare_train_size=1,
        root=root,
        run_dir=run_dir,
    )

    assert (run_dir / "selected_hvg_genes.csv").exists()
    assert (run_dir / "split_assignments.csv").exists()
    assert (run_dir / "artifacts" / "train_predictions.csv").exists()
    assert (run_dir / "artifacts" / "validation_predictions.csv").exists()
    assert (run_dir / "artifacts" / "test_predictions.csv").exists()
    assert (run_dir / "artifacts" / "train_latent.csv").exists()
    assert (run_dir / "artifacts" / "validation_latent.csv").exists()
    assert (run_dir / "artifacts" / "test_latent.csv").exists()


def test_run_inductive_workflow_rebuilds_stage_outputs_across_multiple_runs(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "results"

    monkeypatch.setattr(
        inductive,
        "_iter_slices",
        lambda config, args: iter(
            [
                ("ASDC", "batch_heldout", 42, 20, root),
                ("ASDC", "batch_heldout", 43, 20, root),
            ]
        ),
    )
    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda run_dir: {})
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda root: None)

    def fake_evaluate(*args, run_dir: Path, **kwargs):
        run = run_dir.name
        return {
            "effect_runs": pd.DataFrame([_method_effect_row(run)]),
            "threshold_curve": pd.DataFrame({"run": [run], "threshold": [0.1]}),
            "selected_thresholds": pd.DataFrame({"run": [run], "selected_marker_threshold": [0.1]}),
            "prototype_candidates": pd.DataFrame({"run": [run], "cell_id": [f"{run}_candidate"]}),
            "marker_verified_candidates": pd.DataFrame({"run": [run], "cell_id": [f"{run}_verified"]}),
            "fusion_grid": pd.DataFrame({"run": [run], "temperature": [1.0]}),
        }

    monkeypatch.setattr(inductive, "_evaluate_method_outputs", fake_evaluate)

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline_plus_fusion", reuse_baseline_only=False),
    )

    effect_runs = pd.read_csv(root / "stages" / "inductive_methods" / "five_method_effect_runs.csv")

    assert set(effect_runs["run"]) == {"batch_heldout_seed_42_rare_20", "batch_heldout_seed_43_rare_20"}


def test_run_inductive_workflow_writes_run_and_root_resource_summary(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "results"
    summaries = iter(
        [
            {"wall_time_seconds": 12.5, "peak_rss_mb": 256.0},
            {"wall_time_seconds": 13.5, "peak_rss_mb": 300.0},
        ]
    )

    class FakeResourceMonitor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def summary(self) -> dict[str, float]:
            return next(summaries)

    monkeypatch.setattr(inductive, "ResourceMonitor", FakeResourceMonitor)
    monkeypatch.setattr(
        inductive,
        "_iter_slices",
        lambda config, args: iter(
            [
                ("ASDC", "batch_heldout", 42, 20, root),
                ("ASDC", "batch_heldout", 43, 20, root),
            ]
        ),
    )
    monkeypatch.setattr(inductive, "_missing_baseline_artifacts", lambda run_dir: [])
    monkeypatch.setattr(inductive, "_load_baseline_bundle", lambda run_dir: {})
    monkeypatch.setattr(inductive, "_rebuild_plot_outputs", lambda root: None)
    monkeypatch.setattr(
        inductive,
        "_evaluate_method_outputs",
        lambda *args, run_dir, **kwargs: {
            "effect_runs": pd.DataFrame([_method_effect_row(run_dir.name)]),
        },
    )

    inductive.run_inductive_workflow(
        _config(),
        _args(tmp_path, methods="baseline_plus_fusion", reuse_baseline_only=False),
    )

    read_options = {"dtype": {"rare_train_size": "string"}}
    run_summary = pd.read_csv(root / "runs" / "batch_heldout_seed_42_rare_20" / "resource_summary.csv", **read_options)
    stage_summary = pd.read_csv(root / "stages" / "inductive_methods" / "resource_summary.csv", **read_options)

    assert run_summary.to_dict("records") == [
        {
            "run": "batch_heldout_seed_42_rare_20",
            "split_mode": "batch_heldout",
            "rare_class": "ASDC",
            "seed": 42,
            "rare_train_size": "20",
            "wall_time_seconds": 12.5,
            "peak_memory_mb": 256.0,
        }
    ]
    assert stage_summary.to_dict("records") == [
        {
            "run": "batch_heldout_seed_42_rare_20",
            "split_mode": "batch_heldout",
            "rare_class": "ASDC",
            "seed": 42,
            "rare_train_size": "20",
            "wall_time_seconds": 12.5,
            "peak_memory_mb": 256.0,
        },
        {
            "run": "batch_heldout_seed_43_rare_20",
            "split_mode": "batch_heldout",
            "rare_class": "ASDC",
            "seed": 43,
            "rare_train_size": "20",
            "wall_time_seconds": 13.5,
            "peak_memory_mb": 300.0,
        },
    ]
