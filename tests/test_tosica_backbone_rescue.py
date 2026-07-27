import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.analysis.tosica_backbone_rescue import (
    CONFIGS,
    RARE_TRAIN_SIZES,
    SEEDS,
    TOSICA_EMBED_DIM,
    _annotation_arrays,
    _evaluate_pair,
    build_grid,
    summarize_runs,
)


class FakeTosicaResult:
    def __init__(self):
        self.obs_names = pd.Index(["cell-a", "cell-b"])
        self.obs = pd.DataFrame(
            {
                "Prediction": ["rare", "Unknown"],
                "Probability": [0.9, 0.08],
            },
            index=self.obs_names,
        )
        self.X = np.arange(2 * TOSICA_EMBED_DIM, dtype=np.float32).reshape(
            2, TOSICA_EMBED_DIM
        )


def test_default_grid_is_closed_seed42_eight_dataset_screen():
    grid = build_grid()
    assert len(CONFIGS) == 8
    assert SEEDS == (42,)
    assert RARE_TRAIN_SIZES == ("0.01", "0.05", "0.10", "all")
    assert len(grid) == 32
    assert len(set(grid)) == 32


def test_annotation_arrays_preserve_native_unknown_and_align_latent():
    prediction, confidence, latent = _annotation_arrays(
        FakeTosicaResult(), ["cell-a", "cell-b"]
    )
    assert prediction.tolist() == ["rare", "Unknown"]
    assert np.allclose(confidence, [0.9, 0.08])
    assert latent.shape == (2, 48)


def test_evaluate_pair_counts_true_and_false_rescues():
    result = _evaluate_pair(
        ["rare", "rare", "major", "major"],
        ["major", "major", "major", "major"],
        ["rare", "major", "rare", "major"],
        "rare",
    )
    assert result["true_rescues"] == 1
    assert result["false_rescues"] == 1
    assert result["all_rescues"] == 2
    assert result["rescue_precision"] == 0.5
    assert result["rescue_fdp"] == 0.5
    assert result["incremental_fpr"] == 0.5
    assert result["delta_rare_f1"] > 0


def test_summary_reports_dataset_level_wins_ties_losses():
    runs = pd.DataFrame(
        {
            "dataset": ["x", "x", "x"],
            "status": ["success"] * 3,
            "baseline_rare_f1": [0.1, 0.2, 0.3],
            "refined_rare_f1": [0.2, 0.2, 0.2],
            "delta_rare_f1": [0.1, 0.0, -0.1],
            "baseline_rare_recall": [0.1, 0.2, 0.3],
            "refined_rare_recall": [0.2, 0.2, 0.2],
            "delta_rare_recall": [0.1, 0.0, -0.1],
            "incremental_fpr": [0.0, 0.001, 0.002],
            "alpha_violation": [False, False, False],
            "abstain": [False, True, False],
        }
    )
    summary = summarize_runs(runs).iloc[0]
    assert summary["wins_ties_losses_f1"] == "1/1/1"
    assert summary["n_abstentions"] == 1
    assert summary["incremental_fpr_max"] == 0.002
