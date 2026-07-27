import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.analysis.p4_celltypist_backbone import _annotation_frames, _summary


class FakeAnnotation:
    def __init__(self):
        self.predicted_labels = pd.DataFrame({"predicted_labels": ["b", "a"]})
        self.decision_matrix = pd.DataFrame({"b": [2.0, 0.0], "a": [0.0, 2.0]})


def test_annotation_frames_aligns_native_score_columns():
    pred, score = _annotation_frames(FakeAnnotation(), ["a", "b"])
    assert pred.tolist() == ["b", "a"]
    assert score.columns.tolist() == ["a", "b"]
    assert np.allclose(score.to_numpy(), [[0.0, 2.0], [2.0, 0.0]])


def test_summary_reports_complete_wins_ties_losses():
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
    out = _summary(runs).iloc[0]
    assert out["wins_ties_losses_f1"] == "1/1/1"
    assert out["n_abstentions"] == 1
    assert out["incremental_fpr_max"] == 0.002
