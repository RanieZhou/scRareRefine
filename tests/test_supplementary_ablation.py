import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.analysis.supplementary_ablation import evaluate_predictions


def test_evaluate_predictions_tracks_rescue_precision_and_incremental_fpr():
    result = evaluate_predictions(
        pd.Series(["rare", "major", "major"]),
        pd.Series(["major", "major", "major"]),
        pd.Series(["rare", "rare", "major"]),
        "rare",
    )
    assert result["true_rescues"] == 1
    assert result["false_rescues"] == 1
    assert result["all_rescues"] == 2
    assert result["rescue_precision"] == 0.5
    assert result["incremental_fpr"] == 0.5
    assert result["alpha_violation"] is True


def test_evaluate_predictions_keeps_undefined_rescue_precision_na():
    result = evaluate_predictions(
        pd.Series(["rare", "major"]),
        pd.Series(["major", "major"]),
        pd.Series(["major", "major"]),
        "rare",
    )
    assert np.isnan(result["rescue_precision"])
    assert result["incremental_fpr"] == 0.0
    assert result["alpha_violation"] is False
