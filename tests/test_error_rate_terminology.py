import numpy as np

from tools.analysis.weak_backbone_demo import _metrics as weak_backbone_metrics
from tools.comparison.run_scrarerefine_comparison import (
    _metrics as scrarerefine_metrics,
)


def _assert_alias(metrics):
    assert "incremental_fpr" in metrics
    assert "rescue_ffr" in metrics
    assert metrics["incremental_fpr"] == metrics["rescue_ffr"]


def test_incremental_fpr_is_compatibility_alias_across_active_metric_paths():
    y_true = np.array(["rare", "rare", "major", "major", "major"])
    base_pred = np.array(["major", "rare", "major", "major", "rare"])
    final_pred = np.array(["rare", "rare", "rare", "major", "rare"])

    for metric_fn in (scrarerefine_metrics, weak_backbone_metrics):
        metrics = metric_fn(y_true, final_pred, base_pred, "rare")
        _assert_alias(metrics)
        assert metrics["incremental_fpr"] == round(1 / 3, 6)
        assert metrics["rare_fp_rate"] == round(2 / 3, 6)


def test_incremental_fpr_zero_when_no_new_false_rare_calls_are_added():
    y_true = np.array(["rare", "major", "major"])
    base_pred = np.array(["major", "rare", "major"])
    final_pred = np.array(["rare", "rare", "major"])

    metrics = scrarerefine_metrics(y_true, final_pred, base_pred, "rare")
    _assert_alias(metrics)
    assert metrics["incremental_fpr"] == 0.0
    assert metrics["rare_fp_rate"] == 0.5
