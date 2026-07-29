from __future__ import annotations

import pandas as pd

from tools.analysis.adaptive_gate_stability import (
    classify_stability,
    repeat_decision_seed,
    summarize_repeats,
)


def test_repeat_decision_seed_is_deterministic_and_unique():
    first = [repeat_decision_seed("dataset", 42, "0.01", i) for i in range(20)]
    second = [repeat_decision_seed("dataset", 42, "0.01", i) for i in range(20)]
    assert first == second
    assert len(set(first)) == 20


def test_stability_bands_and_consistency():
    assert classify_stability(True, 0.80) == ("stable_pass", True)
    assert classify_stability(False, 0.20) == ("stable_reject", True)
    assert classify_stability(True, 0.75) == ("unstable", False)
    assert classify_stability(False, 0.85) == ("stable_pass", False)


def test_summarize_repeats_reports_requested_distributions():
    rows = pd.DataFrame(
        {
            "dataset": ["d"] * 5,
            "seed": [42] * 5,
            "rare_train_size": ["0.01"] * 5,
            "separability": [1.2] * 5,
            "original_pass": [True] * 5,
            "adaptive_pass": [True, True, True, True, False],
            "adaptive_reason": ["pass", "pass", "pass", "pass", "gain"],
            "active_folds": [5, 4, 5, 3, 2],
            "oof_ffr_wilson_upper": [0.005, 0.006, 0.007, 0.008, 0.009],
            "oof_delta_f1_lcb": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    summary = summarize_repeats(rows).iloc[0]
    assert summary["pass_rate"] == 0.8
    assert summary["stability_band"] == "stable_pass"
    assert bool(summary["consistent_with_frozen"])
    assert summary["active_folds_min"] == 2
    assert summary["active_folds_max"] == 5
    assert summary["wilson_ucb_min"] == 0.005
    assert summary["delta_f1_lcb_max"] == 0.5

