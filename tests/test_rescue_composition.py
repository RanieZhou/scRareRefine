import numpy as np
import pandas as pd

from tools.analysis.rescue_composition import (
    _expected_labeled_rare,
    _notes,
    composition_from_historical_counts,
    compute_composition,
)


def test_composition_invariants_and_error_rate_alias():
    y_true = pd.Series(["rare", "rare", "rare", "other", "other"])
    baseline = pd.Series(["rare", "other", "other", "other", "other"])
    final = pd.Series(["rare", "rare", "other", "rare", "other"])

    metrics, changed = compute_composition(y_true, baseline, final, "rare")

    assert changed.tolist() == [False, True, False, True, False]
    assert metrics["baseline_missed_rare"] == 2
    assert metrics["true_rescues"] == 1
    assert metrics["remaining_missed_rare"] == 1
    assert metrics["all_rescues"] == 2
    assert metrics["false_rescues"] == 1
    assert metrics["rescue_precision"] + metrics["rescue_fdp"] == 1.0
    assert metrics["incremental_fpr"] == metrics["rescue_ffr"] == 0.5


def test_zero_denominators_are_missing_not_zero():
    metrics, _ = compute_composition(
        pd.Series(["rare", "rare"]),
        pd.Series(["rare", "rare"]),
        pd.Series(["rare", "rare"]),
        "rare",
    )

    assert np.isnan(metrics["recovery_rate"])
    assert np.isnan(metrics["rescue_precision"])
    assert np.isnan(metrics["rescue_fdp"])
    assert np.isnan(metrics["incremental_fpr"])
    assert np.isnan(metrics["rescue_ffr"])


def test_expected_labeled_rare_matches_training_floor_and_cap():
    assert _expected_labeled_rare(120, "0.01") == 5
    assert _expected_labeled_rare(120, "0.05") == 6
    assert _expected_labeled_rare(4, "0.01") == 4
    assert _expected_labeled_rare(120, "all") == 120


def test_notes_accepts_csv_style_boolean_objects():
    frame = pd.DataFrame(
        [
            {
                "status": "success",
                "abstain": True,
                "all_rescues": 0,
                "rare_train_size": "0.01",
                "true_rescues": 0,
                "false_rescues": 0,
                "incremental_fpr": 0.0,
                "reconstruction_basis": "current_formal_replay",
            },
            {
                "status": "success",
                "abstain": False,
                "all_rescues": 2,
                "rare_train_size": "0.05",
                "true_rescues": 2,
                "false_rescues": 0,
                "incremental_fpr": 0.0,
                "reconstruction_basis": "current_formal_replay",
            },
        ],
        dtype=object,
    )

    notes = _notes(frame)

    assert "Non-abstaining runs: 1" in notes


def test_historical_counts_recover_only_traceable_composition():
    metrics = composition_from_historical_counts(
        pd.Series(["rare", "rare", "other", "other"]),
        pd.Series(["other", "other", "other", "other"]),
        "rare",
        n_rescued=2,
        n_false_rescue=1,
        historical_incremental_fpr=0.5,
    )

    assert metrics["true_rescues"] == 1
    assert metrics["remaining_missed_rare"] == 1
    assert metrics["rescue_precision"] == 0.5
    assert metrics["rescue_fdp"] == 0.5
