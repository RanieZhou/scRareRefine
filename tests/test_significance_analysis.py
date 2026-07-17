import numpy as np
import pandas as pd
import pytest

from tools.analysis.significance_test import (
    _assert_complete_grid,
    _cluster_boot_ci,
    _collapse_effective_counts,
    _dataset_effects,
    _sign_test,
)


def test_dataset_effects_weight_datasets_equally():
    paired = pd.DataFrame(
        {"dataset": ["a", "a", "a", "b"], "delta": [1.0, 1.0, 1.0, -1.0]}
    )
    effects = _dataset_effects(paired)
    assert effects.to_dict() == {"a": 1.0, "b": -1.0}
    assert effects.mean() == 0.0


def test_cluster_bootstrap_resamples_dataset_effects():
    lo, hi = _cluster_boot_ci(np.array([1.0, 3.0]), n=2000, seed=7)
    assert lo == 1.0
    assert hi == 3.0


def test_sign_test_excludes_exact_ties():
    positive, zero, negative, p_value = _sign_test(np.array([0.2, 0.1, 0.0, -0.1]))
    assert (positive, zero, negative) == (2, 1, 1)
    assert p_value == 1.0


def test_collapse_aware_merges_repeated_effective_counts():
    paired = pd.DataFrame(
        {
            "dataset": ["a", "a", "a"],
            "rare_train_size": ["0.01", "0.05", "0.10"],
            "seed": [42, 42, 42],
            "delta": [0.2, 0.4, 0.9],
        }
    )
    counts = pd.DataFrame(
        {
            "dataset": ["a", "a", "a"],
            "rare_train_size": ["0.01", "0.05", "0.10"],
            "seed": [42, 42, 42],
            "split": ["train", "train", "train"],
            "train_labeled_rare": [5, 5, 10],
        }
    )
    collapsed = _collapse_effective_counts(paired, counts)
    assert collapsed["effective_rare_labels"].tolist() == [5, 10]
    assert np.allclose(collapsed["delta"].to_numpy(), [0.3, 0.9])


def test_incomplete_matched_grid_fails_closed():
    paired = pd.DataFrame(
        {"dataset": ["a"], "rare_train_size": ["0.01"], "seed": [42], "delta": [0.2]}
    )
    with pytest.raises(ValueError, match="Incomplete matched grid"):
        _assert_complete_grid(paired, {"0.01", "0.05"}, {42}, "scANVI")
