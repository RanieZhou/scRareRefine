import numpy as np
import pandas as pd

from src.rescue import PrototypeRescuer
from tools.analysis.residual_signal import (
    CONTRASTS,
    METRICS,
    assign_primary_groups,
    cliffs_delta,
    prototype_metrics,
)


def test_primary_groups_are_one_hot_and_false_rescue_is_non_target_subset():
    y = pd.Series(["rare", "rare", "rare", "other", "other"])
    baseline = pd.Series(["rare", "other", "other", "other", "other"])
    final = pd.Series(["rare", "rare", "other", "rare", "other"])

    groups, false_rescue = assign_primary_groups(y, baseline, final, "rare")

    assert groups.tolist() == [
        "baseline_correct_rare",
        "true_rescued_rare",
        "unrescued_rare",
        "non_target",
        "non_target",
    ]
    assert false_rescue.tolist() == [False, False, False, True, False]
    assert np.all(y[false_rescue].ne("rare"))


def test_cliffs_delta_direction_and_empty_semantics():
    assert np.isclose(cliffs_delta(np.array([3, 4]), np.array([1, 2])), 1.0)
    assert np.isclose(cliffs_delta(np.array([1, 2]), np.array([3, 4])), -1.0)
    assert np.isnan(cliffs_delta(np.array([]), np.array([1, 2])))


def test_prototype_metrics_and_train_defined_competitor():
    train = np.array([[0.0], [0.2], [-0.2], [3.0], [3.2], [2.8], [8.0], [8.2], [7.8]])
    labels = pd.Series(["rare"] * 3 + ["near"] * 3 + ["far"] * 3)
    proto = PrototypeRescuer("rare")
    proto.fit(train, labels, np.ones(len(train), dtype=bool))

    metrics, competitor = prototype_metrics(proto, np.array([[0.1], [3.1]]))

    assert competitor == "near"
    assert set(METRICS).issubset(metrics.columns)
    assert metrics.loc[0, "rare_rank"] == 1
    assert metrics.loc[0, "prototype_margin"] > 0
    assert metrics.loc[1, "prototype_margin"] < 0


def test_contrast_grid_is_frozen_and_complete():
    assert set(CONTRASTS) == {
        "H1_baseline_correct_vs_true_rescued",
        "H2_true_rescued_vs_unrescued",
        "H3a_true_rescued_vs_non_target",
        "H3b_true_rescued_vs_closest_competitor",
    }
    assert METRICS["rare_membership_score"] == 1
    assert METRICS["rare_rank"] == -1
    assert METRICS["rare_standardized_distance"] == -1
    assert METRICS["standardized_prototype_margin"] == 1
    assert METRICS["nearest_nonrare_distance"] == 1
