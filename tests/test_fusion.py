from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scrare_refine.fusion import (
    confidence_weight,
    evaluate_fusion_effect,
    fuse_predictions,
    prototype_probabilities_from_reference,
    select_best_params,
)


def _make_latent_and_labels():
    """Two well-separated clusters + a few ambiguous cells."""
    rng = np.random.default_rng(0)
    # Class A cluster around [5, 0], class B around [-5, 0]
    latent_a = rng.normal(loc=[5, 0], scale=0.5, size=(40, 2))
    latent_b = rng.normal(loc=[-5, 0], scale=0.5, size=(10, 2))
    # Two ambiguous cells near the boundary
    latent_amb = np.array([[0.5, 0.0], [-0.5, 0.0]])
    latent = np.vstack([latent_a, latent_b, latent_amb])
    labels = pd.Series(["A"] * 40 + ["B"] * 10 + ["A", "B"])
    is_labeled = np.array([True] * 50 + [False, False])
    return latent, labels, is_labeled


class TestPrototypeProbabilities:
    def test_output_shape_and_sums(self):
        latent, labels, is_labeled = _make_latent_and_labels()
        probs = prototype_probabilities_from_reference(
            latent,
            reference_latent=latent,
            reference_labels=labels,
            reference_is_labeled=is_labeled,
            temperature=1.0,
        )
        assert probs.shape == (52, 2)
        np.testing.assert_allclose(probs.sum(axis=1).to_numpy(), 1.0, atol=1e-6)

    def test_labeled_cells_near_own_prototype(self):
        latent, labels, is_labeled = _make_latent_and_labels()
        probs = prototype_probabilities_from_reference(
            latent,
            reference_latent=latent,
            reference_labels=labels,
            reference_is_labeled=is_labeled,
            temperature=1.0,
        )
        # Class A cells (rows 0-39) should have high prob for A
        assert (probs.iloc[:40]["A"] > 0.99).all()
        # Class B cells (rows 40-49) should have high prob for B
        assert (probs.iloc[40:50]["B"] > 0.99).all()


class TestConfidenceWeight:
    def test_range(self):
        margin = np.array([0.0, 0.5, 1.0])
        alpha = confidence_weight(margin, alpha_min=0.4)
        np.testing.assert_allclose(alpha, [0.4, 0.7, 1.0])

    def test_min_clamp(self):
        alpha = confidence_weight(np.array([-0.1, 0.0]), alpha_min=0.3)
        assert alpha[0] == pytest.approx(0.3)
        assert alpha[1] == pytest.approx(0.3)


class TestFusePredictions:
    def test_pure_scanvi_when_alpha_one(self):
        p_s = pd.DataFrame({"A": [0.9, 0.1], "B": [0.1, 0.9]})
        p_p = pd.DataFrame({"A": [0.1, 0.9], "B": [0.9, 0.1]})
        labels, _ = fuse_predictions(p_s, p_p, alpha=np.array([1.0, 1.0]))
        assert list(labels) == ["A", "B"]

    def test_pure_proto_when_alpha_zero(self):
        p_s = pd.DataFrame({"A": [0.9, 0.1], "B": [0.1, 0.9]})
        p_p = pd.DataFrame({"A": [0.1, 0.9], "B": [0.9, 0.1]})
        labels, _ = fuse_predictions(p_s, p_p, alpha=np.array([0.0, 0.0]))
        assert list(labels) == ["B", "A"]


class TestEvaluateFusionEffect:
    def test_no_change(self):
        y_true = pd.Series(["A", "B", "A"])
        baseline = pd.Series(["A", "B", "A"])
        fused = pd.Series(["A", "B", "A"])
        result = evaluate_fusion_effect(y_true, baseline, fused, rare_class="B")
        assert result["n_changed"] == 0
        assert result["rescued_rare_errors"] == 0
        assert result["damaged_correct"] == 0

    def test_rescue_and_damage(self):
        y_true = pd.Series(["B", "A", "A"])
        baseline = pd.Series(["A", "A", "A"])  # B misclassified
        fused = pd.Series(["B", "B", "A"])  # first rescued, second damaged
        result = evaluate_fusion_effect(y_true, baseline, fused, rare_class="B")
        assert result["rescued_rare_errors"] == 1
        assert result["false_rescues"] == 1
        assert result["damaged_correct"] == 1


class TestSelectBestParams:
    def test_selects_highest_f1(self):
        rows = [
            {"temperature": 1.0, "alpha_min": 0.5, "rare_f1": 0.8,
             "overall_accuracy": 0.95, "major_to_rare_false_rescue_rate": 0.001},
            {"temperature": 2.0, "alpha_min": 0.3, "rare_f1": 0.9,
             "overall_accuracy": 0.94, "major_to_rare_false_rescue_rate": 0.002},
        ]
        df = pd.DataFrame(rows)
        t, a, beta = select_best_params(df, baseline_accuracy=0.95, max_accuracy_drop=0.02)
        assert t == 2.0
        assert a == 0.3
        assert beta == 1.0
