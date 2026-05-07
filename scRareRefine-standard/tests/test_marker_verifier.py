import unittest

import numpy as np
import pandas as pd

from scrare.models.marker import (
    choose_marker_threshold,
    compute_marker_signatures,
    evaluate_threshold_rescue,
    marker_scores_for_candidates,
    marker_threshold_curve,
)


class MarkerVerifierTests(unittest.TestCase):
    def test_train_only_signatures_and_candidate_scores(self):
        expression = np.array(
            [
                [10.0, 0.0, 1.0],
                [9.0, 0.0, 1.0],
                [0.0, 8.0, 1.0],
                [0.0, 9.0, 1.0],
                [0.0, 10.0, 1.0],
            ]
        )
        labels = pd.Series(["ASDC", "ASDC", "pDC", "pDC", "ASDC"])
        is_labeled = np.array([True, True, True, True, False])

        signatures = compute_marker_signatures(
            expression,
            gene_names=["ASDC_marker", "pDC_marker", "housekeeping"],
            labels=labels,
            is_labeled=is_labeled,
            top_n=1,
            min_cells=2,
        )

        self.assertEqual(signatures["ASDC"], ["ASDC_marker"])
        self.assertEqual(signatures["pDC"], ["pDC_marker"])

        candidates = pd.DataFrame(
            {
                "true_label": ["ASDC", "pDC"],
                "predicted_label": ["pDC", "pDC"],
            },
            index=[0, 2],
        )
        scores = marker_scores_for_candidates(
            expression,
            candidates,
            signatures=signatures,
            rare_class="ASDC",
            gene_names=["ASDC_marker", "pDC_marker", "housekeeping"],
        )

        self.assertGreater(float(scores.loc[0, "marker_margin"]), 0)
        self.assertLess(float(scores.loc[2, "marker_margin"]), 0)
        self.assertTrue(bool(scores.loc[0, "marker_verified"]))
        self.assertFalse(bool(scores.loc[2, "marker_verified"]))

    def test_marker_threshold_curve_and_recommendation(self):
        predictions = pd.DataFrame(
            {
                "true_label": ["ASDC", "ASDC", "pDC", "pDC"],
                "predicted_label": ["pDC", "pDC", "pDC", "pDC"],
            }
        )
        candidates = pd.DataFrame(
            {
                "marker_margin": [2.0, 0.2, 1.0, -0.5],
            },
            index=[0, 1, 2, 3],
        )

        curve = marker_threshold_curve(
            predictions,
            candidates,
            rare_class="ASDC",
            thresholds=[-1.0, 0.0, 0.5, 1.5],
        )
        loose = curve[curve["marker_threshold"].eq(-1.0)].iloc[0]
        strict = curve[curve["marker_threshold"].eq(1.5)].iloc[0]

        self.assertEqual(int(loose["n_marker_verified"]), 4)
        self.assertEqual(int(strict["n_marker_verified"]), 1)
        self.assertGreater(float(strict["rare_precision"]), float(loose["rare_precision"]))
        self.assertEqual(choose_marker_threshold(curve, max_false_rescue_rate=0.26), 1.5)

    def test_evaluate_threshold_rescue_uses_only_selected_candidates(self):
        predictions = pd.DataFrame(
            {
                "true_label": ["ASDC", "ASDC", "pDC", "pDC"],
                "predicted_label": ["pDC", "pDC", "pDC", "pDC"],
            },
            index=[0, 1, 2, 3],
        )
        candidates = pd.DataFrame(
            {
                "marker_margin": [1.0, 0.1, 0.8],
            },
            index=[0, 1, 2],
        )

        metrics = evaluate_threshold_rescue(
            predictions,
            candidates,
            rare_class="ASDC",
            marker_threshold=0.5,
        )

        self.assertEqual(metrics["n_marker_verified"], 2)
        self.assertEqual(metrics["rescued_rare_errors"], 1)
        self.assertEqual(metrics["false_rescues"], 1)
        self.assertAlmostEqual(metrics["major_to_rare_false_rescue_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
