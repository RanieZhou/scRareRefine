import unittest

import numpy as np
import pandas as pd

from scrare_refine.metrics import classification_tables, compute_uncertainty, topk_review_recall


class MetricsTests(unittest.TestCase):
    def test_classification_tables_include_rare_and_macro_metrics(self):
        y_true = np.array(["ASDC", "ASDC", "pDC", "pDC", "cDC1"])
        y_pred = np.array(["ASDC", "pDC", "pDC", "pDC", "cDC1"])

        overall, per_class = classification_tables(y_true, y_pred, rare_class="ASDC")

        self.assertAlmostEqual(overall["overall_accuracy"], 0.8)
        self.assertAlmostEqual(overall["rare_recall"], 0.5)
        self.assertAlmostEqual(overall["rare_precision"], 1.0)
        self.assertIn("macro_f1", overall)
        self.assertEqual(set(per_class["label"]), {"ASDC", "pDC", "cDC1"})

    def test_uncertainty_and_topk_review_recall(self):
        probs = pd.DataFrame(
            {
                "ASDC": [0.4, 0.1, 0.2, 0.05],
                "pDC": [0.35, 0.8, 0.7, 0.9],
                "cDC1": [0.25, 0.1, 0.1, 0.05],
            }
        )
        uncertainty = compute_uncertainty(probs, rare_class="ASDC")

        self.assertAlmostEqual(float(uncertainty.loc[0, "max_prob"]), 0.4)
        self.assertAlmostEqual(float(uncertainty.loc[0, "margin"]), 0.05)
        self.assertFalse(bool(uncertainty.loc[0, "top2_is_ASDC"]))

        events = np.array([True, False, True, False])
        risk = np.array([0.9, 0.1, 0.8, 0.2])
        review = topk_review_recall(events, risk, ks=[0.25, 0.5])

        self.assertAlmostEqual(review.loc[review["k_fraction"] == 0.25, "event_recall"].iloc[0], 0.5)
        self.assertAlmostEqual(review.loc[review["k_fraction"] == 0.5, "event_recall"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
