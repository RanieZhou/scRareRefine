import unittest

import numpy as np
import pandas as pd

from scrare_refine.prototype import prototype_scores, prototype_scores_from_reference


class PrototypeTests(unittest.TestCase):
    def test_prototypes_use_only_labeled_training_cells(self):
        z = np.array(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [10.0, 10.0],
                [10.0, 12.0],
                [100.0, 100.0],
            ]
        )
        labels = pd.Series(["ASDC", "ASDC", "pDC", "pDC", "ASDC"])
        is_labeled = np.array([True, True, True, True, False])
        predicted = pd.Series(["pDC", "ASDC", "pDC", "pDC", "pDC"])
        margin = np.array([0.1, 0.8, 0.7, 0.6, 0.05])

        scores = prototype_scores(
            z,
            true_labels=labels,
            predicted_labels=predicted,
            is_labeled=is_labeled,
            rare_class="ASDC",
            margin=margin,
        )

        self.assertAlmostEqual(float(scores.loc[0, "distance_to_ASDC"]), 1.0)
        self.assertLess(float(scores.loc[0, "distance_to_ASDC"]), float(scores.loc[0, "distance_to_pred"]))
        self.assertEqual(int(scores.loc[0, "prototype_rank_ASDC"]), 1)
        self.assertTrue(bool(scores.loc[0, "prototype_rescue_candidate"]))
        self.assertFalse(bool(scores.loc[1, "prototype_rescue_candidate"]))

    def test_reference_prototypes_do_not_use_query_labels(self):
        reference_latent = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]])
        reference_labels = pd.Series(["ASDC", "ASDC", "pDC"])
        reference_is_labeled = np.array([True, False, True])
        query_latent = np.array([[0.1, 0.0], [9.8, 0.0]])
        predicted = pd.Series(["pDC", "pDC"])

        scores = prototype_scores_from_reference(
            query_latent,
            reference_latent=reference_latent,
            reference_labels=reference_labels,
            reference_is_labeled=reference_is_labeled,
            predicted_labels=predicted,
            rare_class="ASDC",
            margin=np.array([0.1, 0.9]),
        )

        self.assertEqual(int(scores.loc[0, "prototype_rank_ASDC"]), 1)
        self.assertGreater(float(scores.loc[0, "d_pred_minus_d_ASDC"]), 0.0)
        self.assertEqual(int(scores.loc[1, "prototype_rank_ASDC"]), 2)


if __name__ == "__main__":
    unittest.main()
