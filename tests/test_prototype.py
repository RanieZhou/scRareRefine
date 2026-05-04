import unittest

import numpy as np
import pandas as pd

from scrare_refine.prototype import prototype_scores_from_reference


class PrototypeTests(unittest.TestCase):
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
