import unittest

import anndata as ad
import numpy as np
import pandas as pd

from scrare.evaluation.audit import audit_anndata, matrix_is_integer_like


class AuditTests(unittest.TestCase):
    def test_audit_counts_labels_batches_and_rare_candidates(self):
        adata = ad.AnnData(
            X=np.array([[1, 0], [0, 2], [3, 0], [0, 4], [1, 1]], dtype=np.int64),
            obs=pd.DataFrame(
                {
                    "label": ["major", "major", "rare", "major", "tiny"],
                    "batch": ["b1", "b1", "b2", "b2", "b2"],
                },
                index=[f"cell{i}" for i in range(5)],
            ),
            var=pd.DataFrame(index=["g1", "g2"]),
        )

        summary, class_dist, batch_dist = audit_anndata(
            adata,
            dataset_name="toy",
            label_key="label",
            batch_key="batch",
            rare_threshold=0.25,
            rare_max_cells=1,
        )

        self.assertEqual(summary["dataset"], "toy")
        self.assertEqual(summary["n_cells"], 5)
        self.assertEqual(summary["n_genes"], 2)
        self.assertEqual(summary["n_classes"], 3)
        self.assertEqual(summary["n_batches"], 2)
        self.assertTrue(summary["x_integer_like"])

        rare_rows = class_dist[class_dist["is_rare_candidate"]]
        self.assertEqual(set(rare_rows["label"]), {"rare", "tiny"})
        self.assertEqual(batch_dist.set_index("batch").loc["b2", "n_cells"], 3)

    def test_matrix_is_integer_like_rejects_log_values(self):
        self.assertFalse(matrix_is_integer_like(np.array([[0.0, 1.5], [2.0, 0.0]])))


if __name__ == "__main__":
    unittest.main()
