import unittest

import numpy as np
import pandas as pd

from scrare_refine.splits import make_scanvi_labels


class LabelMaskingTests(unittest.TestCase):
    def test_masks_unlabeled_rare_cells_and_keeps_exact_labeled_count(self):
        obs = pd.DataFrame(
            {
                "cell_type": ["ASDC"] * 5 + ["pDC"] * 4 + ["cDC1"] * 3,
            },
            index=[f"cell{i}" for i in range(12)],
        )

        labels, is_labeled = make_scanvi_labels(
            obs,
            label_key="cell_type",
            rare_class="ASDC",
            rare_train_size=2,
            seed=42,
            unlabeled_category="Unknown",
        )

        self.assertEqual(int(((obs["cell_type"] == "ASDC") & is_labeled).sum()), 2)
        self.assertTrue(is_labeled[obs["cell_type"] != "ASDC"].all())
        self.assertTrue((labels[(obs["cell_type"] == "ASDC") & ~is_labeled] == "Unknown").all())
        self.assertEqual(set(labels[obs["cell_type"] != "ASDC"]), {"pDC", "cDC1"})

    def test_all_keeps_every_rare_cell_labeled(self):
        obs = pd.DataFrame({"cell_type": ["ASDC", "ASDC", "pDC"]})

        labels, is_labeled = make_scanvi_labels(
            obs,
            label_key="cell_type",
            rare_class="ASDC",
            rare_train_size="all",
            seed=42,
            unlabeled_category="Unknown",
        )

        self.assertEqual(labels.tolist(), ["ASDC", "ASDC", "pDC"])
        self.assertTrue(np.all(is_labeled))


if __name__ == "__main__":
    unittest.main()
