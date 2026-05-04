import unittest

import anndata as ad
import numpy as np
import pandas as pd

from scrare_refine.fusion import prototype_probabilities_from_reference
from scrare_refine.inductive import (
    batch_heldout_split,
    cell_stratified_split,
    make_inductive_scanvi_labels,
    select_train_hvg_var_names,
)


class InductiveSplitTests(unittest.TestCase):
    def test_cell_stratified_split_has_disjoint_splits_and_preserves_classes(self):
        obs = pd.DataFrame(
            {
                "label": ["ASDC"] * 20 + ["pDC"] * 40 + ["cDC1"] * 20,
            },
            index=[f"cell{i}" for i in range(80)],
        )

        split = cell_stratified_split(obs, label_key="label", seed=42)

        self.assertEqual(split.index.tolist(), obs.index.tolist())
        self.assertEqual(set(split.unique()), {"train", "validation", "test"})
        for name in ["train", "validation", "test"]:
            labels = set(obs.loc[split.eq(name), "label"])
            self.assertEqual(labels, {"ASDC", "pDC", "cDC1"})
        self.assertEqual(int(split.eq("train").sum()) + int(split.eq("validation").sum()) + int(split.eq("test").sum()), 80)

    def test_batch_heldout_split_keeps_batches_disjoint(self):
        obs = pd.DataFrame(
            {
                "label": ["ASDC", "pDC", "cDC1", "pDC"] * 12,
                "batch": np.repeat([f"b{i}" for i in range(12)], 4),
            },
            index=[f"cell{i}" for i in range(48)],
        )

        split = batch_heldout_split(obs, label_key="label", batch_key="batch", seed=42)

        batch_to_split = obs.assign(split=split).groupby("batch")["split"].nunique()
        self.assertTrue(batch_to_split.eq(1).all())
        self.assertEqual(set(split.unique()), {"train", "validation", "test"})

    def test_batch_heldout_split_seed_changes_batch_assignment(self):
        obs = pd.DataFrame(
            {
                "label": ["ASDC", "pDC", "cDC1", "pDC"] * 18,
                "batch": np.repeat([f"b{i:02d}" for i in range(18)], 4),
            },
            index=[f"cell{i}" for i in range(72)],
        )

        first = batch_heldout_split(obs, label_key="label", batch_key="batch", seed=42)
        second = batch_heldout_split(obs, label_key="label", batch_key="batch", seed=43)

        first_by_batch = obs.assign(split=first).groupby("batch")["split"].first()
        second_by_batch = obs.assign(split=second).groupby("batch")["split"].first()
        self.assertFalse(first_by_batch.equals(second_by_batch))

    def test_batch_heldout_split_keeps_each_split_nonempty_with_few_imbalanced_batches(self):
        batch_label_counts = {
            "celseq": {"acinar": 228, "activated_stellate": 19, "alpha": 191, "beta": 161, "delta": 50, "ductal": 327, "endothelial": 5, "epsilon": 1, "gamma": 18, "macrophage": 1, "mast": 1, "quiescent_stellate": 1, "schwann": 1},
            "celseq2": {"acinar": 274, "activated_stellate": 90, "alpha": 843, "beta": 445, "delta": 203, "ductal": 258, "endothelial": 21, "epsilon": 4, "gamma": 110, "macrophage": 15, "mast": 6, "quiescent_stellate": 12, "schwann": 4},
            "fluidigmc1": {"acinar": 21, "activated_stellate": 16, "alpha": 239, "beta": 258, "delta": 25, "ductal": 36, "endothelial": 14, "epsilon": 1, "gamma": 18, "macrophage": 1, "mast": 3, "quiescent_stellate": 1, "schwann": 5},
            "inDrop1": {"acinar": 110, "activated_stellate": 51, "alpha": 236, "beta": 872, "delta": 214, "ductal": 120, "endothelial": 130, "epsilon": 13, "gamma": 70, "macrophage": 14, "mast": 8, "quiescent_stellate": 92, "schwann": 5, "t_cell": 2},
            "inDrop2": {"acinar": 3, "activated_stellate": 81, "alpha": 676, "beta": 371, "delta": 125, "ductal": 301, "endothelial": 23, "epsilon": 2, "gamma": 86, "macrophage": 17, "mast": 9, "quiescent_stellate": 22, "schwann": 6, "t_cell": 2},
            "inDrop3": {"acinar": 843, "activated_stellate": 100, "alpha": 1130, "beta": 787, "delta": 161, "ductal": 376, "endothelial": 92, "epsilon": 2, "gamma": 36, "macrophage": 14, "mast": 7, "quiescent_stellate": 54, "schwann": 1, "t_cell": 2},
            "inDrop4": {"acinar": 2, "activated_stellate": 52, "alpha": 284, "beta": 495, "delta": 101, "ductal": 280, "endothelial": 7, "epsilon": 1, "gamma": 63, "macrophage": 10, "mast": 1, "quiescent_stellate": 5, "schwann": 1, "t_cell": 1},
            "smarter": {"alpha": 886, "beta": 472, "delta": 49, "gamma": 85},
            "smartseq2": {"acinar": 188, "activated_stellate": 55, "alpha": 1008, "beta": 308, "delta": 127, "ductal": 444, "endothelial": 21, "epsilon": 8, "gamma": 213, "macrophage": 7, "mast": 7, "quiescent_stellate": 6, "schwann": 2},
        }
        rows = []
        for batch, counts in batch_label_counts.items():
            for label, n_cells in counts.items():
                rows.extend({"label": label, "batch": batch} for _ in range(n_cells))
        obs = pd.DataFrame(rows)

        split = batch_heldout_split(obs, label_key="label", batch_key="batch", seed=42)

        self.assertEqual(set(split.unique()), {"train", "validation", "test"})
        batch_to_split = obs.assign(split=split).groupby("batch")["split"].nunique()
        self.assertTrue(batch_to_split.eq(1).all())

    def test_inductive_labels_never_expose_validation_or_test_labels(self):
        obs = pd.DataFrame(
            {
                "label": ["ASDC"] * 6 + ["pDC"] * 6,
            },
            index=[f"cell{i}" for i in range(12)],
        )
        split = pd.Series(["train"] * 8 + ["validation"] * 2 + ["test"] * 2, index=obs.index)

        labels, is_labeled = make_inductive_scanvi_labels(
            obs,
            split,
            label_key="label",
            rare_class="ASDC",
            rare_train_size=2,
            seed=42,
            unlabeled_category="Unknown",
        )

        self.assertTrue((labels[split.ne("train")] == "Unknown").all())
        self.assertFalse(is_labeled[split.ne("train")].any())
        self.assertEqual(int(((obs["label"] == "ASDC") & split.eq("train") & is_labeled).sum()), 2)
        self.assertTrue(is_labeled[(obs["label"] == "pDC") & split.eq("train")].all())

    def test_hvg_names_are_selected_from_train_only(self):
        train = ad.AnnData(
            X=np.array([[100, 0, 1], [90, 0, 1], [80, 0, 1]], dtype=np.float32),
            var=pd.DataFrame(index=["train_high", "test_high", "stable"]),
        )
        full = ad.AnnData(
            X=np.array([[1, 1000, 1], [1, 900, 1], [1, 800, 1]], dtype=np.float32),
            var=pd.DataFrame(index=["train_high", "test_high", "stable"]),
        )

        genes = select_train_hvg_var_names(train, n_top_genes=1)

        self.assertEqual(genes, ["train_high"])
        self.assertEqual(full[:, genes].var_names.tolist(), ["train_high"])


class TrainOnlyPrototypeTests(unittest.TestCase):
    def test_prototype_probabilities_use_reference_labeled_cells_only(self):
        reference_latent = np.array([[0.0, 0.0], [0.2, 0.0], [10.0, 0.0]])
        reference_labels = pd.Series(["rare", "rare", "major"])
        reference_is_labeled = np.array([True, False, True])
        query_latent = np.array([[0.1, 0.0], [9.5, 0.0]])

        probs = prototype_probabilities_from_reference(
            query_latent,
            reference_latent=reference_latent,
            reference_labels=reference_labels,
            reference_is_labeled=reference_is_labeled,
            temperature=1.0,
        )

        self.assertGreater(float(probs.loc[0, "rare"]), 0.99)
        self.assertGreater(float(probs.loc[1, "major"]), 0.99)


if __name__ == "__main__":
    unittest.main()
