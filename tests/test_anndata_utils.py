import unittest

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from scrare.data.preprocess import select_train_hvg_var_names


class AnndataUtilsTests(unittest.TestCase):
    def test_select_train_hvg_var_names_uses_train_matrix_only(self):
        train = ad.AnnData(
            X=np.array(
                [
                    [100, 0, 1],
                    [90, 0, 1],
                    [80, 0, 1],
                ],
                dtype=np.float32,
            ),
            var=pd.DataFrame(index=["train_high", "test_high", "stable"]),
        )

        selected = select_train_hvg_var_names(train, n_top_genes=1)

        self.assertEqual(selected, ["train_high"])

    def test_select_train_hvg_var_names_returns_all_genes_for_non_positive_or_large_cutoff(self):
        adata = ad.AnnData(
            X=np.array([[1, 0, 3], [0, 2, 1]], dtype=np.float32),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )

        self.assertEqual(select_train_hvg_var_names(adata, n_top_genes=None), ["g0", "g1", "g2"])
        self.assertEqual(select_train_hvg_var_names(adata, n_top_genes=0), ["g0", "g1", "g2"])
        self.assertEqual(select_train_hvg_var_names(adata, n_top_genes=3), ["g0", "g1", "g2"])

    def test_select_train_hvg_var_names_supports_sparse_input(self):
        adata = ad.AnnData(
            X=sparse.csr_matrix(np.array([[5, 0, 0], [4, 0, 1], [6, 0, 0]], dtype=np.float32)),
            var=pd.DataFrame(index=["g0", "g1", "g2"]),
        )

        selected = select_train_hvg_var_names(adata, n_top_genes=1)

        self.assertEqual(selected, ["g0"])


if __name__ == "__main__":
    unittest.main()
