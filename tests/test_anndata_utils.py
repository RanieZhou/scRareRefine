import unittest

import anndata as ad
import numpy as np
import pandas as pd

from scrare_refine.anndata_utils import select_top_variable_genes


class AnndataUtilsTests(unittest.TestCase):
    def test_select_top_variable_genes_prefers_precomputed_gene_counts(self):
        adata = ad.AnnData(
            X=np.array(
                [
                    [100, 0, 1, 2],
                    [0, 0, 1, 2],
                    [0, 0, 1, 2],
                ],
                dtype=np.float32,
            ),
            var=pd.DataFrame(
                {"n_cells_by_counts": [1, 4, 3, 2]},
                index=["g0", "g1", "g2", "g3"],
            ),
        )

        selected = select_top_variable_genes(adata, n_top_genes=2)

        self.assertEqual(selected.var_names.tolist(), ["g1", "g2"])


if __name__ == "__main__":
    unittest.main()
