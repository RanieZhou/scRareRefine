import unittest

import pandas as pd

from scrare_refine.prototype_gate import evaluate_gate_rules


class PrototypeGateTests(unittest.TestCase):
    def test_gate_rules_report_relabel_metrics_and_false_rescue(self):
        predictions = pd.DataFrame(
            {
                "cell_id": ["c0", "c1", "c2", "c3"],
                "true_label": ["ASDC", "ASDC", "pDC", "pDC"],
                "predicted_label": ["pDC", "ASDC", "pDC", "pDC"],
                "margin": [0.05, 0.8, 0.03, 0.9],
                "entropy": [1.0, 0.1, 1.1, 0.2],
            }
        )
        prototype = pd.DataFrame(
            {
                "prototype_rank_ASDC": [1, 1, 1, 3],
                "d_pred_minus_d_ASDC": [5.0, 0.0, 4.0, -1.0],
            }
        )

        effect, candidates = evaluate_gate_rules(predictions, prototype, rare_class="ASDC")
        rank1 = effect[effect["gate_name"].eq("rank1")].iloc[0]

        self.assertEqual(int(rank1["n_candidates"]), 2)
        self.assertEqual(int(rank1["rescued_rare_errors"]), 1)
        self.assertEqual(int(rank1["false_rescues"]), 1)
        self.assertAlmostEqual(float(rank1["candidate_precision_for_rare_error"]), 0.5)
        self.assertIn("rank1", set(candidates["gate_name"]))
        self.assertEqual(set(candidates[candidates["gate_name"].eq("rank1")]["cell_id"]), {"c0", "c2"})


if __name__ == "__main__":
    unittest.main()
