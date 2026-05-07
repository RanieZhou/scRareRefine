from pathlib import Path
import unittest

from scrare.infra.paths import artifact_path, root_table_path, stage_table_path
from scrare.workflows import posthoc


class OutputLayoutTests(unittest.TestCase):
    def test_run_artifacts_live_under_artifacts_directory(self):
        root = Path("results/immune_dc/inductive_batch/asdc")
        run_dir = root / "runs" / "batch_heldout_seed_42_rare_20"

        self.assertEqual(
            artifact_path(run_dir, "scanvi_predictions.csv"),
            run_dir / "artifacts" / "scanvi_predictions.csv",
        )

    def test_root_outputs_are_classified_by_kind(self):
        root = Path("results/immune_dc/inductive_batch/asdc")

        self.assertEqual(root_table_path(root, "scanvi_metrics.csv"), root / "tables" / "scanvi_metrics.csv")
        self.assertEqual(
            stage_table_path(root, "prototype_marker_validation", "prototype_marker_effect_summary.csv"),
            root / "stages" / "prototype_marker_validation" / "prototype_marker_effect_summary.csv",
        )

    def test_posthoc_default_output_root_uses_results(self):
        root = posthoc._output_root(
            {"dataset": {"name": "immune_dc"}},
            rare_class="ASDC",
            split_mode="batch_heldout",
        )

        self.assertEqual(root, Path("results/immune_dc/inductive_batch/asdc"))


if __name__ == "__main__":
    unittest.main()
