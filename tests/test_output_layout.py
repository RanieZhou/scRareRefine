import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scrare_refine.output_layout import (
    artifact_path,
    root_table_path,
    stage_table_path,
    legacy_or_artifact_path,
    existing_table_path,
    classify_root_file,
)


class OutputLayoutTests(unittest.TestCase):
    def test_run_artifacts_live_under_artifacts_directory(self):
        root = Path("outputs/immune_dc_p0")
        run_dir = root / "runs" / "seed_42_rare_20"

        self.assertEqual(
            artifact_path(run_dir, "scanvi_predictions.csv"),
            run_dir / "artifacts" / "scanvi_predictions.csv",
        )
        with TemporaryDirectory() as tmp:
            isolated_run = Path(tmp) / "run"
            (isolated_run / "artifacts").mkdir(parents=True)
            legacy = isolated_run / "scanvi_latent.csv"
            artifact = isolated_run / "artifacts" / "scanvi_latent.csv"
            legacy.write_text("legacy", encoding="utf-8")
            self.assertEqual(legacy_or_artifact_path(isolated_run, "scanvi_latent.csv"), legacy)
            artifact.write_text("artifact", encoding="utf-8")
            self.assertEqual(legacy_or_artifact_path(isolated_run, "scanvi_latent.csv"), artifact)

    def test_existing_table_path_falls_back_from_parquet_to_csv_in_artifacts(self):
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            artifact_csv = run_dir / "artifacts" / "scanvi_predictions.csv"
            artifact_csv.parent.mkdir(parents=True)
            artifact_csv.write_text("csv", encoding="utf-8")

            self.assertEqual(existing_table_path(run_dir, "scanvi_predictions.parquet"), artifact_csv)

    def test_root_outputs_are_classified_by_kind(self):
        root = Path("outputs/immune_dc_p0")

        self.assertEqual(root_table_path(root, "scanvi_metrics.csv"), root / "tables" / "scanvi_metrics.csv")
        self.assertEqual(
            stage_table_path(root, "prototype_gate", "gate_effect_summary.csv"),
            root / "stages" / "prototype_gate" / "gate_effect_summary.csv",
        )
        self.assertEqual(classify_root_file("confusion_matrix.png"), "figures")
        self.assertEqual(classify_root_file("run_seed42_rare20.err.log"), "logs")
        self.assertEqual(classify_root_file("scanvi_metrics.csv"), "tables")


if __name__ == "__main__":
    unittest.main()
