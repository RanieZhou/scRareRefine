import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.organize_outputs import plan_output_moves


class OrganizeOutputsTests(unittest.TestCase):
    def test_plans_root_files_and_run_artifacts_without_moving_metrics(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scanvi_metrics.csv").write_text("metrics", encoding="utf-8")
            (root / "confusion_matrix.png").write_text("figure", encoding="utf-8")
            (root / "run.log").write_text("log", encoding="utf-8")
            run = root / "runs" / "seed_42_rare_20"
            run.mkdir(parents=True)
            (run / "scanvi_predictions.csv").write_text("pred", encoding="utf-8")
            (run / "scanvi_latent.csv").write_text("latent", encoding="utf-8")
            (run / "prototype_scores.csv").write_text("proto", encoding="utf-8")
            (run / "per_class_metrics.csv").write_text("per", encoding="utf-8")

            moves = plan_output_moves(root)
            pairs = {(src.relative_to(root).as_posix(), dst.relative_to(root).as_posix()) for src, dst in moves}

            self.assertIn(("scanvi_metrics.csv", "tables/scanvi_metrics.csv"), pairs)
            self.assertIn(("confusion_matrix.png", "figures/confusion_matrix.png"), pairs)
            self.assertIn(("run.log", "logs/run.log"), pairs)
            self.assertIn(
                ("runs/seed_42_rare_20/scanvi_predictions.csv", "runs/seed_42_rare_20/artifacts/scanvi_predictions.csv"),
                pairs,
            )
            self.assertNotIn(
                ("runs/seed_42_rare_20/per_class_metrics.csv", "runs/seed_42_rare_20/artifacts/per_class_metrics.csv"),
                pairs,
            )


if __name__ == "__main__":
    unittest.main()
