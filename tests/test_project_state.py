import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProjectStateTests(unittest.TestCase):
    def test_scanvi_baseline_module_exists(self):
        module = importlib.import_module("scrare.models.scanvi")
        for name in [
            "prediction_outputs",
            "seed_everything",
            "train_reference_scanvi",
            "load_query_model",
        ]:
            self.assertTrue(hasattr(module, name), name)

    def test_scanvi_prediction_outputs_names_probability_columns_from_registry(self):
        module = importlib.import_module("scrare.models.scanvi")

        class DummyManager:
            def get_state_registry(self, key):
                self_key = key
                return SimpleNamespace(categorical_mapping=np.array(["ASDC", "pDC"]))

        class DummyModel:
            adata_manager = DummyManager()

            def predict(self, adata, soft=False):
                if soft:
                    return np.array([[0.8, 0.2], [0.3, 0.7]])
                return np.array(["ASDC", "pDC"])

            def get_latent_representation(self, adata):
                return np.array([[1.0, 0.0], [0.0, 1.0]])

        adata = SimpleNamespace(
            obs_names=pd.Index(["c0", "c1"]),
            obs=pd.DataFrame({"label": ["ASDC", "pDC"]}, index=["c0", "c1"]),
        )

        predictions, latent = module.prediction_outputs(DummyModel(), adata, "label", "ASDC")

        self.assertIn("prob_ASDC", predictions.columns)
        self.assertIn("prob_pDC", predictions.columns)
        self.assertEqual(predictions.loc[0, "top1_label"], "ASDC")
        self.assertEqual(predictions.loc[1, "top1_label"], "pDC")
        self.assertEqual(latent["cell_id"].tolist(), ["c0", "c1"])

    def test_legacy_script_entrypoints_are_removed(self):
        legacy_scripts = [
            "run_scanvi_p0.py",
            "analyze_p0.py",
            "evaluate_prototype_gate.py",
            "evaluate_marker_verifier.py",
            "evaluate_marker_threshold_validation.py",
            "evaluate_fusion.py",
            "audit_dataset.py",
            "run_scanvi_inductive.py",
            "evaluate_inductive_prototype_marker.py",
        ]

        for name in legacy_scripts:
            self.assertFalse((REPO_ROOT / "scripts" / name).exists(), name)

    def test_legacy_scrare_refine_python_implementation_is_removed(self):
        legacy_dir = REPO_ROOT / "scrare_refine"
        self.assertFalse(legacy_dir.exists(), "scrare_refine")

    def test_configs_do_not_use_legacy_p0_output_roots(self):
        legacy_roots = {
            "outputs/immune_dc/p0",
            "outputs/immune_dc/cdc1",
            "outputs/pancreas/p0",
            "outputs/pancreas/epsilon",
        }

        for path in (REPO_ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            output_dir = str(config.get("experiment", {}).get("output_dir", ""))
            self.assertNotIn(output_dir.replace("\\", "/"), legacy_roots, path.name)


if __name__ == "__main__":
    unittest.main()
