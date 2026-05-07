import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anndata as ad
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

    def test_train_reference_scanvi_uses_mps_when_available(self):
        module = importlib.import_module("scrare.models.scanvi")
        train_calls = []

        class DummySCVI:
            @staticmethod
            def setup_anndata(*args, **kwargs):
                return None

            def __init__(self, *args, **kwargs):
                return None

            def train(self, **kwargs):
                train_calls.append(kwargs)

        class DummySCANVI:
            @staticmethod
            def from_scvi_model(*args, **kwargs):
                return DummySCANVI()

            def train(self, **kwargs):
                train_calls.append(kwargs)

        with (
            patch.object(module.torch.backends.mps, "is_available", return_value=True),
            patch.object(module.scvi.model, "SCVI", DummySCVI),
            patch.object(module.scvi.model, "SCANVI", DummySCANVI),
        ):
            module.train_reference_scanvi(
                object(),
                batch_key="batch",
                unlabeled_category="Unknown",
                n_latent=2,
                batch_size=8,
                scvi_epochs=1,
                scanvi_epochs=1,
            )

        self.assertEqual([call["accelerator"] for call in train_calls], ["mps", "mps"])
        self.assertEqual([call["devices"] for call in train_calls], [1, 1])

    def test_scanvi_query_labels_do_not_become_nan_when_unlabeled_category_is_missing_from_train_categories(self):
        module = importlib.import_module("scrare.models.scanvi")
        query_adata = ad.AnnData(
            X=np.ones((2, 1), dtype=np.float32),
            obs=pd.DataFrame(index=["q0", "q1"]),
            var=pd.DataFrame(index=["gene0"]),
        )
        captured = {}

        def fake_load_query_data(query, scanvi_model):
            del scanvi_model
            captured["labels"] = query.obs["scanvi_label"].copy()
            return SimpleNamespace(is_trained_=False)

        with patch.object(module.scvi.model.SCANVI, "load_query_data", side_effect=fake_load_query_data):
            module.load_query_model(
                query_adata,
                object(),
                unlabeled_category="Unknown",
                label_categories=["ASDC", "pDC"],
            )

        self.assertEqual(captured["labels"].astype(str).tolist(), ["Unknown", "Unknown"])
        self.assertFalse(captured["labels"].isna().any())

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

    def test_configs_use_results_output_roots(self):
        for path in (REPO_ROOT / "configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            output_dir = str(config.get("experiment", {}).get("output_dir", "")).replace("\\", "/")
            self.assertTrue(output_dir.startswith("results/"), path.name)
            self.assertNotIn("outputs/", output_dir, path.name)


if __name__ == "__main__":
    unittest.main()
