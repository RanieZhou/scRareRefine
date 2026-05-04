import unittest
from pathlib import Path

import yaml


class ProjectStateTests(unittest.TestCase):
    def test_legacy_transductive_entrypoints_are_removed(self):
        legacy_scripts = [
            "run_scanvi_p0.py",
            "analyze_p0.py",
            "evaluate_prototype_gate.py",
            "evaluate_marker_verifier.py",
            "evaluate_marker_threshold_validation.py",
            "evaluate_fusion.py",
        ]

        for name in legacy_scripts:
            self.assertFalse(Path("scripts", name).exists(), name)

    def test_configs_do_not_use_legacy_p0_output_roots(self):
        legacy_roots = {
            "outputs/immune_dc/p0",
            "outputs/immune_dc/cdc1",
            "outputs/pancreas/p0",
            "outputs/pancreas/epsilon",
        }

        for path in Path("configs").glob("*.yaml"):
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
            output_dir = str(config.get("experiment", {}).get("output_dir", ""))
            self.assertNotIn(output_dir.replace("\\", "/"), legacy_roots, path.name)


if __name__ == "__main__":
    unittest.main()
