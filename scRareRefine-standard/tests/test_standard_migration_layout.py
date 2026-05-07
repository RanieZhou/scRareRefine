from pathlib import Path
import tomllib

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CONFIG_PATHS = {
    "immune_dc.yaml": "data/raw/human_immune_health/human_immune_health_atlas_dc.h5ad",
    "pancreas_epsilon.yaml": "data/raw/pancreas/human_pancreas_norm_complexBatch.h5ad",
    "pancreas_gamma.yaml": "data/raw/pancreas/human_pancreas_norm_complexBatch.h5ad",
}


def test_core_package_layout_is_current_scrare_package() -> None:
    assert (REPO_ROOT / "src" / "scrare" / "__init__.py").is_file()

    pyproject_path = REPO_ROOT / "pyproject.toml"
    assert pyproject_path.is_file()
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    package_find = pyproject["tool"]["setuptools"]["packages"]["find"]

    assert package_find["where"] == ["src"]
    assert "scrare*" in package_find["include"]
    assert "sc_rare_refine*" not in package_find["include"]


def test_copied_configs_reference_existing_raw_data() -> None:
    for config_name, expected_dataset_path in EXPECTED_CONFIG_PATHS.items():
        config_path = REPO_ROOT / "configs" / config_name
        assert config_path.is_file(), config_name

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["dataset"]["path"] == expected_dataset_path
        assert (REPO_ROOT / expected_dataset_path).is_file(), expected_dataset_path


def test_archived_reports_and_figures_live_under_results() -> None:
    reports_dir = REPO_ROOT / "results" / "reports"
    figures_dir = REPO_ROOT / "results" / "figures"

    assert reports_dir.is_dir()
    assert figures_dir.is_dir()
    assert any(path.suffix == ".md" for path in reports_dir.iterdir())
    assert any(
        path.suffix.lower() in {".png", ".svg", ".pdf"}
        for path in figures_dir.iterdir()
    )


def test_operational_docs_name_scrare_package() -> None:
    for doc_name in ["README.md", "PROJECT_STRUCTURE.md", "CLAUDE.md", "AGENTS.md"]:
        doc_path = REPO_ROOT / doc_name
        assert doc_path.is_file(), doc_name
        text = doc_path.read_text(encoding="utf-8")
        assert "src/scrare" in text, doc_name
        assert "src/sc_rare_refine" not in text, doc_name
