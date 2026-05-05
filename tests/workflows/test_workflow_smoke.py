from __future__ import annotations

import argparse

from scrare.cli import run_inductive


def test_run_inductive_cli_loads_config_and_invokes_workflow(monkeypatch) -> None:
    expected_config = {"experiment": {"rare_class": "rare"}}
    observed: dict[str, object] = {}

    def fake_load_config(path: str):
        observed["config_path"] = path
        return expected_config

    def fake_run_inductive_workflow(config: dict, args: argparse.Namespace):
        observed["config"] = config
        observed["args"] = args
        return "workflow-result"

    monkeypatch.setattr(run_inductive, "load_config", fake_load_config)
    monkeypatch.setattr(run_inductive, "run_inductive_workflow", fake_run_inductive_workflow)

    result = run_inductive.main(["--config", "configs/demo.yaml"])

    assert result == "workflow-result"
    assert observed["config_path"] == "configs/demo.yaml"
    assert observed["config"] is expected_config
    assert isinstance(observed["args"], argparse.Namespace)
    assert observed["args"].config == "configs/demo.yaml"
