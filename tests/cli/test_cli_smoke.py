import importlib
from importlib import metadata

import pytest

from scrare.cli import audit


CLI_MODULES = [
    ("scrare-audit", "scrare.cli.audit"),
    ("scrare-run-inductive", "scrare.cli.run_inductive"),
    ("scrare-evaluate-posthoc", "scrare.cli.evaluate_posthoc"),
]


@pytest.mark.parametrize(("script_name", "module_name"), CLI_MODULES)
def test_cli_modules_are_importable(script_name: str, module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert callable(getattr(module, "main"))


def test_project_scripts_point_to_cli_main_functions() -> None:
    console_scripts = {
        entry_point.name: entry_point.value
        for entry_point in metadata.entry_points(group="console_scripts")
        if entry_point.name in {script_name for script_name, _ in CLI_MODULES}
    }

    assert console_scripts == {
        "scrare-audit": "scrare.cli.audit:main",
        "scrare-run-inductive": "scrare.cli.run_inductive:main",
        "scrare-evaluate-posthoc": "scrare.cli.evaluate_posthoc:main",
    }


def test_audit_cli_exposes_parser() -> None:
    parser = audit.build_parser()
    assert parser.prog is not None


@pytest.mark.parametrize(
    ("script_name", "module_name", "expected_code", "expected_stderr", "expected_message"),
    [
        (
            "scrare-audit",
            "scrare.cli.audit",
            2,
            "the following arguments are required: --config",
            None,
        ),
        (
            "scrare-run-inductive",
            "scrare.cli.run_inductive",
            2,
            "the following arguments are required: --config",
            None,
        ),
        (
            "scrare-evaluate-posthoc",
            "scrare.cli.evaluate_posthoc",
            2,
            "the following arguments are required: --config",
            None,
        ),
    ],
)
def test_cli_main_smoke_behaviors(
    script_name: str,
    module_name: str,
    expected_code: int | None,
    expected_stderr: str | None,
    expected_message: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    del script_name
    module = importlib.import_module(module_name)

    if expected_message is not None:
        with pytest.raises(SystemExit, match=expected_message):
            module.main([])
        return

    with pytest.raises(SystemExit) as exc_info:
        module.main([])

    assert exc_info.value.code == expected_code
    captured = capsys.readouterr()
    assert expected_stderr is not None
    assert expected_stderr in captured.err
