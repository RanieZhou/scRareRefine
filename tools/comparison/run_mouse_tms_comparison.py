"""Run full 9-method comparison on the two mouse TMS datasets.

This is the add-on benchmark runner for:
  2 datasets x 3 seeds x 4 rare_train_size x 9 methods = 216 result rows.

It preserves existing human results. Method scripts replace only rows matching
the requested mouse (dataset, seed, rare_train_size, method) keys.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config
from tools.comparison._conda_python import conda_python


ROOT = Path(__file__).resolve().parent.parent.parent
LOG_DIR = ROOT / "results" / "mouse_tms_comparison" / "logs"

DEFAULT_CONFIGS = [
    "configs/mouse_lung_tms_10x.yaml",
    "configs/mouse_pancreas_tms_10x.yaml",
]
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_RTS = ["0.01", "0.05", "0.10", "all"]


@dataclass(frozen=True)
class Method:
    name: str
    env: str
    script: str


METHODS = [
    Method("scANVI", "scanvi311", "tools/comparison/run_scanvi_comparison.py"),
    Method("kNN", "scanvi311", "tools/comparison/run_knn_comparison.py"),
    Method("CellTypist", "scanvi311", "tools/comparison/run_celltypist_comparison.py"),
    Method("scBalance", "scanvi311", "tools/comparison/run_scbalance_comparison.py"),
    Method("ProtoCloud", "sandbox310", "tools/comparison/run_protocloud_comparison.py"),
    Method("HiCat", "sandbox310", "tools/comparison/run_hicat_comparison.py"),
    Method("scCAD", "scanvi311", "tools/comparison/run_scCAD_comparison.py"),
    Method("TOSICA", "sandbox310", "tools/comparison/run_tosica_comparison.py"),
    Method("scRareRefine", "scanvi311", "tools/comparison/run_scrarerefine_comparison.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mouse TMS 9-method comparison.")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--rts", nargs="+", default=DEFAULT_RTS)
    parser.add_argument(
        "--stage",
        choices=["all", "cache", "methods"],
        default="all",
        help="cache = run_pipeline only; methods = comparison scripts only.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[m.name for m in METHODS],
        help="Subset of methods to run in the methods stage.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to run_pipeline cache jobs. Comparison rows are always replaced per method key.",
    )
    return parser.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run_command(cmd: list[str], *, log_path: Path, dry_run: bool) -> int:
    print("  " + quote_cmd(cmd), flush=True)
    if dry_run:
        return 0

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + quote_cmd(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def cache_commands(configs: list[str], seeds: list[int], rts_values: list[str], *, force: bool) -> list[tuple[str, list[str]]]:
    python_exe = conda_python("scanvi311")
    commands: list[tuple[str, list[str]]] = []
    for config_path in configs:
        cfg = load_config(config_path)
        dataset = cfg["dataset"]["name"]
        rare_class = cfg["experiment"]["rare_class"]
        for seed in seeds:
            for rts in rts_values:
                label = f"cache_{dataset}_seed{seed}_rare{rts.replace('.', 'p')}"
                cmd = [
                    python_exe,
                    "-u",
                    "run_pipeline.py",
                    "--config",
                    config_path,
                    "--seed",
                    str(seed),
                    "--rare_class",
                    rare_class,
                    "--rare_train_size",
                    rts,
                ]
                if force:
                    cmd.append("--force")
                commands.append((label, cmd))
    return commands


def method_commands(configs: list[str], seeds: list[int], rts_values: list[str], method_names: list[str]) -> list[tuple[str, list[str]]]:
    requested = set(method_names)
    unknown = requested - {m.name for m in METHODS}
    if unknown:
        raise SystemExit(f"Unknown methods: {sorted(unknown)}")

    commands: list[tuple[str, list[str]]] = []
    for method in METHODS:
        if method.name not in requested:
            continue
        cmd = [
            conda_python(method.env),
            "-u",
            method.script,
            "--configs",
            *configs,
            "--seeds",
            *[str(seed) for seed in seeds],
            "--rts",
            *rts_values,
        ]
        commands.append((f"method_{method.name}", cmd))
    return commands


def run_batch(commands: list[tuple[str, list[str]]], *, dry_run: bool) -> list[tuple[str, int]]:
    failures: list[tuple[str, int]] = []
    total = len(commands)
    for index, (label, cmd) in enumerate(commands, start=1):
        print(f"\n[{index:03d}/{total:03d}] {label}", flush=True)
        log_path = LOG_DIR / f"{label}.log"
        rc = run_command(cmd, log_path=log_path, dry_run=dry_run)
        if rc != 0:
            failures.append((label, rc))
            print(f"[failed] {label} rc={rc}", flush=True)
    return failures


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(
        "[mouse-comparison] "
        f"configs={args.configs} seeds={args.seeds} rts={args.rts} "
        f"stage={args.stage} methods={args.methods} dry_run={args.dry_run} force={args.force}",
        flush=True,
    )

    started = time.time()
    failures: list[tuple[str, int]] = []

    if args.stage in {"all", "cache"}:
        failures.extend(
            run_batch(
                cache_commands(args.configs, args.seeds, args.rts, force=args.force),
                dry_run=args.dry_run,
            )
        )
        if failures and args.stage == "all":
            print("[mouse-comparison] cache failures detected; skip method stage", flush=True)

    if args.stage == "methods" or (args.stage == "all" and not failures):
        failures.extend(
            run_batch(
                method_commands(args.configs, args.seeds, args.rts, args.methods),
                dry_run=args.dry_run,
            )
        )

    elapsed_min = (time.time() - started) / 60
    print(f"\n[mouse-comparison] elapsed_min={elapsed_min:.1f}", flush=True)
    if failures:
        print("[mouse-comparison] failures:", flush=True)
        for label, rc in failures:
            print(f"  {label}: rc={rc}", flush=True)
        raise SystemExit(1)
    print("[mouse-comparison] complete", flush=True)


if __name__ == "__main__":
    main()
