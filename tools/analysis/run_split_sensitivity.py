"""Run split-mode sensitivity experiments for scANVI + scRareRefine.

This runner intentionally does not edit YAML configs. It overrides
``--split_mode cell_stratified`` at runtime so outputs are written under
``outputs/<dataset>/cell_stratified_seed...`` and remain separate from the
primary ``batch_heldout`` results.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PYTHON = Path(r"D:\setup\anaconda\envs\scanvi311\python.exe")
OUT_DIR = ROOT / "results" / "split_sensitivity"
LOG_DIR = OUT_DIR / "logs"

CONFIGS = [
    ("configs/immune_dc.yaml", "ASDC"),
    ("configs/pancreas_baron.yaml", "gamma"),
    ("configs/pancreas_integrated.yaml", "endothelial"),
    ("configs/tabula_lung_endo.yaml", "endothelial cell of lymphatic vessel"),
    ("configs/tabula_sapiens_stomach.yaml", "mast cell"),
    ("configs/tabula_small_intestine.yaml", "intestinal tuft cell"),
]
RTS = ["0.01", "0.05", "0.10", "all"]


@dataclass(frozen=True)
class Job:
    index: int
    config: str
    rare_class: str
    rts: str
    seed: int


def build_jobs(seed: int) -> list[Job]:
    jobs: list[Job] = []
    i = 0
    for config, rare_class in CONFIGS:
        for rts in RTS:
            i += 1
            jobs.append(Job(i, config, rare_class, rts, seed))
    return jobs


def command_for(job: Job, python_exe: Path, *, force: bool) -> list[str]:
    cmd = [
        str(python_exe),
        "-u",
        "run_pipeline.py",
        "--config",
        job.config,
        "--seed",
        str(job.seed),
        "--rare_class",
        job.rare_class,
        "--rare_train_size",
        job.rts,
        "--split_mode",
        "cell_stratified",
    ]
    if force:
        cmd.append("--force")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cell-stratified split sensitivity jobs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON), help="Python executable to use")
    parser.add_argument("--start_at", type=int, default=1)
    parser.add_argument("--end_at", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Pass --force to run_pipeline.py")
    args = parser.parse_args()

    python_exe = Path(args.python)
    jobs = [
        job
        for job in build_jobs(args.seed)
        if job.index >= args.start_at and (args.end_at is None or job.index <= args.end_at)
    ]
    if not jobs:
        raise SystemExit("No jobs selected.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[split-sensitivity] selected_jobs={len(jobs)} seed={args.seed} force={args.force}", flush=True)
    failures: list[tuple[Job, int]] = []
    started = time.time()

    for job in jobs:
        cmd = command_for(job, python_exe, force=args.force)
        log_name = (
            f"{job.index:02d}_{Path(job.config).stem}_seed{job.seed}_"
            f"rare{job.rts.replace('.', 'p')}.log"
        )
        log_path = LOG_DIR / log_name
        print(f"\n[{job.index:02d}/{len(build_jobs(args.seed)):02d}] {job.config} rts={job.rts}", flush=True)
        print("  " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
        if args.dry_run:
            continue

        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
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
            returncode = proc.wait()
        if returncode != 0:
            failures.append((job, returncode))
            print(f"  [failed] exit={returncode} log={log_path.relative_to(ROOT)}", flush=True)
            break
        print(f"  [ok] log={log_path.relative_to(ROOT)}", flush=True)

    elapsed = time.time() - started
    if failures:
        job, code = failures[0]
        raise SystemExit(
            f"Failed at job {job.index} ({job.config}, rts={job.rts}, seed={job.seed}) "
            f"with exit code {code}. See logs in {LOG_DIR.relative_to(ROOT)}."
        )
    print(f"\n[done] selected jobs completed in {elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
