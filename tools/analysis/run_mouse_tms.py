"""Run mouse Tabula Muris Senis scANVI + scRareRefine jobs.

Outputs are naturally separated from the human runs because the dataset names are
``mouse_lung_tms_10x`` and ``mouse_pancreas_tms_10x``. Logs are written under
``results/mouse_tms/logs`` for easier monitoring.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PYTHON = Path(r"D:\setup\anaconda\envs\scanvi311\python.exe")
OUT_DIR = ROOT / "results" / "mouse_tms"
LOG_DIR = OUT_DIR / "logs"

CONFIGS = [
    ("configs/mouse_lung_tms_10x.yaml", "vein endothelial cell"),
    ("configs/mouse_pancreas_tms_10x.yaml", "pancreatic D cell"),
]
RTS = ["0.01", "0.05", "0.10", "all"]


@dataclass(frozen=True)
class Job:
    index: int
    config: str
    rare_class: str
    rts: str
    seed: int


def build_jobs(seeds: list[int]) -> list[Job]:
    jobs: list[Job] = []
    index = 0
    for config, rare_class in CONFIGS:
        for rts in RTS:
            for seed in seeds:
                index += 1
                jobs.append(Job(index, config, rare_class, rts, seed))
    return jobs


def command_for(job: Job, python_exe: Path, *, force: bool, split_mode: str | None) -> list[str]:
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
    ]
    if split_mode:
        cmd.extend(["--split_mode", split_mode])
    if force:
        cmd.append("--force")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mouse TMS experiment jobs.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--start_at", type=int, default=1)
    parser.add_argument("--end_at", type=int, default=None)
    parser.add_argument("--split_mode", choices=["batch_heldout", "cell_stratified"], default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    all_jobs = build_jobs(args.seeds)
    jobs = [
        job
        for job in all_jobs
        if job.index >= args.start_at and (args.end_at is None or job.index <= args.end_at)
    ]
    if not jobs:
        raise SystemExit("No jobs selected.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"[mouse-tms] selected_jobs={len(jobs)} total_jobs={len(all_jobs)} "
        f"seeds={args.seeds} split_mode={args.split_mode or 'config'} force={args.force}",
        flush=True,
    )

    failures: list[tuple[Job, int]] = []
    started = time.time()
    for job in jobs:
        cmd = command_for(job, Path(args.python), force=args.force, split_mode=args.split_mode)
        log_name = (
            f"{job.index:02d}_{Path(job.config).stem}_seed{job.seed}_"
            f"rare{job.rts.replace('.', 'p')}.log"
        )
        log_path = LOG_DIR / log_name
        print(f"\n[{job.index:02d}/{len(all_jobs):02d}] {job.config} rts={job.rts} seed={job.seed}", flush=True)
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

    elapsed_min = (time.time() - started) / 60
    print(f"\n[mouse-tms] elapsed_min={elapsed_min:.1f}", flush=True)
    if failures:
        print("[mouse-tms] failures:", flush=True)
        for job, returncode in failures:
            print(f"  job={job.index} config={job.config} rts={job.rts} seed={job.seed} rc={returncode}", flush=True)
        raise SystemExit(1)
    print("[mouse-tms] complete", flush=True)


if __name__ == "__main__":
    main()
