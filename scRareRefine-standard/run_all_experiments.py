"""Run all comparison experiments across datasets, seeds, and train sizes.

Experiment matrix:
    Datasets / rare classes:
        immune_dc      - ASDC
        immune_dc_cdc1 - cDC1
        pancreas_gamma - gamma
        tabula_spleen  - innate lymphoid cell
    Seeds: 42, 43, 44
    Train sizes: per dataset (see EXPERIMENTS below)

Methods run per combination (via run_pipeline.py):
    - baseline (scANVI)
    - kNN k=15
    - CellTypist
    - scBalance
    - scRareRefine

Already-completed runs are automatically skipped (Stage 2 caching).

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --dry_run     # print commands only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENTS = [
    ("configs/immune_dc.yaml",      "ASDC",                 ["0.01", "0.05", "0.1", "all"]),
    ("configs/immune_dc_cdc1.yaml", "cDC1",                 ["0.01", "0.05", "0.1", "all"]),
    ("configs/pancreas_gamma.yaml", "gamma",                ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_spleen.yaml",  "innate lymphoid cell", ["0.01", "0.05", "0.1", "all"]),
]
SEEDS = [42, 43, 44]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    py = sys.executable
    project_root = Path(__file__).resolve().parent

    total = sum(len(rts_list) * len(SEEDS) for _, _, rts_list in EXPERIMENTS)
    done = 0
    failed: list[str] = []

    for cfg, rare_class, rts_list in EXPERIMENTS:
        for rts in rts_list:
            for seed in SEEDS:
                done += 1
                label = f"[{done}/{total}] {cfg} | {rare_class} | rts={rts} | seed={seed}"
                cmd = [
                    py, "run_pipeline.py",
                    "--config", cfg,
                    "--seed", str(seed),
                    "--rare_class", rare_class,
                    "--rare_train_size", str(rts),
                ]
                print(f"\n{'='*80}")
                print(f">>> {label}")
                print(f"    {' '.join(cmd)}")
                print(f"{'='*80}")

                if args.dry_run:
                    continue

                t0 = time.time()
                result = subprocess.run(cmd, cwd=project_root)
                elapsed = time.time() - t0

                if result.returncode != 0:
                    print(f"  *** FAILED (rc={result.returncode}) — continuing ***")
                    failed.append(label)
                else:
                    print(f"  Finished in {elapsed/60:.1f} min")

    print(f"\n{'='*80}")
    print(f"All {total} runs attempted.")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
    else:
        print("All runs succeeded.")

    if not args.dry_run:
        print("\nRunning summary comparison...")
        subprocess.run([py, "src/09_summary_compare.py"], cwd=project_root)


if __name__ == "__main__":
    main()
