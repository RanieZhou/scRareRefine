"""Run the full scRareRefine pipeline sequentially.

Parameter priority (high → low):
  1. Command-line argument (explicitly passed)
  2. Config file (experiment.rare_class / seeds[0] / rare_train_sizes[0])

Usage:
    # Use all defaults from config
    python run_pipeline.py --config configs/immune_dc.yaml

    # Override specific parameters
    python run_pipeline.py --config configs/immune_dc.yaml --seed 43 --rare_class cDC1 --rare_train_size 50

    # Force re-training (ignore cached embeddings)
    python run_pipeline.py --config configs/immune_dc.yaml --force
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_param(cli_value, config_value):
    """返回 CLI 值（若提供），否则返回 config 值。"""
    return cli_value if cli_value is not None else config_value


def run_command(cmd: list[str], cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("=" * 80 + "\n")
    subprocess.run(cmd, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scRareRefine pipeline. Defaults come from config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/immune_dc.yaml",
                        help="YAML config path (default: configs/immune_dc.yaml)")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed，必须显式指定（config 中 seeds 为列表，无唯一默认值）")
    parser.add_argument("--rare_class", default=None,
                        help="Rare cell class（默认读取 config 中 experiment.rare_class）")
    parser.add_argument("--rare_train_size", required=True,
                        help="Rare class 训练预算，必须显式指定（config 中 rare_train_sizes 为列表，无唯一默认值）")
    parser.add_argument("--split_mode", default="batch_heldout",
                        help="batch_heldout | cell_stratified | lobo（需配合 --test_batch）")
    parser.add_argument("--test_batch", default=None,
                        help="LOBO 模式下留出的 test batch 名称（--split_mode lobo 时必填）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新训练 Stage 2，忽略已有 embedding")
    args = parser.parse_args()

    # ── 读 config，仅用它填充有单值默认的参数（rare_class）────────────────────
    config = load_config(args.config)
    exp = config.get("experiment", {})

    rare_class = resolve_param(args.rare_class, exp.get("rare_class"))

    if rare_class is None:
        parser.error("--rare_class not provided and experiment.rare_class not found in config.")

    if args.split_mode == "lobo":
        if not args.test_batch:
            parser.error("--test_batch is required when --split_mode lobo")
        split_mode = f"lobo_{args.test_batch}"
    else:
        split_mode = args.split_mode

    seed            = args.seed
    rare_train_size = args.rare_train_size

    print(f"\nPipeline parameters:")
    print(f"  config         = {args.config}")
    print(f"  rare_class     = {rare_class}")
    print(f"  seed           = {seed}")
    print(f"  rare_train_size= {rare_train_size}")
    print(f"  split_mode     = {split_mode}")
    print(f"  force          = {args.force}")

    project_root = Path(__file__).resolve().parent
    py = sys.executable

    # ── 公共参数片段 ──────────────────────────────────────────────────────────
    common = [
        "--config", args.config,
        "--seed", str(seed),
        "--rare_class", rare_class,
        "--rare_train_size", str(rare_train_size),
        "--split_mode", split_mode,
    ]

    stage2_extra = ["--force"] if args.force else []

    stages = [
        ("Stage 1: split",               [py, "src/01_split.py",
                                          "--config", args.config,
                                          "--seed", str(seed),
                                          "--split_mode", split_mode]),
        ("Stage 2: scANVI baseline",     [py, "src/02_baseline_scanvi.py", *common, *stage2_extra]),
        ("Stage 3: prototype scores",    [py, "src/03_prototype.py",       *common]),
        ("Stage 4: prototype gate",      [py, "src/04_prototype_gate.py",  *common]),
        ("Stage 5: gate + marker",       [py, "src/05_prototype_gate_marker.py", *common]),
        ("Stage 6: evaluate",            [py, "src/07_evaluate.py",        *common]),
    ]

    for label, cmd in stages:
        print(f"\n>>> {label}")
        run_command(cmd, cwd=project_root)

    print("\nAll stages finished successfully.")


if __name__ == "__main__":
    main()
