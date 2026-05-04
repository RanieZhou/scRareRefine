from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scrare_refine.config import load_config, output_dir
from scrare_refine.io import read_table, write_table
from scrare_refine.output_layout import existing_table_path, stage_table_path
from scrare_refine.prototype_gate import choose_recommended_gate, evaluate_gate_rules, summarize_gate_effect


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate stricter prototype gates from existing scANVI artifacts.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_dir(config)
    rare_class = config["experiment"]["rare_class"]
    all_effect = []
    all_candidates = []

    for run_dir in _run_dirs(root):
        pred = read_table(existing_table_path(run_dir, "scanvi_predictions.parquet"))
        proto = read_table(existing_table_path(run_dir, "prototype_scores.csv"))
        seed = int(pred["seed"].iloc[0])
        rare_train_size = str(pred["rare_train_size"].iloc[0])
        effect, candidates = evaluate_gate_rules(pred, proto, rare_class=rare_class)
        effect.insert(0, "seed", seed)
        effect.insert(1, "rare_train_size", rare_train_size)
        effect.insert(2, "run", run_dir.name)
        all_effect.append(effect)
        if not candidates.empty:
            candidates.insert(0, "seed", seed)
            candidates.insert(1, "rare_train_size", rare_train_size)
            candidates.insert(2, "run", run_dir.name)
            all_candidates.append(candidates)

    if not all_effect:
        raise FileNotFoundError(f"No runs found under {root / 'runs'}")

    effect_runs = pd.concat(all_effect, ignore_index=True)
    summary = summarize_gate_effect(effect_runs)
    recommended = choose_recommended_gate(summary)
    recommended_candidates = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame(columns=["seed", "rare_train_size", "run", "gate_name"])
    )
    recommended_candidates = recommended_candidates[recommended_candidates["gate_name"].eq(recommended)].copy()

    write_table(effect_runs, stage_table_path(root, "prototype_gate", "gate_effect_runs.csv"))
    write_table(summary, stage_table_path(root, "prototype_gate", "gate_effect_summary.csv"))
    write_table(recommended_candidates, stage_table_path(root, "prototype_gate", "selected_gate_candidates.csv"))
    write_table(pd.DataFrame([{"recommended_gate": recommended}]), stage_table_path(root, "prototype_gate", "recommended_gate.csv"))
    print(f"Wrote prototype gate outputs to {root / 'stages' / 'prototype_gate'}")
    print(f"Recommended gate: {recommended}")


if __name__ == "__main__":
    main()
