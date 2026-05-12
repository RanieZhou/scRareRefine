"""Run the Mahalanobis PoC across a panel of representative runs.

Picks one seed per (dataset, rare_class, rts) combination spanning:
  - high-sep / low-baseline  (cDC1, ASDC)        — main-method strong cases
  - high-sep / high-baseline (endothelial cell)   — main-method tie cases
  - low-sep                  (epsilon, gamma)     — main-method abstention cases

Writes a combined CSV so we can eyeball the pattern.

Usage:
    python src/experimental/run_mahalanobis_sweep.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
POC = HERE / "mahalanobis_poc.py"

RUNS = [
    # dataset, run_id, rare_class, regime
    ("immune_dc", "batch_heldout_seed42_cdc1_rare5",  "cDC1",              "high-sep / low-baseline"),
    ("immune_dc", "batch_heldout_seed42_cdc1_rare20", "cDC1",              "high-sep / low-baseline"),
    ("immune_dc", "batch_heldout_seed42_asdc_rare5",  "ASDC",              "high-sep / low-baseline"),
    ("immune_dc", "batch_heldout_seed42_asdc_rare20", "ASDC",              "high-sep / low-baseline"),
    ("tabula_liver", "cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte", "high-sep / low-baseline"),
    ("tabula_kidney", "cell_stratified_seed42_endothelial_cell_rare20", "endothelial cell", "high-sep / high-baseline"),
    ("pancreas", "batch_heldout_seed42_epsilon_rare5",  "epsilon", "low-sep / low-annotation"),
    ("pancreas", "batch_heldout_seed42_epsilon_rare20", "epsilon", "low-sep / low-annotation"),
    ("pancreas", "batch_heldout_seed42_gamma_rare20",   "gamma",   "low-sep / high-baseline"),
    ("tabula_pancreas", "cell_stratified_seed42_type_b_pancreatic_cell_rare20", "type B pancreatic cell", "low-sep / high-baseline"),
]


def main() -> None:
    all_rows = []
    for dataset, run_id, rare_class, regime in RUNS:
        run_dir = ROOT / "outputs" / dataset / run_id
        if not run_dir.exists():
            print(f"[skip] {run_dir} not found")
            continue
        print(f"\n====== {dataset} / {rare_class} / {run_id} — {regime} ======")
        subprocess.run(
            [sys.executable, str(POC), "--run_dir", str(run_dir), "--rare_class", rare_class],
            check=True,
        )
        cmp_path = run_dir / "experimental" / "mahalanobis_poc" / "comparison.csv"
        if cmp_path.exists():
            df = pd.read_csv(cmp_path)
            df.insert(0, "regime", regime)
            df.insert(0, "rare_class", rare_class)
            df.insert(0, "dataset", dataset)
            df.insert(0, "run_id", run_id)
            all_rows.append(df)

    if not all_rows:
        print("No runs succeeded.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = ROOT / "outputs" / "_experimental" / "mahalanobis_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\n\nCombined sweep saved: {out_path}\n")

    print("── Summary: rare_f1 by method × regime ──\n")
    pivot = (
        combined.pivot_table(
            index=["regime", "dataset", "rare_class", "run_id"],
            columns="method",
            values="rare_f1",
        )
        .round(3)
    )
    # Reorder columns for readability
    ordered = [
        "scANVI baseline",
        "euclidean (current method)",
        "mahalanobis (per-class Sigma_c)",
        "mahalanobis (pooled Sigma, LDA-style)",
        "mahalanobis per-class + posterior penalty",
        "mahalanobis pooled + posterior penalty",
    ]
    cols = [c for c in ordered if c in pivot.columns]
    print(pivot[cols].to_string())


if __name__ == "__main__":
    main()
