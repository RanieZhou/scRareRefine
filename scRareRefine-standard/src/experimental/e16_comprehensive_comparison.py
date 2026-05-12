"""E16: Best method per regime — comprehensive comparison.

Collect results from E8-E15. For each (dataset, rare_class, rts):
- Find the best method
- Compute improvement over scANVI baseline
- Compute improvement over current main method (Euclidean+gate+marker)

Generate a comprehensive table and heatmap.

Usage:
    python src/experimental/e16_comprehensive_comparison.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e16_comprehensive_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP_DIR = ROOT / "outputs" / "_experimental"


def load_e8() -> pd.DataFrame:
    """Load E8 aggregated results (mean across seeds)."""
    path = EXP_DIR / "e8_data_efficiency_sweep" / "aggregated_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    # Pivot to wide format
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="mean_rare_f1",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def load_e9() -> pd.DataFrame:
    """Load E9 aggregated results."""
    path = EXP_DIR / "e9_cb_knn_sweep" / "aggregated_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    pivot = df.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method",
        values="mean_rare_f1",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def load_e10() -> pd.DataFrame:
    """Load E10 per-run results (seed42 only)."""
    path = EXP_DIR / "e10_prototype_ensemble" / "per_run_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    return df[["dataset", "rare_class", "rts", "ensemble_test_rare_f1", "best_alpha"]].rename(
        columns={"ensemble_test_rare_f1": "Ensemble"}
    )


def load_e11() -> pd.DataFrame:
    """Load E11 results."""
    path = EXP_DIR / "e11_soft_gate" / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    # Get soft gate full-test results
    soft = df[df["method"].str.startswith("Soft gate full-test")].copy()
    soft["method_short"] = "Soft-gate"
    return soft[["dataset", "rare_class", "rts", "rare_f1", "method_short"]].rename(
        columns={"rare_f1": "Soft-gate"}
    ).drop(columns=["method_short"])


def load_e12() -> pd.DataFrame:
    """Load E12 results."""
    path = EXP_DIR / "e12_temperature_scaling" / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    # Get temperature-scaled results
    ts = df[df["method"].str.startswith("Dist-TempScaled")].copy()
    return ts[["dataset", "rare_class", "rts", "rare_f1"]].rename(
        columns={"rare_f1": "Dist-TempScaled"}
    )


def load_e14() -> pd.DataFrame:
    """Load E14 bootstrap results."""
    path = EXP_DIR / "e14_bootstrap_uncertainty" / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    boot = df[df["method"].str.startswith("Bootstrap")].copy()
    return boot[["dataset", "rare_class", "rts", "rare_f1"]].rename(
        columns={"rare_f1": "Bootstrap"}
    )


def main() -> pd.DataFrame:
    print("Loading results from E8-E15...")

    # E8: core sweep (3 seeds, mean)
    e8 = load_e8()
    print(f"  E8: {len(e8)} rows")

    # E9: CB-kNN sweep (3 seeds, mean)
    e9 = load_e9()
    print(f"  E9: {len(e9)} rows")

    # E10: ensemble (seed42)
    e10 = load_e10()
    print(f"  E10: {len(e10)} rows")

    # E11: soft gate (seed42)
    e11 = load_e11()
    print(f"  E11: {len(e11)} rows")

    # E12: temperature scaling (seed42)
    e12 = load_e12()
    print(f"  E12: {len(e12)} rows")

    # E14: bootstrap (seed42)
    e14 = load_e14()
    print(f"  E14: {len(e14)} rows")

    # Start with E8 as base
    if e8.empty:
        print("E8 results not found, cannot proceed")
        return pd.DataFrame()

    df = e8.copy()

    # Merge E9 CB-kNN
    if not e9.empty and "CB-kNN" in e9.columns:
        df = df.merge(
            e9[["dataset", "rare_class", "rts", "CB-kNN"]],
            on=["dataset", "rare_class", "rts"],
            how="left",
        )

    # Merge E10 ensemble
    if not e10.empty:
        df = df.merge(
            e10[["dataset", "rare_class", "rts", "Ensemble"]],
            on=["dataset", "rare_class", "rts"],
            how="left",
        )

    # Merge E11 soft gate
    if not e11.empty:
        df = df.merge(
            e11[["dataset", "rare_class", "rts", "Soft-gate"]],
            on=["dataset", "rare_class", "rts"],
            how="left",
        )

    # Merge E12 temperature scaling
    if not e12.empty:
        df = df.merge(
            e12[["dataset", "rare_class", "rts", "Dist-TempScaled"]],
            on=["dataset", "rare_class", "rts"],
            how="left",
        )

    # Merge E14 bootstrap
    if not e14.empty:
        df = df.merge(
            e14[["dataset", "rare_class", "rts", "Bootstrap"]],
            on=["dataset", "rare_class", "rts"],
            how="left",
        )

    # Rename E8 columns
    rename_map = {}
    if "scANVI" in df.columns:
        rename_map["scANVI"] = "scANVI"
    if "Euclidean" in df.columns:
        rename_map["Euclidean"] = "Euclidean"
    if "Mahal-pooled" in df.columns:
        rename_map["Mahal-pooled"] = "Mahal-pooled"
    df = df.rename(columns=rename_map)

    # Method columns
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "CB-kNN",
                                "Ensemble", "Soft-gate", "Dist-TempScaled", "Bootstrap"]
                   if c in df.columns]

    # Find best method per row
    df["best_method"] = df[method_cols].idxmax(axis=1)
    df["best_f1"] = df[method_cols].max(axis=1)

    # Improvement over scANVI
    if "scANVI" in df.columns:
        df["improvement_vs_scanvi"] = df["best_f1"] - df["scANVI"]

    # Improvement over Euclidean
    if "Euclidean" in df.columns:
        df["improvement_vs_euclidean"] = df["best_f1"] - df["Euclidean"]

    write_table(df, OUT_DIR / "comprehensive_results.csv")

    print("\n\n=== E16 Comprehensive Comparison ===")
    print(f"Methods compared: {method_cols}")
    print(f"\nTotal configurations: {len(df)}")

    # Best method distribution
    print("\n=== Best method distribution ===")
    print(df["best_method"].value_counts().to_string())

    # Summary by dataset
    print("\n=== Mean best_f1 by dataset ===")
    print(df.groupby("dataset")["best_f1"].mean().sort_values(ascending=False).to_string(
        float_format=lambda x: f"{x:.3f}"
    ))

    # Summary by rts
    print("\n=== Mean best_f1 by rts ===")
    print(df.groupby("rts")["best_f1"].mean().sort_values().to_string(
        float_format=lambda x: f"{x:.3f}"
    ))

    # Full table
    print("\n=== Full comparison table ===")
    display_cols = ["dataset", "rare_class", "rts"] + method_cols + ["best_method", "best_f1"]
    display_cols = [c for c in display_cols if c in df.columns]
    print(df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Mahal-pooled win rate
    if "Mahal-pooled" in df.columns and "Euclidean" in df.columns:
        mahal_wins = (df["Mahal-pooled"] > df["Euclidean"]).sum()
        total = len(df)
        print(f"\n=== Mahal-pooled vs Euclidean ===")
        print(f"Mahal wins: {mahal_wins}/{total} ({100*mahal_wins/total:.1f}%)")
        print(f"Mean delta: {(df['Mahal-pooled'] - df['Euclidean']).mean():.3f}")

    # CB-kNN win rate
    if "CB-kNN" in df.columns and "Euclidean" in df.columns:
        cb_wins = (df["CB-kNN"] > df["Euclidean"]).sum()
        print(f"\n=== CB-kNN vs Euclidean ===")
        print(f"CB-kNN wins: {cb_wins}/{total} ({100*cb_wins/total:.1f}%)")
        print(f"Mean delta: {(df['CB-kNN'] - df['Euclidean']).mean():.3f}")

    return df


if __name__ == "__main__":
    main()
