"""Paired significance tests for scRareRefine comparisons.

Input:
    results/comparison/comparison_summary.csv

Pairing unit:
    (dataset, rare_train_size, seed)

For each baseline, this script compares scRareRefine against the baseline on
rare_f1 using:
    - win / tie / loss counts,
    - paired one-sided Wilcoxon signed-rank test (H1: scRareRefine > baseline),
    - bootstrap 95% CI of mean paired Delta F1, resampling paired units.

Important interpretation:
    The paired units are not fully independent because seeds from the same
    dataset/budget are correlated, and some rare-label budgets collapse to the
    same effective number of rare labels. Use p-values as directional evidence
    and report effect sizes plus confidence intervals in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_CSV = ROOT / "results" / "comparison" / "comparison_summary.csv"
OUT_CSV = ROOT / "results" / "comparison" / "significance_test.csv"

OUR = "scRareRefine"
BASELINES = ["scANVI", "kNN", "CellTypist", "scBalance", "ProtoCloud", "HiCat", "scCAD", "TOSICA"]
TRANSDUCTIVE = {"HiCat"}
SCARCE = ["0.01", "0.05", "0.10"]
KEY = ["dataset", "rare_train_size", "seed"]


def _paired(df: pd.DataFrame, base: str) -> pd.DataFrame:
    ours = df[df.method == OUR].set_index(KEY)["rare_f1"]
    baseline = df[df.method == base].set_index(KEY)["rare_f1"]
    return pd.concat([ours.rename("our"), baseline.rename("base")], axis=1).dropna()


def _boot_ci(delta: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    if len(delta) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _test_block(df: pd.DataFrame, label: str) -> list[dict]:
    rows: list[dict] = []
    print(f"\n========== {label} ==========")
    print(
        f"{'baseline':12s} {'n':>3s} {'win':>3s} {'tie':>3s} {'loss':>4s} "
        f"{'median_d':>9s} {'mean_d':>8s} {'boot95%CI':>18s} {'Wilcoxon p':>13s}"
    )
    for base in BASELINES:
        paired = _paired(df, base)
        delta = (paired["our"] - paired["base"]).to_numpy()
        n = len(delta)
        win = int((delta > 1e-9).sum())
        tie = int((np.abs(delta) <= 1e-9).sum())
        loss = int((delta < -1e-9).sum())
        median_delta = float(np.median(delta)) if n else np.nan
        mean_delta = float(delta.mean()) if n else np.nan
        lo, hi = _boot_ci(delta)
        if n and np.any(np.abs(delta) > 1e-12):
            try:
                p_value = float(stats.wilcoxon(delta, alternative="greater", zero_method="wilcox").pvalue)
            except ValueError:
                p_value = np.nan
        else:
            p_value = np.nan
        tag = " (transductive)" if base in TRANSDUCTIVE else ""
        print(
            f"{base:12s} {n:3d} {win:3d} {tie:3d} {loss:4d} "
            f"{median_delta:+9.4f} {mean_delta:+8.4f} [{lo:+.4f},{hi:+.4f}] {p_value:13.3e}{tag}"
        )
        rows.append(
            {
                "region": label,
                "baseline": base,
                "transductive": base in TRANSDUCTIVE,
                "n_pairs": n,
                "win": win,
                "tie": tie,
                "loss": loss,
                "median_delta": round(median_delta, 4),
                "mean_delta": round(mean_delta, 4),
                "boot_ci_lo": round(lo, 4),
                "boot_ci_hi": round(hi, 4),
                "wilcoxon_p_greater": p_value,
            }
        )
    return rows


def _sort_rts(values: list[str]) -> list[str]:
    def key(v: str) -> tuple[int, float | str]:
        if v == "all":
            return (1, v)
        try:
            return (0, float(v))
        except ValueError:
            return (0, v)

    return sorted(values, key=key)


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
    df = df[df.status == "ok"].copy()

    datasets = sorted(df["dataset"].dropna().unique().tolist())
    rts_all = _sort_rts(df["rare_train_size"].dropna().unique().tolist())
    seeds = sorted(df["seed"].dropna().astype(int).unique().tolist())
    methods = sorted(df["method"].dropna().unique().tolist())

    print(f"datasets={len(datasets)} {datasets}")
    print(f"rts={rts_all} seeds={seeds} methods={methods}")

    rows = []
    rows += _test_block(df, f"ALL rts ({len(datasets)}ds x {len(rts_all)}rts x {len(seeds)}seed)")

    scarce_df = df[df.rare_train_size.isin(SCARCE)].copy()
    scarce_datasets = sorted(scarce_df["dataset"].dropna().unique().tolist())
    rows += _test_block(
        scarce_df,
        f"SCARCE rts<=0.10 ({len(scarce_datasets)}ds x {len(SCARCE)}rts x {len(seeds)}seed)",
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")
    print(
        "Note: paired units are correlated across seeds and collapsed rare-label budgets; "
        "treat p-values as directional evidence. HiCat is marked as transductive."
    )


if __name__ == "__main__":
    main()
