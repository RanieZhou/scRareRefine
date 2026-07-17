"""Dataset-level paired inference for scRareRefine comparisons.

Run-level differences on matched ``(dataset, rare_train_size, seed)`` units are
retained for descriptive win/tie/loss counts. Statistical inference treats the
dataset as the independent unit: paired differences are averaged within each
dataset, and confidence intervals resample those dataset effects as clusters.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_CSV = ROOT / "results" / "comparison" / "comparison_summary.csv"
LABEL_COUNT_CSV = (
    ROOT
    / "results"
    / "comparison"
    / "split_rare_nonrare_by_rts_long_train_label_ratio.csv"
)
OUT_CSV = ROOT / "results" / "comparison" / "significance_test.csv"
EFFECTS_CSV = ROOT / "results" / "comparison" / "significance_dataset_effects.csv"

OUR = "scRareRefine"
BASELINES = [
    "scANVI",
    "kNN",
    "CellTypist",
    "scBalance",
    "ProtoCloud",
    "HiCat",
    "scCAD",
    "TOSICA",
]
TRANSDUCTIVE = {"HiCat"}
SCARCE = ["0.01", "0.05", "0.10"]
KEY = ["dataset", "rare_train_size", "seed"]
TIE_TOL = 1e-9
BOOTSTRAP_ITERATIONS = 10000


def _paired(df: pd.DataFrame, base: str) -> pd.DataFrame:
    ours = df[df.method == OUR].set_index(KEY)["rare_f1"]
    baseline = df[df.method == base].set_index(KEY)["rare_f1"]
    paired = pd.concat([ours.rename("our"), baseline.rename("base")], axis=1).dropna()
    paired["delta"] = paired["our"] - paired["base"]
    return paired.reset_index()


def _dataset_effects(paired: pd.DataFrame) -> pd.Series:
    return paired.groupby("dataset", sort=True)["delta"].mean()


def _assert_complete_grid(
    paired: pd.DataFrame, expected_rts: set[str], expected_seeds: set[int], base: str
) -> None:
    expected = {(rts, seed) for rts in expected_rts for seed in expected_seeds}
    for dataset, group in paired.groupby("dataset"):
        observed = set(
            zip(group["rare_train_size"].astype(str), group["seed"].astype(int))
        )
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"Incomplete matched grid for {base}/{dataset}: missing={missing}, extra={extra}"
            )


def _cluster_boot_ci(
    dataset_effects: np.ndarray, n: int = BOOTSTRAP_ITERATIONS, seed: int = 0
) -> tuple[float, float]:
    effects = np.asarray(dataset_effects, dtype=float)
    if len(effects) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(effects), size=(n, len(effects)))
    means = effects[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _sign_test(effects: np.ndarray) -> tuple[int, int, int, float]:
    effects = np.asarray(effects, dtype=float)
    positive = int((effects > TIE_TOL).sum())
    zero = int((np.abs(effects) <= TIE_TOL).sum())
    negative = int((effects < -TIE_TOL).sum())
    n_nonzero = positive + negative
    p_value = (
        float(
            stats.binomtest(positive, n_nonzero, p=0.5, alternative="two-sided").pvalue
        )
        if n_nonzero
        else np.nan
    )
    return positive, zero, negative, p_value


def _wilcoxon_dataset(effects: np.ndarray) -> tuple[float, float]:
    nonzero = np.asarray(effects, dtype=float)
    nonzero = nonzero[np.abs(nonzero) > TIE_TOL]
    if len(nonzero) == 0:
        return np.nan, np.nan
    try:
        result = stats.wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return np.nan, np.nan


def _lodo_range(effects: np.ndarray) -> tuple[float, float]:
    effects = np.asarray(effects, dtype=float)
    if len(effects) < 2:
        return np.nan, np.nan
    means = np.asarray([np.delete(effects, i).mean() for i in range(len(effects))])
    return float(means.min()), float(means.max())


def _collapse_effective_counts(
    paired: pd.DataFrame, label_counts: pd.DataFrame
) -> pd.DataFrame:
    counts = label_counts[
        (label_counts["split"] == "train")
        & label_counts["rare_train_size"].isin(SCARCE)
    ][KEY + ["train_labeled_rare"]].copy()
    merged = paired.merge(counts, on=KEY, how="left", validate="one_to_one")
    if merged["train_labeled_rare"].isna().any():
        missing = merged.loc[merged["train_labeled_rare"].isna(), KEY]
        raise ValueError(
            f"Missing effective rare-label counts for:\n{missing.to_string(index=False)}"
        )
    return (
        merged.groupby(["dataset", "seed", "train_labeled_rare"], as_index=False)[
            "delta"
        ]
        .mean()
        .rename(columns={"train_labeled_rare": "effective_rare_labels"})
    )


def _analysis_row(
    paired: pd.DataFrame,
    dataset_effects: pd.Series,
    label: str,
    base: str,
    analysis: str,
    seed: int,
) -> dict:
    run_delta = paired["delta"].to_numpy(dtype=float)
    effects = dataset_effects.to_numpy(dtype=float)
    lo, hi = _cluster_boot_ci(effects, seed=seed)
    d_pos, d_zero, d_neg, sign_p = _sign_test(effects)
    wilcoxon_stat, wilcoxon_p = _wilcoxon_dataset(effects)
    lodo_lo, lodo_hi = _lodo_range(effects)
    return {
        "region": label,
        "analysis": analysis,
        "baseline": base,
        "transductive": base in TRANSDUCTIVE,
        "n_run_pairs": len(run_delta),
        "n_independent_datasets": len(effects),
        "run_wins": int((run_delta > TIE_TOL).sum()),
        "run_ties": int((np.abs(run_delta) <= TIE_TOL).sum()),
        "run_losses": int((run_delta < -TIE_TOL).sum()),
        "run_mean_delta": float(run_delta.mean()) if len(run_delta) else np.nan,
        "run_median_delta": float(np.median(run_delta)) if len(run_delta) else np.nan,
        "dataset_mean_delta": float(effects.mean()) if len(effects) else np.nan,
        "dataset_median_delta": float(np.median(effects)) if len(effects) else np.nan,
        "cluster_boot_ci_lo": lo,
        "cluster_boot_ci_hi": hi,
        "dataset_positive": d_pos,
        "dataset_zero": d_zero,
        "dataset_negative": d_neg,
        "exact_sign_p_two_sided": sign_p,
        "dataset_wilcoxon_stat": wilcoxon_stat,
        "dataset_wilcoxon_p_two_sided": wilcoxon_p,
        "lodo_mean_min": lodo_lo,
        "lodo_mean_max": lodo_hi,
        "bootstrap_unit": "dataset",
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "inference_unit": "dataset",
        "tie_tolerance": TIE_TOL,
    }


def _test_block(
    df: pd.DataFrame, label: str, label_counts: pd.DataFrame, seed_offset: int
) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    effect_rows: list[dict] = []
    print(f"\n========== {label} ==========")
    print(
        f"{'baseline':12s} {'runs':>4s} {'W/T/L':>9s} {'ds mean':>8s} {'cluster 95% CI':>20s} {'ds +/-/0':>9s} {'sign p':>9s}"
    )
    for i, base in enumerate(BASELINES):
        paired = _paired(df, base)
        _assert_complete_grid(
            paired,
            set(df["rare_train_size"].astype(str).unique()),
            set(df["seed"].astype(int).unique()),
            base,
        )
        effects = _dataset_effects(paired)
        row = _analysis_row(
            paired, effects, label, base, "nominal_budget", seed_offset + i
        )
        rows.append(row)
        for dataset, effect in effects.items():
            effect_rows.append(
                {
                    "region": label,
                    "analysis": "nominal_budget",
                    "baseline": base,
                    "dataset": dataset,
                    "dataset_effect": effect,
                    "n_within_dataset_pairs": int((paired["dataset"] == dataset).sum()),
                }
            )
        print(
            f"{base:12s} {row['n_run_pairs']:4d} {row['run_wins']:d}/{row['run_ties']:d}/{row['run_losses']:d} {row['dataset_mean_delta']:+8.4f} [{row['cluster_boot_ci_lo']:+.4f},{row['cluster_boot_ci_hi']:+.4f}] {row['dataset_positive']:d}/{row['dataset_negative']:d}/{row['dataset_zero']:d} {row['exact_sign_p_two_sided']:.4f}"
        )

        if set(df["rare_train_size"].unique()).issubset(set(SCARCE)):
            collapsed = _collapse_effective_counts(paired, label_counts)
            collapsed_effects = collapsed.groupby("dataset", sort=True)["delta"].mean()
            collapsed_row = _analysis_row(
                collapsed,
                collapsed_effects,
                label,
                base,
                "collapse_aware_effective_count",
                seed_offset + 100 + i,
            )
            rows.append(collapsed_row)
            for dataset, effect in collapsed_effects.items():
                effect_rows.append(
                    {
                        "region": label,
                        "analysis": "collapse_aware_effective_count",
                        "baseline": base,
                        "dataset": dataset,
                        "dataset_effect": effect,
                        "n_within_dataset_pairs": int(
                            (collapsed["dataset"] == dataset).sum()
                        ),
                    }
                )
    return rows, effect_rows


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
    label_counts = pd.read_csv(LABEL_COUNT_CSV, dtype={"rare_train_size": str})
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    rts_all = _sort_rts(df["rare_train_size"].dropna().unique().tolist())
    seeds = sorted(df["seed"].dropna().astype(int).unique().tolist())
    methods = sorted(df["method"].dropna().unique().tolist())
    print(f"datasets={len(datasets)} {datasets}")
    print(f"rts={rts_all} seeds={seeds} methods={methods}")

    rows: list[dict] = []
    effect_rows: list[dict] = []
    block_rows, block_effects = _test_block(
        df,
        f"ALL rts ({len(datasets)}ds x {len(rts_all)}rts x {len(seeds)}seed)",
        label_counts,
        0,
    )
    rows += block_rows
    effect_rows += block_effects
    scarce_df = df[df.rare_train_size.isin(SCARCE)].copy()
    block_rows, block_effects = _test_block(
        scarce_df,
        f"SCARCE rts<=0.10 ({len(datasets)}ds x {len(SCARCE)}rts x {len(seeds)}seed)",
        label_counts,
        1000,
    )
    rows += block_rows
    effect_rows += block_effects
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(effect_rows).to_csv(EFFECTS_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")
    print(f"[saved] {EFFECTS_CSV}")
    print(
        "Inference uses equally weighted dataset effects (n=8). Run-level W/T/L is descriptive; the sign test evaluates directional consistency, not the arithmetic mean."
    )


if __name__ == "__main__":
    main()
