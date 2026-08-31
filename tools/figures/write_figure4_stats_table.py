"""Write the primary-case statistical summary for Supplementary Table S1.

All numerical values are computed from the archived held-out group scores; the
script contains only the preregistered comparison order and display labels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "biological_case_study" / "v1" / "group_scores.csv"
RESULTS_DIR = ROOT / "results" / "biological_case_study" / "v1"
TABLE_PATH = ROOT / "paper" / "figures" / "tableS1_biological_stats.tex"

METRICS = [
    ("Target-marker score", "rare_marker_score"),
    ("Competitor-marker score", "competitor_marker_score"),
    (r"$\Delta$ similarity (rare - competitor)", "delta_similarity"),
]


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    comparisons = left[:, None] - right[None, :]
    return float((np.sign(comparisons).sum()) / comparisons.size)


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 1.0
    for rank in range(len(p_values) - 1, -1, -1):
        index = order[rank]
        running = min(running, p_values[index] * len(p_values) / (rank + 1))
        adjusted[index] = running
    return adjusted


def fmt(value: float) -> str:
    if abs(value) >= 0.1:
        return f"{value:.3f}"
    return f"{value:.2e}"


def main() -> None:
    scores = pd.read_csv(SOURCE)
    subset = scores[scores["dataset"] == "immune_dc"]
    left_group = subset.loc[subset["group"] == "Rescued TP"]
    right_group = subset.loc[subset["group"] == "Unrescued FN"]
    rows = []
    for display_name, column in METRICS:
        left = left_group[column].to_numpy(dtype=float)
        right = right_group[column].to_numpy(dtype=float)
        test = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic")
        rows.append(
            {
                "metric": display_name,
                "column": column,
                "n_rescued": len(left),
                "n_unrescued": len(right),
                "median_rescued": float(np.median(left)),
                "median_unrescued": float(np.median(right)),
                "median_difference": float(np.median(left) - np.median(right)),
                "cliffs_delta": cliffs_delta(left, right),
                "mannwhitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    table = pd.DataFrame(rows)
    table["q_value_bh"] = bh_adjust(table["p_value"].to_numpy(dtype=float))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "primary_comparison_stats.csv", index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Primary-case statistical comparisons between Rescued TP and Unrescued FN cells in the held-out Immune DC--ASDC case. Two-sided Wilcoxon rank-sum tests were applied to the three preregistered measures; $q_{\mathrm{BH}}$ values were adjusted across these three comparisons. Cliff's $\delta$ is reported as the effect size.}",
        r"\label{tab:biological-stats}",
        r"\small",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Measure & $n_R$ & $n_U$ & Median$_R$ & Median$_U$ & $\Delta$ median & Cliff's $\delta$ & $p$ & $q_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        label = row.metric.replace("$", "")
        if row.column == "delta_similarity":
            label = r"$\Delta$ similarity (rare - competitor)"
        lines.append(
            f"{label} & {row.n_rescued} & {row.n_unrescued} & "
            f"{fmt(row.median_rescued)} & {fmt(row.median_unrescued)} & "
            f"{fmt(row.median_difference)} & {fmt(row.cliffs_delta)} & "
            f"{fmt(row.p_value)} & {fmt(row.q_value_bh)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {RESULTS_DIR / 'primary_comparison_stats.csv'}")
    print(f"[saved] {TABLE_PATH}")


if __name__ == "__main__":
    main()
