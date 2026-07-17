from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import GRAY, GREEN, LIGHT_GRAY, RUST, panel_label, save_figure

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "comparison" / "comparison_summary.csv"
SCARCE = ["0.01", "0.05", "0.10"]


def dataset_cluster_ci(
    values: pd.Series, datasets: pd.Series, seed: int = 0, n: int = 10000
) -> tuple[float, float]:
    values = (
        pd.DataFrame({"value": values, "dataset": datasets})
        .groupby("dataset")["value"]
        .mean()
        .to_numpy(dtype=float)
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    df = pd.read_csv(SUMMARY, dtype={"rare_train_size": str})
    df = df[(df["status"] == "ok") & (df["rare_train_size"].isin(SCARCE))].copy()

    rows = []
    for method, group in df.groupby("method"):
        f1 = group["rare_f1"].astype(float)
        dataset_f1 = group.assign(_f1=f1).groupby("dataset")["_f1"].mean()
        lo, hi = dataset_cluster_ci(f1, group["dataset"])
        rows.append(
            {
                "method": method,
                "f1_mean": dataset_f1.mean(),
                "f1_lo": lo,
                "f1_hi": hi,
                "fp_mean": group["rare_fp_rate"].astype(float).mean(),
                "fp_max": group["rare_fp_rate"].astype(float).max(),
            }
        )
    stats = (
        pd.DataFrame(rows)
        .sort_values("f1_mean", ascending=False)
        .reset_index(drop=True)
    )

    labels = stats["method"].tolist()
    x = np.arange(len(labels))
    colors = [GREEN if m == "scRareRefine" else "#9a9a9a" for m in labels]
    fp_colors = [GREEN if v <= 0.01 + 1e-12 else RUST for v in stats["fp_max"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.25, 1.0]}
    )

    err = np.vstack(
        [stats["f1_mean"] - stats["f1_lo"], stats["f1_hi"] - stats["f1_mean"]]
    )
    axes[0].bar(
        x,
        stats["f1_mean"],
        yerr=err,
        color=colors,
        edgecolor="#333333",
        linewidth=0.4,
        capsize=2,
    )
    axes[0].set_ylabel("Rare-cell F1")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylim(0, 0.9)
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.7)
    panel_label(axes[0], "A")

    axes[1].bar(x, stats["fp_max"], color=fp_colors, edgecolor="#333333", linewidth=0.4)
    axes[1].axhline(0.01, color="#333333", linewidth=0.8, linestyle="--")
    axes[1].text(
        len(labels) - 0.3,
        0.0115,
        r"$\alpha=0.01$",
        ha="right",
        va="bottom",
        fontsize=7.5,
    )
    axes[1].set_ylabel("Max false rare-call rate")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].set_ylim(0, max(0.08, stats["fp_max"].max() * 1.12))
    axes[1].grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.7)
    panel_label(axes[1], "B")

    fig.tight_layout(w_pad=1.4)
    save_figure(fig, "fig3_scarce_benchmark")


if __name__ == "__main__":
    main()
