"""Generate Figure 3: component roles and candidate-rank sensitivity.

Panel A combines the paired performance effect and the safety cost of the
same four leave-one-component-out variants.  Panel B is a compact profile of
fixed candidate ranks versus the validation-adaptive rank.  All summaries are
read from the ablation CSV; the historical ``ffr`` column is retained as the
data-field alias, while figure text uses ``iFPR``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "ablation" / "ablation_summary.csv"
OUT_DIR = ROOT / "paper" / "figures"

RISK_TARGET = 0.01
EXPECTED_DATASETS = {
    "immune_dc",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
}
# Representative display subset for the main ablation figure.  The complete
# six-human-dataset grid remains validated and available in the source CSV.
DISPLAY_DATASETS = {
    "pancreas_baron",
    "tabula_sapiens_stomach",
    "pancreas_integrated",
}
FULL_VARIANT = "A5_full"
ABLATIONS = [
    ("A3_minus_adaptive_rank", "Adaptive rank $\\rightarrow$ fixed $k=1$"),
    ("A4_minus_tau", "No score threshold $\\tau$"),
    ("A2_minus_necessity", "No necessity guard"),
    ("A1_minus_sep", "No separability gate"),
]
# Two-line labels keep the full ablation names readable without long rotated
# text crossing into the lower plot.  They are only a display choice; the
# underlying variant names and the statistical definitions are unchanged.
ABLATION_DISPLAY_LABELS = [
    "Adaptive rank\n→ fixed $k=1$",
    "No score threshold\n$\\tau$",
    "No necessity\nguard",
    "No separability\ngate",
]
RANKS = [
    ("R1_rank1", "k=1"),
    ("R2_rank2", "k=2"),
    ("R3_rank3", "k=3"),
    ("R_adaptive", "Adaptive"),
]

# Match the Figure 2 scRareRefine emphasis color.  The risk color is used
# only for points that exceed the empirical target in a risk display.
FULL_COLOR = "#D8736B"
FULL_EDGE = "#A9514B"
NEUTRAL_COLOR = "#A2A9AF"
NEUTRAL_EDGE = "#68727A"
RISK_COLOR = "#B67C75"
RISK_EDGE = "#8E5953"
POINT_COLOR = "#8D989F"
GRID_COLOR = "#E5E8EA"


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.4,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.065,
        1.08,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color="#222222",
    )


def load_ablation_data() -> pd.DataFrame:
    df = pd.read_csv(SOURCE)
    required = {"dataset", "seed", "rts", "variant", "group", "rare_f1", "ffr"}
    missing = required.difference(df.columns)
    if missing:
        raise AssertionError(f"Ablation file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["rare_f1"] = pd.to_numeric(df["rare_f1"], errors="raise")
    df["ffr"] = pd.to_numeric(df["ffr"], errors="raise")
    df["unit_key"] = (
        df["dataset"].astype(str)
        + "|"
        + df["rts"].astype(str)
        + "|"
        + df["seed"].astype(str)
    )

    expected_variants = [FULL_VARIANT] + [key for key, _ in ABLATIONS] + [key for key, _ in RANKS]
    for variant in expected_variants:
        sub = df[df["variant"] == variant]
        if len(sub) != 72:
            raise AssertionError(f"{variant}: expected 72 observations, got {len(sub)}")
        if set(sub["dataset"]) != EXPECTED_DATASETS:
            raise AssertionError(f"{variant}: dataset set does not match six human datasets")
        if sub["unit_key"].nunique() != 72:
            raise AssertionError(f"{variant}: duplicate or missing dataset-budget-seed units")
        if sub["seed"].nunique() != 3 or sub["rts"].nunique() != 4:
            raise AssertionError(f"{variant}: expected 3 seeds and 4 budgets")

    return df


def unit_values(df: pd.DataFrame, variant: str, unit_order: list[str], column: str) -> np.ndarray:
    sub = df[df["variant"] == variant].set_index("unit_key")
    return sub.loc[unit_order, column].to_numpy(dtype=float)


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    bootstrap_means = values[indices].mean(axis=1)
    mean_value = float(values.mean())
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean_value, float(lower), float(upper)


def signed_delta_formatter(value: float, _position: int) -> str:
    if abs(value) < 1e-10:
        return "0"
    return f"{value:+.2f}"


def clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_component_summary(
    ax_impact: plt.Axes, ax_cost: plt.Axes, df: pd.DataFrame
) -> None:
    """Draw Panel A as two vertically aligned summaries.

    The upper plot uses a conventional boxplot for the paired F1 changes.
    The lower plot keeps the maximum-iFPR summary as one point per ablation,
    because the plotted quantity is a single maximum for each variant rather
    than a distribution of observations.
    """

    unit_order = sorted(df[df["variant"] == FULL_VARIANT]["unit_key"].unique())
    full_values = unit_values(df, FULL_VARIANT, unit_order, "rare_f1")
    positions = np.arange(len(ABLATIONS))
    delta_values = []

    for variant, _ in ABLATIONS:
        ablation_values = unit_values(df, variant, unit_order, "rare_f1")
        delta_values.append(ablation_values - full_values)

    # Upper half: standard boxplots of the paired ablation effects. The
    # default 1.5-IQR whiskers and fliers retain conventional boxplot
    # semantics without adding a second summary layer.
    boxplot = ax_impact.boxplot(
        delta_values,
        positions=positions,
        widths=0.56,
        patch_artist=True,
        whis=1.5,
        showfliers=True,
        boxprops={
            "facecolor": "#D9DEE1",
            "edgecolor": "#68727A",
            "linewidth": 0.9,
        },
        medianprops={"color": "#39454D", "linewidth": 1.15},
        whiskerprops={"color": "#68727A", "linewidth": 0.85},
        capprops={"color": "#68727A", "linewidth": 0.85},
        flierprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#8D989F",
            "markeredgewidth": 0.55,
            "markersize": 2.8,
            "alpha": 0.72,
        },
        zorder=3,
    )
    for patch in boxplot["boxes"]:
        patch.set_alpha(0.9)

    ax_impact.axhline(
        0.0, color="#68727A", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1
    )
    ax_impact.set_xlim(-0.5, len(ABLATIONS) - 0.5)
    ax_impact.set_xticks(positions)
    # The shared category labels are shown only below the lower summary to
    # avoid duplicating long ablation names between the two aligned plots.
    ax_impact.tick_params(axis="x", labelbottom=False, length=0)
    ax_impact.set_ylim(-0.045, 0.045)
    ax_impact.set_yticks([-0.04, -0.02, 0.0, 0.02, 0.04])
    ax_impact.yaxis.set_major_formatter(FuncFormatter(signed_delta_formatter))
    ax_impact.set_ylabel("$\\Delta$ Rare-cell F1 vs full method", labelpad=5)
    ax_impact.set_title("Performance impact", pad=4, fontweight="bold")
    ax_impact.set_axisbelow(True)
    clean_axes(ax_impact)
    add_panel_label(ax_impact, "A")

    # Lower half: one maximum-iFPR point per ablation and the empirical target.
    risk_values = np.array(
        [
            float(df.loc[df["variant"] == variant, "ffr"].max())
            for variant, _ in ABLATIONS
        ]
    )
    exceeds = risk_values > RISK_TARGET
    ax_cost.axhline(
        RISK_TARGET,
        color="#68727A",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    for position, risk_value, exceeds_target in zip(positions, risk_values, exceeds):
        ax_cost.scatter(
            [position],
            [risk_value],
            s=32,
            facecolor=RISK_COLOR if exceeds_target else NEUTRAL_COLOR,
            edgecolor=RISK_EDGE if exceeds_target else NEUTRAL_EDGE,
            linewidth=0.85,
            zorder=4,
        )
        ax_cost.annotate(
            f"{risk_value:.4f}",
            xy=(position, risk_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.2,
            color=RISK_EDGE if exceeds_target else NEUTRAL_EDGE,
        )

    ax_cost.text(
        len(ABLATIONS) - 0.55,
        RISK_TARGET + 0.0010,
        r"$\alpha=0.01$",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#59636B",
    )
    ax_cost.set_xlim(-0.5, len(ABLATIONS) - 0.5)
    ax_cost.set_xticks(positions)
    ax_cost.set_xticklabels(ABLATION_DISPLAY_LABELS, rotation=0, ha="center")
    ax_cost.set_ylim(0.0, 0.020)
    ax_cost.set_yticks([0.000, 0.005, 0.010, 0.015, 0.020])
    ax_cost.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_cost.set_ylabel("Maximum iFPR", labelpad=5)
    ax_cost.set_xlabel("Ablation variant", labelpad=3)
    ax_cost.set_title("Maximum observed iFPR", pad=4, fontweight="bold")
    ax_cost.set_axisbelow(True)
    clean_axes(ax_cost)

def draw_rank_profile(ax_top: plt.Axes, ax_bottom: plt.Axes, df: pd.DataFrame) -> None:
    """Draw Panel B with a visual gap before the adaptive strategy."""

    fixed_x = np.array([0.0, 1.0, 2.0])
    adaptive_x = 3.45
    x_all = np.array([0.0, 1.0, 2.0, adaptive_x])
    mean_f1 = np.array(
        [float(df.loc[df["variant"] == variant, "rare_f1"].mean()) for variant, _ in RANKS]
    )
    max_ffr = np.array(
        [float(df.loc[df["variant"] == variant, "ffr"].max()) for variant, _ in RANKS]
    )

    # Only fixed-k points are connected; Adaptive is a validation-selected
    # strategy rather than a fourth point on the fixed-rank trajectory.
    ax_top.plot(fixed_x, mean_f1[:3], color=NEUTRAL_EDGE, linewidth=1.0, zorder=2)
    ax_bottom.plot(fixed_x, max_ffr[:3], color=NEUTRAL_EDGE, linewidth=1.0, zorder=2)

    for index, x_value in enumerate(x_all):
        adaptive = index == 3
        face = FULL_COLOR if adaptive else NEUTRAL_COLOR
        edge = FULL_EDGE if adaptive else NEUTRAL_EDGE
        size = 46 if adaptive else 34

        ax_top.scatter(
            [x_value], [mean_f1[index]], s=size, color=face, edgecolor=edge, linewidth=0.8, zorder=4
        )
        ax_bottom.scatter(
            [x_value], [max_ffr[index]], s=size, color=face, edgecolor=edge, linewidth=0.8, zorder=4
        )

        top_offset = 7 if index != 2 else -10
        bottom_offset = 7 if index not in (2,) else -11
        ax_top.annotate(
            f"{mean_f1[index]:.3f}",
            xy=(x_value, mean_f1[index]),
            xytext=(0, top_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.1,
            color=edge,
        )
        ax_bottom.annotate(
            f"{max_ffr[index]:.4f}",
            xy=(x_value, max_ffr[index]),
            xytext=(0, bottom_offset),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=6.1,
            color=edge,
        )

    for ax in (ax_top, ax_bottom):
        ax.set_xlim(-0.35, adaptive_x + 0.35)
        ax.set_xticks(x_all)
        ax.set_xticklabels([label for _, label in RANKS])
        clean_axes(ax)

    ax_top.set_ylabel("Mean Rare-cell F1", labelpad=6)
    ax_top.set_ylim(0.848, 0.892)
    ax_top.set_yticks([0.85, 0.87, 0.89])
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_top.tick_params(axis="x", labelbottom=False, length=0)
    add_panel_label(ax_top, "B")

    ax_bottom.axhline(
        RISK_TARGET,
        color="#68727A",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=1,
    )
    ax_bottom.text(
        2.10,
        RISK_TARGET + 0.0012,
        r"$\alpha=0.01$",
        ha="left",
        va="bottom",
        fontsize=6.1,
        color="#59636B",
    )
    ax_bottom.set_ylabel("Maximum iFPR", labelpad=5)
    ax_bottom.set_xlabel("Candidate rank strategy", labelpad=3)
    ax_bottom.set_ylim(0.0, 0.053)
    ax_bottom.set_yticks([0.00, 0.02, 0.04])
    ax_bottom.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bottom.tick_params(axis="x", pad=2)


def main() -> None:
    set_publication_style()
    df = load_ablation_data()
    display_df = df[df["dataset"].isin(DISPLAY_DATASETS)].copy()

    figure = plt.figure(figsize=(7.8, 5.05), facecolor="white")
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.62, 1.0],
        left=0.095,
        right=0.985,
        bottom=0.20,
        top=0.88,
        wspace=0.48,
    )
    component = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.50)
    rank = outer[1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.10)

    ax_impact = figure.add_subplot(component[0, 0])
    ax_cost = figure.add_subplot(component[1, 0], sharex=ax_impact)
    ax_rank_top = figure.add_subplot(rank[0, 0])
    ax_rank_bottom = figure.add_subplot(rank[1, 0], sharex=ax_rank_top)

    draw_component_summary(ax_impact, ax_cost, display_df)
    # Keep the compact rank profile on the complete six-dataset grid so that
    # its values remain aligned with the aggregate rank-sensitivity results.
    draw_rank_profile(ax_rank_top, ax_rank_bottom, df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in {
        "pdf": {"format": "pdf"},
        "svg": {"format": "svg"},
        "png": {"format": "png", "dpi": 300},
    }.items():
        path = OUT_DIR / f"figure3.{suffix}"
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04, **options)
        print(f"saved {path} ({path.stat().st_size} bytes)")

    unit_count = len(DISPLAY_DATASETS) * 4 * 3
    print(
        f"paired component observations shown: {len(ABLATIONS)} variants x "
        f"{unit_count} units ({len(DISPLAY_DATASETS)} representative datasets)"
    )
    for variant, label in ABLATIONS:
        sub = df[df["variant"] == variant]
        print(f"{label}: max iFPR={sub['ffr'].max():.6f}")
    for variant, label in RANKS:
        sub = df[df["variant"] == variant]
        print(
            f"{label}: mean F1={sub['rare_f1'].mean():.6f}; "
            f"max iFPR={sub['ffr'].max():.6f}"
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
