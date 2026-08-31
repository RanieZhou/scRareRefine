"""Generate Figure 5: split sensitivity and secondary-backbone audit.

The figure reads the split-sensitivity and TOSICA run-level result files
directly.  The historical CSV fields ``rescue_ffr`` and ``incremental_fpr``
are reported as iFPR in the manuscript; the two fields are validated to
agree before plotting.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SPLIT_BATCH_SOURCES = [
    ROOT / "results" / "adaptive_separability_gate" / "v1" / "human_run_level.csv",
    ROOT / "results" / "adaptive_separability_gate" / "v1" / "mouse_run_level.csv",
]
SPLIT_CELL_SOURCE = ROOT / "results" / "split_sensitivity" / "cell_stratified_followup_run_level.csv"
TOSICA_SOURCE = ROOT / "results" / "tosica_backbone_rescue" / "v1" / "run_level.csv"
OUT_DIR = ROOT / "paper" / "figures"

RISK_TARGET = 0.01
TOLERANCE = 1e-12
BUDGETS = ["0.01", "0.05", "0.10"]
BUDGET_LABELS = {"0.01": "1%", "0.05": "5%", "0.10": "10%"}

SPLIT_DATASETS = [
    "immune_dc",
    "mouse_lung_tms_10x",
    "mouse_pancreas_tms_10x",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
]
TOSICA_DATASETS = [
    "immune_dc",
    "mouse_lung_tms_10x",
    "mouse_pancreas_tms_10x",
    "pancreas_baron",
    "pancreas_integrated",
    "tabula_lung_endo",
    "tabula_sapiens_stomach",
    "tabula_small_intestine",
]
DISPLAY_NAMES = {
    "immune_dc": "Immune DC",
    "mouse_lung_tms_10x": "Mouse lung",
    "mouse_pancreas_tms_10x": "Mouse pancreas",
    "pancreas_baron": "Baron pancreas",
    "pancreas_integrated": "Integrated pancreas",
    "tabula_lung_endo": "Lung endothelium",
    "tabula_sapiens_stomach": "Stomach",
    "tabula_small_intestine": "Small intestine",
}

FULL_COLOR = "#D8736B"
FULL_EDGE = "#A9514B"
BASELINE_COLOR = "#8A939C"
BASELINE_EDGE = "#626B74"
NEUTRAL_EDGE = "#68727A"
RISK_EDGE = "#7E4F4A"
BUDGET_COLORS = {
    "0.01": "#B9CBE0",
    "0.05": "#C7D9D0",
    "0.10": "#E2B9B2",
}

DELTA_CMAP = LinearSegmentedColormap.from_list(
    "soft_delta",
    ["#6F94C4", "#B7CEE5", "#F8F7F2", "#F1C1B4", "#D8736B"],
)


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.4,
            "axes.labelsize": 8.0,
            "axes.titlesize": 7.9,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.105,
        1.07,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#222222",
        clip_on=False,
    )


def normalize_budget(value: object) -> str:
    text = str(value).strip().lower()
    if text == "all":
        return "all"
    try:
        return f"{float(text):.2f}"
    except ValueError as exc:
        raise AssertionError(f"Unrecognized rare-label budget: {value!r}") from exc


def load_split_data() -> pd.DataFrame:
    batch_frames = [pd.read_csv(path) for path in SPLIT_BATCH_SOURCES]
    batch = pd.concat(batch_frames, ignore_index=True)
    required_batch = {
        "dataset", "seed", "rare_train_size", "variant",
        "baseline_rare_f1", "rare_f1", "delta_rare_f1",
    }
    missing = required_batch.difference(batch.columns)
    if missing:
        raise AssertionError(f"Batch-heldout result is missing columns: {sorted(missing)}")
    batch = batch[batch["variant"] == "adaptive_sep_gate"].copy()
    batch["budget"] = batch["rare_train_size"].map(normalize_budget)
    if len(batch) != 96:
        raise AssertionError(f"Expected 96 adaptive batch-heldout units, got {len(batch)}")
    if set(batch["seed"]) != {42, 43, 44}:
        raise AssertionError(f"Expected batch seeds 42--44, got {sorted(batch['seed'].unique())}")
    if set(batch["dataset"]) != set(SPLIT_DATASETS):
        raise AssertionError("Batch-heldout dataset set does not match eight datasets")
    if batch[["dataset", "seed", "budget"]].duplicated().any():
        raise AssertionError("Duplicate adaptive batch-heldout dataset--seed--budget unit")
    batch = batch.rename(
        columns={
            "baseline_rare_f1": "scANVI_f1_batch_heldout",
            "rare_f1": "scRareRefine_f1_batch_heldout",
            "delta_rare_f1": "delta_f1_batch_heldout",
        }
    )
    batch = batch[[
        "dataset", "seed", "budget", "scANVI_f1_batch_heldout",
        "scRareRefine_f1_batch_heldout", "delta_f1_batch_heldout",
    ]]

    cell = pd.read_csv(SPLIT_CELL_SOURCE)
    required_cell = {
        "dataset", "seed", "rare_train_size", "status",
        "baseline_rare_f1", "refined_rare_f1", "delta_rare_f1",
    }
    missing = required_cell.difference(cell.columns)
    if missing:
        raise AssertionError(f"Cell-stratified result is missing columns: {sorted(missing)}")
    cell = cell[cell["status"] == "ok"].copy()
    cell["budget"] = cell["rare_train_size"].map(normalize_budget)
    if len(cell) != 96:
        raise AssertionError(f"Expected 96 cell-stratified units, got {len(cell)}")
    if set(cell["seed"]) != {42, 43, 44}:
        raise AssertionError(f"Expected cell-stratified seeds 42--44, got {sorted(cell['seed'].unique())}")
    if set(cell["dataset"]) != set(SPLIT_DATASETS):
        raise AssertionError("Cell-stratified dataset set does not match eight datasets")
    if cell[["dataset", "seed", "budget"]].duplicated().any():
        raise AssertionError("Duplicate cell-stratified dataset--seed--budget unit")
    cell = cell.rename(
        columns={
            "baseline_rare_f1": "scANVI_f1_cell_stratified",
            "refined_rare_f1": "scRareRefine_f1_cell_stratified",
            "delta_rare_f1": "delta_f1_cell_stratified",
        }
    )
    cell = cell[[
        "dataset", "seed", "budget", "scANVI_f1_cell_stratified",
        "scRareRefine_f1_cell_stratified", "delta_f1_cell_stratified",
    ]]

    df = batch.merge(cell, on=["dataset", "seed", "budget"], how="inner", validate="one_to_one")
    if len(df) != 96:
        raise AssertionError(f"Expected 96 matched split-sensitivity units, got {len(df)}")
    for split in ("batch_heldout", "cell_stratified"):
        baseline_col = f"scANVI_f1_{split}"
        refined_col = f"scRareRefine_f1_{split}"
        reported_delta = f"delta_f1_{split}"
        for column in (baseline_col, refined_col, reported_delta):
            df[column] = pd.to_numeric(df[column], errors="raise")
        computed_delta = df[refined_col] - df[baseline_col]
        if not np.allclose(computed_delta, df[reported_delta], atol=TOLERANCE, rtol=0):
            raise AssertionError(f"{split}: reported and computed delta F1 disagree")
        df[f"computed_delta_{split}"] = computed_delta
    return df


def load_tosica_data() -> pd.DataFrame:
    df = pd.read_csv(TOSICA_SOURCE)
    required = {
        "dataset",
        "seed",
        "rare_train_size",
        "baseline_rare_f1",
        "refined_rare_f1",
        "delta_rare_f1",
        "incremental_fpr",
        "rescue_ffr",
    }
    missing = required.difference(df.columns)
    if missing:
        raise AssertionError(f"TOSICA file is missing columns: {sorted(missing)}")

    df = df.copy()
    df["budget"] = df["rare_train_size"].map(normalize_budget)
    if set(df["seed"]) != {42, 43, 44}:
        raise AssertionError(f"Expected TOSICA seeds 42--44, got {sorted(df['seed'].unique())}")
    if set(df["dataset"]) != set(TOSICA_DATASETS):
        raise AssertionError("TOSICA dataset set does not match the eight benchmark datasets")
    if len(df) != 96 or df[["dataset", "seed", "budget"]].duplicated().any():
        raise AssertionError("Expected one row per TOSICA dataset, seed and budget")

    for column in (
        "baseline_rare_f1",
        "refined_rare_f1",
        "delta_rare_f1",
        "incremental_fpr",
        "rescue_ffr",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    computed_delta = df["refined_rare_f1"] - df["baseline_rare_f1"]
    if not np.allclose(computed_delta, df["delta_rare_f1"], atol=TOLERANCE, rtol=0):
        raise AssertionError("TOSICA: reported and computed delta F1 disagree")
    if not np.allclose(df["incremental_fpr"], df["rescue_ffr"], atol=TOLERANCE, rtol=0):
        raise AssertionError("TOSICA: incremental_fpr and rescue_ffr disagree")
    df["computed_delta"] = computed_delta
    df["risk"] = df["incremental_fpr"]
    return df


def scarce_split(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["budget"].isin(BUDGETS)].copy()
    if len(sub) != 72:
        raise AssertionError(f"Expected 72 split-sensitivity scarce units per split, got {len(sub)}")
    return sub


def scarce_tosica(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["budget"].isin(BUDGETS)].copy()
    if len(sub) != 72:
        raise AssertionError(f"Expected 72 TOSICA scarce units, got {len(sub)}")
    return sub


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap CI for the sample mean."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise AssertionError("Bootstrap input must be a non-empty one-dimensional array")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(10000, values.size))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def draw_panel_a(ax: plt.Axes, split_df: pd.DataFrame) -> None:
    """Show split-level paired gains without duplicating the absolute scores."""
    split_specs = [
        ("batch_heldout", "Batch-heldout", "#D5DCE1"),
        ("cell_stratified", "Cell-stratified", "#E0B5AE"),
    ]
    positions = np.arange(len(split_specs), dtype=float)
    delta_values = []

    for split, _, _ in split_specs:
        values = split_df[f"computed_delta_{split}"].to_numpy(dtype=float)
        delta_values.append(values)

    box = ax.boxplot(
        delta_values,
        positions=positions,
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        boxprops={"edgecolor": NEUTRAL_EDGE, "linewidth": 0.8},
        whiskerprops={"color": NEUTRAL_EDGE, "linewidth": 0.75},
        capprops={"color": NEUTRAL_EDGE, "linewidth": 0.75},
        medianprops={"color": "#39434A", "linewidth": 1.05},
    )
    for patch, (_, _, color) in zip(box["boxes"], split_specs):
        patch.set_facecolor(color)
        patch.set_alpha(0.82)

    for position, (values, (split, _, _)) in enumerate(zip(delta_values, split_specs)):
        mean = float(values.mean())
        ci_low, ci_high = bootstrap_mean_ci(values, seed=9100 + position)
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - ci_low], [ci_high - mean]],
            fmt="none",
            ecolor=FULL_EDGE,
            elinewidth=0.85,
            capsize=2.7,
            capthick=0.85,
            zorder=5,
        )
        ax.scatter(
            [position],
            [mean],
            marker="D",
            s=31,
            color=FULL_COLOR,
            edgecolor=FULL_EDGE,
            linewidth=0.8,
            zorder=6,
        )
        ax.annotate(
            f"{mean:+.3f}",
            xy=(position, mean),
            # Put the value outside the right edge of the box so that it does
            # not cover the box, mean diamond, CI, or whiskers.
            xytext=(position + 0.23, mean),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=6.3,
            color=FULL_EDGE,
            zorder=7,
        )

    ax.axhline(0.0, color="#788188", linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(-0.10, 1.05)
    ax.set_ylabel("$\\Delta$ Rare-cell F1")
    ax.set_xlabel("Data split")
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _ in split_specs])
    ax.set_yticks(np.arange(-0.1, 1.01, 0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    add_panel_label(ax, "A")


def heatmap_matrix(df: pd.DataFrame, split: str) -> np.ndarray:
    column = f"computed_delta_{split}"
    averaged = (
        df.groupby(["dataset", "budget"], as_index=False)[column]
        .mean()
    )
    values = (
        averaged.pivot(index="dataset", columns="budget", values=column)
        .reindex(index=SPLIT_DATASETS, columns=BUDGETS)
        .to_numpy(dtype=float)
    )
    if np.isnan(values).any():
        raise AssertionError(f"Missing values in {split} split heatmap")
    return values


def draw_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    show_y_labels: bool,
    annotate: bool = False,
) -> object:
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    image = ax.imshow(values, cmap=DELTA_CMAP, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=7.5, fontweight="bold", color="#4E575E", pad=5)
    ax.set_xticks(range(len(BUDGETS)))
    ax.set_xticklabels([BUDGET_LABELS[budget] for budget in BUDGETS])
    ax.set_yticks(range(len(SPLIT_DATASETS)))
    if show_y_labels:
        ax.set_yticklabels([DISPLAY_NAMES[dataset] for dataset in SPLIT_DATASETS])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=2.0)
    if annotate:
        for row_index in range(values.shape[0]):
            for column_index in range(values.shape[1]):
                value = values[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="#26313A" if abs(value) < 0.55 else "white",
                )
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def draw_supplementary_heatmaps(figure: plt.Figure, split_df: pd.DataFrame) -> None:
    nested = figure.add_gridspec(
        1,
        2,
        left=0.14,
        right=0.88,
        bottom=0.16,
        top=0.88,
        wspace=0.15,
    )
    ax_left = figure.add_subplot(nested[0, 0])
    ax_right = figure.add_subplot(nested[0, 1])
    left_values = heatmap_matrix(split_df, "batch_heldout")
    right_values = heatmap_matrix(split_df, "cell_stratified")
    image = draw_heatmap(ax_left, left_values, "Batch-heldout", True, annotate=True)
    draw_heatmap(ax_right, right_values, "Cell-stratified", False, annotate=True)
    colorbar = figure.colorbar(
        image,
        ax=[ax_left, ax_right],
        fraction=0.045,
        pad=0.06,
        shrink=0.88,
        aspect=22,
    )
    colorbar.set_label("$\\Delta$ Rare-cell F1\n(scRareRefine $-$ scANVI)", fontsize=7.0)
    colorbar.ax.tick_params(labelsize=6.3, width=0.5, length=2.0)
    add_panel_label(ax_left, "A")
    add_panel_label(ax_right, "B")


def wtl_counts(values: pd.Series) -> tuple[int, int, int]:
    array = values.to_numpy(dtype=float)
    wins = int((array > TOLERANCE).sum())
    ties = int(np.isclose(array, 0.0, atol=TOLERANCE, rtol=0).sum())
    losses = int((array < -TOLERANCE).sum())
    if wins + ties + losses != len(array):
        raise AssertionError("W/T/L counts do not cover all observations")
    return wins, ties, losses


def aggregate_tosica_for_figure(tosica_df: pd.DataFrame) -> pd.DataFrame:
    """Average the three seeds within each dataset--budget group for plotting."""
    aggregated = (
        tosica_df.groupby(["dataset", "budget"], as_index=False)
        .agg(
            delta=("computed_delta", "mean"),
            risk=("risk", "mean"),
        )
    )
    if len(aggregated) != 24:
        raise AssertionError(f"Expected 24 dataset--budget aggregates, got {len(aggregated)}")
    counts = aggregated.groupby("budget").size().reindex(BUDGETS)
    if counts.isna().any() or not np.all(counts.to_numpy(dtype=int) == 8):
        raise AssertionError("Expected eight dataset-level aggregates per budget")
    return aggregated


def draw_budget_boxplots(
    axis: plt.Axes,
    values_by_budget: list[np.ndarray],
    ylabel: str,
    ylim: tuple[float, float],
) -> None:
    positions = np.arange(1, len(BUDGETS) + 1, dtype=float)
    box = axis.boxplot(
        values_by_budget,
        positions=positions,
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),
        boxprops={"edgecolor": NEUTRAL_EDGE, "linewidth": 0.8},
        whiskerprops={"color": NEUTRAL_EDGE, "linewidth": 0.75},
        capprops={"color": NEUTRAL_EDGE, "linewidth": 0.75},
        medianprops={"color": "#39434A", "linewidth": 1.05},
    )
    for patch, budget in zip(box["boxes"], BUDGETS):
        patch.set_facecolor(BUDGET_COLORS[budget])
        patch.set_alpha(0.86)
    axis.set_xlim(0.5, len(BUDGETS) + 0.5)
    axis.set_ylim(*ylim)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions)
    axis.set_xticklabels([BUDGET_LABELS[budget] for budget in BUDGETS])
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def draw_panel_b(ax_top: plt.Axes, ax_bottom: plt.Axes, tosica_df: pd.DataFrame) -> None:
    """Show standard seed-aggregated TOSICA effect and risk distributions."""
    aggregated = aggregate_tosica_for_figure(tosica_df)
    delta_values = [
        aggregated.loc[aggregated["budget"] == budget, "delta"].to_numpy(dtype=float)
        for budget in BUDGETS
    ]
    risk_values = [
        aggregated.loc[aggregated["budget"] == budget, "risk"].to_numpy(dtype=float)
        for budget in BUDGETS
    ]

    delta_min = min(float(values.min()) for values in delta_values)
    delta_max = max(float(values.max()) for values in delta_values)
    draw_budget_boxplots(
        ax_top,
        delta_values,
        "$\\Delta$ Rare-cell F1",
        (min(-0.05, delta_min - 0.015), max(0.12, delta_max + 0.02)),
    )
    ax_top.axhline(0.0, color="#788188", linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)
    ax_top.tick_params(axis="x", labelbottom=False)
    add_panel_label(ax_top, "B")

    risk_max = max(float(values.max()) for values in risk_values)
    draw_budget_boxplots(
        ax_bottom,
        risk_values,
        "iFPR",
        (0.0, max(0.03, risk_max * 1.12)),
    )
    ax_bottom.axhline(
        RISK_TARGET,
        color="#788188",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=2,
    )
    ax_bottom.text(
        len(BUDGETS) + 0.43,
        RISK_TARGET + 0.0008,
        r"$\alpha=0.01$",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#59636B",
    )
    ax_bottom.set_xlabel("Rare-label budget")


def draw_panel_b_legacy(ax_top: plt.Axes, ax_bottom: plt.Axes, tosica_df: pd.DataFrame) -> None:
    """Legacy unit-level audit retained only for provenance."""
    ordered = tosica_df.copy()
    ordered["_dataset_order"] = ordered["dataset"].map(
        {dataset: index for index, dataset in enumerate(TOSICA_DATASETS)}
    )
    ordered["_budget_order"] = ordered["budget"].map(
        {budget: index for index, budget in enumerate(BUDGETS)}
    )
    ordered = ordered.sort_values(
        by=["computed_delta", "_dataset_order", "_budget_order"]
    ).reset_index(drop=True)

    if len(ordered) != 72:
        raise AssertionError(f"Expected 72 ordered TOSICA units, got {len(ordered)}")

    delta = ordered["computed_delta"].to_numpy(dtype=float)
    risk = ordered["risk"].to_numpy(dtype=float)
    x = np.arange(1, len(ordered) + 1, dtype=float)
    negative = delta < -TOLERANCE
    ties = np.isclose(delta, 0.0, atol=TOLERANCE, rtol=0)
    positive = delta > TOLERANCE
    violations = risk > RISK_TARGET + TOLERANCE

    for axis, values in ((ax_top, delta), (ax_bottom, risk)):
        for x_value, y_value in zip(x, values):
            axis.vlines(
                x_value,
                0.0,
                y_value,
                color="#D9DEE2",
                linewidth=0.55,
                alpha=0.85,
                zorder=1,
            )
        axis.set_xlim(0.35, len(ordered) + 0.65)
        axis.set_xticks([1, 12, 24, 36, 48, 60, 72])
        axis.grid(False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if positive.any():
        ax_top.scatter(
            x[positive],
            delta[positive],
            s=22,
            color=FULL_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            zorder=4,
        )
    if ties.any():
        ax_top.scatter(
            x[ties],
            delta[ties],
            s=18,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=4,
        )
    if negative.any():
        ax_top.scatter(
            x[negative],
            delta[negative],
            s=44,
            marker="x",
            color=RISK_EDGE,
            linewidth=1.35,
            zorder=5,
        )
    ax_top.axhline(0.0, color="#788188", linestyle=(0, (3, 2)), linewidth=0.8, zorder=2)

    ax_top.set_ylim(min(-0.10, float(delta.min()) - 0.025), max(0.24, float(delta.max()) + 0.03))
    ax_top.set_ylabel("$\\Delta$ Rare-cell F1")
    ax_top.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_top.tick_params(axis="x", labelbottom=False)
    add_panel_label(ax_top, "B")

    safe_positive = positive & ~violations
    safe_tie = ties & ~violations
    safe_negative = negative & ~violations
    risk_positive_or_tie = ~negative & violations
    risk_negative = negative & violations
    if safe_positive.any():
        ax_bottom.scatter(
            x[safe_positive],
            risk[safe_positive],
            s=21,
            color=FULL_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.75,
            zorder=4,
        )
    if safe_tie.any():
        ax_bottom.scatter(
            x[safe_tie],
            risk[safe_tie],
            s=18,
            color=BASELINE_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.9,
            zorder=4,
        )
    if safe_negative.any():
        ax_bottom.scatter(
            x[safe_negative],
            risk[safe_negative],
            s=42,
            marker="x",
            color=RISK_EDGE,
            linewidth=1.25,
            zorder=5,
        )
    if risk_positive_or_tie.any():
        ax_bottom.scatter(
            x[risk_positive_or_tie],
            risk[risk_positive_or_tie],
            s=35,
            facecolor="white",
            edgecolor=RISK_EDGE,
            linewidth=1.05,
            zorder=5,
        )
    if risk_negative.any():
        ax_bottom.scatter(
            x[risk_negative],
            risk[risk_negative],
            s=46,
            marker="x",
            color=RISK_EDGE,
            linewidth=1.4,
            zorder=6,
        )

    ax_bottom.axhline(
        RISK_TARGET,
        color="#788188",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        zorder=2,
    )
    ax_bottom.text(
        len(ordered) + 0.45,
        RISK_TARGET + 0.0013,
        r"$\alpha=0.01$",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color="#59636B",
    )
    ax_bottom.set_ylim(0.0, max(0.065, float(risk.max()) * 1.12))
    ax_bottom.set_ylabel("iFPR")
    ax_bottom.set_xlabel("TOSICA test units ordered by $\\Delta$F1")
    ax_bottom.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_bottom.set_xticklabels(["1", "12", "24", "36", "48", "60", "72"])


def save_figure_outputs(figure: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, options in {
        "pdf": {"format": "pdf"},
        "svg": {"format": "svg"},
        "png": {"format": "png", "dpi": 300},
    }.items():
        path = OUT_DIR / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", pad_inches=0.04, **options)
        print(f"saved {path} ({path.stat().st_size} bytes)")


def main() -> None:
    set_publication_style()
    split_df = scarce_split(load_split_data())
    tosica_df = scarce_tosica(load_tosica_data())

    figure = plt.figure(figsize=(7.6, 4.65), facecolor="white")
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[0.82, 1.18],
        left=0.085,
        right=0.975,
        bottom=0.14,
        top=0.93,
        wspace=0.52,
    )
    ax_a = figure.add_subplot(outer[0, 0])
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.13)
    ax_b_top = figure.add_subplot(right[0, 0])
    ax_b_bottom = figure.add_subplot(right[1, 0], sharex=ax_b_top)

    draw_panel_a(ax_a, split_df)
    draw_panel_b(ax_b_top, ax_b_bottom, tosica_df)
    save_figure_outputs(figure, "figure5")

    supplementary = plt.figure(figsize=(7.0, 4.6), facecolor="white")
    draw_supplementary_heatmaps(supplementary, split_df)
    save_figure_outputs(supplementary, "fig_split_sensitivity")

    wins, ties, losses = wtl_counts(tosica_df["computed_delta"])
    print("split units: 8 datasets x 3 seeds x 3 scarce budgets = 72 per split")
    print("TOSICA units: 8 datasets x 3 seeds x 3 scarce budgets = 72")
    print(
        "split means: "
        f"batch {split_df['scANVI_f1_batch_heldout'].mean():.6f} -> "
        f"{split_df['scRareRefine_f1_batch_heldout'].mean():.6f}; "
        f"cell-stratified {split_df['scANVI_f1_cell_stratified'].mean():.6f} -> "
        f"{split_df['scRareRefine_f1_cell_stratified'].mean():.6f}"
    )
    print(
        f"TOSICA mean F1: {tosica_df['baseline_rare_f1'].mean():.6f} -> "
        f"{tosica_df['refined_rare_f1'].mean():.6f}; W/T/L={wins}/{ties}/{losses}; "
        f"max iFPR={tosica_df['risk'].max():.6f}; "
        f"violations={(tosica_df['risk'] > RISK_TARGET + TOLERANCE).sum()}; "
        f"negative={(tosica_df['computed_delta'] < -TOLERANCE).sum()}"
    )
    plt.close(figure)
    plt.close(supplementary)


if __name__ == "__main__":
    main()
