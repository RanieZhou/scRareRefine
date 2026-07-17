"""Cache-only characterization of the formal rescue selection pathway."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rescue import (  # noqa: E402
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
    DEFAULT_CONFORMAL_ALPHA,
    MIN_VAL_MISSED,
    PrototypeRescuer,
    conformal_rescue,
)
from src.utils import git_sha, load_config, make_run_dir, parse_rare_train_size  # noqa: E402
from tools.analysis.rescue_composition import (  # noqa: E402
    CONFIGS,
    RARE_TRAIN_SIZES,
    SEEDS,
    _expected_labeled_rare,
    _latent,
    _load_aligned_split,
)

OUT = ROOT / "results" / "residual_signal" / "v1"
LOG_DIR = ROOT / "logs" / "residual_signal"
COMPOSITION_PATH = ROOT / "results" / "rescue_composition" / "v1" / "run_level.csv"
PRIMARY_GROUPS = (
    "baseline_correct_rare",
    "true_rescued_rare",
    "unrescued_rare",
    "non_target",
)
METRICS = {
    "rare_membership_score": 1,
    "rare_rank": -1,
    "rare_standardized_distance": -1,
    "standardized_prototype_margin": 1,
    "rare_prototype_distance": -1,
    "nearest_nonrare_distance": 1,
    "prototype_margin": 1,
}
CONTRASTS = {
    "H1_baseline_correct_vs_true_rescued": (
        "baseline_correct_rare",
        "true_rescued_rare",
    ),
    "H2_true_rescued_vs_unrescued": ("true_rescued_rare", "unrescued_rare"),
    "H3a_true_rescued_vs_non_target": ("true_rescued_rare", "non_target"),
    "H3b_true_rescued_vs_closest_competitor": (
        "true_rescued_rare",
        "closest_competitor",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_dirty() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        )
        return bool(output.strip())
    except Exception:
        return None


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def assign_primary_groups(
    y_true: pd.Series,
    baseline: pd.Series,
    final: pd.Series,
    rare_class: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = pd.Series(y_true).astype(str).reset_index(drop=True)
    base = pd.Series(baseline).astype(str).reset_index(drop=True)
    refined = pd.Series(final).astype(str).reset_index(drop=True)
    if not (len(y) == len(base) == len(refined)):
        raise ValueError("group inputs have different lengths")
    rare = y.eq(rare_class).to_numpy()
    base_rare = base.eq(rare_class).to_numpy()
    final_rare = refined.eq(rare_class).to_numpy()
    changed = refined.ne(base).to_numpy()
    invalid = changed & ~(~base_rare & final_rare)
    if invalid.any():
        raise AssertionError("formal rescue introduced an invalid label transition")
    masks = {
        "baseline_correct_rare": rare & base_rare,
        "true_rescued_rare": rare & ~base_rare & final_rare,
        "unrescued_rare": rare & ~final_rare,
        "non_target": ~rare,
    }
    membership = np.column_stack(list(masks.values())).sum(axis=1)
    if not np.all(membership == 1):
        raise AssertionError("primary groups are not mutually exclusive and exhaustive")
    groups = np.empty(len(y), dtype=object)
    for name, mask in masks.items():
        groups[mask] = name
    false_rescue = (~rare) & changed
    return groups.astype(str), false_rescue


def prototype_metrics(
    proto: PrototypeRescuer, query_latent: np.ndarray
) -> tuple[pd.DataFrame, str]:
    classes = list(proto.classes)
    rare_index = classes.index(proto.rare_class)
    prototypes = np.vstack([proto.prototypes[label] for label in classes])
    radii = np.asarray([proto.radii[label] for label in classes], dtype=float)
    distances = np.sqrt(
        ((query_latent[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2)
    )
    standardized = distances / radii[None, :]
    nonrare_indices = [
        i for i, label in enumerate(classes) if label != proto.rare_class
    ]
    if not nonrare_indices:
        raise ValueError("prototype analysis requires at least one non-target class")
    rare_distance = distances[:, rare_index]
    rare_standardized = standardized[:, rare_index]
    nearest_nonrare = distances[:, nonrare_indices].min(axis=1)
    nearest_nonrare_standardized = standardized[:, nonrare_indices].min(axis=1)
    rare_prototype = prototypes[rare_index]
    competitor_index = min(
        nonrare_indices,
        key=lambda i: float(np.linalg.norm(rare_prototype - prototypes[i])),
    )
    metrics = pd.DataFrame(
        {
            "rare_membership_score": proto.rare_membership_score(query_latent),
            "rare_rank": proto.rare_rank(query_latent).astype(int),
            "rare_standardized_distance": rare_standardized,
            "standardized_prototype_margin": (
                nearest_nonrare_standardized - rare_standardized
            ),
            "rare_prototype_distance": rare_distance,
            "nearest_nonrare_distance": nearest_nonrare,
            "prototype_margin": nearest_nonrare - rare_distance,
        }
    )
    if not np.isfinite(metrics.to_numpy(dtype=float)).all():
        raise ValueError("prototype metrics contain non-finite values")
    return metrics, classes[competitor_index]


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    statistic = mannwhitneyu(
        x, y, alternative="two-sided", method="asymptotic"
    ).statistic
    return float(2.0 * statistic / (len(x) * len(y)) - 1.0)


def _historical_identity_lookup() -> dict[tuple[str, int, str], dict]:
    composition = pd.read_csv(COMPOSITION_PATH, dtype={"rare_train_size": str})
    keys = ["dataset", "seed", "rare_train_size"]
    if composition.duplicated(keys).any() or len(composition) != 96:
        raise ValueError("rescue-composition ledger is not a unique 96-row grid")
    return {
        (str(row.dataset), int(row.seed), str(row.rare_train_size)): row._asdict()
        for row in composition.itertuples(index=False)
    }


def _analyze_run(
    config_path: str,
    seed: int,
    rare_train_size: str,
    historical: dict[tuple[str, int, str], dict],
) -> tuple[dict, pd.DataFrame | None, dict]:
    config = load_config(ROOT / config_path)
    dataset = config["dataset"]["name"]
    experiment = config["experiment"]
    rare = experiment["rare_class"]
    split_mode = experiment.get("split_mode", "batch_heldout")
    run_dir = ROOT / make_run_dir(
        config, split_mode, seed, rare, parse_rare_train_size(rare_train_size)
    )
    row = {
        "dataset": dataset,
        "seed": seed,
        "rare_train_size": rare_train_size,
        "rare_class": rare,
        "split_mode": split_mode,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status": "analysis_failed",
        "error": "",
        "current_replay_available": False,
        "historical_cell_identity_available": False,
    }
    provenance = {"run_dir": row["run_dir"], "files": {}}
    embeddings = run_dir / "embeddings"
    required = [
        embeddings / f"{split}_{kind}.csv"
        for split in ("train", "validation", "test")
        for kind in ("predictions", "latent")
    ]
    if not all(path.exists() for path in required):
        missing = [
            str(path.relative_to(ROOT)) for path in required if not path.exists()
        ]
        row.update(status="missing_cache", error="; ".join(missing))
        return row, None, provenance
    try:
        frames = {
            split: _load_aligned_split(embeddings, split)
            for split in ("train", "validation", "test")
        }
        id_sets = {split: set(frame["cell_id"]) for split, frame in frames.items()}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            if id_sets[left].intersection(id_sets[right]):
                raise ValueError(f"cell_id overlap between {left} and {right}")
        if len({_latent(frame).shape[1] for frame in frames.values()}) != 1:
            raise ValueError("latent dimensions differ across splits")
        train = frames["train"]
        validation = frames["validation"]
        test = frames["test"]
        labeled = train["is_labeled_for_scanvi"]
        if labeled.isna().any():
            raise ValueError("train labeled mask contains missing values")
        is_labeled = labeled.astype(bool).to_numpy()
        train_true = train["true_label"].astype(str)
        n_train_rare = int(train_true.eq(rare).sum())
        observed_labeled_rare = int((train_true.eq(rare).to_numpy() & is_labeled).sum())
        expected_labeled_rare = _expected_labeled_rare(n_train_rare, rare_train_size)
        if observed_labeled_rare != expected_labeled_rare:
            raise ValueError(
                "labeled rare count mismatch: "
                f"observed={observed_labeled_rare}, expected={expected_labeled_rare}"
            )
        proto = PrototypeRescuer(rare)
        proto.fit(_latent(train), train_true, is_labeled)
        baseline = test["predicted_label"].astype(str).reset_index(drop=True)
        final, rescue_summary = conformal_rescue(
            proto,
            baseline,
            validation["predicted_label"].astype(str),
            validation["true_label"].astype(str),
            _latent(validation),
            _latent(test),
        )
        groups, false_rescue = assign_primary_groups(
            test["true_label"], baseline, final, rare
        )
        metrics, closest_competitor = prototype_metrics(proto, _latent(test))
        probability_column = f"prob_{rare}"
        sc_prob = (
            pd.to_numeric(test[probability_column], errors="coerce").to_numpy(
                dtype=float
            )
            if probability_column in test.columns
            else np.full(len(test), np.nan)
        )
        if np.isfinite(sc_prob).any() and (
            np.nanmin(sc_prob) < -1e-8 or np.nanmax(sc_prob) > 1.0 + 1e-8
        ):
            raise ValueError("cached scANVI rare probability falls outside [0, 1]")
        y = test["true_label"].astype(str).reset_index(drop=True)
        cell = pd.DataFrame(
            {
                "dataset": dataset,
                "seed": seed,
                "rare_train_size": rare_train_size,
                "rare_class": rare,
                "cell_id": test["cell_id"].astype(str),
                "true_label": y,
                "baseline_label": baseline,
                "current_final_label": final.astype(str),
                "primary_group": groups,
                "false_rescue": false_rescue,
                "closest_competitor_class": closest_competitor,
                "is_closest_competitor": y.eq(closest_competitor).to_numpy(),
                "scANVI_rare_probability": sc_prob,
            }
        )
        cell = pd.concat([cell, metrics], axis=1)
        if not cell["primary_group"].isin(PRIMARY_GROUPS).all():
            raise AssertionError("unknown primary group")
        historical_row = historical[(dataset, seed, rare_train_size)]
        historical_identity_available = (
            historical_row["reconstruction_basis"] == "current_formal_replay"
        )
        current_true_rescues = int((groups == "true_rescued_rare").sum())
        current_false_rescues = int(false_rescue.sum())
        historical_count_match = bool(
            current_true_rescues == int(historical_row["true_rescues"])
            and current_false_rescues == int(historical_row["false_rescues"])
        )
        row.update(
            status="success",
            current_replay_available=True,
            historical_cell_identity_available=historical_identity_available,
            historical_reconstruction_basis=historical_row["reconstruction_basis"],
            historical_count_match=historical_count_match,
            separability=proto.separability_ratio,
            n_train=len(train),
            n_validation=len(validation),
            n_test=len(test),
            n_train_rare=n_train_rare,
            n_labeled_rare=observed_labeled_rare,
            expected_labeled_rare=expected_labeled_rare,
            closest_competitor_class=closest_competitor,
            scANVI_probability_available=bool(np.isfinite(sc_prob).any()),
            val_missed=rescue_summary.get("val_missed", np.nan),
            abstain=rescue_summary["abstain"],
            abstain_reason=rescue_summary["reason"] or "rescue_applied",
            chosen_rank=rescue_summary["chosen_rank"],
            tau=rescue_summary["tau"],
            current_true_rescues=current_true_rescues,
            current_false_rescues=current_false_rescues,
            **{f"n_{group}": int((groups == group).sum()) for group in PRIMARY_GROUPS},
            n_closest_competitor=int(cell["is_closest_competitor"].sum()),
        )
        for path in required:
            provenance["files"][str(path.relative_to(ROOT))] = _sha256(path)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            provenance["files"][str(manifest_path.relative_to(ROOT))] = _sha256(
                manifest_path
            )
        return row, cell, provenance
    except Exception as exc:
        row.update(status="invalid_cache", error=f"{type(exc).__name__}: {exc}")
        return row, None, provenance


def group_summaries(cells: pd.DataFrame) -> pd.DataFrame:
    long = cells.melt(
        id_vars=["dataset", "seed", "rare_train_size", "primary_group"],
        value_vars=[*METRICS, "scANVI_rare_probability"],
        var_name="metric",
        value_name="value",
    )
    grouped = long.groupby(
        ["dataset", "seed", "rare_train_size", "primary_group", "metric"],
        observed=True,
        sort=False,
    )["value"]
    return grouped.agg(
        n="count",
        median="median",
        q25=lambda values: values.quantile(0.25),
        q75=lambda values: values.quantile(0.75),
    ).reset_index()


def contrast_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    run_keys = ["dataset", "seed", "rare_train_size"]
    metrics = [*METRICS, "scANVI_rare_probability"]
    for key, frame in cells.groupby(run_keys, sort=False):
        competitor = frame[frame["is_closest_competitor"]]
        for contrast, (left_group, right_group) in CONTRASTS.items():
            left = frame[frame["primary_group"] == left_group]
            right = (
                competitor
                if right_group == "closest_competitor"
                else frame[frame["primary_group"] == right_group]
            )
            for metric in metrics:
                x = left[metric].to_numpy(dtype=float)
                y = right[metric].to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                y = y[np.isfinite(y)]
                raw_delta = cliffs_delta(x, y)
                raw_median_difference = (
                    float(np.median(x) - np.median(y)) if len(x) and len(y) else np.nan
                )
                direction = METRICS.get(metric, np.nan)
                rows.append(
                    {
                        "dataset": key[0],
                        "seed": key[1],
                        "rare_train_size": key[2],
                        "contrast": contrast,
                        "left_group": left_group,
                        "right_group": right_group,
                        "metric": metric,
                        "n_left": len(x),
                        "n_right": len(y),
                        "median_left": float(np.median(x)) if len(x) else np.nan,
                        "median_right": float(np.median(y)) if len(y) else np.nan,
                        "raw_median_difference": raw_median_difference,
                        "raw_cliffs_delta": raw_delta,
                        "hypothesis_direction": direction,
                        "oriented_median_difference": (
                            raw_median_difference * direction
                            if np.isfinite(direction)
                            else np.nan
                        ),
                        "oriented_cliffs_delta": (
                            raw_delta * direction if np.isfinite(direction) else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_contrasts(contrasts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    defined = contrasts[contrasts["oriented_cliffs_delta"].notna()].copy()
    seed_summary = (
        defined.groupby(
            ["dataset", "rare_train_size", "contrast", "metric"], sort=False
        )
        .agg(
            n_seeds=("seed", "nunique"),
            median_oriented_delta=("oriented_cliffs_delta", "median"),
            min_oriented_delta=("oriented_cliffs_delta", "min"),
            max_oriented_delta=("oriented_cliffs_delta", "max"),
            seed_direction_rate=(
                "oriented_cliffs_delta",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index()
    )
    dataset_summary = (
        seed_summary.groupby(["rare_train_size", "contrast", "metric"], sort=False)
        .agg(
            n_datasets=("dataset", "nunique"),
            median_dataset_effect=("median_oriented_delta", "median"),
            q25_dataset_effect=(
                "median_oriented_delta",
                lambda values: values.quantile(0.25),
            ),
            q75_dataset_effect=(
                "median_oriented_delta",
                lambda values: values.quantile(0.75),
            ),
            dataset_direction_rate=(
                "median_oriented_delta",
                lambda values: float((values > 0).mean()),
            ),
        )
        .reset_index()
    )
    return seed_summary, dataset_summary


def _centered_sample(
    cells: pd.DataFrame, metric: str, max_per_group: int = 750
) -> pd.DataFrame:
    data = cells[["dataset", "seed", "rare_train_size", "primary_group", metric]].copy()
    data = data.rename(columns={metric: "value"}).dropna(subset=["value"])
    run_median = data.groupby(["dataset", "seed", "rare_train_size"])[
        "value"
    ].transform("median")
    data["centered_value"] = data["value"] - run_median
    sampled = []
    for _, frame in data.groupby(
        ["dataset", "seed", "rare_train_size", "primary_group"], sort=False
    ):
        sampled.append(
            frame.sample(min(len(frame), max_per_group), random_state=42)
            if len(frame) > max_per_group
            else frame
        )
    return pd.concat(sampled, ignore_index=True)


def make_figures(
    cells: pd.DataFrame, contrasts: pd.DataFrame, output_dir: Path
) -> list[Path]:
    colors = {
        "baseline_correct_rare": "#0072B2",
        "true_rescued_rare": "#009E73",
        "unrescued_rare": "#D55E00",
        "non_target": "#999999",
    }
    labels = {
        "baseline_correct_rare": "Baseline-correct rare",
        "true_rescued_rare": "True rescued rare",
        "unrescued_rare": "Unrescued rare",
        "non_target": "Non-target",
    }
    plot_metrics = [
        ("rare_membership_score", "Run-centered rare score"),
        ("rare_rank", "Run-centered rare rank"),
        ("standardized_prototype_margin", "Run-centered standardized margin"),
        ("scANVI_rare_probability", "Run-centered scANVI rare probability"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    rng = np.random.default_rng(42)
    for ax, (metric, ylabel) in zip(axes.flat, plot_metrics):
        data = _centered_sample(cells, metric)
        values = [
            data.loc[data["primary_group"] == group, "centered_value"].to_numpy()
            for group in PRIMARY_GROUPS
        ]
        violin = ax.violinplot(values, showmedians=True, showextrema=False)
        for body, group in zip(violin["bodies"], PRIMARY_GROUPS):
            body.set_facecolor(colors[group])
            body.set_edgecolor("black")
            body.set_alpha(0.65)
        violin["cmedians"].set_color("black")
        for position, (group, group_values) in enumerate(
            zip(PRIMARY_GROUPS, values), 1
        ):
            if len(group_values):
                chosen = rng.choice(
                    group_values, size=min(len(group_values), 250), replace=False
                )
                jitter = rng.normal(position, 0.045, len(chosen))
                ax.scatter(jitter, chosen, s=4, color="black", alpha=0.12, linewidths=0)
        ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(1, len(PRIMARY_GROUPS) + 1))
        ax.set_xticklabels(
            [labels[group] for group in PRIMARY_GROUPS], rotation=20, ha="right"
        )
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Formal rescue selection-pathway characteristics")
    fig.text(
        0.01,
        0.01,
        "Prototype metrics partly reflect the selection rule by construction. scANVI probability is a frozen non-selection model readout, not biological validation. Points are deterministic display samples; effects are computed from all cells.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    distribution_png = output_dir / "figures" / "selection_pathway_distributions.png"
    distribution_pdf = output_dir / "figures" / "selection_pathway_distributions.pdf"
    fig.savefig(distribution_png, dpi=300, bbox_inches="tight")
    fig.savefig(distribution_pdf, bbox_inches="tight")
    plt.close(fig)

    margin = contrasts[
        (contrasts["metric"] == "standardized_prototype_margin")
        & contrasts["oriented_cliffs_delta"].notna()
    ].copy()
    margin = (
        margin.groupby(["dataset", "rare_train_size", "contrast"], sort=False)[
            "oriented_cliffs_delta"
        ]
        .median()
        .reset_index()
    )
    contrast_order = list(CONTRASTS)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    budget_colors = {
        "0.01": "#0072B2",
        "0.05": "#009E73",
        "0.10": "#D55E00",
        "all": "#CC79A7",
    }
    for ax, contrast in zip(axes.flat, contrast_order):
        subset = margin[margin["contrast"] == contrast]
        dataset_order = list(dict.fromkeys(subset["dataset"]))
        for index, dataset in enumerate(dataset_order):
            dataset_rows = subset[subset["dataset"] == dataset]
            values = dataset_rows["oriented_cliffs_delta"].to_numpy()
            jitter = rng.normal(index, 0.055, len(values))
            for x_value, y_value, budget in zip(
                values, jitter, dataset_rows["rare_train_size"]
            ):
                ax.scatter(
                    x_value,
                    y_value,
                    s=24,
                    alpha=0.75,
                    color=budget_colors[str(budget)],
                    edgecolor="none",
                )
            if len(values):
                ax.scatter(
                    np.median(values),
                    index,
                    marker="D",
                    s=34,
                    color="#D55E00",
                    edgecolor="black",
                    linewidth=0.4,
                )
        ax.axvline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_yticks(range(len(dataset_order)))
        ax.set_yticklabels(dataset_order)
        ax.set_title(contrast.replace("_", " "), loc="left", fontsize=9)
        ax.set_xlabel("Oriented Cliff's delta (positive supports ordering)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Dataset-budget standardized prototype-margin contrasts")
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color=color,
            label=budget,
            markersize=5,
        )
        for budget, color in budget_colors.items()
    ]
    fig.legend(
        handles=handles,
        title="Rare-label budget",
        frameon=False,
        ncol=4,
        loc="lower center",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    margin_png = output_dir / "figures" / "prototype_margin_contrasts.png"
    margin_pdf = output_dir / "figures" / "prototype_margin_contrasts.pdf"
    fig.savefig(margin_png, dpi=300, bbox_inches="tight")
    fig.savefig(margin_pdf, bbox_inches="tight")
    plt.close(fig)
    return [distribution_png, distribution_pdf, margin_png, margin_pdf]


def _notes(
    run_level: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    cells: pd.DataFrame,
) -> str:
    status = run_level["status"].value_counts().to_dict()
    historical_unavailable = int(
        (~run_level["historical_cell_identity_available"].astype(bool)).sum()
    )
    rare_probability_available = int(run_level["scANVI_probability_available"].sum())
    group_totals = {
        group: int((cells["primary_group"] == group).sum()) for group in PRIMARY_GROUPS
    }
    false_rescues = int(cells["false_rescue"].sum())
    primary = dataset_summary[
        dataset_summary["metric"].isin(
            [
                "rare_membership_score",
                "rare_rank",
                "rare_standardized_distance",
                "standardized_prototype_margin",
            ]
        )
    ]
    lines = [
        "# Residual-signal selection-pathway analysis notes",
        "",
        "## Scope and interpretation",
        "",
        "- This is a cache-only audit of the formal rescue selection pathway, not independent biological validation.",
        "- Rare score, rank, distance, and prototype margin share the geometry used by rescue and therefore partly reflect selection by construction.",
        "- Cached scANVI rare probability is a frozen non-selection model readout; disagreement is scientifically expected in the target failure mode.",
        "- Test labels are used only for final group characterization and historical agreement diagnostics, never for current rescue selection or run eligibility.",
        "",
        "## Completeness",
        "",
        f"- Expected runs: {len(CONFIGS) * len(SEEDS) * len(RARE_TRAIN_SIZES)}; status counts: `{json.dumps(status, sort_keys=True)}`.",
        f"- Cell-level current-code replay rows: {len(cells):,} cells across {int(run_level['current_replay_available'].sum())} runs.",
        f"- Historical cell identity unavailable: {historical_unavailable} runs; current replay is reported separately and is not presented as historical reconstruction.",
        f"- Cached scANVI rare probability available in {rare_probability_available} runs.",
        f"- Current-code replay group totals: baseline-correct rare={group_totals['baseline_correct_rare']}, true rescued rare={group_totals['true_rescued_rare']}, unrescued rare={group_totals['unrescued_rare']}, non-target={group_totals['non_target']}, false rescues={false_rescues}.",
        "",
        "## Prespecified ordering summary",
        "",
    ]
    for contrast in CONTRASTS:
        subset = primary[primary["contrast"] == contrast]
        if subset.empty:
            lines.append(f"- {contrast}: no informative dataset-level effects.")
            continue
        rates = subset.groupby("metric")["dataset_direction_rate"].median()
        formatted = ", ".join(
            f"{metric}={value:.3f}" for metric, value in rates.items()
        )
        lines.append(
            f"- {contrast}: median dataset direction rates by budget: {formatted}."
        )
    probability = contrast_table(cells)
    probability = probability[
        probability["metric"].eq("scANVI_rare_probability")
        & probability["raw_cliffs_delta"].notna()
    ]
    lines.extend(["", "## Frozen scANVI probability readout", ""])
    for contrast in CONTRASTS:
        values = probability.loc[
            probability["contrast"] == contrast, "raw_cliffs_delta"
        ]
        if values.empty:
            lines.append(f"- {contrast}: unavailable.")
        else:
            lines.append(
                f"- {contrast}: n_runs={len(values)}, median raw Cliff's delta={values.median():.4f}, positive-direction rate={(values > 0).mean():.3f}."
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Direction rates are descriptive; datasets, not pooled cells, are the main evidence units.",
            "- Empty groups yield missing effects and are not counted as failures or successes.",
            "- Current-code replay can differ from historical count-only outputs; both provenance fields are retained.",
            "- Biological marker validation remains a separate P2 analysis.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    for path in (output_dir, output_dir / "tables", output_dir / "figures", LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)
    historical = _historical_identity_lookup()
    rows = []
    cell_frames = []
    provenance = []
    log_lines = []
    expected_keys = []
    for config_path in CONFIGS:
        dataset = load_config(ROOT / config_path)["dataset"]["name"]
        for seed in SEEDS:
            for rare_train_size in RARE_TRAIN_SIZES:
                expected_keys.append((dataset, seed, rare_train_size))
                row, cell, run_provenance = _analyze_run(
                    config_path, seed, rare_train_size, historical
                )
                rows.append(row)
                provenance.append(run_provenance)
                if cell is not None:
                    cell_frames.append(cell)
                log_lines.append(
                    f"{dataset}\t{seed}\t{rare_train_size}\t{row['status']}\t{row.get('error', '')}"
                )
    run_level = pd.DataFrame(rows)
    keys = ["dataset", "seed", "rare_train_size"]
    if run_level.duplicated(keys).any():
        raise AssertionError("run ledger has duplicate keys")
    if set(map(tuple, run_level[keys].itertuples(index=False, name=None))) != set(
        expected_keys
    ):
        raise AssertionError("run ledger does not match the expected 96 keys")
    run_path = output_dir / "run_level.csv"
    run_level.to_csv(run_path, index=False)
    failures = run_level[run_level["status"] != "success"]
    if not failures.empty:
        raise SystemExit(f"Analysis incomplete: {len(failures)} configurations failed")
    cells = pd.concat(cell_frames, ignore_index=True)
    cell_path = output_dir / "cell_level.parquet"
    cells.to_parquet(cell_path, index=False, compression="zstd")
    group_summary = group_summaries(cells)
    group_summary_path = output_dir / "tables" / "group_summaries.csv"
    group_summary.to_csv(group_summary_path, index=False)
    contrasts = contrast_table(cells)
    contrast_path = output_dir / "tables" / "group_contrasts.csv"
    contrasts.to_csv(contrast_path, index=False)
    seed_summary, dataset_summary = aggregate_contrasts(contrasts)
    seed_summary_path = output_dir / "tables" / "seed_aggregated_contrasts.csv"
    seed_summary.to_csv(seed_summary_path, index=False)
    summary_path = output_dir / "summary.csv"
    dataset_summary.to_csv(summary_path, index=False)
    figure_paths = make_figures(cells, contrasts, output_dir)
    notes_path = output_dir / "analysis_notes.md"
    notes_path.write_text(_notes(run_level, dataset_summary, cells), encoding="utf-8")
    log_path = LOG_DIR / "residual_signal_v1.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    source_paths = [
        ROOT / "tools" / "analysis" / "residual_signal.py",
        ROOT / "tools" / "analysis" / "rescue_composition.py",
        ROOT / "src" / "rescue.py",
        ROOT / "src" / "utils.py",
        ROOT / "tests" / "test_residual_signal.py",
        *(ROOT / config for config in CONFIGS),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
    }
    outputs = [
        run_path,
        cell_path,
        group_summary_path,
        contrast_path,
        seed_summary_path,
        summary_path,
        notes_path,
        *figure_paths,
    ]
    script_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/residual_signal.py",
        "working_directory": str(ROOT),
        "sources": source_hashes,
        "inputs": {str(COMPOSITION_PATH.relative_to(ROOT)): _sha256(COMPOSITION_PATH)},
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    script_manifest = output_dir / "_script_manifest.jsonl"
    script_manifest.write_text(
        json.dumps(script_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    outputs.append(script_manifest)
    manifest = {
        "analysis": "residual_signal_selection_pathway",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/residual_signal.py",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in (
                    "numpy",
                    "pandas",
                    "matplotlib",
                    "scipy",
                    "scikit-learn",
                    "pyarrow",
                )
            },
        },
        "git": {"sha": git_sha(), "dirty": _git_dirty()},
        "source_hashes": source_hashes,
        "parameters": {
            "configs": list(CONFIGS),
            "seeds": list(SEEDS),
            "rare_train_sizes": list(RARE_TRAIN_SIZES),
            "alpha": DEFAULT_CONFORMAL_ALPHA,
            "low_separability": CONFORMAL_LOW_SEP,
            "rank_grid": list(CONFORMAL_RANK_GRID),
            "min_val_missed": MIN_VAL_MISSED,
            "primary_groups": list(PRIMARY_GROUPS),
            "metrics_and_directions": METRICS,
            "contrasts": {key: list(value) for key, value in CONTRASTS.items()},
        },
        "interpretation": "selection-pathway characterization; not independent biological validation",
        "test_label_usage": "final grouping and historical agreement diagnostics only; never selection or current-replay eligibility",
        "overwrite_status": "versioned analysis outputs only; historical outputs untouched",
        "expected_configurations": len(expected_keys),
        "status_counts": run_level["status"].value_counts().to_dict(),
        "inputs": {
            "rescue_composition": {
                "path": str(COMPOSITION_PATH.relative_to(ROOT)),
                "sha256": _sha256(COMPOSITION_PATH),
            },
            "runs": provenance,
        },
        "outputs": {str(path.relative_to(ROOT)): _sha256(path) for path in outputs},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(run_level)} run rows and {len(cells):,} cell rows")
    print(f"Status counts: {run_level['status'].value_counts().to_dict()}")
    print(
        "Historical identity unavailable: "
        f"{int((~run_level['historical_cell_identity_available'].astype(bool)).sum())}"
    )


if __name__ == "__main__":
    main()
