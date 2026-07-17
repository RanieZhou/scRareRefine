"""Eight-dataset component and rank ablation for the frozen P1 protocol."""

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
from src.utils import (
    classification_tables,
    load_config,
    make_run_dir,
    parse_rare_train_size,
)  # noqa: E402
from tools.analysis.ablation import _conformal_with_overrides  # noqa: E402
from tools.analysis.label_budget import normalize_bool, sha256_file  # noqa: E402
from tools.analysis.rescue_composition import _latent, _load_aligned_split  # noqa: E402


CONFIGS = (
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/mouse_lung_tms_10x.yaml",
    "configs/mouse_pancreas_tms_10x.yaml",
)
SEEDS = (42, 43, 44)
RARE_TRAIN_SIZES = ("0.01", "0.05", "0.10", "all")
ALPHA = DEFAULT_CONFORMAL_ALPHA
OUT = ROOT / "results" / "supplementary_ablation" / "v1"
LOG_DIR = ROOT / "logs" / "supplementary_ablation"
LABEL_BUDGET_PATH = ROOT / "results" / "label_budget" / "v1" / "run_level.csv"

FULL = dict(
    low_sep=CONFORMAL_LOW_SEP,
    enforce_necessity=True,
    rank_grid=CONFORMAL_RANK_GRID,
    use_conformal_tau=True,
)
VARIANTS = (
    ("baseline", "component", None),
    ("minus_separability_gate", "component", {**FULL, "low_sep": 0.0}),
    ("minus_necessity_gate", "component", {**FULL, "enforce_necessity": False}),
    ("fixed_rank_1", "component", {**FULL, "rank_grid": (1,)}),
    ("minus_conformal_tau", "component", {**FULL, "use_conformal_tau": False}),
    ("full_method", "component", "full"),
    ("rank_1", "rank", {**FULL, "rank_grid": (1,)}),
    ("rank_2", "rank", {**FULL, "rank_grid": (2,)}),
    ("rank_3", "rank", {**FULL, "rank_grid": (3,)}),
    ("adaptive_rank", "rank", "full"),
)
COMPONENT_ORDER = [variant for variant, group, _ in VARIANTS if group == "component"]
RANK_ORDER = [variant for variant, group, _ in VARIANTS if group == "rank"]
METRICS = (
    "rare_f1",
    "rare_recall",
    "rescue_precision",
    "incremental_fpr",
    "abstain",
    "alpha_violation",
)


def _git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return None


def _package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def evaluate_predictions(
    y_true: pd.Series,
    baseline: pd.Series,
    final: pd.Series,
    rare_class: str,
) -> dict:
    y = pd.Series(y_true).astype(str).reset_index(drop=True)
    base = pd.Series(baseline).astype(str).reset_index(drop=True)
    refined = pd.Series(final).astype(str).reset_index(drop=True)
    changed = refined.ne(base)
    invalid = changed & ~(base.ne(rare_class) & refined.eq(rare_class))
    if invalid.any():
        raise AssertionError("ablation introduced a non-rescue label transition")
    true_rescues = int((changed & y.eq(rare_class)).sum())
    false_rescues = int((changed & y.ne(rare_class)).sum())
    all_rescues = int(changed.sum())
    n_nonrare = int(y.ne(rare_class).sum())
    metrics, _ = classification_tables(y, refined, rare_class=rare_class)
    incremental_fpr = false_rescues / n_nonrare if n_nonrare else np.nan
    return {
        "rare_f1": float(metrics["rare_f1"]),
        "rare_recall": float(metrics["rare_recall"]),
        "rare_precision": float(metrics["rare_precision"]),
        "true_rescues": true_rescues,
        "false_rescues": false_rescues,
        "all_rescues": all_rescues,
        "rescue_precision": true_rescues / all_rescues if all_rescues else np.nan,
        "incremental_fpr": incremental_fpr,
        "rescue_ffr": incremental_fpr,
        "alpha_violation": bool(
            np.isfinite(incremental_fpr) and incremental_fpr > ALPHA
        ),
    }


def run_variant(
    spec: object,
    proto: PrototypeRescuer,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    rare = proto.rare_class
    baseline = test["predicted_label"].astype(str).reset_index(drop=True)
    val_base = validation["predicted_label"].astype(str).reset_index(drop=True)
    val_true = validation["true_label"].astype(str).reset_index(drop=True)
    if spec is None:
        final = baseline.copy()
        decision = {
            "abstain": False,
            "reason": "baseline_no_rescue",
            "chosen_rank": 0,
            "tau": np.nan,
            "val_missed": int((val_true.eq(rare) & val_base.ne(rare)).sum()),
            "n_candidate": 0,
            "n_rescued": 0,
        }
    elif spec == "full":
        final, decision = conformal_rescue(
            proto,
            baseline,
            val_base,
            val_true,
            _latent(validation),
            _latent(test),
            alpha=ALPHA,
        )
    else:
        final, decision = _conformal_with_overrides(
            proto,
            baseline,
            val_base,
            val_true,
            _latent(validation),
            _latent(test),
            **spec,
        )
    metrics = evaluate_predictions(test["true_label"], baseline, final, rare)
    return {
        **metrics,
        "abstain": bool(decision.get("abstain", False)),
        "abstain_reason": str(decision.get("reason", "") or "rescue_applied"),
        "chosen_rank": int(decision.get("chosen_rank", 0)),
        "tau": float(decision.get("tau", np.nan)),
        "val_missed": int(decision.get("val_missed", 0)),
        "n_candidate": int(decision.get("n_candidate", 0)),
    }


def analyze_run(
    config_path: str,
    seed: int,
    rare_train_size: str,
    budget_lookup: dict[tuple[str, int, str], dict],
) -> tuple[list[dict], dict, str]:
    config = load_config(ROOT / config_path)
    experiment = config["experiment"]
    dataset = config["dataset"]["name"]
    rare = str(experiment["rare_class"])
    split_mode = experiment.get("split_mode", "batch_heldout")
    run_dir = ROOT / make_run_dir(
        config, split_mode, seed, rare, parse_rare_train_size(rare_train_size)
    )
    base = {
        "dataset": dataset,
        "seed": seed,
        "rare_train_size": rare_train_size,
        "rare_class": rare,
        "split_mode": split_mode,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    provenance: dict[str, object] = {"run_dir": base["run_dir"], "files": {}}
    try:
        embeddings = run_dir / "embeddings"
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
        latent_dims = {_latent(frame).shape[1] for frame in frames.values()}
        if len(latent_dims) != 1:
            raise ValueError("latent dimensions differ across splits")

        budget = budget_lookup.get((dataset, seed, rare_train_size))
        if budget is None or budget["status"] != "success":
            raise ValueError("P0 label-budget provenance missing or unsuccessful")
        if str(budget["run_dir"]).replace("/", "\\") != base["run_dir"].replace(
            "/", "\\"
        ):
            raise ValueError("P0 run directory differs from P1 run directory")

        train = frames["train"]
        is_labeled = normalize_bool(train["is_labeled_for_scanvi"]).to_numpy()
        proto = PrototypeRescuer(rare)
        proto.fit(_latent(train), train["true_label"].astype(str), is_labeled)
        rows = []
        for variant, group, spec in VARIANTS:
            result = run_variant(spec, proto, frames["validation"], frames["test"])
            rows.append(
                {
                    **base,
                    "variant": variant,
                    "variant_group": group,
                    "status": "success",
                    "error_reason": "",
                    "separability": float(proto.separability_ratio),
                    "alpha": ALPHA,
                    "split_hash": budget["split_hash"],
                    "actual_training_labeled_rare_count": int(
                        budget["actual_training_labeled_rare_count"]
                    ),
                    "labeled_rare_id_sha256": budget["labeled_rare_id_sha256"],
                    "effective_budget_key": budget["effective_budget_key"],
                    **result,
                }
            )
        component_fixed = next(row for row in rows if row["variant"] == "fixed_rank_1")
        rank_fixed = next(row for row in rows if row["variant"] == "rank_1")
        component_full = next(row for row in rows if row["variant"] == "full_method")
        rank_full = next(row for row in rows if row["variant"] == "adaptive_rank")
        compare_columns = [
            "rare_f1",
            "rare_recall",
            "rare_precision",
            "true_rescues",
            "false_rescues",
            "all_rescues",
            "incremental_fpr",
            "abstain",
            "chosen_rank",
        ]
        for left, right, label in (
            (component_fixed, rank_fixed, "fixed rank 1 duplicate"),
            (component_full, rank_full, "full/adaptive duplicate"),
        ):
            for column in compare_columns:
                lv, rv = left[column], right[column]
                if pd.isna(lv) and pd.isna(rv):
                    continue
                if isinstance(lv, (float, np.floating)):
                    equal = bool(np.isclose(lv, rv, equal_nan=True))
                else:
                    equal = lv == rv
                if not equal:
                    raise AssertionError(f"{label} differs in {column}")
        for path in (
            *(
                embeddings / f"{split}_{kind}.csv"
                for split in frames
                for kind in ("predictions", "latent")
            ),
            run_dir / "manifest.json",
        ):
            provenance["files"][str(path.relative_to(ROOT))] = sha256_file(path)
        return rows, provenance, ""
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        failed = [
            {
                **base,
                "variant": variant,
                "variant_group": group,
                "status": "failed",
                "error_reason": error,
            }
            for variant, group, _ in VARIANTS
        ]
        return failed, provenance, error


def collapse_identity(run_level: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "rare_class",
        "seed",
        "variant",
        "variant_group",
        "split_hash",
        "actual_training_labeled_rare_count",
        "labeled_rare_id_sha256",
    ]
    metric_columns = [
        "separability",
        "rare_f1",
        "rare_recall",
        "rare_precision",
        "true_rescues",
        "false_rescues",
        "all_rescues",
        "rescue_precision",
        "incremental_fpr",
        "alpha_violation",
        "abstain",
        "chosen_rank",
        "tau",
        "val_missed",
        "n_candidate",
    ]
    rows = []
    for key, group in run_level.groupby(keys, sort=False, dropna=False):
        row = dict(zip(keys, key))
        row["n_folded_nominal_budgets"] = len(group)
        row["folded_nominal_budgets"] = json.dumps(
            group["rare_train_size"].tolist(), separators=(",", ":")
        )
        row["abstain_reason"] = " | ".join(sorted(group["abstain_reason"].unique()))
        for column in metric_columns:
            values = group[column].to_numpy()
            if column in ("alpha_violation", "abstain"):
                if len(set(bool(value) for value in values)) != 1:
                    raise AssertionError(f"identity-equivalent runs differ in {column}")
                row[column] = bool(values[0])
            else:
                numeric = values.astype(float)
                if not np.allclose(numeric, numeric[0], equal_nan=True):
                    raise AssertionError(f"identity-equivalent runs differ in {column}")
                row[column] = values[0]
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(identity_level: pd.DataFrame) -> pd.DataFrame:
    unit_keys = [
        "dataset",
        "rare_class",
        "actual_training_labeled_rare_count",
        "variant",
        "variant_group",
        "seed",
    ]
    seed_rows = []
    numeric = ["rare_f1", "rare_recall", "rescue_precision", "incremental_fpr"]
    for key, group in identity_level.groupby(unit_keys, sort=False, dropna=False):
        row = dict(zip(unit_keys, key))
        row["n_identity_runs"] = len(group)
        for column in numeric:
            row[column] = group[column].astype(float).mean()
        row["abstain"] = group["abstain"].astype(float).mean()
        row["alpha_violation"] = group["alpha_violation"].astype(float).mean()
        seed_rows.append(row)
    seed_units = pd.DataFrame(seed_rows)
    if seed_units.duplicated(unit_keys).any():
        raise AssertionError("duplicate seed-count-variant aggregation unit")

    summary_keys = [
        "dataset",
        "rare_class",
        "actual_training_labeled_rare_count",
        "variant",
        "variant_group",
    ]
    summary_rows = []
    for key, group in seed_units.groupby(summary_keys, sort=False, dropna=False):
        row = dict(zip(summary_keys, key))
        row["n_seeds"] = group["seed"].nunique()
        for column in numeric:
            values = group[column].astype(float)
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_min"] = values.min()
            row[f"{column}_max"] = values.max()
            row[f"{column}_n_defined"] = int(values.notna().sum())
        row["abstention_rate"] = group["abstain"].mean()
        row["alpha_violation_rate"] = group["alpha_violation"].mean()
        row["n_alpha_violations"] = int(group["alpha_violation"].sum())
        summary_rows.append(row)
    return seed_units, pd.DataFrame(summary_rows)


def dataset_equal_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group), frame in summary.groupby(
        ["variant", "variant_group"], sort=False
    ):
        dataset_means = frame.groupby("dataset", sort=False)[
            [
                "rare_f1_mean",
                "rare_recall_mean",
                "rescue_precision_mean",
                "incremental_fpr_mean",
                "abstention_rate",
                "alpha_violation_rate",
            ]
        ].mean()
        row = {
            "variant": variant,
            "variant_group": group,
            "n_datasets": len(dataset_means),
            "n_effective_budget_units": len(frame),
            "n_alpha_violations": int(frame["n_alpha_violations"].sum()),
        }
        for column in dataset_means.columns:
            row[f"dataset_equal_{column}"] = dataset_means[column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(dataset_equal: pd.DataFrame, output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    panels = (
        ("component", COMPONENT_ORDER, "Component leave-one-out"),
        ("rank", RANK_ORDER, "Fixed versus adaptive rank"),
    )
    colors = {"rare_f1": "#1F6F8B", "rare_recall": "#C45A3C"}
    for ax, (group, order, title) in zip(axes, panels):
        frame = (
            dataset_equal[dataset_equal["variant_group"].eq(group)]
            .set_index("variant")
            .reindex(order)
        )
        x = np.arange(len(frame))
        width = 0.36
        ax.bar(
            x - width / 2,
            frame["dataset_equal_rare_f1_mean"],
            width,
            color=colors["rare_f1"],
            label="Rare F1",
        )
        ax.bar(
            x + width / 2,
            frame["dataset_equal_rare_recall_mean"],
            width,
            color=colors["rare_recall"],
            label="Rare recall",
        )
        for position, violations in zip(x, frame["n_alpha_violations"]):
            if violations:
                ax.text(
                    position,
                    1.01,
                    f"{int(violations)} alpha violations",
                    rotation=90,
                    va="bottom",
                    ha="center",
                    fontsize=7,
                    color="#8A2F20",
                )
        ax.set_xticks(
            x,
            [value.replace("_", " ") for value in frame.index],
            rotation=32,
            ha="right",
        )
        ax.set_ylim(0, 1.18)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color="#D7D2C8", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Dataset-equal mean across effective budgets")
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Eight-dataset scRareRefine ablation", x=0.02, ha="left", fontsize=15)
    fig.text(
        0.02,
        0.005,
        "Each dataset contributes equal weight after within-seed identity collapse. Alpha violations count empirical test incremental FPR > 0.01 and are safety outcomes, not parameter-selection criteria.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    png = output_dir / "figures" / "supplementary_ablation.png"
    pdf = output_dir / "figures" / "supplementary_ablation.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def notes(
    run_level: pd.DataFrame, identity: pd.DataFrame, summary: pd.DataFrame
) -> str:
    violations = (
        run_level.groupby("variant", sort=False)["alpha_violation"].sum().astype(int)
    )
    abstentions = run_level.groupby("variant", sort=False)["abstain"].sum().astype(int)
    return (
        "\n".join(
            [
                "# Supplementary ablation notes",
                "",
                "## Scope",
                "",
                "- Current-code cache replay over the frozen eight datasets, four nominal budgets, and three seeds.",
                "- Prototypes use labeled training cells; all gates, rank choices, and tau use validation only; test labels are used only for final empirical metrics.",
                "- This P1 replay does not reconstruct unavailable historical mouse-pancreas cell identities; it evaluates every variant consistently under the current frozen implementation.",
                "",
                "## Completeness",
                "",
                f"- Expected run-variant rows: {len(CONFIGS) * len(SEEDS) * len(RARE_TRAIN_SIZES) * len(VARIANTS)}; observed: {len(run_level)}.",
                f"- Identity-collapsed run-variant rows: {len(identity)}.",
                f"- Dataset-effective-budget-variant summary rows: {len(summary)}.",
                f"- Empirical alpha violations by variant: `{json.dumps(violations.to_dict())}`.",
                f"- Abstentions by variant: `{json.dumps(abstentions.to_dict())}`.",
                "",
                "## Interpretation limits",
                "",
                "- Rescue precision is NA when no cells are rescued and is never replaced with zero.",
                "- Empirical test incremental-FPR exceedance under split shift is reported completely but does not alter the frozen defaults.",
                "- Dataset-equal summaries prevent large datasets or duplicated floor budgets from dominating the headline aggregation.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    for path in (output_dir, output_dir / "tables", output_dir / "figures", LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

    budget_frame = pd.read_csv(LABEL_BUDGET_PATH, dtype={"rare_train_size": str})
    budget_lookup = {
        (str(row.dataset), int(row.seed), str(row.rare_train_size)): row._asdict()
        for row in budget_frame.itertuples(index=False)
    }
    all_rows = []
    provenance = []
    log_lines = []
    expected_run_keys = []
    for config_path in CONFIGS:
        dataset = load_config(ROOT / config_path)["dataset"]["name"]
        for seed in SEEDS:
            for rare_train_size in RARE_TRAIN_SIZES:
                expected_run_keys.append((dataset, seed, rare_train_size))
                rows, run_provenance, error = analyze_run(
                    config_path, seed, rare_train_size, budget_lookup
                )
                all_rows.extend(rows)
                provenance.append(run_provenance)
                log_lines.append(
                    f"{dataset}\t{seed}\t{rare_train_size}\t{'failed' if error else 'success'}\t{error}"
                )

    run_level = pd.DataFrame(all_rows)
    expected_rows = len(expected_run_keys) * len(VARIANTS)
    keys = ["dataset", "seed", "rare_train_size", "variant"]
    if len(run_level) != expected_rows or run_level.duplicated(keys).any():
        raise AssertionError("run-variant ledger does not match the frozen grid")
    run_path = output_dir / "run_level.csv"
    run_level.to_csv(run_path, index=False)
    log_path = LOG_DIR / "supplementary_ablation_v1.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    failures = run_level[~run_level["status"].eq("success")]
    if not failures.empty:
        raise SystemExit(f"Fail-closed ledger written; {len(failures)} rows failed")

    identity = collapse_identity(run_level)
    seed_units, summary = aggregate(identity)
    dataset_equal = dataset_equal_summary(summary)
    identity_path = output_dir / "tables" / "identity_collapsed.csv"
    seed_path = output_dir / "tables" / "seed_count_variant_units.csv"
    summary_path = output_dir / "summary.csv"
    overall_path = output_dir / "tables" / "dataset_equal_summary.csv"
    reasons_path = output_dir / "tables" / "abstention_reasons.csv"
    identity.to_csv(identity_path, index=False)
    seed_units.to_csv(seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    dataset_equal.to_csv(overall_path, index=False)
    (
        run_level.groupby(["variant", "abstain", "abstain_reason"], dropna=False)
        .size()
        .rename("n_runs")
        .reset_index()
        .to_csv(reasons_path, index=False)
    )
    figure_paths = make_figure(dataset_equal, output_dir)
    notes_path = output_dir / "analysis_notes.md"
    notes_path.write_text(notes(run_level, identity, summary), encoding="utf-8")

    source_paths = [
        ROOT / "tools" / "analysis" / "supplementary_ablation.py",
        ROOT / "tools" / "analysis" / "ablation.py",
        ROOT / "src" / "rescue.py",
        ROOT / "results" / "supplementary_program" / "v1" / "methodology.md",
        ROOT / "results" / "label_budget" / "v1" / "run_level.csv",
        *(ROOT / path for path in CONFIGS),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
    }
    outputs = [
        run_path,
        identity_path,
        seed_path,
        summary_path,
        overall_path,
        reasons_path,
        notes_path,
        log_path,
        *figure_paths,
    ]
    script_manifest_path = output_dir / "_script_manifest.jsonl"
    script_manifest_path.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "command": "python tools/analysis/supplementary_ablation.py",
                "working_directory": str(ROOT),
                "sources": source_hashes,
                "outputs": [str(path.relative_to(ROOT)) for path in outputs],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    outputs.append(script_manifest_path)
    manifest = {
        "analysis": "supplementary_ablation",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/supplementary_ablation.py",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "pandas", "matplotlib", "scipy", "scikit-learn")
            },
        },
        "git": {
            "sha": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "dirty": _git_dirty(),
        },
        "source_hashes": source_hashes,
        "parameters": {
            "configs": list(CONFIGS),
            "seeds": list(SEEDS),
            "rare_train_sizes": list(RARE_TRAIN_SIZES),
            "alpha": ALPHA,
            "low_sep": CONFORMAL_LOW_SEP,
            "rank_grid": list(CONFORMAL_RANK_GRID),
            "min_val_missed": MIN_VAL_MISSED,
            "variants": [variant for variant, _, _ in VARIANTS],
        },
        "test_label_usage": "final rare metrics, rescue composition, and empirical alpha violation only; never gates, rank, tau, or parameter selection",
        "overwrite_status": "versioned P1 outputs only; caches and historical ablation outputs untouched",
        "expected_run_variant_rows": expected_rows,
        "status_counts": run_level["status"].value_counts().to_dict(),
        "inputs": {"runs": provenance},
        "outputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in outputs},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(run_level)} run-variant rows")
    print(f"Identity-collapsed rows: {len(identity)}")
    print(f"Summary rows: {len(summary)}")
    print(f"Dataset-equal rows: {len(dataset_equal)}")


if __name__ == "__main__":
    main()
