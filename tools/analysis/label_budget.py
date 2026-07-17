"""Fail-closed rare-label budget accounting for the 96 formal runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from src.utils import (  # noqa: E402
    compute_split_hash,
    git_sha,
    load_config,
    make_run_dir,
    parse_rare_train_size,
)


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
SPLITS = ("train", "validation", "test")
RATIO_COLUMNS = (
    "training_rare_label_fraction",
    "training_rare_share_of_all_split_rare",
    "total_supervised_rare_share_of_all_split_rare",
    "training_rare_label_share_of_training_cells",
)
OUT = ROOT / "results" / "label_budget" / "v1"
LOG_DIR = ROOT / "logs" / "label_budget"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id_identity(ids: pd.Series) -> tuple[list[str], str, str]:
    if ids.isna().any():
        raise ValueError("canonical labeled rare IDs contain missing values")
    values = ids.astype(str)
    if values.duplicated().any():
        raise ValueError("canonical labeled rare IDs contain duplicates")
    ordered = sorted(values.tolist())
    serialized = json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return ordered, serialized, digest


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("labeled mask contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    normalized = series.astype(str).str.strip().str.lower().map(mapping)
    if normalized.isna().any():
        invalid = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"labeled mask contains invalid values: {invalid}")
    return normalized.astype(bool)


def expected_labeled_rare(n_train_rare: int, rare_train_size: str) -> int:
    parsed = parse_rare_train_size(rare_train_size)
    if parsed == "all":
        return n_train_rare
    requested = (
        max(5, int(parsed * n_train_rare)) if isinstance(parsed, float) else int(parsed)
    )
    return min(n_train_rare, requested)


def load_prediction_split(path: Path, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path.relative_to(ROOT)))
    frame = pd.read_csv(path, low_memory=False)
    required = {"cell_id", "true_label"}
    if split == "train":
        required.add("is_labeled_for_scanvi")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{split} predictions missing columns: {sorted(missing)}")
    if frame["cell_id"].isna().any():
        raise ValueError(f"{split} predictions contain missing cell_id")
    frame = frame.copy()
    frame["cell_id"] = frame["cell_id"].astype(str)
    if frame["cell_id"].duplicated().any():
        raise ValueError(f"{split} predictions contain duplicate cell_id")
    if frame["true_label"].isna().any():
        raise ValueError(f"{split} predictions contain missing true_label")
    frame["true_label"] = frame["true_label"].astype(str)
    return frame


def _budget_equal(observed: object, expected: str) -> bool:
    left = parse_rare_train_size(str(observed))
    right = parse_rare_train_size(expected)
    if left == "all" or right == "all":
        return left == right
    return bool(math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12))


def validate_manifest(
    manifest: dict,
    config: dict,
    seed: int,
    rare_train_size: str,
    frames: dict[str, pd.DataFrame],
) -> str:
    experiment = config["experiment"]
    checks = {
        "dataset": config["dataset"]["name"],
        "dataset_path": config["dataset"]["path"],
        "label_key": config["dataset"].get("label_key"),
        "batch_key": config["dataset"].get("batch_key"),
        "split_mode": experiment.get("split_mode", "batch_heldout"),
        "seed": seed,
        "rare_class": experiment["rare_class"],
    }
    mismatches = [
        (key, manifest.get(key), value)
        for key, value in checks.items()
        if str(manifest.get(key)) != str(value)
    ]
    if not _budget_equal(manifest.get("rare_train_size"), rare_train_size):
        mismatches.append(
            ("rare_train_size", manifest.get("rare_train_size"), rare_train_size)
        )
    for split, manifest_key in (
        ("train", "n_train"),
        ("validation", "n_val"),
        ("test", "n_test"),
    ):
        if int(manifest.get(manifest_key, -1)) != len(frames[split]):
            mismatches.append(
                (manifest_key, manifest.get(manifest_key), len(frames[split]))
            )
    cached_hash = compute_split_hash(frames)
    if str(manifest.get("split_hash")) != cached_hash:
        mismatches.append(("split_hash", manifest.get("split_hash"), cached_hash))
    if mismatches:
        raise ValueError(f"manifest mismatch: {mismatches}")
    return cached_hash


def analyze_run(config_path: str, seed: int, rare_train_size: str) -> tuple[dict, dict]:
    config = load_config(ROOT / config_path)
    experiment = config["experiment"]
    dataset = config["dataset"]["name"]
    rare_class = str(experiment["rare_class"])
    split_mode = experiment.get("split_mode", "batch_heldout")
    run_dir = ROOT / make_run_dir(
        config,
        split_mode,
        seed,
        rare_class,
        parse_rare_train_size(rare_train_size),
    )
    row = {
        "dataset": dataset,
        "seed": seed,
        "rare_train_size": rare_train_size,
        "rare_class": rare_class,
        "split_mode": split_mode,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status": "failed",
        "error_reason": "",
        "identity_status": "identity_unverifiable",
    }
    provenance: dict[str, object] = {"run_dir": row["run_dir"], "files": {}}
    try:
        prediction_paths = {
            split: run_dir / "embeddings" / f"{split}_predictions.csv"
            for split in SPLITS
        }
        frames = {
            split: load_prediction_split(path, split)
            for split, path in prediction_paths.items()
        }
        id_sets = {split: set(frame["cell_id"]) for split, frame in frames.items()}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = id_sets[left].intersection(id_sets[right])
            if overlap:
                raise ValueError(
                    f"cell_id overlap between {left} and {right}: {len(overlap)}"
                )

        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(str(manifest_path.relative_to(ROOT)))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_hash = validate_manifest(manifest, config, seed, rare_train_size, frames)

        train = frames["train"]
        labeled = normalize_bool(train["is_labeled_for_scanvi"])
        train_rare_mask = train["true_label"].eq(rare_class)
        labeled_rare_mask = train_rare_mask & labeled
        labeled_ids, serialized_ids, identity_hash = canonical_id_identity(
            train.loc[labeled_rare_mask, "cell_id"]
        )
        actual_labeled_rare = int(labeled_rare_mask.sum())
        if len(labeled_ids) != actual_labeled_rare:
            raise AssertionError("labeled rare ID-set size differs from observed count")
        if not set(labeled_ids).issubset(id_sets["train"]):
            raise AssertionError("labeled rare identity contains non-training cell IDs")

        train_rare_ids = set(train.loc[train_rare_mask, "cell_id"])
        validation_rare_ids = set(
            frames["validation"].loc[
                frames["validation"]["true_label"].eq(rare_class), "cell_id"
            ]
        )
        test_rare_ids = set(
            frames["test"].loc[frames["test"]["true_label"].eq(rare_class), "cell_id"]
        )
        all_split_rare_ids = train_rare_ids | validation_rare_ids | test_rare_ids
        train_rare_pool = len(train_rare_ids)
        validation_rare = len(validation_rare_ids)
        test_rare = len(test_rare_ids)
        all_split_rare = len(all_split_rare_ids)
        all_training_cells = len(id_sets["train"])
        expected = expected_labeled_rare(train_rare_pool, rare_train_size)
        if actual_labeled_rare != expected:
            raise ValueError(
                "labeled rare count mismatch: "
                f"observed={actual_labeled_rare}, expected={expected}"
            )
        if train_rare_pool == 0 or all_split_rare == 0 or all_training_cells == 0:
            raise ValueError("required accounting denominator is zero")

        row.update(
            status="success",
            identity_status="verified",
            split_hash=split_hash,
            all_training_cells=all_training_cells,
            total_labeled_training_cells=int(labeled.sum()),
            train_rare_pool=train_rare_pool,
            actual_training_labeled_rare_count=actual_labeled_rare,
            expected_training_labeled_rare_count=expected,
            validation_rare=validation_rare,
            test_rare=test_rare,
            all_split_rare=all_split_rare,
            total_rare_supervision=actual_labeled_rare + validation_rare,
            labeled_rare_id_json=serialized_ids,
            labeled_rare_id_sha256=identity_hash,
            effective_budget_key=json.dumps(
                [dataset, rare_class, split_hash, actual_labeled_rare, identity_hash],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            training_rare_label_fraction=actual_labeled_rare / train_rare_pool,
            training_rare_share_of_all_split_rare=actual_labeled_rare / all_split_rare,
            total_supervised_rare_share_of_all_split_rare=(
                actual_labeled_rare + validation_rare
            )
            / all_split_rare,
            training_rare_label_share_of_training_cells=actual_labeled_rare
            / all_training_cells,
            test_label_usage="support accounting only",
        )
        for path in (*prediction_paths.values(), manifest_path):
            provenance["files"][str(path.relative_to(ROOT))] = sha256_file(path)
        provenance["cache_manifest"] = manifest
    except Exception as exc:
        row["error_reason"] = f"{type(exc).__name__}: {exc}"
    return row, provenance


def classify_collapses(run_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = run_level.copy()
    result["count_collapse"] = False
    result["identity_collapse"] = False
    result["collapse_class"] = "not_evaluable"
    ok = result["status"].eq("success")
    count_keys = [
        "dataset",
        "rare_class",
        "seed",
        "actual_training_labeled_rare_count",
    ]
    identity_keys = [*count_keys, "split_hash", "labeled_rare_id_sha256"]
    count_sizes = result.loc[ok].groupby(count_keys, dropna=False).size()
    identity_sizes = result.loc[ok].groupby(identity_keys, dropna=False).size()
    for index in result.index[ok]:
        count_key = tuple(result.loc[index, count_keys])
        identity_key = tuple(result.loc[index, identity_keys])
        count_collapse = int(count_sizes.loc[count_key]) > 1
        identity_collapse = int(identity_sizes.loc[identity_key]) > 1
        result.loc[index, "count_collapse"] = count_collapse
        result.loc[index, "identity_collapse"] = identity_collapse
        if identity_collapse:
            collapse_class = "identity_collapse"
        elif count_collapse:
            collapse_class = "count_only_collision"
        else:
            collapse_class = "unique_budget"
        result.loc[index, "collapse_class"] = collapse_class

    count_rows = []
    for key, group in result.loc[ok].groupby(count_keys, sort=False, dropna=False):
        hashes = sorted(group["labeled_rare_id_sha256"].unique().tolist())
        count_rows.append(
            {
                **dict(zip(count_keys, key)),
                "n_nominal_runs": len(group),
                "n_distinct_identities": len(hashes),
                "nominal_budgets": json.dumps(
                    group["rare_train_size"].tolist(), separators=(",", ":")
                ),
                "identity_hashes": json.dumps(hashes, separators=(",", ":")),
                "count_collapse": len(group) > 1,
                "identity_collapse": len(hashes) == 1 and len(group) > 1,
                "count_only_collision": len(hashes) > 1,
            }
        )
    return result, pd.DataFrame(count_rows)


def identity_collapse(run_level: pd.DataFrame) -> pd.DataFrame:
    ok = run_level[run_level["status"].eq("success")].copy()
    keys = [
        "dataset",
        "rare_class",
        "seed",
        "split_hash",
        "actual_training_labeled_rare_count",
        "labeled_rare_id_sha256",
    ]
    rows = []
    for key, group in ok.groupby(keys, sort=False, dropna=False):
        first = group.iloc[0]
        row = {column: value for column, value in zip(keys, key)}
        row.update(
            n_folded_nominal_budgets=len(group),
            folded_nominal_budgets=json.dumps(
                group["rare_train_size"].tolist(), separators=(",", ":")
            ),
            source_run_dirs=json.dumps(
                group["run_dir"].tolist(), ensure_ascii=False, separators=(",", ":")
            ),
        )
        for column in (
            "all_training_cells",
            "total_labeled_training_cells",
            "train_rare_pool",
            "validation_rare",
            "test_rare",
            "all_split_rare",
            "total_rare_supervision",
            *RATIO_COLUMNS,
        ):
            values = group[column].astype(float).to_numpy()
            if not np.allclose(values, values[0], equal_nan=True):
                raise AssertionError(
                    f"identity-equivalent runs differ in accounting column {column}"
                )
            row[column] = first[column]
        rows.append(row)
    return pd.DataFrame(rows)


def seed_count_units(identity_level: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "dataset",
        "rare_class",
        "actual_training_labeled_rare_count",
        "seed",
    ]
    rows = []
    numeric = [
        "all_training_cells",
        "total_labeled_training_cells",
        "train_rare_pool",
        "validation_rare",
        "test_rare",
        "all_split_rare",
        "total_rare_supervision",
        *RATIO_COLUMNS,
    ]
    for key, group in identity_level.groupby(keys, sort=False, dropna=False):
        row = {column: value for column, value in zip(keys, key)}
        row.update(
            n_identity_runs=len(group),
            split_hashes=json.dumps(
                sorted(group["split_hash"].unique().tolist()), separators=(",", ":")
            ),
            identity_hashes=json.dumps(
                sorted(group["labeled_rare_id_sha256"].unique().tolist()),
                separators=(",", ":"),
            ),
            folded_nominal_budgets=json.dumps(
                sorted(
                    {
                        budget
                        for payload in group["folded_nominal_budgets"]
                        for budget in json.loads(payload)
                    },
                    key=lambda value: RARE_TRAIN_SIZES.index(value),
                ),
                separators=(",", ":"),
            ),
        )
        for column in numeric:
            row[column] = group[column].astype(float).mean()
        rows.append(row)
    units = pd.DataFrame(rows)
    if units.duplicated(keys).any():
        raise AssertionError("a seed contributes more than once to a count unit")
    return units


def aggregate_summary(units: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "rare_class", "actual_training_labeled_rare_count"]
    rows = []
    numeric = [
        "train_rare_pool",
        "validation_rare",
        "test_rare",
        "all_split_rare",
        "total_rare_supervision",
        *RATIO_COLUMNS,
    ]
    for key, group in units.groupby(keys, sort=False, dropna=False):
        row = {column: value for column, value in zip(keys, key)}
        row.update(
            n_seeds=group["seed"].nunique(),
            n_seed_count_units=len(group),
            n_identity_runs=int(group["n_identity_runs"].sum()),
        )
        for column in numeric:
            values = group[column].astype(float)
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_std"] = values.std(ddof=1)
            row[f"{column}_min"] = values.min()
            row[f"{column}_max"] = values.max()
        rows.append(row)
    return pd.DataFrame(rows)


def make_figure(run_level: pd.DataFrame, output_dir: Path) -> list[Path]:
    ok = run_level[run_level["status"].eq("success")].copy()
    dataset_order = [load_config(ROOT / path)["dataset"]["name"] for path in CONFIGS]
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.4), sharex=True, sharey=True)
    colors = ("#1F6F8B", "#C45A3C")
    labels = (
        "Training rare labels / train rare pool",
        "Training rare labels + validation rare / all rare",
    )
    x = np.arange(len(RARE_TRAIN_SIZES))
    for ax, dataset in zip(axes.flat, dataset_order):
        subset = ok[ok["dataset"].eq(dataset)]
        grouped = subset.groupby("rare_train_size", sort=False)
        for column, color, label in zip(
            (
                "training_rare_label_fraction",
                "total_supervised_rare_share_of_all_split_rare",
            ),
            colors,
            labels,
        ):
            mean = grouped[column].mean().reindex(RARE_TRAIN_SIZES)
            low = grouped[column].min().reindex(RARE_TRAIN_SIZES)
            high = grouped[column].max().reindex(RARE_TRAIN_SIZES)
            ax.plot(x, mean, marker="o", linewidth=1.8, color=color, label=label)
            ax.fill_between(x, low, high, color=color, alpha=0.14, linewidth=0)
        ax.set_title(dataset.replace("_", " "), loc="left", fontsize=10)
        ax.set_xticks(x, ("1%", "5%", "10%", "all"))
        ax.set_ylim(-0.02, 1.02)
        ax.grid(axis="y", color="#D7D2C8", alpha=0.7, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Observed rare-label share")
    axes[1, 0].set_ylabel("Observed rare-label share")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Observed training and calibration rare-label support",
        fontsize=15,
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.005,
        "Points are seed means; ribbons span seed minima to maxima. Validation labels are reported as calibration support and are not training labels.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    png = output_dir / "figures" / "rare_label_budget_accounting.png"
    pdf = output_dir / "figures" / "rare_label_budget_accounting.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def analysis_notes(
    run_level: pd.DataFrame,
    count_table: pd.DataFrame,
    identity_level: pd.DataFrame,
    units: pd.DataFrame,
) -> str:
    status_counts = run_level["status"].value_counts().to_dict()
    successful = run_level[run_level["status"].eq("success")]
    identity_groups = int(count_table["identity_collapse"].sum())
    collision_groups = int(count_table["count_only_collision"].sum())
    return (
        "\n".join(
            [
                "# Rare-label budget accounting notes",
                "",
                "## Scope",
                "",
                "- The ledger covers the frozen 8 datasets x 3 seeds x 4 nominal budgets grid.",
                "- Training labels and validation calibration labels are reported separately.",
                "- Test labels contribute only split-support counts; they are not used for model selection, calibration, thresholds, or collapse.",
                "",
                "## Completeness and identity",
                "",
                f"- Expected ledger rows: {len(CONFIGS) * len(SEEDS) * len(RARE_TRAIN_SIZES)}.",
                f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`.",
                f"- Verified labeled-rare identity hashes: {successful['labeled_rare_id_sha256'].notna().sum()}.",
                f"- Within-seed identity-collapse groups: {identity_groups}.",
                f"- Within-seed count-only collision groups: {collision_groups}.",
                f"- Rows after identity collapse: {len(identity_level)}; seed-count units: {len(units)}.",
                "",
                "## Interpretation",
                "",
                "- Nominal budgets are requests; observed labeled-rare counts and fractions are the auditable supervision quantities.",
                "- Equal counts do not imply equal labeled-cell identity. Count-only collisions retain every identity hash and are averaged only at the frozen seed-count aggregation step.",
                "- Cross-dataset pooled cell counts are not inferential summaries because dataset sizes and rare-cell prevalence differ.",
            ]
        )
        + "\n"
    )


def package_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return "unknown"


def git_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    for path in (output_dir, output_dir / "tables", output_dir / "figures", LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)

    rows = []
    run_provenance = []
    expected_keys = []
    log_lines = []
    for config_path in CONFIGS:
        dataset = load_config(ROOT / config_path)["dataset"]["name"]
        for seed in SEEDS:
            for rare_train_size in RARE_TRAIN_SIZES:
                expected_keys.append((dataset, seed, rare_train_size))
                row, provenance = analyze_run(config_path, seed, rare_train_size)
                rows.append(row)
                run_provenance.append(provenance)
                log_lines.append(
                    f"{dataset}\t{seed}\t{rare_train_size}\t{row['status']}\t{row['error_reason']}"
                )

    run_level = pd.DataFrame(rows)
    key_columns = ["dataset", "seed", "rare_train_size"]
    if run_level.duplicated(key_columns).any():
        raise AssertionError("ledger contains duplicate expected keys")
    observed_keys = set(
        map(tuple, run_level[key_columns].itertuples(index=False, name=None))
    )
    if observed_keys != set(expected_keys):
        raise AssertionError("ledger does not match the frozen 96-key grid")

    run_level, count_table = classify_collapses(run_level)
    run_path = output_dir / "run_level.csv"
    run_level.to_csv(run_path, index=False)
    log_path = LOG_DIR / "label_budget_v1.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    failures = run_level[~run_level["status"].eq("success")]
    if not failures.empty:
        raise SystemExit(
            f"Fail-closed ledger written; {len(failures)} configurations failed"
        )

    identity_level = identity_collapse(run_level)
    units = seed_count_units(identity_level)
    summary = aggregate_summary(units)
    identity_path = output_dir / "tables" / "identity_collapsed.csv"
    count_path = output_dir / "tables" / "count_collapse.csv"
    units_path = output_dir / "tables" / "seed_count_units.csv"
    summary_path = output_dir / "summary.csv"
    identity_level.to_csv(identity_path, index=False)
    count_table.to_csv(count_path, index=False)
    units.to_csv(units_path, index=False)
    summary.to_csv(summary_path, index=False)
    figure_paths = make_figure(run_level, output_dir)
    notes_path = output_dir / "analysis_notes.md"
    notes_path.write_text(
        analysis_notes(run_level, count_table, identity_level, units),
        encoding="utf-8",
    )

    source_paths = [
        ROOT / "tools" / "analysis" / "label_budget.py",
        ROOT / "results" / "supplementary_program" / "v1" / "methodology.md",
        *(ROOT / config_path for config_path in CONFIGS),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path) for path in source_paths
    }
    outputs = [
        run_path,
        summary_path,
        identity_path,
        count_path,
        units_path,
        notes_path,
        log_path,
        *figure_paths,
    ]
    script_manifest_path = output_dir / "_script_manifest.jsonl"
    script_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/label_budget.py",
        "working_directory": str(ROOT),
        "sources": source_hashes,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    script_manifest_path.write_text(
        json.dumps(script_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    outputs.append(script_manifest_path)
    manifest = {
        "analysis": "label_budget",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/label_budget.py",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: package_version(name)
                for name in ("numpy", "pandas", "matplotlib", "pyyaml")
            },
        },
        "git": {"sha": git_sha(), "dirty": git_dirty()},
        "source_hashes": source_hashes,
        "parameters": {
            "configs": list(CONFIGS),
            "seeds": list(SEEDS),
            "rare_train_sizes": list(RARE_TRAIN_SIZES),
            "canonical_id_serialization": "sorted unique strings; compact UTF-8 JSON; ensure_ascii=False; separators=(',', ':')",
            "within_seed_effective_budget_key": [
                "dataset",
                "rare_class",
                "split_hash",
                "actual_training_labeled_rare_count",
                "labeled_rare_id_sha256",
            ],
            "cross_seed_unit": [
                "dataset",
                "rare_class",
                "actual_training_labeled_rare_count",
            ],
        },
        "test_label_usage": "split-support accounting only; never training, selection, calibration, thresholds, or collapse",
        "overwrite_status": "versioned P0 outputs only; formal caches and historical results untouched",
        "expected_configurations": len(expected_keys),
        "status_counts": run_level["status"].value_counts().to_dict(),
        "inputs": {"runs": run_provenance},
        "outputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in outputs},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(run_level)} successful ledger rows to {run_path}")
    print(f"Identity-collapsed rows: {len(identity_level)}")
    print(f"Seed-count units: {len(units)}")
    print(f"Summary rows: {len(summary)}")


if __name__ == "__main__":
    main()
