"""Cache-only rescue composition analysis for the formal scRareRefine grid."""

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
from src.utils import (  # noqa: E402
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
OUT = ROOT / "results" / "rescue_composition" / "v1"
LOG_DIR = ROOT / "logs" / "rescue_composition"
COMPARISON_PATH = ROOT / "results" / "comparison" / "comparison_summary.csv"
SCRIPT_MANIFEST_PATH = OUT / "_script_manifest.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _expected_labeled_rare(n_train_rare: int, rare_train_size: str) -> int:
    parsed = parse_rare_train_size(rare_train_size)
    if parsed == "all":
        return n_train_rare
    if isinstance(parsed, float):
        requested = max(5, int(parsed * n_train_rare))
    else:
        requested = int(parsed)
    return min(n_train_rare, requested)


def _load_aligned_split(embeddings: Path, split: str) -> pd.DataFrame:
    pred_path = embeddings / f"{split}_predictions.csv"
    latent_path = embeddings / f"{split}_latent.csv"
    pred = pd.read_csv(pred_path, low_memory=False)
    latent = pd.read_csv(latent_path, low_memory=False)
    required_pred = {"cell_id", "true_label", "predicted_label"}
    if split == "train":
        required_pred.add("is_labeled_for_scanvi")
    missing = required_pred.difference(pred.columns)
    if missing:
        raise ValueError(f"{split} predictions missing columns: {sorted(missing)}")
    if "cell_id" not in latent.columns:
        raise ValueError(f"{split} latent missing cell_id")
    latent_cols = [column for column in latent.columns if column.startswith("latent_")]
    if not latent_cols:
        raise ValueError(f"{split} latent has no latent_* columns")
    for name, frame in (("predictions", pred), ("latent", latent)):
        if (
            frame["cell_id"].isna().any()
            or frame["cell_id"].astype(str).duplicated().any()
        ):
            raise ValueError(f"{split} {name} has missing or duplicate cell_id")
    pred_ids = set(pred["cell_id"].astype(str))
    latent_ids = set(latent["cell_id"].astype(str))
    if pred_ids != latent_ids:
        raise ValueError(f"{split} prediction/latent cell_id sets differ")
    pred = pred.copy()
    latent = latent.copy()
    pred["cell_id"] = pred["cell_id"].astype(str)
    latent["cell_id"] = latent["cell_id"].astype(str)
    aligned = pred.merge(
        latent[["cell_id", *latent_cols]],
        on="cell_id",
        how="inner",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(pred):
        raise ValueError(f"{split} alignment changed row count")
    values = aligned[latent_cols].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{split} latent contains non-finite values")
    if aligned[["true_label", "predicted_label"]].isna().any().any():
        raise ValueError(f"{split} labels contain missing values")
    return aligned


def _latent(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame.columns if column.startswith("latent_")]
    return frame[columns].to_numpy(dtype=float)


def compute_composition(
    y_true: pd.Series,
    baseline: pd.Series,
    final: pd.Series,
    rare_class: str,
) -> tuple[dict, np.ndarray]:
    y = pd.Series(y_true).astype(str).reset_index(drop=True)
    base = pd.Series(baseline).astype(str).reset_index(drop=True)
    refined = pd.Series(final).astype(str).reset_index(drop=True)
    if not (len(y) == len(base) == len(refined)):
        raise ValueError("composition inputs have different lengths")
    baseline_missed = y.eq(rare_class) & base.ne(rare_class)
    changed = refined.ne(base)
    invalid_change = changed & ~(base.ne(rare_class) & refined.eq(rare_class))
    if invalid_change.any():
        raise AssertionError("formal rescue introduced a non-target-label transition")
    true_rescue = changed & y.eq(rare_class)
    false_rescue = changed & y.ne(rare_class)
    remaining_missed = y.eq(rare_class) & refined.ne(rare_class)
    n_missed = int(baseline_missed.sum())
    n_true = int(true_rescue.sum())
    n_false = int(false_rescue.sum())
    n_all = int(changed.sum())
    n_nonrare = int(y.ne(rare_class).sum())
    if n_missed != n_true + int(remaining_missed.sum()):
        raise AssertionError("baseline missed rare decomposition failed")
    if n_all != n_true + n_false:
        raise AssertionError("rescue decomposition failed")
    precision = n_true / n_all if n_all else np.nan
    fdp = n_false / n_all if n_all else np.nan
    if n_all and not np.isclose(precision + fdp, 1.0):
        raise AssertionError("rescue precision/FDP complement failed")
    incremental_fpr = n_false / n_nonrare if n_nonrare else np.nan
    metrics = {
        "n_test": len(y),
        "true_rare": int(y.eq(rare_class).sum()),
        "true_nonrare": n_nonrare,
        "baseline_missed_rare": n_missed,
        "true_rescues": n_true,
        "false_rescues": n_false,
        "all_rescues": n_all,
        "remaining_missed_rare": int(remaining_missed.sum()),
        "recovery_rate": n_true / n_missed if n_missed else np.nan,
        "rescue_precision": precision,
        "rescue_fdp": fdp,
        "incremental_fpr": incremental_fpr,
        "rescue_ffr": incremental_fpr,
    }
    return metrics, changed.to_numpy()


def composition_from_historical_counts(
    y_true: pd.Series,
    baseline: pd.Series,
    rare_class: str,
    n_rescued: int,
    n_false_rescue: int,
    historical_incremental_fpr: float,
) -> dict:
    y = pd.Series(y_true).astype(str).reset_index(drop=True)
    base = pd.Series(baseline).astype(str).reset_index(drop=True)
    n_missed = int((y.eq(rare_class) & base.ne(rare_class)).sum())
    n_nonrare = int(y.ne(rare_class).sum())
    n_true = int(n_rescued) - int(n_false_rescue)
    if n_true < 0 or n_true > n_missed:
        raise AssertionError(
            "historical rescue counts are incompatible with baseline misses"
        )
    expected_fpr = n_false_rescue / n_nonrare if n_nonrare else np.nan
    if not (
        (np.isnan(expected_fpr) and np.isnan(historical_incremental_fpr))
        or np.isclose(expected_fpr, historical_incremental_fpr, atol=5e-7)
    ):
        raise AssertionError(
            "historical incremental FPR does not match false-rescue count"
        )
    precision = n_true / n_rescued if n_rescued else np.nan
    fdp = n_false_rescue / n_rescued if n_rescued else np.nan
    return {
        "n_test": len(y),
        "true_rare": int(y.eq(rare_class).sum()),
        "true_nonrare": n_nonrare,
        "baseline_missed_rare": n_missed,
        "true_rescues": n_true,
        "false_rescues": int(n_false_rescue),
        "all_rescues": int(n_rescued),
        "remaining_missed_rare": n_missed - n_true,
        "recovery_rate": n_true / n_missed if n_missed else np.nan,
        "rescue_precision": precision,
        "rescue_fdp": fdp,
        "incremental_fpr": expected_fpr,
        "rescue_ffr": expected_fpr,
    }


def _historical_lookup() -> dict[tuple[str, int, str], dict]:
    if not COMPARISON_PATH.exists():
        return {}
    comparison = pd.read_csv(COMPARISON_PATH, dtype={"rare_train_size": str})
    subset = comparison[
        (comparison["method"] == "scRareRefine") & (comparison["status"] == "ok")
    ].copy()
    keys = ["dataset", "seed", "rare_train_size"]
    if subset.duplicated(keys).any():
        raise ValueError("historical comparison has duplicate scRareRefine keys")
    return {
        (str(row.dataset), int(row.seed), str(row.rare_train_size)): row._asdict()
        for row in subset.itertuples(index=False)
    }


def _analyze_run(
    config_path: str,
    seed: int,
    rare_train_size: str,
    historical: dict[tuple[str, int, str], dict],
) -> tuple[dict, dict]:
    config = load_config(ROOT / config_path)
    experiment = config["experiment"]
    dataset = config["dataset"]["name"]
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
    }
    provenance = {"run_dir": row["run_dir"], "files": {}}
    embeddings = run_dir / "embeddings"
    required = [
        embeddings / f"{split}_{kind}.csv"
        for split in SPLITS
        for kind in ("predictions", "latent")
    ]
    if not all(path.exists() for path in required):
        missing = [
            str(path.relative_to(ROOT)) for path in required if not path.exists()
        ]
        row.update(status="missing_cache", error="; ".join(missing))
        return row, provenance
    try:
        frames = {split: _load_aligned_split(embeddings, split) for split in SPLITS}
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
        train = frames["train"]
        labeled = train["is_labeled_for_scanvi"]
        if labeled.isna().any():
            raise ValueError("train labeled mask contains missing values")
        is_labeled = labeled.astype(bool).to_numpy()
        train_true = train["true_label"].astype(str)
        n_train_rare = int(train_true.eq(rare).sum())
        n_labeled_rare = int((train_true.eq(rare).to_numpy() & is_labeled).sum())
        expected_rare = _expected_labeled_rare(n_train_rare, rare_train_size)
        if n_labeled_rare != expected_rare:
            raise ValueError(
                f"labeled rare count mismatch: observed={n_labeled_rare}, expected={expected_rare}"
            )
        proto = PrototypeRescuer(rare)
        proto.fit(_latent(train), train_true, is_labeled)
        validation = frames["validation"]
        test = frames["test"]
        baseline = test["predicted_label"].astype(str).reset_index(drop=True)
        final, rescue_summary = conformal_rescue(
            proto,
            baseline,
            validation["predicted_label"].astype(str),
            validation["true_label"].astype(str),
            _latent(validation),
            _latent(test),
        )
        composition, changed = compute_composition(
            test["true_label"], baseline, final, rare
        )
        abstain = bool(rescue_summary["abstain"])
        chosen_rank = int(rescue_summary["chosen_rank"])
        raw_candidate = None
        if not abstain and chosen_rank > 0:
            raw_candidate = proto.rank_candidate(
                _latent(test), baseline, max_rank=chosen_rank
            )
            if np.any(changed & ~raw_candidate):
                raise AssertionError("rescued cells are not a subset of raw candidates")
        if abstain:
            if chosen_rank != 0 or changed.any():
                raise AssertionError("abstention did not preserve baseline predictions")
        historical_row = historical.get((dataset, seed, rare_train_size))
        historical_match = np.nan
        reconstruction_basis = "current_formal_replay"
        if historical_row is not None:
            historical_match = bool(
                int(float(historical_row["n_rescued"])) == composition["all_rescues"]
                and int(float(historical_row["n_false_rescue"]))
                == composition["false_rescues"]
                and np.isclose(
                    float(historical_row["rescue_ffr"]),
                    composition["incremental_fpr"],
                    atol=5e-7,
                )
            )
            if not historical_match:
                composition = composition_from_historical_counts(
                    test["true_label"],
                    baseline,
                    rare,
                    int(float(historical_row["n_rescued"])),
                    int(float(historical_row["n_false_rescue"])),
                    float(historical_row["rescue_ffr"]),
                )
                reconstruction_basis = "historical_counts_only"
                rescue_summary = {
                    "val_missed": np.nan,
                    "abstain": np.nan,
                    "reason": "unavailable_historical_decision_metadata",
                    "chosen_rank": np.nan,
                    "tau": np.nan,
                }
                raw_candidate = None
        y = test["true_label"].astype(str).reset_index(drop=True)
        row.update(
            status="success",
            separability=proto.separability_ratio,
            n_train=len(train),
            n_validation=len(validation),
            n_train_rare=n_train_rare,
            n_labeled_total=int(is_labeled.sum()),
            n_labeled_rare=n_labeled_rare,
            expected_labeled_rare=expected_rare,
            labeled_mask_source="train_predictions.is_labeled_for_scanvi",
            val_missed=rescue_summary.get("val_missed", np.nan),
            abstain=(
                bool(rescue_summary["abstain"])
                if pd.notna(rescue_summary["abstain"])
                else np.nan
            ),
            abstain_reason=rescue_summary["reason"] or "rescue_applied",
            chosen_rank=rescue_summary["chosen_rank"],
            tau=rescue_summary["tau"],
            raw_candidates=(
                int(raw_candidate.sum()) if raw_candidate is not None else np.nan
            ),
            raw_candidate_true_rare=(
                int((raw_candidate & y.eq(rare).to_numpy()).sum())
                if raw_candidate is not None
                else np.nan
            ),
            raw_candidate_nonrare=(
                int((raw_candidate & y.ne(rare).to_numpy()).sum())
                if raw_candidate is not None
                else np.nan
            ),
            historical_comparison_available=historical_row is not None,
            historical_comparison_match=historical_match,
            reconstruction_basis=reconstruction_basis,
            **composition,
        )
        for path in required:
            provenance["files"][str(path.relative_to(ROOT))] = _sha256(path)
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            provenance["files"][str(manifest_path.relative_to(ROOT))] = _sha256(
                manifest_path
            )
            provenance["cache_manifest"] = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        return row, provenance
    except Exception as exc:
        row.update(status="invalid_cache", error=f"{type(exc).__name__}: {exc}")
        return row, provenance


def _aggregate(run_level: pd.DataFrame) -> pd.DataFrame:
    ok = run_level[run_level["status"] == "success"].copy()
    numeric = [
        "baseline_missed_rare",
        "true_rescues",
        "false_rescues",
        "all_rescues",
        "remaining_missed_rare",
        "recovery_rate",
        "rescue_precision",
        "rescue_fdp",
        "incremental_fpr",
    ]
    grouped = ok.groupby(
        ["dataset", "rare_class", "rare_train_size"], sort=False, dropna=False
    )
    summary = grouped[numeric].agg(["mean", "std", "min", "max"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]
    counts = grouped.agg(
        n_seeds=("seed", "nunique"),
        n_abstain_known=("abstain", lambda values: int(values.notna().sum())),
        n_abstain_true=("abstain", lambda values: int(values.eq(True).sum())),
        n_abstain_false=("abstain", lambda values: int(values.eq(False).sum())),
        n_abstain_unknown=("abstain", lambda values: int(values.isna().sum())),
        n_current_replay=(
            "reconstruction_basis",
            lambda values: int(values.eq("current_formal_replay").sum()),
        ),
        n_historical_counts_only=(
            "reconstruction_basis",
            lambda values: int(values.eq("historical_counts_only").sum()),
        ),
        n_rescue_runs=("all_rescues", lambda values: int((values > 0).sum())),
        n_recovery_rate_defined=(
            "recovery_rate",
            lambda values: int(values.notna().sum()),
        ),
        n_rescue_precision_defined=(
            "rescue_precision",
            lambda values: int(values.notna().sum()),
        ),
        n_rescue_fdp_defined=("rescue_fdp", lambda values: int(values.notna().sum())),
    ).reset_index()
    return summary.merge(
        counts, on=["dataset", "rare_class", "rare_train_size"], validate="one_to_one"
    )


def _make_figure(run_level: pd.DataFrame, output_dir: Path) -> list[Path]:
    ok = run_level[run_level["status"] == "success"].copy()
    order = [load_config(ROOT / path)["dataset"]["name"] for path in CONFIGS]
    scarce = ok[ok["rare_train_size"].isin(("0.01", "0.05", "0.10"))]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.0), sharex=True)
    for ax, budget in zip(axes, ("0.01", "0.05", "0.10")):
        plot_data = (
            scarce[scarce["rare_train_size"] == budget]
            .groupby("dataset")[["true_rescues", "remaining_missed_rare"]]
            .mean()
            .reindex(order)
        )
        x = np.arange(len(plot_data))
        rescued = plot_data["true_rescues"].to_numpy()
        remaining = plot_data["remaining_missed_rare"].to_numpy()
        ax.bar(x, rescued, color="#2A7F62", label="True rescues")
        ax.bar(
            x,
            remaining,
            bottom=rescued,
            color="#D8A24A",
            label="Remaining missed rare",
        )
        ax.set_ylabel("Mean cells")
        ax.set_title(f"Nominal rare-label budget = {budget}", loc="left")
        ax.grid(axis="y", color="#D9D4C7", linewidth=0.7, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(np.arange(len(order)))
    axes[-1].set_xticklabels(order, rotation=32, ha="right")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Composition of baseline-missed rare cells by label budget")
    fig.text(
        0.01,
        0.01,
        "Each bar averages three seeds within one dataset-budget group. Absolute counts are not directly comparable across datasets. Mouse-pancreas composition uses traceable historical rescue counts; decision metadata are unavailable.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    png = output_dir / "figures" / "rescue_composition_scarce.png"
    pdf = output_dir / "figures" / "rescue_composition_scarce.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def _notes(run_level: pd.DataFrame) -> str:
    status_counts = run_level["status"].value_counts().to_dict()
    ok = run_level[run_level["status"] == "success"].copy()
    applied = ok[ok["abstain"].eq(False)]
    rescued = ok[ok["all_rescues"] > 0]
    scarce = ok[ok["rare_train_size"].isin(("0.01", "0.05", "0.10"))]
    total_true = int(scarce["true_rescues"].sum())
    total_false = int(scarce["false_rescues"].sum())
    pooled_precision = (
        total_true / (total_true + total_false) if total_true + total_false else np.nan
    )
    lines = [
        "# Rescue composition analysis notes",
        "",
        "## Scope",
        "",
        "- Cache-only reconstruction using the formal train-only prototype and validation-calibrated conformal rescue implementation.",
        "- Test labels are used only to characterize true rescues, false rescues, and remaining misses.",
        "- Raw candidates are undefined for abstention runs because chosen_rank=0 is a sentinel, not an active rank.",
        "",
        "## Completeness",
        "",
        f"- Expected configurations: {len(CONFIGS) * len(SEEDS) * len(RARE_TRAIN_SIZES)}.",
        f"- Status counts: `{json.dumps(status_counts, sort_keys=True)}`.",
        f"- Non-abstaining runs: {len(applied)}; runs with at least one rescue: {len(rescued)}.",
        f"- Historical-count-only rows: {int(ok['reconstruction_basis'].eq('historical_counts_only').sum())}; rank, tau, raw candidates, and abstention metadata are unavailable for these rows.",
        "",
        "## Scarce-label descriptive composition",
        "",
        f"- Across all successful scarce-label runs, true rescues={total_true}, false rescues={total_false}.",
        f"- Pooled rescue precision across those rescue events={pooled_precision:.4f}."
        if np.isfinite(pooled_precision)
        else "- No scarce-label rescue events were observed.",
        f"- Maximum run-level incremental FPR={scarce['incremental_fpr'].max():.6f}.",
        "",
        "## Interpretation limits",
        "",
        "- Pooled counts are descriptive and do not replace dataset-level inference.",
        "- Rescue precision/FDP is undefined when no cells are rescued; recovery rate is undefined when the baseline has no missed rare cells.",
        "- Empirical test error rates under batch shift are safety outcomes, not unconditional conformal guarantees.",
    ]
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
    historical = _historical_lookup()
    rows = []
    run_provenance = []
    log_lines = []
    expected_keys = []
    for config_path in CONFIGS:
        dataset = load_config(ROOT / config_path)["dataset"]["name"]
        for seed in SEEDS:
            for rare_train_size in RARE_TRAIN_SIZES:
                expected_keys.append((dataset, seed, rare_train_size))
                row, provenance = _analyze_run(
                    config_path, seed, rare_train_size, historical
                )
                rows.append(row)
                run_provenance.append(provenance)
                log_lines.append(
                    f"{dataset}\t{seed}\t{rare_train_size}\t{row['status']}\t{row.get('error', '')}"
                )
    run_level = pd.DataFrame(rows)
    key_columns = ["dataset", "seed", "rare_train_size"]
    if run_level.duplicated(key_columns).any():
        raise AssertionError("analysis ledger contains duplicate expected keys")
    observed_keys = set(
        map(tuple, run_level[key_columns].itertuples(index=False, name=None))
    )
    if observed_keys != set(expected_keys):
        raise AssertionError("analysis ledger does not match the 96 expected keys")
    run_path = output_dir / "run_level.csv"
    run_level.to_csv(run_path, index=False)
    summary = _aggregate(run_level)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    reason_path = output_dir / "tables" / "abstention_reasons.csv"
    (
        run_level[run_level["status"] == "success"]
        .groupby(["abstain", "abstain_reason"], dropna=False)
        .size()
        .rename("n_runs")
        .reset_index()
        .to_csv(reason_path, index=False)
    )
    figure_paths = _make_figure(run_level, output_dir)
    notes_path = output_dir / "analysis_notes.md"
    notes_path.write_text(_notes(run_level), encoding="utf-8")
    log_path = LOG_DIR / "rescue_composition_v1.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    source_paths = [
        ROOT / "tools" / "analysis" / "rescue_composition.py",
        ROOT / "src" / "rescue.py",
        ROOT / "src" / "utils.py",
        ROOT / "tests" / "test_rescue_composition.py",
        *(ROOT / config for config in CONFIGS),
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
    }
    script_record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/rescue_composition.py",
        "working_directory": str(ROOT),
        "sources": source_hashes,
        "inputs": {str(COMPARISON_PATH.relative_to(ROOT)): _sha256(COMPARISON_PATH)},
        "outputs": [
            str(path.relative_to(ROOT))
            for path in (run_path, summary_path, reason_path, notes_path, *figure_paths)
        ],
    }
    SCRIPT_MANIFEST_PATH.write_text(
        json.dumps(script_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    output_paths = [
        run_path,
        summary_path,
        reason_path,
        notes_path,
        SCRIPT_MANIFEST_PATH,
        *figure_paths,
    ]
    manifest = {
        "analysis": "rescue_composition",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python tools/analysis/rescue_composition.py",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: _package_version(name)
                for name in ("numpy", "pandas", "matplotlib", "scipy", "scikit-learn")
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
        },
        "test_label_usage": "final composition and empirical metrics only; never selection",
        "overwrite_status": "versioned analysis outputs only; historical outputs untouched",
        "expected_configurations": len(expected_keys),
        "status_counts": run_level["status"].value_counts().to_dict(),
        "inputs": {
            "comparison_summary": {
                "path": str(COMPARISON_PATH.relative_to(ROOT)),
                "sha256": _sha256(COMPARISON_PATH),
            },
            "runs": run_provenance,
        },
        "outputs": {
            str(path.relative_to(ROOT)): _sha256(path) for path in output_paths
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    failures = run_level[run_level["status"] != "success"]
    print(f"Saved {len(run_level)} ledger rows to {run_path}")
    print(f"Status counts: {run_level['status'].value_counts().to_dict()}")
    if not failures.empty:
        raise SystemExit(f"Analysis incomplete: {len(failures)} configurations failed")


if __name__ == "__main__":
    main()
