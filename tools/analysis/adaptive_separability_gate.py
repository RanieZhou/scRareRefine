"""Cross-fitted adaptive separability gate for scRareRefine.

This is an isolated, cache-only experiment.  It deliberately does not modify
``src.rescue.conformal_rescue``.  The fixed ``S_min=1.3`` rule remains the
primary method until this experiment passes its pre-declared safety criteria.

The adaptive rule changes behaviour only when the train-only separability
statistic is below the fixed cutoff.  It then uses out-of-fold validation
predictions to decide whether relaxing the gate is both useful and safe.  Test
labels are never accepted by :func:`adaptive_separability_rescue`; they are
used only by the experiment runner for final frozen evaluation.

Examples
--------
Development (writes the frozen policy manifest after completing 6 human data):

    D:/setup/anaconda/envs/scanvi311/python.exe \
      tools/analysis/adaptive_separability_gate.py --stage human

Confirmatory mouse evaluation (refuses to run if code or policy changed):

    D:/setup/anaconda/envs/scanvi311/python.exe \
      tools/analysis/adaptive_separability_gate.py --stage mouse
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

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
    classification_tables,
    load_config,
    make_run_dir,
    parse_rare_train_size,
)
from tools.analysis.ablation import _conformal_with_overrides  # noqa: E402


HUMAN_CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
]
MOUSE_CONFIGS = [
    "configs/mouse_lung_tms_10x.yaml",
    "configs/mouse_pancreas_tms_10x.yaml",
]
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_RTS = ("0.01", "0.05", "0.10", "all")
VARIANTS = ("fixed_s1.3", "no_sep_gate", "adaptive_sep_gate")


@dataclass(frozen=True)
class AdaptiveGatePolicy:
    """Pre-declared adaptive-gate policy.

    ``min_active_folds`` is deliberately conservative.  A run with fewer than
    three usable folds cannot override the fixed gate.
    """

    alpha: float = DEFAULT_CONFORMAL_ALPHA
    low_sep: float = CONFORMAL_LOW_SEP
    n_splits: int = 5
    min_active_folds: int = 3
    min_val_missed: int = MIN_VAL_MISSED
    bootstrap_reps: int = 2000
    bootstrap_alpha: float = 0.05
    wilson_z: float = 1.96
    rank_grid: tuple[int, ...] = CONFORMAL_RANK_GRID


DEFAULT_POLICY = AdaptiveGatePolicy()


def _rare_f1(y_true: Iterable[str], y_pred: Iterable[str], rare: str) -> float:
    metrics, _ = classification_tables(y_true, y_pred, rare_class=rare)
    return float(metrics["rare_f1"])


def wilson_upper(false_count: int, n_nonrare: int, *, z: float = 1.96) -> float:
    """Wilson upper confidence bound used by the existing rank selector."""

    if n_nonrare <= 0:
        return 1.0
    p = float(false_count) / float(n_nonrare)
    n = float(n_nonrare)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return float(min(1.0, center + half))


def _paired_stratified_bootstrap_delta_f1(
    y_true: Iterable[str],
    baseline_pred: Iterable[str],
    refined_pred: Iterable[str],
    *,
    rare: str,
    reps: int,
    lower_alpha: float,
    seed: int,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for F1 change using two multinomial strata.

    Resampling rare and non-rare cells separately prevents the very large
    non-rare population from randomly erasing the rare support.  The four
    paired prediction states are sampled jointly, preserving correlation
    between baseline and refined predictions.
    """

    y = np.asarray(y_true).astype(str)
    base = np.asarray(baseline_pred).astype(str)
    refined = np.asarray(refined_pred).astype(str)
    if not (len(y) == len(base) == len(refined)):
        raise ValueError("y_true, baseline_pred, and refined_pred must align")
    if reps <= 0:
        raise ValueError("bootstrap reps must be positive")

    is_rare = y == rare
    if int(is_rare.sum()) == 0 or int((~is_rare).sum()) == 0:
        return float("-inf"), float("nan"), float("nan")

    state = ((base == rare).astype(int) << 1) | (refined == rare).astype(int)
    rare_counts = np.bincount(state[is_rare], minlength=4)
    nonrare_counts = np.bincount(state[~is_rare], minlength=4)
    rng = np.random.default_rng(int(seed))
    rare_draw = rng.multinomial(
        int(rare_counts.sum()), rare_counts / rare_counts.sum(), size=int(reps)
    )
    nonrare_draw = rng.multinomial(
        int(nonrare_counts.sum()),
        nonrare_counts / nonrare_counts.sum(),
        size=int(reps),
    )

    # State bits: bit 1 = baseline predicts rare; bit 0 = refined predicts rare.
    base_tp = rare_draw[:, 2] + rare_draw[:, 3]
    refined_tp = rare_draw[:, 1] + rare_draw[:, 3]
    base_fn = rare_draw.sum(axis=1) - base_tp
    refined_fn = rare_draw.sum(axis=1) - refined_tp
    base_fp = nonrare_draw[:, 2] + nonrare_draw[:, 3]
    refined_fp = nonrare_draw[:, 1] + nonrare_draw[:, 3]

    def _f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
        denom = 2.0 * tp + fp + fn
        return np.divide(
            2.0 * tp,
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom > 0,
        )

    delta = _f1(refined_tp, refined_fp, refined_fn) - _f1(
        base_tp, base_fp, base_fn
    )
    return (
        float(np.quantile(delta, lower_alpha)),
        float(np.quantile(delta, 0.5)),
        float(np.quantile(delta, 1.0 - lower_alpha)),
    )


def _safe_series(values: Iterable[str]) -> pd.Series:
    return pd.Series(values).astype(str).reset_index(drop=True)


def adaptive_separability_rescue(
    proto: PrototypeRescuer,
    base_pred_test: pd.Series,
    val_pred_labels: pd.Series,
    val_true: pd.Series,
    val_latent: np.ndarray,
    test_latent: np.ndarray,
    *,
    policy: AdaptiveGatePolicy = DEFAULT_POLICY,
    decision_seed: int = 42,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply fixed rescue or a cross-fitted low-S override.

    The function intentionally has no ``test_true`` argument.  Its decision is
    a pure function of train-derived ``proto``, validation data, and test
    features/predictions.
    """

    base_test = _safe_series(base_pred_test)
    val_pred = _safe_series(val_pred_labels)
    val_y = _safe_series(val_true)
    val_lat = np.asarray(val_latent)
    test_lat = np.asarray(test_latent)
    if len(val_pred) != len(val_y) or len(val_y) != len(val_lat):
        raise ValueError("validation predictions, labels, and latent rows must align")
    if len(base_test) != len(test_lat):
        raise ValueError("test predictions and latent rows must align")

    rare = str(proto.rare_class)
    sep = float(proto.separability_ratio)
    summary: dict[str, Any] = {
        "gate_mode": "fixed_pass" if sep >= policy.low_sep else "adaptive_audit",
        "separability": sep,
        "adaptive_pass": False,
        "adaptive_reason": "",
        "requested_folds": int(policy.n_splits),
        "actual_folds": 0,
        "active_folds": 0,
        "val_missed": int((val_y.eq(rare) & val_pred.ne(rare)).sum()),
        "oof_baseline_f1": float("nan"),
        "oof_refined_f1": float("nan"),
        "oof_delta_f1": float("nan"),
        "oof_delta_f1_lcb": float("nan"),
        "oof_delta_f1_median": float("nan"),
        "oof_delta_f1_ucb": float("nan"),
        "oof_false_rescues": 0,
        "oof_incremental_fpr": float("nan"),
        "oof_ffr_wilson_upper": float("nan"),
        "fold_chosen_ranks": [],
    }

    if sep >= policy.low_sep:
        final, fixed = conformal_rescue(
            proto,
            base_test,
            val_pred,
            val_y,
            val_lat,
            test_lat,
            alpha=policy.alpha,
            rank_grid=policy.rank_grid,
        )
        summary.update(fixed)
        summary.update(
            adaptive_pass=not bool(fixed.get("abstain", False)),
            adaptive_reason="fixed_s_gate_passed",
        )
        return final, summary

    if summary["val_missed"] < policy.min_val_missed:
        summary.update(
            abstain=True,
            reason="adaptive_val_support",
            adaptive_reason=(
                f"val_missed={summary['val_missed']} < {policy.min_val_missed}"
            ),
            chosen_rank=0,
            n_candidate=0,
            n_rescued=0,
        )
        return base_test.copy(), summary

    binary = val_y.eq(rare).astype(int).to_numpy()
    class_counts = np.bincount(binary, minlength=2)
    actual_folds = min(int(policy.n_splits), int(class_counts.min()))
    summary["actual_folds"] = actual_folds
    if actual_folds < policy.min_active_folds:
        summary.update(
            abstain=True,
            reason="adaptive_insufficient_folds",
            adaptive_reason=(
                f"actual_folds={actual_folds} < min_active_folds="
                f"{policy.min_active_folds}"
            ),
            chosen_rank=0,
            n_candidate=0,
            n_rescued=0,
        )
        return base_test.copy(), summary

    splitter = StratifiedKFold(
        n_splits=actual_folds, shuffle=True, random_state=int(decision_seed)
    )
    oof = val_pred.copy()
    active_folds = 0
    fold_ranks: list[int] = []
    fold_reasons: list[str] = []
    for calibrate_idx, audit_idx in splitter.split(np.zeros(len(binary)), binary):
        fold_pred, fold_summary = _conformal_with_overrides(
            proto,
            val_pred.iloc[audit_idx].reset_index(drop=True),
            val_pred.iloc[calibrate_idx].reset_index(drop=True),
            val_y.iloc[calibrate_idx].reset_index(drop=True),
            val_lat[calibrate_idx],
            val_lat[audit_idx],
            low_sep=0.0,
            enforce_necessity=True,
            min_val_missed=policy.min_val_missed,
            rank_grid=policy.rank_grid,
            use_conformal_tau=True,
        )
        oof.iloc[audit_idx] = fold_pred.to_numpy()
        if not bool(fold_summary.get("abstain", False)):
            active_folds += 1
        fold_ranks.append(int(fold_summary.get("chosen_rank", 0)))
        fold_reasons.append(str(fold_summary.get("reason", "")))

    summary["active_folds"] = active_folds
    summary["fold_chosen_ranks"] = fold_ranks
    summary["fold_reasons"] = fold_reasons

    base_f1 = _rare_f1(val_y, val_pred, rare)
    refined_f1 = _rare_f1(val_y, oof, rare)
    delta_f1 = refined_f1 - base_f1
    nonrare = val_y.ne(rare).to_numpy()
    false_mask = (
        oof.ne(val_pred).to_numpy() & oof.eq(rare).to_numpy() & nonrare
    )
    n_nonrare = int(nonrare.sum())
    n_false = int(false_mask.sum())
    ffr = n_false / max(n_nonrare, 1)
    ffr_upper = wilson_upper(n_false, n_nonrare, z=policy.wilson_z)
    lcb, median, ucb = _paired_stratified_bootstrap_delta_f1(
        val_y,
        val_pred,
        oof,
        rare=rare,
        reps=policy.bootstrap_reps,
        lower_alpha=policy.bootstrap_alpha,
        seed=decision_seed,
    )
    summary.update(
        oof_baseline_f1=base_f1,
        oof_refined_f1=refined_f1,
        oof_delta_f1=delta_f1,
        oof_delta_f1_lcb=lcb,
        oof_delta_f1_median=median,
        oof_delta_f1_ucb=ucb,
        oof_false_rescues=n_false,
        oof_incremental_fpr=ffr,
        oof_ffr_wilson_upper=ffr_upper,
    )

    checks = {
        "active_folds": active_folds >= policy.min_active_folds,
        "ffr": ffr_upper <= policy.alpha,
        "gain": lcb > 0.0,
    }
    if not all(checks.values()):
        failed = ",".join(name for name, passed in checks.items() if not passed)
        summary.update(
            abstain=True,
            reason="adaptive_audit_failed",
            adaptive_reason=f"failed:{failed}",
            chosen_rank=0,
            n_candidate=0,
            n_rescued=0,
        )
        return base_test.copy(), summary

    final, full_summary = _conformal_with_overrides(
        proto,
        base_test,
        val_pred,
        val_y,
        val_lat,
        test_lat,
        low_sep=0.0,
        enforce_necessity=True,
        min_val_missed=policy.min_val_missed,
        rank_grid=policy.rank_grid,
        use_conformal_tau=True,
    )
    if bool(full_summary.get("abstain", False)):
        summary.update(full_summary)
        summary.update(
            adaptive_pass=False,
            adaptive_reason=f"full_validation_abstain:{full_summary.get('reason', '')}",
        )
        return base_test.copy(), summary

    summary.update(full_summary)
    summary.update(adaptive_pass=True, adaptive_reason="oof_audit_passed")
    return final, summary


def _latent(frame: pd.DataFrame) -> np.ndarray:
    cols = [c for c in frame.columns if c.startswith("latent_")]
    if not cols:
        raise ValueError("latent file has no latent_* columns")
    return frame[cols].to_numpy()


def _script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _policy_payload(policy: AdaptiveGatePolicy) -> dict[str, Any]:
    payload = asdict(policy)
    payload["rank_grid"] = list(policy.rank_grid)
    return payload


def _stable_decision_seed(dataset: str, seed: int, rts: str) -> int:
    raw = f"{dataset}|{seed}|{rts}|adaptive-sep-v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def _variant_metrics(
    y_true: pd.Series,
    baseline: pd.Series,
    prediction: pd.Series,
    *,
    rare: str,
) -> dict[str, Any]:
    metrics, _ = classification_tables(y_true, prediction, rare_class=rare)
    base_metrics, _ = classification_tables(y_true, baseline, rare_class=rare)
    y = y_true.astype(str).to_numpy()
    b = baseline.astype(str).to_numpy()
    p = prediction.astype(str).to_numpy()
    changed_to_rare = (p != b) & (p == rare)
    false_rescues = int((changed_to_rare & (y != rare)).sum())
    true_rescues = int((changed_to_rare & (y == rare)).sum())
    n_nonrare = int((y != rare).sum())
    ffr = false_rescues / max(n_nonrare, 1)
    return {
        "baseline_rare_f1": float(base_metrics["rare_f1"]),
        "rare_f1": float(metrics["rare_f1"]),
        "rare_recall": float(metrics["rare_recall"]),
        "rare_precision": float(metrics["rare_precision"]),
        "delta_rare_f1": float(metrics["rare_f1"] - base_metrics["rare_f1"]),
        "true_rescues": true_rescues,
        "false_rescues": false_rescues,
        "all_rescues": int(changed_to_rare.sum()),
        "incremental_fpr": float(ffr),
        "alpha_violation": bool(ffr > DEFAULT_CONFORMAL_ALPHA),
    }


def _load_run(config_path: str, seed: int, rts: str, split_mode: str):
    config = load_config(ROOT / config_path)
    rare = str(config["experiment"]["rare_class"])
    run_dir = ROOT / make_run_dir(
        config, split_mode, seed, rare, parse_rare_train_size(rts)
    )
    emb = run_dir / "embeddings"
    required = [
        emb / "train_predictions.csv",
        emb / "train_latent.csv",
        emb / "validation_predictions.csv",
        emb / "validation_latent.csv",
        emb / "test_predictions.csv",
        emb / "test_latent.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(missing))
    train_pred = pd.read_csv(
        required[0], usecols=["true_label", "is_labeled_for_scanvi"]
    )
    train_lat = _latent(pd.read_csv(required[1]))
    val_pred = pd.read_csv(
        required[2], usecols=["true_label", "predicted_label"]
    )
    val_lat = _latent(pd.read_csv(required[3]))
    test_pred = pd.read_csv(
        required[4], usecols=["true_label", "predicted_label"]
    )
    test_lat = _latent(pd.read_csv(required[5]))
    return config, rare, run_dir, train_pred, train_lat, val_pred, val_lat, test_pred, test_lat


def evaluate_run(
    config_path: str,
    seed: int,
    rts: str,
    split_mode: str,
    *,
    policy: AdaptiveGatePolicy,
) -> list[dict[str, Any]]:
    (
        config,
        rare,
        run_dir,
        train_pred,
        train_lat,
        val_pred,
        val_lat,
        test_pred,
        test_lat,
    ) = _load_run(config_path, seed, rts, split_mode)
    dataset = str(config["dataset"]["name"])
    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    proto = PrototypeRescuer(rare)
    proto.fit(
        train_lat,
        train_pred["true_label"].astype(str),
        is_labeled,
    )
    base_test = test_pred["predicted_label"].astype(str).reset_index(drop=True)
    test_true = test_pred["true_label"].astype(str).reset_index(drop=True)
    val_base = val_pred["predicted_label"].astype(str).reset_index(drop=True)
    val_true = val_pred["true_label"].astype(str).reset_index(drop=True)

    fixed_pred, fixed_summary = conformal_rescue(
        proto,
        base_test,
        val_base,
        val_true,
        val_lat,
        test_lat,
        alpha=policy.alpha,
        rank_grid=policy.rank_grid,
    )
    nogate_pred, nogate_summary = _conformal_with_overrides(
        proto,
        base_test,
        val_base,
        val_true,
        val_lat,
        test_lat,
        low_sep=0.0,
        enforce_necessity=True,
        min_val_missed=policy.min_val_missed,
        rank_grid=policy.rank_grid,
        use_conformal_tau=True,
    )
    adaptive_pred, adaptive_summary = adaptive_separability_rescue(
        proto,
        base_test,
        val_base,
        val_true,
        val_lat,
        test_lat,
        policy=policy,
        decision_seed=_stable_decision_seed(dataset, seed, rts),
    )
    by_variant = {
        "fixed_s1.3": (fixed_pred, fixed_summary),
        "no_sep_gate": (nogate_pred, nogate_summary),
        "adaptive_sep_gate": (adaptive_pred, adaptive_summary),
    }
    rows: list[dict[str, Any]] = []
    for variant, (prediction, decision) in by_variant.items():
        row = {
            "dataset": dataset,
            "config": config_path,
            "seed": int(seed),
            "rare_train_size": str(rts),
            "rare_class": rare,
            "split_mode": split_mode,
            "run_dir": str(run_dir.relative_to(ROOT)),
            "variant": variant,
            "separability": float(proto.separability_ratio),
            "status": "success",
            **_variant_metrics(test_true, base_test, prediction, rare=rare),
            "abstain": bool(decision.get("abstain", False)),
            "reason": str(decision.get("reason", "")),
            "chosen_rank": int(decision.get("chosen_rank", 0)),
            "n_candidate": int(decision.get("n_candidate", 0)),
            "n_rescued": int(decision.get("n_rescued", 0)),
        }
        if variant == "adaptive_sep_gate":
            for key in [
                "gate_mode",
                "adaptive_pass",
                "adaptive_reason",
                "requested_folds",
                "actual_folds",
                "active_folds",
                "val_missed",
                "oof_baseline_f1",
                "oof_refined_f1",
                "oof_delta_f1",
                "oof_delta_f1_lcb",
                "oof_delta_f1_median",
                "oof_delta_f1_ucb",
                "oof_false_rescues",
                "oof_incremental_fpr",
                "oof_ffr_wilson_upper",
            ]:
                row[key] = decision.get(key)
            row["fold_chosen_ranks"] = json.dumps(
                decision.get("fold_chosen_ranks", [])
            )
            row["fold_reasons"] = json.dumps(
                decision.get("fold_reasons", []), ensure_ascii=False
            )
        rows.append(row)
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for region, subset in [
        ("ALL", rows),
        ("SCARCE", rows[rows["rare_train_size"].isin(["0.01", "0.05", "0.10"])]),
    ]:
        fixed = subset[subset["variant"] == "fixed_s1.3"].set_index(
            ["dataset", "seed", "rare_train_size"]
        )
        for variant, group in subset.groupby("variant"):
            group = group.copy()
            indexed = group.set_index(["dataset", "seed", "rare_train_size"])
            paired_delta = indexed["rare_f1"] - fixed["rare_f1"]
            output.append(
                {
                    "region": region,
                    "variant": variant,
                    "n": int(len(group)),
                    "rare_f1_mean": float(group["rare_f1"].mean()),
                    "delta_vs_baseline_mean": float(group["delta_rare_f1"].mean()),
                    "delta_vs_fixed_mean": float(paired_delta.mean()),
                    "wins_vs_fixed": int((paired_delta > 1e-12).sum()),
                    "ties_vs_fixed": int((paired_delta.abs() <= 1e-12).sum()),
                    "losses_vs_fixed": int((paired_delta < -1e-12).sum()),
                    "incremental_fpr_max": float(group["incremental_fpr"].max()),
                    "n_alpha_violations": int(group["alpha_violation"].sum()),
                    "n_abstain": int(group["abstain"].sum()),
                }
            )
    return pd.DataFrame(output)


def _write_policy_manifest(path: Path, policy: AdaptiveGatePolicy) -> None:
    payload = {
        "schema": "adaptive-separability-policy-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": _script_sha256(),
        "policy": _policy_payload(policy),
        "development_scope": "6-human batch-heldout, seeds 42/43/44, 4 rts",
        "confirmatory_scope": "2-mouse batch-heldout, seeds 42/43/44, 4 rts",
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _validate_policy_manifest(path: Path, policy: AdaptiveGatePolicy) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"confirmatory run requires frozen policy manifest: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("script_sha256") != _script_sha256():
        raise RuntimeError(
            "adaptive gate script changed after policy freeze; rerun human development "
            "under a new version instead of using the confirmatory mouse stage"
        )
    if payload.get("policy") != _policy_payload(policy):
        raise RuntimeError("adaptive gate policy differs from frozen manifest")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["human", "mouse"], required=True)
    parser.add_argument("--split-mode", default="batch_heldout")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rts", nargs="+", default=list(DEFAULT_RTS))
    parser.add_argument("--only-low-sep", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument(
        "--output-dir", default="results/adaptive_separability_gate/v1"
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    policy = AdaptiveGatePolicy(bootstrap_reps=int(args.bootstrap_reps))
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_manifest = out_dir / "policy_manifest.json"
    if args.stage == "mouse":
        _validate_policy_manifest(policy_manifest, policy)

    configs = HUMAN_CONFIGS if args.stage == "human" else MOUSE_CONFIGS
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for config_path in configs:
        for seed in args.seeds:
            for rts in args.rts:
                try:
                    unit_rows = evaluate_run(
                        config_path,
                        int(seed),
                        str(rts),
                        args.split_mode,
                        policy=policy,
                    )
                    if args.only_low_sep and unit_rows[0]["separability"] >= policy.low_sep:
                        continue
                    rows.extend(unit_rows)
                    adaptive = unit_rows[2]
                    print(
                        f"[{args.stage}] {adaptive['dataset']} seed={seed} rts={rts} "
                        f"S={adaptive['separability']:.3f} "
                        f"fixed={unit_rows[0]['rare_f1']:.3f} "
                        f"adaptive={adaptive['rare_f1']:.3f} "
                        f"pass={adaptive.get('adaptive_pass', False)} "
                        f"reason={adaptive.get('adaptive_reason', '')}"
                    )
                except FileNotFoundError as exc:
                    missing.append(
                        {
                            "config": config_path,
                            "seed": int(seed),
                            "rare_train_size": str(rts),
                            "error": str(exc),
                        }
                    )
                    if not args.allow_missing:
                        raise

    if not rows:
        raise RuntimeError("no experiment rows were collected")
    frame = pd.DataFrame(rows)
    summary = _summarize(frame)
    audit = frame[frame["variant"] == "adaptive_sep_gate"].copy()
    prefix = f"{args.stage}{'_low_s' if args.only_low_sep else ''}"
    frame.to_csv(out_dir / f"{prefix}_run_level.csv", index=False)
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    audit.to_csv(out_dir / f"{prefix}_decision_audit.csv", index=False)
    if missing:
        pd.DataFrame(missing).to_csv(out_dir / f"{prefix}_missing.csv", index=False)

    run_manifest = {
        "schema": "adaptive-separability-run-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": args.stage,
        "split_mode": args.split_mode,
        "seeds": list(args.seeds),
        "rts": list(args.rts),
        "only_low_sep": bool(args.only_low_sep),
        "script_sha256": _script_sha256(),
        "policy": _policy_payload(policy),
        "n_rows": int(len(frame)),
        "n_missing": int(len(missing)),
    }
    (out_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.stage == "human" and not args.only_low_sep:
        _write_policy_manifest(policy_manifest, policy)
        print(f"[frozen] {policy_manifest}")
    print(summary.to_string(index=False))
    print(f"[saved] {out_dir / f'{prefix}_run_level.csv'}")


if __name__ == "__main__":
    main()
