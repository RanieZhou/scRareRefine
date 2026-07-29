"""Decision-seed stability audit for the frozen adaptive separability gate.

This script is deliberately separate from ``adaptive_separability_gate.py``.
It verifies the frozen policy/script manifest, finds every S<1.3 unit across
the 8-dataset batch-heldout benchmark, and repeats the complete validation
cross-fitting decision under 20 deterministic fold seeds.  Test labels are
never loaded; a single test feature/prediction is used only because the frozen
API applies the full-validation rule after the OOF decision passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rescue import PrototypeRescuer  # noqa: E402
from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402
from tools.analysis import adaptive_separability_gate as gate  # noqa: E402


DEFAULT_OUTPUT_DIR = (
    ROOT / "results" / "adaptive_separability_gate" / "v1" / "stability_20seeds"
)
FROZEN_MANIFEST = (
    ROOT / "results" / "adaptive_separability_gate" / "v1" / "policy_manifest.json"
)
PASS_RATE_HIGH = 0.80
PASS_RATE_LOW = 0.20


def repeat_decision_seed(dataset: str, seed: int, rts: str, repeat: int) -> int:
    """Return a deterministic, unit-specific 32-bit seed for a repeat."""

    raw = f"{dataset}|{seed}|{rts}|adaptive-sep-v1-stability|{repeat}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def classify_stability(original_pass: bool, pass_rate: float) -> tuple[str, bool]:
    """Classify a unit without changing the frozen decision rule."""

    if pass_rate >= PASS_RATE_HIGH:
        band = "stable_pass"
    elif pass_rate <= PASS_RATE_LOW:
        band = "stable_reject"
    else:
        band = "unstable"
    consistent = (original_pass and band == "stable_pass") or (
        (not original_pass) and band == "stable_reject"
    )
    return band, bool(consistent)


def _latent(path: Path, *, nrows: int | None = None) -> np.ndarray:
    frame = pd.read_csv(path, nrows=nrows)
    cols = [column for column in frame.columns if column.startswith("latent_")]
    if not cols:
        raise ValueError(f"latent file has no latent_* columns: {path}")
    return frame[cols].to_numpy()


def _load_decision_inputs(config_path: str, seed: int, rts: str) -> dict[str, Any]:
    """Load train/validation labels and one test feature, never test labels."""

    config = load_config(ROOT / config_path)
    dataset = str(config["dataset"]["name"])
    rare = str(config["experiment"]["rare_class"])
    run_dir = ROOT / make_run_dir(
        config,
        "batch_heldout",
        int(seed),
        rare,
        parse_rare_train_size(rts),
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
    train_lat = _latent(required[1])
    val_pred = pd.read_csv(
        required[2], usecols=["true_label", "predicted_label"]
    )
    val_lat = _latent(required[3])
    # Deliberately exclude true_label from the stability audit.
    test_pred = pd.read_csv(required[4], usecols=["predicted_label"], nrows=1)
    test_lat = _latent(required[5], nrows=1)
    return {
        "dataset": dataset,
        "rare": rare,
        "run_dir": run_dir,
        "train_pred": train_pred,
        "train_lat": train_lat,
        "val_pred": val_pred,
        "val_lat": val_lat,
        "test_pred": test_pred,
        "test_lat": test_lat,
    }


def _fit_proto(inputs: dict[str, Any]) -> PrototypeRescuer:
    proto = PrototypeRescuer(str(inputs["rare"]))
    train_pred = inputs["train_pred"]
    proto.fit(
        inputs["train_lat"],
        train_pred["true_label"].astype(str),
        train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
    )
    return proto


def _run_decision(
    inputs: dict[str, Any], proto: PrototypeRescuer, *, decision_seed: int
) -> dict[str, Any]:
    val_pred = inputs["val_pred"]
    _, summary = gate.adaptive_separability_rescue(
        proto,
        inputs["test_pred"]["predicted_label"].astype(str),
        val_pred["predicted_label"].astype(str),
        val_pred["true_label"].astype(str),
        inputs["val_lat"],
        inputs["test_lat"],
        policy=gate.DEFAULT_POLICY,
        decision_seed=int(decision_seed),
    )
    return summary


def _finite_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_q95": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(data)),
        f"{prefix}_q05": float(np.quantile(data, 0.05)),
        f"{prefix}_median": float(np.median(data)),
        f"{prefix}_q95": float(np.quantile(data, 0.95)),
        f"{prefix}_max": float(np.max(data)),
    }


def summarize_repeats(rows: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    keys = ["dataset", "seed", "rare_train_size", "separability", "original_pass"]
    for key, group in rows.groupby(keys, dropna=False, sort=False):
        dataset, seed, rts, separability, original_pass = key
        pass_rate = float(group["adaptive_pass"].mean())
        stability_band, consistent = classify_stability(bool(original_pass), pass_rate)
        reasons = group["adaptive_reason"].value_counts().to_dict()
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": int(seed),
            "rare_train_size": str(rts),
            "separability": float(separability),
            "original_pass": bool(original_pass),
            "n_repeats": int(len(group)),
            "pass_count": int(group["adaptive_pass"].sum()),
            "pass_rate": pass_rate,
            "stability_band": stability_band,
            "consistent_with_frozen": consistent,
            "reason_counts": json.dumps(reasons, ensure_ascii=False, sort_keys=True),
            "active_folds_min": int(group["active_folds"].min()),
            "active_folds_median": float(group["active_folds"].median()),
            "active_folds_max": int(group["active_folds"].max()),
        }
        row.update(_finite_stats(group["oof_ffr_wilson_upper"], "wilson_ucb"))
        row.update(_finite_stats(group["oof_delta_f1_lcb"], "delta_f1_lcb"))
        output.append(row)
    return pd.DataFrame(output).sort_values(
        ["dataset", "seed", "rare_train_size"], kind="stable"
    )


def _unit_label(row: pd.Series) -> str:
    return f"{row['dataset']} | s{int(row['seed'])} | {row['rare_train_size']}"


def plot_stability(rows: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    order = summary.copy().reset_index(drop=True)
    order["unit"] = order.apply(_unit_label, axis=1)
    rows = rows.copy()
    rows["unit"] = rows.apply(_unit_label, axis=1)
    categories = order["unit"].tolist()
    ymap = {unit: index for index, unit in enumerate(categories)}
    y = np.arange(len(categories))

    fig, axes = plt.subplots(1, 4, figsize=(18, max(7, 0.42 * len(categories))))
    colors = np.where(order["original_pass"], "#1b9e77", "#6c757d")

    axes[0].barh(y, order["pass_rate"], color=colors, alpha=0.9)
    axes[0].axvspan(PASS_RATE_LOW, PASS_RATE_HIGH, color="#f3d7a6", alpha=0.45)
    axes[0].axvline(PASS_RATE_LOW, color="#b36b00", linestyle="--", linewidth=1)
    axes[0].axvline(PASS_RATE_HIGH, color="#b36b00", linestyle="--", linewidth=1)
    axes[0].set(xlim=(0, 1), xlabel="Adaptive-gate pass rate", title="a  Decision stability")
    axes[0].set_yticks(y, categories, fontsize=8)

    for column, axis, title, threshold, xlim in [
        ("active_folds", axes[1], "b  Active folds", 3.0, (-0.2, 5.2)),
        ("oof_ffr_wilson_upper", axes[2], "c  Wilson UCB", 0.01, None),
        ("oof_delta_f1_lcb", axes[3], "d  OOF ΔF1 LCB", 0.0, None),
    ]:
        for _, point in rows.iterrows():
            value = float(point[column])
            if np.isfinite(value):
                axis.scatter(value, ymap[point["unit"]], s=13, color="#377eb8", alpha=0.5)
        axis.axvline(threshold, color="#d62728", linestyle="--", linewidth=1)
        axis.set(xlabel=title.split("  ", 1)[-1], title=title)
        axis.set_yticks(y, [])
        if xlim is not None:
            axis.set_xlim(*xlim)

    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.invert_yaxis()
    fig.suptitle("Frozen adaptive separability gate: 20 decision-seed repeats", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
    stable = int(summary["consistent_with_frozen"].sum())
    total = int(len(summary))
    unstable = summary[~summary["consistent_with_frozen"]]
    lines = [
        "# Adaptive Gate Decision-Seed Stability",
        "",
        "Frozen v1 was rerun under 20 deterministic fold/bootstrap seeds for every batch-heldout unit with `S < 1.3`.",
        "Test labels were not loaded and no frozen rule or threshold was modified.",
        "",
        f"- Low-S units: {total}",
        f"- Consistent with frozen decision: {stable}/{total}",
        f"- Unstable/inconsistent units: {total - stable}",
        f"- Stable-pass definition: pass rate >= {PASS_RATE_HIGH:.2f}",
        f"- Stable-reject definition: pass rate <= {PASS_RATE_LOW:.2f}",
        "",
        "## Unit-level decisions",
        "",
        "| dataset | seed | rts | S | frozen | pass rate | band | consistent | active folds | Wilson UCB q05–q95 | ΔF1 LCB q05–q95 |",
        "|---|---:|---:|---:|---|---:|---|---|---|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {dataset} | {seed} | {rts} | {sep:.3f} | {frozen} | {rate:.2f} | "
            "{band} | {consistent} | {afmin}/{afmed:.1f}/{afmax} | {w05:.4f}–{w95:.4f} | "
            "{d05:.4f}–{d95:.4f} |".format(
                dataset=row["dataset"],
                seed=int(row["seed"]),
                rts=row["rare_train_size"],
                sep=row["separability"],
                frozen="pass" if row["original_pass"] else "reject",
                rate=row["pass_rate"],
                band=row["stability_band"],
                consistent="yes" if row["consistent_with_frozen"] else "no",
                afmin=int(row["active_folds_min"]),
                afmed=row["active_folds_median"],
                afmax=int(row["active_folds_max"]),
                w05=row["wilson_ucb_q05"],
                w95=row["wilson_ucb_q95"],
                d05=row["delta_f1_lcb_q05"],
                d95=row["delta_f1_lcb_q95"],
            )
        )
    if not unstable.empty:
        lines.extend(
            [
                "",
                "## Units requiring review",
                "",
                ", ".join(unstable.apply(_unit_label, axis=1).tolist()),
            ]
        )
    (out_dir / "stability_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR.relative_to(ROOT)))
    args = parser.parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2")

    frozen = gate._validate_policy_manifest(FROZEN_MANIFEST, gate.DEFAULT_POLICY)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    configs = [*gate.HUMAN_CONFIGS, *gate.MOUSE_CONFIGS]
    for config_path in configs:
        for seed in gate.DEFAULT_SEEDS:
            for rts in gate.DEFAULT_RTS:
                inputs = _load_decision_inputs(config_path, int(seed), str(rts))
                proto = _fit_proto(inputs)
                separability = float(proto.separability_ratio)
                if separability >= gate.DEFAULT_POLICY.low_sep:
                    continue
                dataset = str(inputs["dataset"])
                frozen_decision = _run_decision(
                    inputs,
                    proto,
                    decision_seed=gate._stable_decision_seed(dataset, int(seed), str(rts)),
                )
                print(
                    f"[unit] {dataset} seed={seed} rts={rts} S={separability:.3f} "
                    f"frozen_pass={bool(frozen_decision.get('adaptive_pass', False))}",
                    flush=True,
                )
                for repeat in range(int(args.repeats)):
                    decision_seed = repeat_decision_seed(dataset, int(seed), str(rts), repeat)
                    decision = _run_decision(
                        inputs, proto, decision_seed=decision_seed
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": int(seed),
                            "rare_train_size": str(rts),
                            "separability": separability,
                            "repeat": int(repeat),
                            "decision_seed": int(decision_seed),
                            "original_pass": bool(
                                frozen_decision.get("adaptive_pass", False)
                            ),
                            "adaptive_pass": bool(decision.get("adaptive_pass", False)),
                            "adaptive_reason": str(decision.get("adaptive_reason", "")),
                            "actual_folds": int(decision.get("actual_folds", 0)),
                            "active_folds": int(decision.get("active_folds", 0)),
                            "val_missed": int(decision.get("val_missed", 0)),
                            "oof_delta_f1": decision.get("oof_delta_f1"),
                            "oof_delta_f1_lcb": decision.get("oof_delta_f1_lcb"),
                            "oof_ffr_wilson_upper": decision.get(
                                "oof_ffr_wilson_upper"
                            ),
                            "oof_false_rescues": int(
                                decision.get("oof_false_rescues", 0)
                            ),
                        }
                    )

    repeats = pd.DataFrame(rows)
    if repeats.empty:
        raise RuntimeError("No S<1.3 units were found")
    summary = summarize_repeats(repeats)
    repeats.to_csv(out_dir / "stability_repeats.csv", index=False)
    summary.to_csv(out_dir / "stability_summary.csv", index=False)
    plot_stability(repeats, summary, out_dir / "stability_summary.png")
    _write_report(summary, out_dir)

    manifest = {
        "schema": "adaptive-gate-stability-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repeats": int(args.repeats),
        "scope": "8 datasets, batch-heldout, all S<1.3 units",
        "split_mode": "batch_heldout",
        "test_labels_loaded": False,
        "stability_thresholds": {
            "stable_pass_min": PASS_RATE_HIGH,
            "stable_reject_max": PASS_RATE_LOW,
        },
        "frozen_policy_manifest": str(FROZEN_MANIFEST.relative_to(ROOT)),
        "frozen_script_sha256": frozen["script_sha256"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    consistent = int(summary["consistent_with_frozen"].sum())
    print(
        f"[done] low_s_units={len(summary)} consistent={consistent}/{len(summary)} "
        f"output={out_dir.relative_to(ROOT)}",
        flush=True,
    )


if __name__ == "__main__":
    main()

