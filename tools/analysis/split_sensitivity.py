"""Summarize cell-stratified split sensitivity results.

The script compares the primary batch-heldout cache against the
cell-stratified sensitivity cache for scANVI and scRareRefine. It is read-only
with respect to ``outputs/`` and writes tables under ``results/split_sensitivity``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rescue import CONFORMAL_LOW_SEP, DEFAULT_CONFORMAL_ALPHA  # noqa: E402
from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402

CONFIGS = [
    ("configs/immune_dc.yaml", "ASDC"),
    ("configs/pancreas_baron.yaml", "gamma"),
    ("configs/pancreas_integrated.yaml", "endothelial"),
    ("configs/tabula_lung_endo.yaml", "endothelial cell of lymphatic vessel"),
    ("configs/tabula_sapiens_stomach.yaml", "mast cell"),
    ("configs/tabula_small_intestine.yaml", "intestinal tuft cell"),
]
RTS = ["0.01", "0.05", "0.10", "all"]
OUT_DIR = ROOT / "results" / "split_sensitivity"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "None"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def _metric_row(run_dir: Path, *, split_mode: str, dataset: str, seed: int, rts: str, rare_class: str) -> dict[str, Any]:
    metrics_path = run_dir / "metrics" / "final_metrics.csv"
    if not metrics_path.exists():
        return {
            "dataset": dataset,
            "split_mode": split_mode,
            "seed": seed,
            "rare_train_size": rts,
            "rare_class": rare_class,
            "status": "missing",
        }

    metrics = pd.read_csv(metrics_path)
    by_method = {str(row["method"]): row for _, row in metrics.iterrows()}
    baseline = by_method.get("baseline")
    refined = by_method.get("scRareRefine")
    if baseline is None or refined is None:
        return {
            "dataset": dataset,
            "split_mode": split_mode,
            "seed": seed,
            "rare_train_size": rts,
            "rare_class": rare_class,
            "status": "missing_method",
        }

    emb = run_dir / "embeddings"
    split_counts = {}
    rare_counts = {}
    for split in ["train", "validation", "test"]:
        path = emb / f"{split}_predictions.csv"
        if path.exists():
            pred = pd.read_csv(path, usecols=["cell_id", "true_label"])
            split_counts[split] = len(pred)
            rare_counts[split] = int((pred["true_label"].astype(str) == rare_class).sum())
        else:
            split_counts[split] = 0
            rare_counts[split] = 0
    total = sum(split_counts.values())

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    sep = float(refined.get("sep", float("nan"))) if "sep" in refined else float("nan")
    rescue_ffr = float(refined.get("major_to_rare_false_rescue_rate", 0.0))
    n_rescued = int(refined.get("n_rescued", 0))
    n_false = int(refined.get("n_false_rescues", 0))

    return {
        "dataset": dataset,
        "split_mode": split_mode,
        "seed": seed,
        "rare_train_size": rts,
        "rare_class": rare_class,
        "status": "ok",
        "n_train": split_counts["train"],
        "n_val": split_counts["validation"],
        "n_test": split_counts["test"],
        "train_pct": split_counts["train"] / total if total else float("nan"),
        "val_pct": split_counts["validation"] / total if total else float("nan"),
        "test_pct": split_counts["test"] / total if total else float("nan"),
        "rare_train": rare_counts["train"],
        "rare_val": rare_counts["validation"],
        "rare_test": rare_counts["test"],
        "split_hash": manifest.get("split_hash", ""),
        "git_sha": manifest.get("git_sha", ""),
        "scANVI_f1": float(baseline["rare_f1"]),
        "scANVI_recall": float(baseline["rare_recall"]),
        "scANVI_precision": float(baseline["rare_precision"]),
        "scRareRefine_f1": float(refined["rare_f1"]),
        "scRareRefine_recall": float(refined["rare_recall"]),
        "scRareRefine_precision": float(refined["rare_precision"]),
        "delta_f1": float(refined["rare_f1"]) - float(baseline["rare_f1"]),
        "delta_recall": float(refined["rare_recall"]) - float(baseline["rare_recall"]),
        "rescue_ffr": rescue_ffr,
        "n_rescued": n_rescued,
        "n_false_rescue": n_false,
        "abstained": bool(n_rescued == 0),
        "sep": sep,
    }


def collect(seed: int) -> pd.DataFrame:
    rows = []
    for cfg_path, rare_class in CONFIGS:
        config = load_config(ROOT / cfg_path)
        dataset = config["dataset"]["name"]
        for split_mode in ["batch_heldout", "cell_stratified"]:
            for rts in RTS:
                size = parse_rare_train_size(rts)
                run_dir = ROOT / make_run_dir(config, split_mode, seed, rare_class, size)
                rows.append(
                    _metric_row(
                        run_dir,
                        split_mode=split_mode,
                        dataset=dataset,
                        seed=seed,
                        rts=rts,
                        rare_class=rare_class,
                    )
                )
    return pd.DataFrame(rows)


def write_report(df: pd.DataFrame, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / f"cell_stratified_seed{seed}_summary.csv"
    compare_path = OUT_DIR / f"batch_vs_cell_stratified_seed{seed}.csv"
    report_path = OUT_DIR / f"split_sensitivity_seed{seed}_report.md"

    df.to_csv(summary_path, index=False)
    ok = df[df["status"] == "ok"].copy()
    pivot = ok.pivot_table(
        index=["dataset", "seed", "rare_train_size", "rare_class"],
        columns="split_mode",
        values=[
            "scANVI_f1",
            "scANVI_recall",
            "scRareRefine_f1",
            "scRareRefine_recall",
            "delta_f1",
            "rescue_ffr",
            "train_pct",
            "val_pct",
            "test_pct",
            "rare_train",
            "rare_val",
            "rare_test",
            "n_rescued",
            "n_false_rescue",
        ],
    )
    pivot.columns = [f"{metric}_{split}" for metric, split in pivot.columns]
    pivot = pivot.reset_index()
    pivot.to_csv(compare_path, index=False)

    scarce = ok[ok["rare_train_size"].isin(["0.01", "0.05", "0.10"])]
    agg = (
        scarce.groupby("split_mode")
        .agg(
            n=("delta_f1", "size"),
            scANVI_f1=("scANVI_f1", "mean"),
            scRareRefine_f1=("scRareRefine_f1", "mean"),
            delta_f1=("delta_f1", "mean"),
            scANVI_recall=("scANVI_recall", "mean"),
            scRareRefine_recall=("scRareRefine_recall", "mean"),
            delta_recall=("delta_recall", "mean"),
            rescue_ffr_max=("rescue_ffr", "max"),
            n_abstain=("abstained", "sum"),
        )
        .reset_index()
    )

    cell = scarce[scarce["split_mode"] == "cell_stratified"]
    wins = int((cell["delta_f1"] > 1e-12).sum())
    ties = int((cell["delta_f1"].abs() <= 1e-12).sum())
    losses = int((cell["delta_f1"] < -1e-12).sum())
    missing = df[df["status"] != "ok"]

    lines = [
        f"# Split Sensitivity Report (seed={seed})",
        "",
        "Scope: scANVI + scRareRefine only. Primary claims remain based on `batch_heldout`; `cell_stratified` is a supplementary easier-setting sensitivity analysis.",
        "",
        f"- Conformal alpha: {DEFAULT_CONFORMAL_ALPHA}",
        f"- Low-separability gate: {CONFORMAL_LOW_SEP}",
        f"- Completed rows: {len(ok)}/{len(df)}",
        f"- Missing/non-ok rows: {len(missing)}",
        "",
        "## Scarce-region aggregate",
        "",
        _markdown_table(agg),
        "",
        "## Cell-stratified paired delta vs scANVI",
        "",
        f"- scarce-region wins/ties/losses: {wins}/{ties}/{losses}",
        f"- scarce-region mean delta F1: {cell['delta_f1'].mean():.4f}" if not cell.empty else "- no cell-stratified scarce rows",
        f"- scarce-region max rescue FFR: {cell['rescue_ffr'].max():.6f}" if not cell.empty else "- no cell-stratified scarce rows",
        "",
        "## Output files",
        "",
        f"- `{summary_path.relative_to(ROOT)}`",
        f"- `{compare_path.relative_to(ROOT)}`",
    ]
    if not missing.empty:
        lines.extend(["", "## Non-ok rows", "", _markdown_table(missing)])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize split sensitivity results.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = collect(args.seed)
    write_report(df, args.seed)


if __name__ == "__main__":
    main()
