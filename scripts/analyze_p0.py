from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from scrare_refine.config import load_config, output_dir
from scrare_refine.io import read_table, write_table
from scrare_refine.metrics import classification_tables, topk_review_recall
from scrare_refine.output_layout import (
    artifact_path,
    existing_table_path,
    legacy_or_artifact_path,
    root_figure_path,
    root_table_path,
)
from scrare_refine.prototype import prototype_scores


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    if not runs.exists():
        return []
    return sorted(path for path in runs.iterdir() if path.is_dir())


def _load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return read_table(existing_table_path(run_dir, "scanvi_predictions.parquet")), read_table(
        existing_table_path(run_dir, "scanvi_latent.csv")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate P0 scANVI outputs and review metrics.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    exp = config["experiment"]
    analysis = config.get("analysis", {})
    rare_class = exp["rare_class"]
    root = output_dir(config)
    root.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    per_class_rows = []
    rare_error_rows = []
    uncertainty_rows = []
    prototype_rows = []
    all_true = []
    all_pred = []

    for run_dir in _run_dirs(root):
        pred, latent_df = _load_run(run_dir)
        seed = int(pred["seed"].iloc[0])
        rare_train_size = str(pred["rare_train_size"].iloc[0])

        overall, per_class = classification_tables(
            pred["true_label"], pred["predicted_label"], rare_class=rare_class
        )
        resources_path = run_dir / "run_resources.csv"
        metrics_path = run_dir / "scanvi_metrics.csv"
        if resources_path.exists():
            resources = pd.read_csv(resources_path).iloc[0].to_dict()
            overall["wall_time_seconds"] = resources.get("wall_time_seconds")
            overall["peak_rss_mb"] = resources.get("peak_rss_mb")
        elif metrics_path.exists():
            prior_metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
            overall["wall_time_seconds"] = prior_metrics.get("wall_time_seconds")
            overall["peak_rss_mb"] = prior_metrics.get("peak_rss_mb")
        overall.update({"seed": seed, "rare_train_size": rare_train_size, "run": run_dir.name})
        metrics_rows.append(overall)
        per_class.insert(0, "seed", seed)
        per_class.insert(1, "rare_train_size", rare_train_size)
        per_class_rows.append(per_class)

        rare_errors = (pred["true_label"] == rare_class) & (pred["predicted_label"] != rare_class)
        rare_error_summary = (
            pred.loc[rare_errors, "predicted_label"]
            .value_counts()
            .rename_axis("predicted_label")
            .reset_index(name="n_errors")
        )
        rare_error_summary.insert(0, "seed", seed)
        rare_error_summary.insert(1, "rare_train_size", rare_train_size)
        rare_error_rows.append(rare_error_summary)

        review = topk_review_recall(
            rare_errors,
            pred["entropy"],
            ks=[float(k) for k in analysis.get("review_fractions", [0.01, 0.05, 0.10, 0.20])],
        )
        review.insert(0, "seed", seed)
        review.insert(1, "rare_train_size", rare_train_size)
        review.insert(2, "risk_score", "entropy")
        uncertainty_rows.append(review)

        latent_cols = [col for col in latent_df.columns if col.startswith("latent_")]
        proto = prototype_scores(
            latent_df[latent_cols].to_numpy(),
            true_labels=pred["true_label"],
            predicted_labels=pred["predicted_label"],
            is_labeled=pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            rare_class=rare_class,
            margin=pred["margin"].to_numpy(),
        )
        proto_events = rare_errors.to_numpy()
        proto_risk = proto[f"d_pred_minus_d_{rare_class}"].fillna(-np.inf).to_numpy()
        proto_review = topk_review_recall(
            proto_events,
            proto_risk,
            ks=[float(k) for k in analysis.get("review_fractions", [0.01, 0.05, 0.10, 0.20])],
        )
        proto_review.insert(0, "seed", seed)
        proto_review.insert(1, "rare_train_size", rare_train_size)
        proto_review.insert(2, "risk_score", f"d_pred_minus_d_{rare_class}")
        prototype_rows.append(proto_review)

        proto_out = pd.concat([pred[["cell_id", "true_label", "predicted_label", "margin"]], proto], axis=1)
        write_table(proto_out, artifact_path(run_dir, "prototype_scores.csv"))
        all_true.extend(pred["true_label"].astype(str).tolist())
        all_pred.extend(pred["predicted_label"].astype(str).tolist())

    if not metrics_rows:
        raise FileNotFoundError(f"No run outputs found under {root / 'runs'}")

    write_table(pd.DataFrame(metrics_rows), root_table_path(root, "scanvi_metrics.csv"))
    write_table(pd.concat(per_class_rows, ignore_index=True), root_table_path(root, "per_class_metrics.csv"))
    write_table(pd.concat(rare_error_rows, ignore_index=True), root_table_path(root, "rare_error_analysis.csv"))
    write_table(pd.concat(uncertainty_rows, ignore_index=True), root_table_path(root, "uncertainty_review_recall.csv"))
    write_table(pd.concat(prototype_rows, ignore_index=True), root_table_path(root, "prototype_review_recall.csv"))

    labels = sorted(set(all_true) | set(all_pred))
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.8), max(6, len(labels) * 0.8)))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    root_figure_path(root, "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(root_figure_path(root, "confusion_matrix.png"), dpi=180)
    plt.close(fig)

    print(f"Wrote aggregate analysis to {root}")


if __name__ == "__main__":
    main()
