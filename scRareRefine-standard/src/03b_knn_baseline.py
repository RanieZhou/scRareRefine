"""Stage 3b: k-NN baseline classifier on latent embedding.

Uses labeled training cells as reference; predicts test cell labels by
k-nearest-neighbor majority vote in latent space.  Intended as a comparison
baseline against the prototype-based rescue pipeline.

Reads:
    outputs/{dataset}/{run_id}/embeddings/

Writes:
    outputs/{dataset}/{run_id}/knn/
        test_predictions.csv    predicted_label, true_label, knn_predicted_label per test cell
        test_metrics.csv        rare_f1, rare_precision, rare_recall, overall_accuracy

Usage:
    python src/03b_knn_baseline.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from utils import classification_tables, load_config, make_run_dir, parse_rare_train_size, read_table, write_table


def _plot(metrics_df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    cols = ["rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    labels = {"baseline": "Baseline\n(scANVI)"}
    colors = {"baseline": "#8da0cb"}
    for m in metrics_df["method"]:
        if m != "baseline":
            labels[m] = m.replace("_", " ")
            colors[m] = "#66c2a5"

    methods = metrics_df["method"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    seed_val = out_path.parent.parent.name  # run_id as title fallback
    fig.suptitle(f"kNN vs Baseline  |  {rare_class}", fontsize=10, fontweight="bold")
    for ax, col in zip(axes, cols):
        vals = [float(metrics_df.loc[metrics_df["method"] == m, col].iloc[0])
                if col in metrics_df.columns else 0.0 for m in methods]
        bars = ax.bar(range(len(methods)), vals,
                      color=[colors.get(m, "#aaa") for m in methods],
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


def knn_predict(
    query_latent: np.ndarray,
    *,
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    labeled = np.asarray(reference_is_labeled, dtype=bool)
    ref = np.asarray(reference_latent, dtype=float)[labeled]
    labs = pd.Series(reference_labels).astype(str).to_numpy()[labeled]

    clf = KNeighborsClassifier(n_neighbors=min(k, len(ref)), metric="euclidean", n_jobs=1)
    clf.fit(ref, labs)
    return clf.predict(np.asarray(query_latent, dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b: kNN baseline on latent embedding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--k", type=int, default=15, help="Number of nearest neighbors (default: 15)")
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    knn_dir = run_dir / "knn"

    train_pred = read_table(emb_dir / "train_predictions.csv")
    train_latent = read_table(emb_dir / "train_latent.csv")
    latent_cols = [c for c in train_latent.columns if c.startswith("latent_")]
    ref_lat = train_latent[latent_cols].to_numpy(dtype=float)

    for split_name in ["test"]:
        pred = read_table(emb_dir / f"{split_name}_predictions.csv")
        latent = read_table(emb_dir / f"{split_name}_latent.csv")
        query_lat = latent[latent_cols].to_numpy(dtype=float)

        knn_preds = knn_predict(
            query_lat,
            reference_latent=ref_lat,
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            k=args.k,
        )

        out_pred = pd.DataFrame({
            "cell_id": pred["cell_id"].astype(str) if "cell_id" in pred.columns else np.arange(len(pred)),
            "true_label": pred["true_label"].astype(str),
            "baseline_predicted": pred["predicted_label"].astype(str),
            "knn_predicted": knn_preds,
        })

        knn_metrics, _ = classification_tables(
            out_pred["true_label"], out_pred["knn_predicted"], rare_class=rare_class
        )
        baseline_metrics, _ = classification_tables(
            out_pred["true_label"], out_pred["baseline_predicted"], rare_class=rare_class
        )
        metrics_df = pd.DataFrame([
            {"method": "baseline", **baseline_metrics},
            {"method": f"knn_k{args.k}", "k": args.k, **knn_metrics},
        ])

        write_table(out_pred, knn_dir / f"{split_name}_predictions.csv")
        write_table(metrics_df, knn_dir / f"{split_name}_metrics.csv")
        _plot(metrics_df, knn_dir / "comparison.png", rare_class=rare_class)

        tp = ((out_pred["true_label"] == rare_class) & (out_pred["knn_predicted"] == rare_class)).sum()
        fn = ((out_pred["true_label"] == rare_class) & (out_pred["knn_predicted"] != rare_class)).sum()
        fp = ((out_pred["true_label"] != rare_class) & (out_pred["knn_predicted"] == rare_class)).sum()
        print(f"  baseline: F1={baseline_metrics.get('rare_f1', 0):.3f}")
        print(f"  kNN k={args.k}: tp={tp} fn={fn} fp={fp}  F1={knn_metrics.get('rare_f1', 0):.3f}")

    print(f"Done. kNN results in {knn_dir}")


if __name__ == "__main__":
    main()
