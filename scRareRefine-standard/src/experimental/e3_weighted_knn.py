"""E3: Class-balanced weighted KNN in latent space.

Innovation: weight each neighbor's vote by 1 / (class_count * distance^2).
This gives rare-class neighbors disproportionately high weight.

Compare k ∈ {5, 15, 30} with:
  - standard kNN (majority vote)
  - distance-weighted kNN
  - class-balanced distance-weighted kNN (our innovation)

Run on: cDC1 rare5, ASDC rare5, epsilon rare20 (seed42).

Usage:
    python src/experimental/e3_weighted_knn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import _latent

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e3_weighted_knn"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_VALUES = [5, 15, 30]

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
]


def knn_predict(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    test_z: np.ndarray,
    k: int,
    mode: str,  # "standard", "distance_weighted", "class_balanced"
    class_counts: dict[str, int] | None = None,
) -> np.ndarray:
    """Predict labels using kNN with different weighting schemes."""
    k_eff = min(k, len(train_z))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="ball_tree", n_jobs=-1)
    nbrs.fit(train_z)
    nn_dists_all, nn_idx_all = nbrs.kneighbors(test_z)

    preds = []
    eps = 1e-10
    for i in range(len(test_z)):
        nn_idx    = nn_idx_all[i]
        nn_labels = train_labels[nn_idx]
        nn_dists  = nn_dists_all[i]

        if mode == "standard":
            counts = Counter(nn_labels)
            preds.append(max(counts, key=counts.get))

        elif mode == "distance_weighted":
            weights = 1.0 / (nn_dists ** 2 + eps)
            vote: dict[str, float] = {}
            for lbl, w in zip(nn_labels, weights):
                vote[lbl] = vote.get(lbl, 0.0) + w
            preds.append(max(vote, key=vote.get))

        elif mode == "class_balanced":
            vote = {}
            for lbl, d in zip(nn_labels, nn_dists):
                n_c = class_counts.get(lbl, 1)
                w = 1.0 / (n_c * d ** 2 + eps)
                vote[lbl] = vote.get(lbl, 0.0) + w
            preds.append(max(vote, key=vote.get))

    return np.array(preds)


def run_one(run_dir: Path, rare_class: str) -> list[dict]:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"  WARNING: {emb_dir} not found, skipping.")
        return []

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  WARNING: missing file {e}, skipping {run_dir}")
        return []

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    # Use only labeled training cells as reference
    ref_z      = train_z[is_labeled]
    ref_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    # Class counts for class-balanced weighting
    class_counts = dict(Counter(ref_labels))

    rows = []
    for k in K_VALUES:
        for mode in ["standard", "distance_weighted", "class_balanced"]:
            pred = knn_predict(ref_z, ref_labels, test_z, k, mode, class_counts)
            m, _ = classification_tables(y_test, pd.Series(pred), rare_class=rare_class)
            rows.append({
                "run": run_dir.name,
                "rare_class": rare_class,
                "k": k,
                "mode": mode,
                "rare_f1": m["rare_f1"],
                "rare_recall": m["rare_recall"],
                "rare_precision": m["rare_precision"],
                "overall_accuracy": m["overall_accuracy"],
            })
            print(f"    k={k:2d}  {mode:25s}  rare_f1={m['rare_f1']:.3f}")

    return rows


def main() -> pd.DataFrame:
    all_rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        rows = run_one(run_dir, rare_class)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E3 Results (kNN variants, seed42) ===")
    pivot = df.pivot_table(
        index=["rare_class", "k"],
        columns="mode",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
