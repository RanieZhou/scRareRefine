"""Stage 3: Compute prototype-distance scores for validation and test cells.

Reads:
    outputs/{dataset}/{run_id}/embeddings/

Writes:
    outputs/{dataset}/{run_id}/prototype/
        validation_scores.csv
        test_scores.csv

Usage:
    python src/03_prototype.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import load_config, make_run_dir, parse_rare_train_size, read_table, write_table


def _latent_matrix(latent_df: pd.DataFrame) -> np.ndarray:
    return latent_df[[c for c in latent_df.columns if c.startswith("latent_")]].to_numpy()


def prototype_scores(
    query_latent: np.ndarray,
    *,
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    predicted_labels: pd.Series,
    rare_class: str,
    margin: np.ndarray,
    margin_quantile: float = 0.25,
) -> pd.DataFrame:
    query_latent = np.asarray(query_latent, dtype=float)
    reference_latent = np.asarray(reference_latent, dtype=float)
    reference_labels = pd.Series(reference_labels).astype(str).reset_index(drop=True)
    predicted_labels = pd.Series(predicted_labels).astype(str).reset_index(drop=True)
    reference_is_labeled = np.asarray(reference_is_labeled, dtype=bool)
    margin = np.asarray(margin, dtype=float)

    classes = sorted(reference_labels[reference_is_labeled].unique())
    if rare_class not in classes:
        raise ValueError(f"Rare class '{rare_class}' has no labeled reference cells")

    prototypes = np.vstack([
        reference_latent[reference_is_labeled & reference_labels.eq(cls).to_numpy()].mean(axis=0)
        for cls in classes
    ])
    diff = query_latent[:, None, :] - prototypes[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    rare_idx = class_to_idx[rare_class]
    pred_dist = np.array([
        distances[i, class_to_idx[pred]] if pred in class_to_idx else np.nan
        for i, pred in enumerate(predicted_labels)
    ])
    rare_dist = distances[:, rare_idx]
    ranks = np.argsort(np.argsort(distances, axis=1), axis=1)[:, rare_idx] + 1
    threshold = float(np.quantile(margin, margin_quantile))
    candidates = (predicted_labels.to_numpy() != rare_class) & (ranks <= 2) & (margin <= threshold)

    return pd.DataFrame({
        f"distance_to_{rare_class}": rare_dist,
        "distance_to_pred": pred_dist,
        f"prototype_rank_{rare_class}": ranks,
        f"d_pred_minus_d_{rare_class}": pred_dist - rare_dist,
        "prototype_rescue_candidate": candidates,
    })


def separability_metrics(
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    rare_class: str,
) -> dict:
    """Quantify how well the rare class is separated from majority classes in prototype space."""
    labeled_mask = np.asarray(reference_is_labeled, dtype=bool)
    labels = pd.Series(reference_labels).astype(str).reset_index(drop=True)
    ref = np.asarray(reference_latent, dtype=float)
    classes = sorted(labels[labeled_mask].unique())

    prototypes = {
        cls: ref[labeled_mask & labels.eq(cls).to_numpy()].mean(axis=0)
        for cls in classes
    }
    rare_proto = prototypes[rare_class]
    rare_cells = ref[labeled_mask & labels.eq(rare_class).to_numpy()]
    intra_radius = float(np.sqrt(((rare_cells - rare_proto) ** 2).sum(axis=1)).mean())

    inter_dists = {
        cls: float(np.sqrt(((rare_proto - p) ** 2).sum()))
        for cls, p in prototypes.items()
        if cls != rare_class
    }
    nearest_class = min(inter_dists, key=inter_dists.get)
    dist_nearest = inter_dists[nearest_class]
    mean_dist = float(np.mean(list(inter_dists.values())))

    sep_ratio = round(dist_nearest / max(intra_radius, 1e-10), 4)
    if sep_ratio >= 1.5:
        rescue_confidence = "HIGH"
    elif sep_ratio >= 1.0:
        rescue_confidence = "MEDIUM"
    else:
        rescue_confidence = "LOW"

    return {
        "rare_class": rare_class,
        "n_rare_train": int((labeled_mask & labels.eq(rare_class).to_numpy()).sum()),
        "intra_rare_radius": round(intra_radius, 4),
        "dist_to_nearest_majority": round(dist_nearest, 4),
        "nearest_majority_class": nearest_class,
        "mean_dist_to_majority": round(mean_dist, 4),
        "separability_ratio": sep_ratio,
        "rescue_confidence": rescue_confidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: compute prototype scores")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    proto_dir = run_dir / "prototype"

    train_pred = read_table(emb_dir / "train_predictions.csv")
    train_latent = read_table(emb_dir / "train_latent.csv")

    sep = separability_metrics(
        _latent_matrix(train_latent),
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        rare_class=rare_class,
    )
    sep_df = pd.DataFrame([sep])
    write_table(sep_df, proto_dir / "separability.csv")
    print(f"  Separability ratio: {sep['separability_ratio']:.3f}  "
          f"[rescue_confidence={sep['rescue_confidence']}]  "
          f"(intra_radius={sep['intra_rare_radius']:.3f}, "
          f"nearest={sep['nearest_majority_class']} d={sep['dist_to_nearest_majority']:.3f})")

    for split_name in ["validation", "test"]:
        pred = read_table(emb_dir / f"{split_name}_predictions.csv")
        latent = read_table(emb_dir / f"{split_name}_latent.csv")
        scores = prototype_scores(
            _latent_matrix(latent),
            reference_latent=_latent_matrix(train_latent),
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            predicted_labels=pred["predicted_label"],
            rare_class=rare_class,
            margin=pred["margin"].to_numpy(),
        )
        write_table(scores, proto_dir / f"{split_name}_scores.csv")
        n_candidates = int(scores["prototype_rescue_candidate"].sum())
        print(f"  {split_name}: {n_candidates} prototype rescue candidates")

    print(f"Done. Prototype scores in {proto_dir}")


if __name__ == "__main__":
    main()
