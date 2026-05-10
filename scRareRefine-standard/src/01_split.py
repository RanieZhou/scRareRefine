"""Stage 1: Generate and persist train/val/test split.

Usage:
    python src/01_split.py --config configs/immune_dc.yaml --seed 42
    python src/01_split.py --config configs/immune_dc.yaml --seed 42 --split_mode cell_stratified

Output:
    data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
    Columns: cell_id, split, original_label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_config, load_adata, make_split_path, write_table


def cell_stratified_split(
    obs: pd.DataFrame,
    *,
    label_key: str,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.Series:
    labels = obs[label_key].astype(str)
    indices = np.arange(len(obs))
    train_idx, heldout_idx = train_test_split(
        indices, train_size=train_fraction, random_state=seed, stratify=labels,
    )
    heldout_labels = labels.iloc[heldout_idx]
    val_share = validation_fraction / (validation_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        heldout_idx, train_size=val_share, random_state=seed + 1, stratify=heldout_labels,
    )
    split = pd.Series(index=obs.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "validation"
    split.iloc[test_idx] = "test"
    return split.astype(str)


def batch_heldout_split(
    obs: pd.DataFrame,
    *,
    label_key: str,
    batch_key: str,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.Series:
    labels = obs[label_key].astype(str)
    batches = obs[batch_key].astype(str)
    classes = sorted(labels.unique())
    targets = {
        "train": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * train_fraction,
        "validation": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * validation_fraction,
        "test": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * test_fraction,
    }
    split_counts = {name: np.zeros(len(classes), dtype=float) for name in targets}
    batch_counts = pd.crosstab(batches, labels).reindex(columns=classes, fill_value=0)
    batch_counts["_n"] = batch_counts.sum(axis=1)
    if len(batch_counts) < 3:
        raise ValueError("batch_heldout_split requires at least 3 batches")
    rng = np.random.default_rng(seed)
    batch_counts["_tie"] = rng.random(len(batch_counts))
    batch_counts = batch_counts.sort_values(["_n", "_tie"], ascending=[False, True])
    ordered_batches = batch_counts.index.to_numpy()
    split_order = ["train", "validation", "test"]

    batch_to_split: dict[str, str] = {}
    for batch in ordered_batches:
        counts = batch_counts.loc[batch, classes].to_numpy(dtype=float)
        scores = []
        for name in split_order:
            new_counts = split_counts[name] + counts
            target = targets[name]
            denom = np.maximum(target, 1.0)
            score = float((((new_counts - target) / denom) ** 2).sum())
            score += float(max(new_counts.sum() - target.sum(), 0.0) / max(target.sum(), 1.0))
            scores.append(score)
        chosen = split_order[int(np.argmin(scores))]
        batch_to_split[str(batch)] = chosen
        split_counts[chosen] += counts

    def _total_score(counts_by_split: dict[str, np.ndarray]) -> float:
        score = 0.0
        for name in split_order:
            target = targets[name]
            denom = np.maximum(target, 1.0)
            new_counts = counts_by_split[name]
            score += float((((new_counts - target) / denom) ** 2).sum())
            score += float(max(new_counts.sum() - target.sum(), 0.0) / max(target.sum(), 1.0))
        return score

    for missing in [name for name in split_order if name not in set(batch_to_split.values())]:
        best_move = None
        for batch, source in batch_to_split.items():
            if source == missing:
                continue
            if sum(assigned == source for assigned in batch_to_split.values()) <= 1:
                continue
            counts = batch_counts.loc[batch, classes].to_numpy(dtype=float)
            proposed = {name: values.copy() for name, values in split_counts.items()}
            proposed[source] -= counts
            proposed[missing] += counts
            candidate = (_total_score(proposed), batch, source, counts)
            if best_move is None or candidate[0] < best_move[0]:
                best_move = candidate
        if best_move is None:
            raise ValueError("Unable to assign at least one batch to every split")
        _, batch, source, counts = best_move
        batch_to_split[batch] = missing
        split_counts[source] -= counts
        split_counts[missing] += counts

    return batches.map(batch_to_split).astype(str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: generate train/val/test split")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--split_mode", default="batch_heldout", choices=["batch_heldout", "cell_stratified"])
    parser.add_argument("--train_fraction", type=float, default=0.70)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = config["dataset"]
    label_key = dataset.get("label_key", "label")
    batch_key = dataset.get("batch_key", "batch")

    print(f"Loading data from {dataset['path']} ...")
    adata = load_adata(config)
    adata.obs_names_make_unique()

    fractions = dict(
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    print(f"Running {args.split_mode} split with seed={args.seed} ...")
    if args.split_mode == "cell_stratified":
        split = cell_stratified_split(adata.obs, label_key=label_key, seed=args.seed, **fractions)
    else:
        split = batch_heldout_split(adata.obs, label_key=label_key, batch_key=batch_key, seed=args.seed, **fractions)

    out = pd.DataFrame({
        "cell_id": adata.obs_names.astype(str),
        "split": split.to_numpy(),
        "original_label": adata.obs[label_key].astype(str).to_numpy(),
    })
    out_path = make_split_path(config, args.split_mode, args.seed)
    write_table(out, out_path)
    print(f"Saved to {out_path}")
    print(split.value_counts().to_string())


if __name__ == "__main__":
    main()
