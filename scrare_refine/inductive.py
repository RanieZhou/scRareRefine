from __future__ import annotations

from typing import Literal

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split

from scrare_refine.splits import RareTrainSize, parse_rare_train_size

SplitName = Literal["train", "validation", "test"]


def _validate_fractions(train_fraction: float, validation_fraction: float, test_fraction: float) -> None:
    total = train_fraction + validation_fraction + test_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive")
    if not np.isclose(total, 1.0):
        raise ValueError("train/validation/test fractions must sum to 1.0")


def cell_stratified_split(
    obs: pd.DataFrame,
    *,
    label_key: str,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.Series:
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if label_key not in obs:
        raise KeyError(f"Missing label column in obs: {label_key}")

    labels = obs[label_key].astype(str)
    indices = np.arange(len(obs))
    train_idx, heldout_idx = train_test_split(
        indices,
        train_size=train_fraction,
        random_state=seed,
        stratify=labels,
    )
    heldout_labels = labels.iloc[heldout_idx]
    validation_share = validation_fraction / (validation_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        heldout_idx,
        train_size=validation_share,
        random_state=seed + 1,
        stratify=heldout_labels,
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
    _validate_fractions(train_fraction, validation_fraction, test_fraction)
    if label_key not in obs:
        raise KeyError(f"Missing label column in obs: {label_key}")
    if batch_key not in obs:
        raise KeyError(f"Missing batch column in obs: {batch_key}")

    labels = obs[label_key].astype(str)
    batches = obs[batch_key].astype(str)
    classes = sorted(labels.unique())
    targets = {
        "train": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * train_fraction,
        "validation": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * validation_fraction,
        "test": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * test_fraction,
    }
    split_counts = {name: np.zeros(len(classes), dtype=float) for name in targets}
    batch_counts = (
        pd.crosstab(batches, labels)
        .reindex(columns=classes, fill_value=0)
        .assign(_n=lambda df: df.sum(axis=1))
        .sort_values("_n", ascending=False)
    )
    if len(batch_counts) < 3:
        raise ValueError("batch_heldout_split requires at least 3 batches for train/validation/test")
    rng = np.random.default_rng(seed)
    ordered_batches = batch_counts.index.to_numpy()
    jitter = rng.random(len(ordered_batches)) * 1e-6
    ordered_batches = ordered_batches[np.argsort(np.arange(len(ordered_batches)) + jitter)]

    batch_to_split: dict[str, str] = {}
    split_order = ["train", "validation", "test"]
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

    def total_score(counts_by_split: dict[str, np.ndarray]) -> float:
        score = 0.0
        for name in split_order:
            target = targets[name]
            denom = np.maximum(target, 1.0)
            new_counts = counts_by_split[name]
            score += float((((new_counts - target) / denom) ** 2).sum())
            score += float(max(new_counts.sum() - target.sum(), 0.0) / max(target.sum(), 1.0))
        return score

    for missing in [name for name in split_order if name not in set(batch_to_split.values())]:
        best_move: tuple[float, str, str, np.ndarray] | None = None
        for batch, source in batch_to_split.items():
            if source == missing:
                continue
            if sum(assigned == source for assigned in batch_to_split.values()) <= 1:
                continue
            counts = batch_counts.loc[batch, classes].to_numpy(dtype=float)
            proposed = {name: values.copy() for name, values in split_counts.items()}
            proposed[source] -= counts
            proposed[missing] += counts
            candidate = (total_score(proposed), batch, source, counts)
            if best_move is None or candidate[0] < best_move[0]:
                best_move = candidate
        if best_move is None:
            raise ValueError("Unable to allocate at least one held-out batch to every split")
        _, batch, source, counts = best_move
        batch_to_split[batch] = missing
        split_counts[source] -= counts
        split_counts[missing] += counts

    split = batches.map(batch_to_split)
    return split.astype(str)


def make_inductive_scanvi_labels(
    obs: pd.DataFrame,
    split: pd.Series,
    *,
    label_key: str,
    rare_class: str,
    rare_train_size: RareTrainSize | str | int,
    seed: int,
    unlabeled_category: str,
) -> tuple[pd.Series, np.ndarray]:
    if label_key not in obs:
        raise KeyError(f"Missing label column in obs: {label_key}")

    split = pd.Series(split, index=obs.index).astype(str)
    true_labels = obs[label_key].astype(str)
    labels = pd.Series(unlabeled_category, index=obs.index, dtype=object)
    is_labeled = np.zeros(len(obs), dtype=bool)

    train_mask = split.eq("train")
    train_major = train_mask & true_labels.ne(rare_class)
    labels.loc[train_major] = true_labels.loc[train_major]
    is_labeled[train_major.to_numpy()] = True

    rare_train_size = parse_rare_train_size(rare_train_size)
    rare_train = train_mask & true_labels.eq(rare_class)
    rare_indices = np.flatnonzero(rare_train.to_numpy())
    if rare_train_size == "all":
        selected = rare_indices
    else:
        rng = np.random.default_rng(seed)
        selected = rng.choice(rare_indices, size=min(int(rare_train_size), len(rare_indices)), replace=False)
    labels.iloc[selected] = rare_class
    is_labeled[selected] = True
    return labels.astype(str), is_labeled


def select_train_hvg_var_names(train_adata: ad.AnnData, *, n_top_genes: int | None) -> list[str]:
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= train_adata.n_vars:
        return train_adata.var_names.astype(str).tolist()
    x = train_adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
    variance = mean_sq - mean * mean
    top_idx = np.argsort(-variance)[:n_top_genes]
    return train_adata.var_names[np.sort(top_idx)].astype(str).tolist()
