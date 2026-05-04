from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

RareTrainSize = int | Literal["all"]


def parse_rare_train_size(value: str | int) -> RareTrainSize:
    if isinstance(value, int):
        return value
    if str(value).lower() == "all":
        return "all"
    return int(value)


def make_scanvi_labels(
    obs: pd.DataFrame,
    *,
    label_key: str,
    rare_class: str,
    rare_train_size: RareTrainSize,
    seed: int,
    unlabeled_category: str,
) -> tuple[pd.Series, np.ndarray]:
    if label_key not in obs:
        raise KeyError(f"Missing label column in obs: {label_key}")

    true_labels = obs[label_key].astype(str)
    labels = true_labels.copy()
    rare_mask = true_labels == rare_class
    is_labeled = np.ones(len(obs), dtype=bool)

    rare_indices = np.flatnonzero(rare_mask.to_numpy())
    if rare_train_size == "all":
        selected_rare = rare_indices
    else:
        if rare_train_size < 0:
            raise ValueError("rare_train_size must be non-negative or 'all'")
        rng = np.random.default_rng(seed)
        selected_rare = rng.choice(
            rare_indices,
            size=min(int(rare_train_size), len(rare_indices)),
            replace=False,
        )

    rare_labeled = np.zeros(len(obs), dtype=bool)
    rare_labeled[selected_rare] = True
    unlabeled_rare = rare_mask.to_numpy() & ~rare_labeled
    labels.iloc[unlabeled_rare] = unlabeled_category
    is_labeled[unlabeled_rare] = False
    return labels, is_labeled

