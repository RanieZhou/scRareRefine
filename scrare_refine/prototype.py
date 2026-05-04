from __future__ import annotations

import numpy as np
import pandas as pd


def _euclidean_distances(z: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    diff = z[:, None, :] - prototypes[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def prototype_scores(
    latent: np.ndarray,
    *,
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    is_labeled: np.ndarray,
    rare_class: str,
    margin: np.ndarray,
    margin_quantile: float = 0.25,
) -> pd.DataFrame:
    latent = np.asarray(latent, dtype=float)
    true_labels = pd.Series(true_labels).astype(str).reset_index(drop=True)
    predicted_labels = pd.Series(predicted_labels).astype(str).reset_index(drop=True)
    is_labeled = np.asarray(is_labeled, dtype=bool)
    margin = np.asarray(margin, dtype=float)

    classes = sorted(true_labels[is_labeled].unique())
    if rare_class not in classes:
        raise ValueError(f"Rare class has no labeled cells for prototype: {rare_class}")

    prototypes = np.vstack([latent[is_labeled & (true_labels == cls)].mean(axis=0) for cls in classes])
    distances = _euclidean_distances(latent, prototypes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    rare_idx = class_to_idx[rare_class]
    pred_dist = np.array(
        [
            distances[i, class_to_idx[pred]] if pred in class_to_idx else np.nan
            for i, pred in enumerate(predicted_labels)
        ]
    )
    rare_dist = distances[:, rare_idx]
    ranks = np.argsort(np.argsort(distances, axis=1), axis=1)[:, rare_idx] + 1
    threshold = float(np.quantile(margin, margin_quantile))
    candidates = (predicted_labels.to_numpy() != rare_class) & (ranks <= 2) & (margin <= threshold)

    return pd.DataFrame(
        {
            "distance_to_" + rare_class: rare_dist,
            "distance_to_pred": pred_dist,
            "prototype_rank_" + rare_class: ranks,
            "d_pred_minus_d_" + rare_class: pred_dist - rare_dist,
            "prototype_rescue_candidate": candidates,
        }
    )


def prototype_scores_from_reference(
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
        raise ValueError(f"Rare class has no labeled cells for prototype: {rare_class}")

    prototypes = np.vstack(
        [
            reference_latent[reference_is_labeled & reference_labels.eq(cls).to_numpy()].mean(axis=0)
            for cls in classes
        ]
    )
    distances = _euclidean_distances(query_latent, prototypes)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    rare_idx = class_to_idx[rare_class]
    pred_dist = np.array(
        [
            distances[i, class_to_idx[pred]] if pred in class_to_idx else np.nan
            for i, pred in enumerate(predicted_labels)
        ]
    )
    rare_dist = distances[:, rare_idx]
    ranks = np.argsort(np.argsort(distances, axis=1), axis=1)[:, rare_idx] + 1
    threshold = float(np.quantile(margin, margin_quantile))
    candidates = (predicted_labels.to_numpy() != rare_class) & (ranks <= 2) & (margin <= threshold)

    return pd.DataFrame(
        {
            "distance_to_" + rare_class: rare_dist,
            "distance_to_pred": pred_dist,
            "prototype_rank_" + rare_class: ranks,
            "d_pred_minus_d_" + rare_class: pred_dist - rare_dist,
            "prototype_rescue_candidate": candidates,
        }
    )
