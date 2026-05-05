"""Inductive probability fusion for rare-cell refinement."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_tables(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    rare_class: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    rare_row = per_class[per_class["label"] == rare_class]
    if rare_row.empty:
        rare_precision = rare_recall = rare_f1 = 0.0
    else:
        rare_precision = float(rare_row["precision"].iloc[0])
        rare_recall = float(rare_row["recall"].iloc[0])
        rare_f1 = float(rare_row["f1"].iloc[0])

    overall = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rare_precision": rare_precision,
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
    }
    return overall, per_class


def prototype_probabilities_from_reference(
    query_latent: np.ndarray,
    *,
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Convert distances to train-set prototypes into query-cell probabilities."""
    query_latent = np.asarray(query_latent, dtype=float)
    reference_latent = np.asarray(reference_latent, dtype=float)
    reference_labels = pd.Series(reference_labels).astype(str).reset_index(drop=True)
    reference_is_labeled = np.asarray(reference_is_labeled, dtype=bool)

    classes = sorted(reference_labels[reference_is_labeled].unique())
    if not classes:
        raise ValueError("No labeled reference cells available for prototypes")
    proto_vecs = np.vstack(
        [
            reference_latent[reference_is_labeled & reference_labels.eq(cls).to_numpy()].mean(axis=0)
            for cls in classes
        ]
    )
    distances = np.sqrt(((query_latent[:, None, :] - proto_vecs[None, :, :]) ** 2).sum(axis=2))
    logits = -distances / max(temperature, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=classes)


def confidence_weight(
    margin: np.ndarray,
    *,
    alpha_min: float = 0.5,
) -> np.ndarray:
    """Compute the scANVI probability weight from the prediction margin."""
    margin = np.asarray(margin, dtype=float)
    return alpha_min + (1.0 - alpha_min) * np.clip(margin, 0.0, 1.0)


def disagreement_aware_weight(
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    *,
    margin: np.ndarray,
    alpha_min: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """Lower the scANVI weight when prototype and scANVI top labels disagree."""
    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    s_top = np.array(common)[p_scanvi[common].to_numpy(dtype=float).argmax(axis=1)]
    p_top = np.array(common)[p_proto[common].to_numpy(dtype=float).argmax(axis=1)]
    disagree = s_top != p_top

    alpha = confidence_weight(margin, alpha_min=alpha_min)
    alpha[disagree] *= beta
    return np.clip(alpha, 0.0, 1.0)


def fuse_predictions(
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    *,
    alpha: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    """Blend scANVI probabilities with train-reference prototype probabilities."""
    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    if not common:
        raise ValueError("No overlapping class columns between p_scanvi and p_proto")

    s_arr = p_scanvi[common].to_numpy(dtype=float)
    p_arr = p_proto[common].to_numpy(dtype=float)
    a = np.asarray(alpha, dtype=float)[:, None]

    fused = a * s_arr + (1.0 - a) * p_arr
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12

    fused_df = pd.DataFrame(fused, columns=common, index=p_scanvi.index)
    fused_labels = pd.Series([common[i] for i in fused.argmax(axis=1)], index=p_scanvi.index)
    return fused_labels, fused_df


def evaluate_fusion_effect(
    y_true: pd.Series,
    baseline_pred: pd.Series,
    fused_pred: pd.Series,
    *,
    rare_class: str,
) -> dict[str, float]:
    """Compute classification and rescue metrics on held-out cells."""
    overall, _ = classification_tables(y_true, fused_pred, rare_class=rare_class)

    bl = baseline_pred.astype(str)
    fu = fused_pred.astype(str)
    yt = y_true.astype(str)

    changed = bl.ne(fu)
    rare_errors = yt.eq(rare_class) & bl.ne(rare_class)
    non_rare = yt.ne(rare_class)

    rescued = int((changed & rare_errors).sum())
    false_rescues = int((changed & non_rare & fu.eq(rare_class)).sum())
    damaged = int((changed & bl.eq(yt) & fu.ne(yt)).sum())
    n_changed = int(changed.sum())
    n_total = len(y_true)

    overall.update(
        {
            "n_changed": n_changed,
            "modification_rate": n_changed / n_total if n_total else 0.0,
            "rescued_rare_errors": rescued,
            "false_rescues": false_rescues,
            "damaged_correct": damaged,
            "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
            "major_to_rare_false_rescue_rate": (
                false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0
            ),
        }
    )
    return overall


def fuse_and_evaluate(
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    *,
    margin: np.ndarray,
    y_true: pd.Series,
    baseline_pred: pd.Series,
    rare_class: str,
    temperature: float,
    alpha_min: float,
    beta: float = 1.0,
) -> dict[str, float]:
    """Fuse probabilities and evaluate the resulting labels."""
    if beta < 1.0:
        alpha = disagreement_aware_weight(
            p_scanvi,
            p_proto,
            margin=margin,
            alpha_min=alpha_min,
            beta=beta,
        )
    else:
        alpha = confidence_weight(margin, alpha_min=alpha_min)
    fused_labels, _ = fuse_predictions(p_scanvi, p_proto, alpha=alpha)
    result = evaluate_fusion_effect(y_true, baseline_pred, fused_labels, rare_class=rare_class)
    result["temperature"] = temperature
    result["alpha_min"] = alpha_min
    result["beta"] = beta
    return result


def select_best_params(
    val_results: pd.DataFrame,
    *,
    baseline_accuracy: float,
    max_accuracy_drop: float = 0.005,
    max_false_rescue_rate: float = 0.005,
) -> tuple[float, float, float]:
    """Select fusion parameters from validation metrics only."""
    eligible = val_results[
        val_results["overall_accuracy"].ge(baseline_accuracy - max_accuracy_drop)
        & val_results["major_to_rare_false_rescue_rate"].le(max_false_rescue_rate)
    ]
    if eligible.empty:
        eligible = val_results.copy()

    best = eligible.sort_values(
        ["rare_f1", "overall_accuracy", "major_to_rare_false_rescue_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["temperature"]), float(best["alpha_min"]), float(best.get("beta", 1.0))
