"""Confidence-aware probability fusion for rare-cell refinement.

Instead of binary rescue decisions, this module blends scANVI's
softmax probabilities with prototype-distance-based probabilities.
The blending weight α is derived from scANVI's own confidence:

    p_fused(y|x) = α(x) · p_scanvi(y|x) + (1 - α(x)) · p_proto(y|x)

When scANVI is confident (high margin), α ≈ 1 and fusion defaults
to the baseline prediction.  When scANVI is uncertain, prototype
geometry gets more influence, enabling rare-class rescue without
harming already-correct predictions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scrare_refine.metrics import classification_tables


# ---------------------------------------------------------------------------
# Prototype probability distribution
# ---------------------------------------------------------------------------

def prototype_probabilities(
    latent: np.ndarray,
    *,
    labels: pd.Series,
    is_labeled: np.ndarray,
    temperature: float = 1.0,
) -> pd.DataFrame:
    """Convert Euclidean distances to class prototypes into probabilities.

    Uses softmax over negative distances, analogous to prototypical
    networks:  p(y=c|x) = exp(-d(z_x, μ_c)/τ) / Σ_k exp(-d(z_x, μ_k)/τ)

    Parameters
    ----------
    latent : (n_cells, n_latent) array
    labels : true labels for each cell
    is_labeled : boolean mask indicating labeled (training) cells
    temperature : softmax temperature τ.  Lower → peakier, higher → flatter.

    Returns
    -------
    DataFrame of shape (n_cells, n_classes) with probability columns.
    """
    latent = np.asarray(latent, dtype=float)
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    is_labeled = np.asarray(is_labeled, dtype=bool)

    classes = sorted(labels[is_labeled].unique())
    proto_vecs = np.vstack(
        [latent[is_labeled & labels.eq(cls).to_numpy()].mean(axis=0) for cls in classes]
    )

    # Euclidean distances: (n_cells, n_classes)
    diff = latent[:, None, :] - proto_vecs[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2))

    # Softmax over negative distances / temperature
    logits = -distances / max(temperature, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return pd.DataFrame(probs, columns=classes)


def prototype_probabilities_from_reference(
    query_latent: np.ndarray,
    *,
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    temperature: float = 1.0,
) -> pd.DataFrame:
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
    diff = query_latent[:, None, :] - proto_vecs[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=2))
    logits = -distances / max(temperature, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=classes)


# ---------------------------------------------------------------------------
# Confidence-aware fusion weight
# ---------------------------------------------------------------------------

def confidence_weight(
    margin: np.ndarray,
    *,
    alpha_min: float = 0.5,
) -> np.ndarray:
    """Compute per-cell fusion weight α ∈ [alpha_min, 1.0].

    α = alpha_min + (1 - alpha_min) · margin

    - margin ≈ 0 (maximally uncertain) → α ≈ alpha_min  (prototype has influence)
    - margin ≈ 1 (maximally confident) → α ≈ 1.0        (scANVI dominates)
    """
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
    """Compute per-cell α that drops when prototype disagrees with scANVI.

    1. Base α from scANVI margin  (same as confidence_weight).
    2. If prototype top-prediction ≠ scANVI top-prediction, multiply α
       by a discount factor β ∈ (0, 1].

    This allows prototype to override even high-margin scANVI predictions
    when the latent geometry strongly points to a different class.

    Parameters
    ----------
    beta : discount factor applied when scANVI and prototype disagree.
           0 → always trust prototype on disagreement.
           1 → ignore disagreement (same as confidence_weight).
    """
    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    s_top = np.array(common)[p_scanvi[common].to_numpy(dtype=float).argmax(axis=1)]
    p_top = np.array(common)[p_proto[common].to_numpy(dtype=float).argmax(axis=1)]
    disagree = s_top != p_top

    alpha = confidence_weight(margin, alpha_min=alpha_min)
    alpha[disagree] *= beta
    return np.clip(alpha, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Probability blending
# ---------------------------------------------------------------------------

def fuse_predictions(
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    *,
    alpha: np.ndarray,
) -> tuple[pd.Series, pd.DataFrame]:
    """Blend scANVI and prototype probabilities.

    p_fused = α · p_scanvi + (1 - α) · p_proto

    Returns (fused_labels, fused_probabilities).
    """
    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    if not common:
        raise ValueError("No overlapping class columns between p_scanvi and p_proto")

    s_arr = p_scanvi[common].to_numpy(dtype=float)
    p_arr = p_proto[common].to_numpy(dtype=float)
    a = np.asarray(alpha, dtype=float)[:, None]

    fused = a * s_arr + (1.0 - a) * p_arr
    # Re-normalise in case of floating-point drift
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12

    fused_df = pd.DataFrame(fused, columns=common, index=p_scanvi.index)
    fused_labels = pd.Series(
        [common[i] for i in fused.argmax(axis=1)],
        index=p_scanvi.index,
    )
    return fused_labels, fused_df


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_fusion_effect(
    y_true: pd.Series,
    baseline_pred: pd.Series,
    fused_pred: pd.Series,
    *,
    rare_class: str,
) -> dict[str, float]:
    """Compute classification metrics and rescue/damage statistics."""
    overall, _ = classification_tables(y_true, fused_pred, rare_class=rare_class)

    bl = baseline_pred.astype(str)
    fu = fused_pred.astype(str)
    yt = y_true.astype(str)

    changed = bl.ne(fu)
    rare_errors = yt.eq(rare_class) & bl.ne(rare_class)
    non_rare = yt.ne(rare_class)

    rescued = int((changed & rare_errors).sum())
    false_rescues = int((changed & non_rare & fu.eq(rare_class)).sum())
    # Cells that were correct but got changed to wrong label
    damaged = int((changed & bl.eq(yt) & fu.ne(yt)).sum())
    n_changed = int(changed.sum())
    n_total = len(y_true)

    overall.update({
        "n_changed": n_changed,
        "modification_rate": n_changed / n_total if n_total else 0.0,
        "rescued_rare_errors": rescued,
        "false_rescues": false_rescues,
        "damaged_correct": damaged,
        "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
        "major_to_rare_false_rescue_rate": (
            false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0
        ),
    })
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
    """End-to-end: compute α → fuse → evaluate.  Returns metrics dict.

    When beta < 1.0, uses disagreement-aware weighting so prototype can
    override confident-but-wrong scANVI predictions.
    """
    if beta < 1.0:
        alpha = disagreement_aware_weight(
            p_scanvi, p_proto,
            margin=margin, alpha_min=alpha_min, beta=beta,
        )
    else:
        alpha = confidence_weight(margin, alpha_min=alpha_min)
    fused_labels, _ = fuse_predictions(p_scanvi, p_proto, alpha=alpha)
    result = evaluate_fusion_effect(
        y_true, baseline_pred, fused_labels, rare_class=rare_class,
    )
    result["temperature"] = temperature
    result["alpha_min"] = alpha_min
    result["beta"] = beta
    return result


# ---------------------------------------------------------------------------
# Parameter selection
# ---------------------------------------------------------------------------

def select_best_params(
    val_results: pd.DataFrame,
    *,
    baseline_accuracy: float,
    max_accuracy_drop: float = 0.005,
    max_false_rescue_rate: float = 0.005,
) -> tuple[float, float, float]:
    """Pick (temperature, alpha_min, beta) that maximise rare_f1 on validation
    subject to accuracy and false-rescue constraints."""
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
