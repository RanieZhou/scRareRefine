from __future__ import annotations

import numpy as np
import pandas as pd

from scrare_refine.metrics import classification_tables


def _safe_quantile(values: pd.Series, q: float) -> float:
    return float(pd.to_numeric(values, errors="coerce").dropna().quantile(q))


def gate_masks(predictions: pd.DataFrame, prototype: pd.DataFrame, *, rare_class: str) -> dict[str, pd.Series]:
    rank_col = f"prototype_rank_{rare_class}"
    d_col = f"d_pred_minus_d_{rare_class}"
    base = predictions["predicted_label"].astype(str).ne(rare_class)
    rank = pd.to_numeric(prototype[rank_col], errors="coerce")
    d_score = pd.to_numeric(prototype[d_col], errors="coerce")
    margin = pd.to_numeric(predictions["margin"], errors="coerce")
    entropy = pd.to_numeric(predictions["entropy"], errors="coerce")
    margin_q25 = _safe_quantile(margin, 0.25)
    entropy_q50 = _safe_quantile(entropy, 0.50)
    d_q90 = _safe_quantile(d_score, 0.90)

    # Identify the top-2 most frequent predicted classes (excluding the rare class)
    # as "neighbor major" classes — these are the most likely absorbers.
    pred_counts = predictions["predicted_label"].astype(str).value_counts()
    neighbor_major_classes = [c for c in pred_counts.index if c != rare_class][:2]
    neighbor_major = predictions["predicted_label"].astype(str).isin(neighbor_major_classes)

    return {
        "rank1": base & rank.le(1),
        "rank2_margin_q25": base & rank.le(2) & margin.le(margin_q25),
        "rank2_dscore_q90": base & rank.le(2) & d_score.ge(d_q90),
        "rank2_margin_q25_entropy_q50": base & rank.le(2) & margin.le(margin_q25) & entropy.ge(entropy_q50),
        "rank2_margin_q25_neighbor_major": base & rank.le(2) & margin.le(margin_q25) & neighbor_major,
    }


def evaluate_gate_rules(
    predictions: pd.DataFrame,
    prototype: pd.DataFrame,
    *,
    rare_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = gate_masks(predictions, prototype, rare_class=rare_class)
    y_true = predictions["true_label"].astype(str)
    baseline_pred = predictions["predicted_label"].astype(str)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    rows = []
    candidate_rows = []

    for gate_name, mask in masks.items():
        mask = mask.fillna(False).astype(bool)
        relabeled = baseline_pred.copy()
        relabeled.loc[mask] = rare_class
        overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
        rescued = int((mask & rare_errors).sum())
        false_rescues = int((mask & non_rare).sum())
        n_candidates = int(mask.sum())
        rows.append(
            {
                "gate_name": gate_name,
                **overall,
                "n_candidates": n_candidates,
                "rescued_rare_errors": rescued,
                "false_rescues": false_rescues,
                "candidate_precision_for_rare_error": rescued / n_candidates if n_candidates else 0.0,
                "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
                "modification_rate": n_candidates / len(predictions) if len(predictions) else 0.0,
                "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
            }
        )
        if n_candidates:
            candidate = predictions.loc[mask, ["cell_id", "true_label", "predicted_label", "margin", "entropy"]].copy()
            candidate.insert(0, "gate_name", gate_name)
            for col in [f"prototype_rank_{rare_class}", f"d_pred_minus_d_{rare_class}", f"distance_to_{rare_class}", "distance_to_pred"]:
                if col in prototype:
                    candidate[col] = prototype.loc[mask, col].to_numpy()
            candidate_rows.append(candidate)

    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return pd.DataFrame(rows), candidates


def summarize_gate_effect(effect: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["rare_train_size", "gate_name"]
    metric_cols = [
        "overall_accuracy",
        "macro_f1",
        "rare_precision",
        "rare_recall",
        "rare_f1",
        "n_candidates",
        "rescued_rare_errors",
        "false_rescues",
        "candidate_precision_for_rare_error",
        "rare_error_recall",
        "modification_rate",
        "major_to_rare_false_rescue_rate",
    ]
    return effect.groupby(group_cols)[metric_cols].agg(["mean", "std", "count"]).reset_index()


def choose_recommended_gate(summary: pd.DataFrame) -> str:
    flat = summary.copy()
    flat.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in flat.columns]
    eligible = flat[
        (flat["major_to_rare_false_rescue_rate_mean"] <= 0.01)
        & (flat["rare_train_size"].astype(str).isin(["20", "50", "100"]))
    ].copy()
    if eligible.empty:
        eligible = flat[flat["rare_train_size"].astype(str).isin(["20", "50", "100"])].copy()
    scores = eligible.groupby("gate_name")["rare_f1_mean"].mean().sort_values(ascending=False)
    return str(scores.index[0])
