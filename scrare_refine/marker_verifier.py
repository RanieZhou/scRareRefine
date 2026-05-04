from __future__ import annotations

import numpy as np
import pandas as pd

from scrare_refine.metrics import classification_tables


def compute_marker_signatures(
    expression: np.ndarray,
    *,
    gene_names: list[str],
    labels: pd.Series,
    is_labeled: np.ndarray,
    top_n: int = 25,
    min_cells: int = 5,
) -> dict[str, list[str]]:
    labels = pd.Series(labels).astype(str).reset_index(drop=True)
    is_labeled = np.asarray(is_labeled, dtype=bool)
    expr = np.asarray(expression, dtype=float)
    signatures: dict[str, list[str]] = {}

    for label in sorted(labels[is_labeled].unique()):
        in_class = is_labeled & labels.eq(label).to_numpy()
        out_class = is_labeled & ~labels.eq(label).to_numpy()
        if int(in_class.sum()) < min_cells or int(out_class.sum()) == 0:
            continue
        diff = expr[in_class].mean(axis=0) - expr[out_class].mean(axis=0)
        top_idx = np.argsort(-diff)[:top_n]
        signatures[label] = [gene_names[i] for i in top_idx if diff[i] > 0]
    return signatures


def marker_scores_for_candidates(
    expression: np.ndarray,
    candidates: pd.DataFrame,
    *,
    signatures: dict[str, list[str]],
    rare_class: str,
    gene_names: list[str],
) -> pd.DataFrame:
    expr = np.asarray(expression, dtype=float)
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    rare_genes = [gene_to_idx[g] for g in signatures.get(rare_class, []) if g in gene_to_idx]
    rows = []
    for idx, row in candidates.iterrows():
        pred = str(row["predicted_label"])
        pred_genes = [gene_to_idx[g] for g in signatures.get(pred, []) if g in gene_to_idx]
        rare_score = float(expr[idx, rare_genes].mean()) if rare_genes else 0.0
        pred_score = float(expr[idx, pred_genes].mean()) if pred_genes else 0.0
        margin = rare_score - pred_score
        rows.append(
            {
                f"marker_score_{rare_class}": rare_score,
                "marker_score_predicted": pred_score,
                "marker_margin": margin,
                "marker_verified": margin > 0,
            }
        )
    return pd.DataFrame(rows, index=candidates.index)


def marker_threshold_curve(
    predictions: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    *,
    rare_class: str,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    y_true = predictions["true_label"].astype(str)
    baseline_pred = predictions["predicted_label"].astype(str)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce")

    for threshold in thresholds:
        verified = margins.ge(threshold).fillna(False)
        relabeled = baseline_pred.copy()
        relabeled.loc[scored_candidates.index[verified]] = rare_class
        overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
        verified_indices = scored_candidates.index[verified]
        n_verified = int(verified.sum())
        rescued = int(rare_errors.loc[verified_indices].sum()) if n_verified else 0
        false_rescues = int(non_rare.loc[verified_indices].sum()) if n_verified else 0
        overall.update(
            {
                "marker_threshold": float(threshold),
                "n_candidates": int(len(scored_candidates)),
                "n_marker_verified": n_verified,
                "rescued_rare_errors": rescued,
                "false_rescues": false_rescues,
                "candidate_precision_for_rare_error": rescued / n_verified if n_verified else 0.0,
                "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
                "modification_rate": n_verified / len(predictions) if len(predictions) else 0.0,
                "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
            }
        )
        rows.append(overall)
    return pd.DataFrame(rows)


def evaluate_threshold_rescue(
    predictions: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    *,
    rare_class: str,
    marker_threshold: float | None = None,
) -> dict[str, float]:
    y_true = predictions["true_label"].astype(str)
    baseline_pred = predictions["predicted_label"].astype(str)
    if marker_threshold is None:
        verified = pd.Series(True, index=scored_candidates.index)
    else:
        margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce")
        verified = margins.ge(marker_threshold).fillna(False)

    relabeled = baseline_pred.copy()
    verified_indices = scored_candidates.index[verified]
    relabeled.loc[verified_indices] = rare_class
    overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)

    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    n_verified = int(verified.sum())
    rescued = int(rare_errors.loc[verified_indices].sum()) if n_verified else 0
    false_rescues = int(non_rare.loc[verified_indices].sum()) if n_verified else 0
    overall.update(
        {
            "marker_threshold": float(marker_threshold) if marker_threshold is not None else np.nan,
            "n_candidates": int(len(scored_candidates)),
            "n_marker_verified": n_verified,
            "rescued_rare_errors": rescued,
            "false_rescues": false_rescues,
            "candidate_precision_for_rare_error": rescued / n_verified if n_verified else 0.0,
            "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
            "modification_rate": n_verified / len(predictions) if len(predictions) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
        }
    )
    return overall


def choose_marker_threshold(curve: pd.DataFrame, *, max_false_rescue_rate: float = 0.001) -> float:
    eligible = curve[curve["major_to_rare_false_rescue_rate"].le(max_false_rescue_rate)].copy()
    if eligible.empty:
        eligible = curve.copy()
    eligible = eligible.sort_values(
        ["rare_f1", "rare_recall", "rare_precision", "marker_threshold"],
        ascending=[False, False, False, True],
    )
    return float(eligible["marker_threshold"].iloc[0])


def default_marker_thresholds(scored_candidates: pd.DataFrame) -> list[float]:
    margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce").dropna()
    if margins.empty:
        return [0.0]
    quantiles = margins.quantile([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).tolist()
    thresholds = sorted({float(x) for x in quantiles + [-1.0, -0.5, 0.0, 0.5, 1.0]})
    return thresholds
