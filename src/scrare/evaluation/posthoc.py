from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from scrare.evaluation.metrics import classification_tables
from scrare.models.fusion import (
    evaluate_fusion_effect,
    fuse_predictions,
    prototype_probabilities_from_reference,
    select_best_params,
)
from scrare.models.marker import (
    choose_marker_threshold,
    compute_marker_signatures,
    default_marker_thresholds,
    evaluate_threshold_rescue,
    marker_scores_for_candidates,
    marker_threshold_curve,
)
from scrare.models.prototype import prototype_scores_from_reference
from scrare.models.prototype_gate import evaluate_gate_rules, gate_masks

METHOD_ORDER = [
    "baseline",
    "baseline_plus_prototype",
    "baseline_plus_prototype_gate",
    "baseline_plus_prototype_gate_plus_marker",
    "baseline_plus_fusion",
]

METHOD_LABELS = {
    "baseline": "scANVI baseline",
    "baseline_plus_prototype": "prototype candidate",
    "baseline_plus_prototype_gate": "prototype rank1 gate",
    "baseline_plus_prototype_gate_plus_marker": "validation-tuned marker",
    "baseline_plus_fusion": "fusion",
}


def _latent_matrix(latent_df: pd.DataFrame) -> np.ndarray:
    return latent_df[[col for col in latent_df.columns if col.startswith("latent_")]].to_numpy()


def _baseline_metrics(pred: pd.DataFrame, *, rare_class: str) -> dict[str, float]:
    overall, _ = classification_tables(pred["true_label"], pred["predicted_label"], rare_class=rare_class)
    overall.update(
        {
            "marker_threshold": np.nan,
            "n_candidates": 0,
            "n_marker_verified": 0,
            "rescued_rare_errors": 0,
            "false_rescues": 0,
            "candidate_precision_for_rare_error": 0.0,
            "rare_error_recall": 0.0,
            "modification_rate": 0.0,
            "major_to_rare_false_rescue_rate": 0.0,
        }
    )
    return overall


def _log1p_cpm_dense(x: Any) -> np.ndarray:
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        normalized = x.multiply(10000.0 / row_sum[:, None])
        return np.log1p(normalized.toarray()).astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    row_sum = arr.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(arr * (10000.0 / row_sum[:, None])).astype(np.float32)


def _expression_for_cells(adata: Any, *, cell_ids: pd.Series, genes: list[str]) -> np.ndarray:
    subset = adata[cell_ids.astype(str).tolist(), genes]
    return _log1p_cpm_dense(subset.X)


def _score_candidates(
    expression: np.ndarray,
    predictions: pd.DataFrame,
    candidate_mask: pd.Series,
    *,
    signatures: dict[str, list[str]],
    rare_class: str,
    gene_names: list[str],
) -> pd.DataFrame:
    candidates = predictions.loc[candidate_mask.fillna(False).astype(bool)].copy()
    if candidates.empty:
        return candidates.assign(marker_margin=pd.Series(dtype=float))
    scores = marker_scores_for_candidates(
        expression,
        candidates,
        signatures=signatures,
        rare_class=rare_class,
        gene_names=gene_names,
    )
    return pd.concat([candidates, scores], axis=1)


def _effect_from_mask(
    predictions: pd.DataFrame,
    candidate_mask: pd.Series,
    *,
    rare_class: str,
) -> dict[str, float]:
    mask = candidate_mask.fillna(False).astype(bool)
    y_true = predictions["true_label"].astype(str)
    baseline_pred = predictions["predicted_label"].astype(str)
    relabeled = baseline_pred.copy()
    relabeled.loc[mask] = rare_class
    overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    n_candidates = int(mask.sum())
    rescued = int((mask & rare_errors).sum())
    false_rescues = int((mask & non_rare).sum())
    overall.update(
        {
            "marker_threshold": np.nan,
            "n_candidates": n_candidates,
            "n_marker_verified": 0,
            "rescued_rare_errors": rescued,
            "false_rescues": false_rescues,
            "candidate_precision_for_rare_error": rescued / n_candidates if n_candidates else 0.0,
            "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
            "modification_rate": n_candidates / len(predictions) if len(predictions) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
        }
    )
    return overall


def _summarize(effect_runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "overall_accuracy",
        "macro_f1",
        "rare_precision",
        "rare_recall",
        "rare_f1",
        "n_candidates",
        "n_marker_verified",
        "rescued_rare_errors",
        "false_rescues",
        "candidate_precision_for_rare_error",
        "rare_error_recall",
        "modification_rate",
        "major_to_rare_false_rescue_rate",
    ]
    return (
        effect_runs.groupby(["split_mode", "rare_class", "rare_train_size", "method"], dropna=False)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _with_run_metadata(
    df: pd.DataFrame,
    *,
    seed: int,
    rare_train_size: str,
    rare_class: str,
    split_mode: str,
    run: str,
) -> pd.DataFrame:
    out = df.copy()
    for col in ["seed", "rare_train_size", "rare_class", "split_mode", "run"]:
        if col in out.columns:
            out = out.drop(columns=[col])
    out.insert(0, "seed", seed)
    out.insert(1, "rare_train_size", rare_train_size)
    out.insert(2, "rare_class", rare_class)
    out.insert(3, "split_mode", split_mode)
    out.insert(4, "run", run)
    return out


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    return out


def _csv_values(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _fusion_grid() -> list[tuple[float, float, float]]:
    return [
        (temperature, alpha_min, beta)
        for temperature in [0.5, 1.0, 2.0]
        for alpha_min in [0.3, 0.5, 0.7]
        for beta in [0.5, 1.0]
    ]


def _fusion_with_params(
    scanvi_probs: pd.DataFrame,
    proto_probs: pd.DataFrame,
    *,
    margin: np.ndarray,
    temperature: float,
    alpha_min: float,
    beta: float,
) -> pd.Series:
    del temperature
    common = sorted(set(scanvi_probs.columns) & set(proto_probs.columns))
    if not common:
        raise ValueError("No overlapping class columns between scanvi and prototype probabilities")
    alpha = np.clip(alpha_min + (1.0 - alpha_min) * np.clip(np.asarray(margin, dtype=float), 0.0, 1.0), 0.0, 1.0)
    if beta < 1.0:
        s_top = np.array(common)[scanvi_probs[common].to_numpy(dtype=float).argmax(axis=1)]
        p_top = np.array(common)[proto_probs[common].to_numpy(dtype=float).argmax(axis=1)]
        alpha[s_top != p_top] *= beta
    fused_labels, _ = fuse_predictions(scanvi_probs[common], proto_probs[common], alpha=alpha)
    return fused_labels


def evaluate_four_stage_methods(
    adata: Any,
    *,
    train_pred: pd.DataFrame,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    train_latent: pd.DataFrame,
    val_latent: pd.DataFrame,
    test_latent: pd.DataFrame,
    genes: list[str],
    rare_class: str,
    split_mode: str,
    seed: int,
    rare_train_size: str,
    run: str,
    max_false_rescue_rate: float,
    top_n: int,
    min_cells: int,
) -> dict[str, pd.DataFrame]:
    common = {
        "seed": seed,
        "rare_train_size": rare_train_size,
        "rare_class": rare_class,
        "split_mode": split_mode,
        "run": run,
    }

    effect_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []
    verified_rows: list[pd.DataFrame] = []
    fusion_grid_rows: list[dict[str, Any]] = []

    baseline = _baseline_metrics(test_pred, rare_class=rare_class)
    effect_rows.append({**baseline, **common, "method_key": "baseline", "method": METHOD_LABELS["baseline"], "gate_name": ""})

    proto_val = prototype_scores_from_reference(
        _latent_matrix(val_latent),
        reference_latent=_latent_matrix(train_latent),
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        predicted_labels=val_pred["predicted_label"],
        rare_class=rare_class,
        margin=val_pred["margin"].to_numpy(),
    )
    proto_test = prototype_scores_from_reference(
        _latent_matrix(test_latent),
        reference_latent=_latent_matrix(train_latent),
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        predicted_labels=test_pred["predicted_label"],
        rare_class=rare_class,
        margin=test_pred["margin"].to_numpy(),
    )

    gate_effect, gate_candidates = evaluate_gate_rules(test_pred, proto_test, rare_class=rare_class)
    prototype_candidate_mask = pd.Series(proto_test["prototype_rescue_candidate"], index=test_pred.index)
    prototype_candidate_effect = _effect_from_mask(test_pred, prototype_candidate_mask, rare_class=rare_class)
    effect_rows.append(
        {
            **prototype_candidate_effect,
            **common,
            "method_key": "baseline_plus_prototype",
            "method": METHOD_LABELS["baseline_plus_prototype"],
            "gate_name": "candidate",
        }
    )
    rank1 = gate_effect[gate_effect["gate_name"].eq("rank1")].iloc[0].to_dict()
    effect_rows.append(
        {
            **rank1,
            **common,
            "method_key": "baseline_plus_prototype_gate",
            "method": METHOD_LABELS["baseline_plus_prototype_gate"],
        }
    )
    prototype_candidates = test_pred.loc[prototype_candidate_mask.fillna(False).astype(bool), ["cell_id", "true_label", "predicted_label", "margin", "entropy"]].copy()
    if not prototype_candidates.empty:
        for col in [f"prototype_rank_{rare_class}", f"d_pred_minus_d_{rare_class}", f"distance_to_{rare_class}", "distance_to_pred"]:
            if col in proto_test:
                prototype_candidates[col] = proto_test.loc[prototype_candidate_mask.fillna(False).astype(bool), col].to_numpy()
        candidate_rows.append(
            _with_run_metadata(
                prototype_candidates,
                seed=seed,
                rare_train_size=rare_train_size,
                rare_class=rare_class,
                split_mode=split_mode,
                run=run,
            )
        )
    if not gate_candidates.empty:
        gate_candidates = gate_candidates[gate_candidates["gate_name"].eq("rank1")].copy()
        candidate_rows.append(
            _with_run_metadata(
                gate_candidates,
                seed=seed,
                rare_train_size=rare_train_size,
                rare_class=rare_class,
                split_mode=split_mode,
                run=run,
            )
        )

    train_expr = _expression_for_cells(adata, cell_ids=train_pred["cell_id"], genes=genes)
    val_expr = _expression_for_cells(adata, cell_ids=val_pred["cell_id"], genes=genes)
    test_expr = _expression_for_cells(adata, cell_ids=test_pred["cell_id"], genes=genes)
    signatures = compute_marker_signatures(
        train_expr,
        gene_names=genes,
        labels=train_pred["true_label"],
        is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        top_n=top_n,
        min_cells=min_cells,
    )

    val_rank1_mask = gate_masks(val_pred, proto_val, rare_class=rare_class)["rank1"]
    test_rank1_mask = gate_masks(test_pred, proto_test, rare_class=rare_class)["rank1"]
    scored_val = _score_candidates(
        val_expr,
        val_pred,
        val_rank1_mask,
        signatures=signatures,
        rare_class=rare_class,
        gene_names=genes,
    )
    scored_test = _score_candidates(
        test_expr,
        test_pred,
        test_rank1_mask,
        signatures=signatures,
        rare_class=rare_class,
        gene_names=genes,
    )

    if scored_val.empty:
        selected_threshold = np.inf
        curve = pd.DataFrame()
    else:
        curve = marker_threshold_curve(
            val_pred,
            scored_val,
            rare_class=rare_class,
            thresholds=default_marker_thresholds(scored_val),
        )
        selected_threshold = choose_marker_threshold(curve, max_false_rescue_rate=max_false_rescue_rate)
        curve.insert(0, "seed", seed)
        curve.insert(1, "rare_train_size", rare_train_size)
        curve.insert(2, "rare_class", rare_class)
        curve.insert(3, "split_mode", split_mode)
        curve.insert(4, "run", run)
        threshold_rows.append(curve)

    marker_effect = evaluate_threshold_rescue(
        test_pred,
        scored_test,
        rare_class=rare_class,
        marker_threshold=selected_threshold,
    )
    effect_rows.append(
        {
            **marker_effect,
            **common,
            "method_key": "baseline_plus_prototype_gate_plus_marker",
            "method": METHOD_LABELS["baseline_plus_prototype_gate_plus_marker"],
            "gate_name": "rank1",
        }
    )
    selected_rows.append(
        {
            **common,
            "gate_name": "rank1",
            "selected_marker_threshold": selected_threshold,
            "max_false_rescue_rate": max_false_rescue_rate,
        }
    )
    verified = scored_test[
        pd.to_numeric(scored_test.get("marker_margin", pd.Series(dtype=float)), errors="coerce")
        .ge(selected_threshold)
        .fillna(False)
    ].copy()
    if not verified.empty:
        verified_rows.append(
            _with_run_metadata(
                verified,
                seed=seed,
                rare_train_size=rare_train_size,
                rare_class=rare_class,
                split_mode=split_mode,
                run=run,
            )
        )

    probability_cols = [col for col in test_pred.columns if col.startswith("prob_")]
    val_probability_cols = [col for col in val_pred.columns if col.startswith("prob_")]
    if probability_cols and val_probability_cols:
        scanvi_test = test_pred[probability_cols].rename(columns=lambda col: col.removeprefix("prob_"))
        scanvi_val = val_pred[val_probability_cols].rename(columns=lambda col: col.removeprefix("prob_"))
        proto_prob_val = prototype_probabilities_from_reference(
            _latent_matrix(val_latent),
            reference_latent=_latent_matrix(train_latent),
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        )
        proto_prob_test = prototype_probabilities_from_reference(
            _latent_matrix(test_latent),
            reference_latent=_latent_matrix(train_latent),
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        )

        val_results = []
        baseline_accuracy = float(classification_tables(val_pred["true_label"], val_pred["predicted_label"], rare_class=rare_class)[0]["overall_accuracy"])
        for temperature, alpha_min, beta in _fusion_grid():
            fused_val = _fusion_with_params(
                scanvi_val,
                proto_prob_val,
                margin=val_pred["margin"].to_numpy(),
                temperature=temperature,
                alpha_min=alpha_min,
                beta=beta,
            )
            metrics = evaluate_fusion_effect(
                val_pred["true_label"],
                val_pred["predicted_label"],
                fused_val,
                rare_class=rare_class,
            )
            row = {**common, **metrics, "temperature": temperature, "alpha_min": alpha_min, "beta": beta}
            val_results.append(row)
            fusion_grid_rows.append(row)
        val_results_df = pd.DataFrame(val_results)
        temperature, alpha_min, beta = select_best_params(
            val_results_df,
            baseline_accuracy=baseline_accuracy,
            max_false_rescue_rate=max_false_rescue_rate,
        )
        fused_test = _fusion_with_params(
            scanvi_test,
            proto_prob_test,
            margin=test_pred["margin"].to_numpy(),
            temperature=temperature,
            alpha_min=alpha_min,
            beta=beta,
        )
        fusion_effect = evaluate_fusion_effect(
            test_pred["true_label"],
            test_pred["predicted_label"],
            fused_test,
            rare_class=rare_class,
        )
        effect_rows.append(
            {
                **fusion_effect,
                **common,
                "method_key": "baseline_plus_fusion",
                "method": METHOD_LABELS["baseline_plus_fusion"],
                "temperature": temperature,
                "alpha_min": alpha_min,
                "beta": beta,
            }
        )
    else:
        effect_rows.append(
            {
                **baseline,
                **common,
                "method_key": "baseline_plus_fusion",
                "method": METHOD_LABELS["baseline_plus_fusion"],
                "temperature": np.nan,
                "alpha_min": np.nan,
                "beta": np.nan,
            }
        )

    effect_runs = pd.DataFrame(effect_rows)
    return {
        "effect_runs": effect_runs,
        "effect_summary": summarize_four_stage_methods(effect_runs),
        "threshold_curve": pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(),
        "selected_thresholds": pd.DataFrame(selected_rows),
        "prototype_candidates": pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame(),
        "marker_verified_candidates": pd.concat(verified_rows, ignore_index=True) if verified_rows else pd.DataFrame(),
        "fusion_grid": pd.DataFrame(fusion_grid_rows),
    }


def summarize_four_stage_methods(effect_runs: pd.DataFrame) -> pd.DataFrame:
    return _flatten_summary(_summarize(effect_runs))


__all__ = [
    "METHOD_LABELS",
    "METHOD_ORDER",
    "evaluate_four_stage_methods",
    "summarize_four_stage_methods",
]
