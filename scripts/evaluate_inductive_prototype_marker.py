from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from scrare_refine.anndata_utils import adata_from_config, ensure_unique_names
from scrare_refine.config import load_config
from scrare_refine.io import read_table, write_table
from scrare_refine.marker_verifier import (
    choose_marker_threshold,
    compute_marker_signatures,
    default_marker_thresholds,
    evaluate_threshold_rescue,
    marker_scores_for_candidates,
    marker_threshold_curve,
)
from scrare_refine.metrics import classification_tables
from scrare_refine.output_layout import artifact_path, stage_table_path
from scrare_refine.prototype import prototype_scores_from_reference
from scrare_refine.prototype_gate import evaluate_gate_rules, gate_masks


def _csv_values(value: str | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _safe_class_name(name: str) -> str:
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def _output_root(config: dict[str, Any], *, rare_class: str, split_mode: str) -> Path:
    dataset_name = config["dataset"].get("name", "dataset")
    split_name = "inductive_cell" if split_mode == "cell_stratified" else "inductive_batch"
    return Path("outputs") / dataset_name / split_name / _safe_class_name(rare_class)


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


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


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    return out


def _log1p_cpm_dense(x) -> np.ndarray:
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        normalized = x.multiply(10000.0 / row_sum[:, None])
        return np.log1p(normalized.toarray()).astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    row_sum = arr.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(arr * (10000.0 / row_sum[:, None])).astype(np.float32)


def _expression_for_cells(adata, *, cell_ids: pd.Series, genes: list[str]) -> np.ndarray:
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


def evaluate_root(
    config: dict[str, Any],
    adata,
    *,
    rare_class: str,
    split_mode: str,
    max_false_rescue_rate: float,
    top_n: int,
    min_cells: int,
) -> None:
    root = _output_root(config, rare_class=rare_class, split_mode=split_mode)
    stage = "prototype_marker_validation"
    effect_rows: list[dict[str, Any]] = []
    threshold_rows: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []
    candidate_rows: list[pd.DataFrame] = []
    verified_rows: list[pd.DataFrame] = []

    for run_dir in _run_dirs(root):
        train_pred = read_table(artifact_path(run_dir, "train_predictions.csv")).reset_index(drop=True)
        val_pred = read_table(artifact_path(run_dir, "validation_predictions.csv")).reset_index(drop=True)
        test_pred = read_table(artifact_path(run_dir, "test_predictions.csv")).reset_index(drop=True)
        train_latent = read_table(artifact_path(run_dir, "train_latent.csv")).reset_index(drop=True)
        val_latent = read_table(artifact_path(run_dir, "validation_latent.csv")).reset_index(drop=True)
        test_latent = read_table(artifact_path(run_dir, "test_latent.csv")).reset_index(drop=True)
        genes = read_table(run_dir / "selected_hvg_genes.csv")["gene"].astype(str).tolist()

        seed = int(test_pred["seed"].iloc[0])
        rare_train_size = str(test_pred["rare_train_size"].iloc[0])
        common = {
            "seed": seed,
            "rare_train_size": rare_train_size,
            "rare_class": rare_class,
            "split_mode": split_mode,
            "run": run_dir.name,
        }

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

        baseline = _baseline_metrics(test_pred, rare_class=rare_class)
        effect_rows.append({**baseline, **common, "method": "scANVI baseline", "gate_name": ""})

        gate_effect, gate_candidates = evaluate_gate_rules(test_pred, proto_test, rare_class=rare_class)
        rank1 = gate_effect[gate_effect["gate_name"].eq("rank1")].iloc[0].to_dict()
        effect_rows.append({**rank1, **common, "method": "prototype rank1 gate"})
        if not gate_candidates.empty:
            gate_candidates = gate_candidates[gate_candidates["gate_name"].eq("rank1")].copy()
            candidate_rows.append(
                _with_run_metadata(
                    gate_candidates,
                    seed=seed,
                    rare_train_size=rare_train_size,
                    rare_class=rare_class,
                    split_mode=split_mode,
                    run=run_dir.name,
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
            curve.insert(4, "run", run_dir.name)
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
                "method": "validation-tuned marker",
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
        verified = scored_test[pd.to_numeric(scored_test.get("marker_margin", pd.Series(dtype=float)), errors="coerce").ge(selected_threshold).fillna(False)].copy()
        if not verified.empty:
            verified_rows.append(
                _with_run_metadata(
                    verified,
                    seed=seed,
                    rare_train_size=rare_train_size,
                    rare_class=rare_class,
                    split_mode=split_mode,
                    run=run_dir.name,
                )
            )

        print(
            f"{run_dir.name}: baseline F1={baseline['rare_f1']:.3f}, "
            f"rank1 F1={rank1['rare_f1']:.3f}, marker F1={marker_effect['rare_f1']:.3f}"
        )

    if not effect_rows:
        raise FileNotFoundError(f"No inductive runs found under {root / 'runs'}")

    effect_runs = pd.DataFrame(effect_rows)
    summary = _summarize(effect_runs)
    write_table(effect_runs, stage_table_path(root, stage, "prototype_marker_effect_runs.csv"))
    write_table(_flatten_summary(summary), stage_table_path(root, stage, "prototype_marker_effect_summary.csv"))
    write_table(pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame(), stage_table_path(root, stage, "validation_marker_threshold_curve.csv"))
    write_table(pd.DataFrame(selected_rows), stage_table_path(root, stage, "selected_marker_thresholds.csv"))
    write_table(pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame(), stage_table_path(root, stage, "prototype_rank1_test_candidates.csv"))
    write_table(pd.concat(verified_rows, ignore_index=True) if verified_rows else pd.DataFrame(), stage_table_path(root, stage, "marker_verified_test_candidates.csv"))
    print(f"Wrote prototype/marker outputs to {root / 'stages' / stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inductive prototype rank1 and validation-tuned marker on held-out test cells.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare-class", default="ASDC,cDC1")
    parser.add_argument("--split-mode", default="batch_heldout")
    parser.add_argument("--max-false-rescue-rate", type=float, default=0.001)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    adata = adata_from_config(config)
    ensure_unique_names(adata)
    for rare_class in _csv_values(args.rare_class, [config["experiment"]["rare_class"]]):
        for split_mode in _csv_values(args.split_mode, ["batch_heldout"]):
            evaluate_root(
                config,
                adata,
                rare_class=rare_class,
                split_mode=split_mode,
                max_false_rescue_rate=args.max_false_rescue_rate,
                top_n=args.top_n,
                min_cells=args.min_cells,
            )


if __name__ == "__main__":
    main()
