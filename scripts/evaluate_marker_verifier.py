from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from scrare_refine.anndata_utils import adata_from_config, ensure_unique_names, select_top_variable_genes
from scrare_refine.config import load_config, output_dir
from scrare_refine.io import read_table, write_table
from scrare_refine.marker_verifier import (
    choose_marker_threshold,
    compute_marker_signatures,
    default_marker_thresholds,
    evaluate_marker_verified_rescue,
    marker_scores_for_candidates,
    marker_threshold_curve,
)
from scrare_refine.output_layout import existing_table_path, stage_table_path
from scrare_refine.prototype_gate import gate_masks


def _run_dirs(root: Path) -> list[Path]:
    runs = root / "runs"
    return sorted(path for path in runs.iterdir() if path.is_dir()) if runs.exists() else []


def _log1p_cpm_dense(adata) -> np.ndarray:
    x = adata.X
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        normalized = x.multiply(10000.0 / row_sum[:, None])
        return np.log1p(normalized.toarray()).astype(np.float32)
    arr = np.asarray(x, dtype=np.float32)
    row_sum = arr.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(arr * (10000.0 / row_sum[:, None])).astype(np.float32)


def _flatten_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in out.columns]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate train-only marker verification for prototype-gated candidates.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gate", default=None, help="Prototype gate to verify; defaults to recommended_gate.csv")
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--min-cells", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_dir(config)
    rare_class = config["experiment"]["rare_class"]
    gate = args.gate
    if gate is None:
        gate_path = stage_table_path(root, "prototype_gate", "recommended_gate.csv")
        gate = pd.read_csv(gate_path)["recommended_gate"].iloc[0] if gate_path.exists() else "rank1"

    adata = adata_from_config(config)
    ensure_unique_names(adata)
    adata = select_top_variable_genes(adata, n_top_genes=int(config["model"].get("n_top_hvg", 3000)))
    expression = _log1p_cpm_dense(adata)
    gene_names = adata.var_names.astype(str).tolist()

    effect_rows = []
    threshold_rows = []
    recommended_rows = []
    candidate_rows = []
    all_scored_candidate_rows = []
    signature_rows = []

    for run_dir in _run_dirs(root):
        pred = read_table(existing_table_path(run_dir, "scanvi_predictions.parquet"))
        proto = read_table(existing_table_path(run_dir, "prototype_scores.csv"))
        seed = int(pred["seed"].iloc[0])
        rare_train_size = str(pred["rare_train_size"].iloc[0])
        pred = pred.reset_index(drop=True)
        proto = proto.reset_index(drop=True)
        masks = gate_masks(pred, proto, rare_class=rare_class)
        candidate_mask = masks[gate].fillna(False).astype(bool)
        candidates = pred.loc[candidate_mask].copy()
        if candidates.empty:
            continue

        signatures = compute_marker_signatures(
            expression,
            gene_names=gene_names,
            labels=pred["true_label"],
            is_labeled=pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            top_n=args.top_n,
            min_cells=args.min_cells,
        )
        for cell_type, genes in signatures.items():
            signature_rows.append(
                {
                    "seed": seed,
                    "rare_train_size": rare_train_size,
                    "run": run_dir.name,
                    "cell_type": cell_type,
                    "genes": ";".join(genes),
                }
            )

        scored = marker_scores_for_candidates(
            expression,
            candidates,
            signatures=signatures,
            rare_class=rare_class,
            gene_names=gene_names,
        )
        scored_candidates = pd.concat([candidates, scored], axis=1)
        for col in ["seed", "rare_train_size", "run", "gate_name"]:
            if col in scored_candidates.columns:
                scored_candidates = scored_candidates.drop(columns=[col])
        scored_candidates.insert(0, "seed", seed)
        scored_candidates.insert(1, "rare_train_size", rare_train_size)
        scored_candidates.insert(2, "run", run_dir.name)
        scored_candidates.insert(3, "gate_name", gate)
        all_scored_candidate_rows.append(scored_candidates)

        thresholds = default_marker_thresholds(scored_candidates)
        curve = marker_threshold_curve(pred, scored_candidates, rare_class=rare_class, thresholds=thresholds)
        curve.insert(0, "seed", seed)
        curve.insert(1, "rare_train_size", rare_train_size)
        curve.insert(2, "run", run_dir.name)
        curve.insert(3, "gate_name", gate)
        threshold_rows.append(curve)
        recommended_threshold = choose_marker_threshold(curve, max_false_rescue_rate=0.001)
        recommended_rows.append(
            {
                "seed": seed,
                "rare_train_size": rare_train_size,
                "run": run_dir.name,
                "gate_name": gate,
                "recommended_marker_threshold": recommended_threshold,
            }
        )

        effect, verified_candidates = evaluate_marker_verified_rescue(
            pred,
            candidates,
            expression,
            signatures=signatures,
            rare_class=rare_class,
            gene_names=gene_names,
        )
        effect.update({"seed": seed, "rare_train_size": rare_train_size, "run": run_dir.name, "gate_name": gate})
        effect_rows.append(effect)
        if not verified_candidates.empty:
            for col in ["seed", "rare_train_size", "run", "gate_name"]:
                if col in verified_candidates.columns:
                    verified_candidates = verified_candidates.drop(columns=[col])
            verified_candidates.insert(0, "seed", seed)
            verified_candidates.insert(1, "rare_train_size", rare_train_size)
            verified_candidates.insert(2, "run", run_dir.name)
            verified_candidates.insert(3, "gate_name", gate)
            candidate_rows.append(verified_candidates)

    effect_runs = pd.DataFrame(effect_rows)
    threshold_curve = pd.concat(threshold_rows, ignore_index=True) if threshold_rows else pd.DataFrame()
    recommended_thresholds = pd.DataFrame(recommended_rows)
    summary = (
        effect_runs.groupby(["rare_train_size", "gate_name"])[
            [
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
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    write_table(effect_runs, stage_table_path(root, "marker_verifier", "marker_effect_runs.csv"))
    write_table(_flatten_summary(summary), stage_table_path(root, "marker_verifier", "marker_effect_summary.csv"))
    write_table(threshold_curve, stage_table_path(root, "marker_verifier", "marker_threshold_curve_runs.csv"))
    write_table(recommended_thresholds, stage_table_path(root, "marker_verifier", "recommended_marker_thresholds.csv"))
    write_table(pd.DataFrame(signature_rows), stage_table_path(root, "marker_verifier", "marker_signatures.csv"))
    scored_all = pd.concat(all_scored_candidate_rows, ignore_index=True) if all_scored_candidate_rows else pd.DataFrame()
    write_table(scored_all, stage_table_path(root, "marker_verifier", "marker_candidate_scores.csv"))
    verified = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    write_table(verified, stage_table_path(root, "marker_verifier", "marker_verified_candidates.csv"))
    print(f"Wrote marker verifier outputs to {root / 'stages' / 'marker_verifier'}")
    print(f"Gate verified: {gate}")


if __name__ == "__main__":
    main()
