"""Stage 5: Marker-verified rescue using prototype rank1 candidates.

Threshold is selected from validation, then applied to test (inductive constraint preserved).

Reads:
    h5ad (for expression)
    outputs/{dataset}/{run_id}/embeddings/
    outputs/{dataset}/{run_id}/prototype/
    outputs/{dataset}/{run_id}/selected_hvg_genes.csv

Writes:
    outputs/{dataset}/{run_id}/gate_marker/
        validation_scored.csv       rank1 candidates with marker scores
        test_scored.csv
        marker_threshold_curve.csv  val metrics at each threshold
        selected_thresholds.csv     threshold chosen on val

Usage:
    python src/05_prototype_gate_marker.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from utils import (
    classification_tables,
    load_config,
    load_adata,
    log1p_cpm,
    make_run_dir,
    parse_rare_train_size,
    read_table,
    write_table,
)


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


def score_candidates(
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
    for row_num, (_, row) in enumerate(candidates.iterrows()):
        pred = str(row["predicted_label"])
        pred_genes = [gene_to_idx[g] for g in signatures.get(pred, []) if g in gene_to_idx]
        rare_score = float(expr[row_num, rare_genes].mean()) if rare_genes else 0.0
        pred_score = float(expr[row_num, pred_genes].mean()) if pred_genes else 0.0
        margin = rare_score - pred_score
        rows.append({
            f"marker_score_{rare_class}": rare_score,
            "marker_score_predicted": pred_score,
            "marker_margin": margin,
            "marker_verified": margin > 0,
        })
    return pd.DataFrame(rows, index=candidates.index)


def marker_threshold_curve(
    predictions: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    *,
    rare_class: str,
    thresholds: list[float],
) -> pd.DataFrame:
    y_true = predictions["true_label"].astype(str)
    baseline_pred = predictions["predicted_label"].astype(str)
    rare_errors = y_true.eq(rare_class) & baseline_pred.ne(rare_class)
    non_rare = y_true.ne(rare_class)
    margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce")
    rows = []
    for threshold in thresholds:
        verified = margins.ge(threshold).fillna(False)
        verified_ids = set(scored_candidates.loc[verified, "cell_id"].astype(str)) if "cell_id" in scored_candidates.columns else set()
        relabeled = baseline_pred.copy()
        if "cell_id" in predictions.columns:
            relabeled.loc[predictions["cell_id"].astype(str).isin(verified_ids)] = rare_class
        overall, _ = classification_tables(y_true, relabeled, rare_class=rare_class)
        n_verified = int(verified.sum())
        rescued = int(rare_errors.loc[predictions["cell_id"].astype(str).isin(verified_ids)].sum()) if "cell_id" in predictions.columns and n_verified else 0
        false_rescues = int(non_rare.loc[predictions["cell_id"].astype(str).isin(verified_ids)].sum()) if "cell_id" in predictions.columns and n_verified else 0
        overall.update({
            "marker_threshold": float(threshold),
            "n_candidates": len(scored_candidates),
            "n_marker_verified": n_verified,
            "rescued_rare_errors": rescued,
            "false_rescues": false_rescues,
            "candidate_precision_for_rare_error": rescued / n_verified if n_verified else 0.0,
            "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
            "modification_rate": n_verified / len(predictions) if len(predictions) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
        })
        rows.append(overall)
    return pd.DataFrame(rows)


def choose_threshold(curve: pd.DataFrame, *, max_false_rescue_rate: float = 0.001) -> float:
    eligible = curve[curve["major_to_rare_false_rescue_rate"].le(max_false_rescue_rate)].copy()
    if eligible.empty:
        eligible = curve.copy()
    eligible = eligible.sort_values(
        ["rare_f1", "rare_recall", "rare_precision", "marker_threshold"],
        ascending=[False, False, False, True],
    )
    return float(eligible["marker_threshold"].iloc[0])


def default_thresholds(scored_candidates: pd.DataFrame) -> list[float]:
    margins = pd.to_numeric(scored_candidates["marker_margin"], errors="coerce").dropna()
    if margins.empty:
        return [0.0]
    quantiles = margins.quantile([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).tolist()
    return sorted({float(x) for x in quantiles + [-1.0, -0.5, 0.0, 0.5, 1.0]})


def _rank1_candidate_ids(predictions: pd.DataFrame, prototype_scores: pd.DataFrame, *, rare_class: str) -> pd.Series:
    rank_col = f"prototype_rank_{rare_class}"
    base = predictions["predicted_label"].astype(str).ne(rare_class)
    rank = pd.to_numeric(prototype_scores[rank_col], errors="coerce")
    return (base & rank.le(1)).fillna(False)


def _load_expression_for_cells(adata, cell_ids: list[str], genes: list[str]) -> np.ndarray:
    subset = adata[cell_ids, genes]
    return log1p_cpm(subset.X)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: prototype gate + marker verification")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", choices=["batch_heldout", "cell_stratified"])
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--max_false_rescue_rate", type=float, default=0.001)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    proto_dir = run_dir / "prototype"
    out_dir = run_dir / "gate_marker"

    genes = read_table(run_dir / "selected_hvg_genes.csv")["gene"].astype(str).tolist()
    train_pred = read_table(emb_dir / "train_predictions.csv")

    print("Loading expression data for marker signatures ...")
    adata = load_adata(config)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    train_cell_ids = train_pred["cell_id"].astype(str).tolist()
    train_expr = _load_expression_for_cells(adata, train_cell_ids, genes)
    signatures = compute_marker_signatures(
        train_expr,
        gene_names=genes,
        labels=train_pred["true_label"],
        is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
    )
    print(f"  Marker signatures computed for {len(signatures)} cell types")
    # Save marker signatures for downstream analysis/visualization
    sig_rows = [{"cell_type": ct, "gene": g, "rank": i + 1}
                for ct, genes_list in signatures.items() for i, g in enumerate(genes_list)]
    if sig_rows:
        write_table(pd.DataFrame(sig_rows), out_dir / "marker_signatures.csv")

    # ── Validation: score candidates, build threshold curve, select threshold ──
    val_pred = read_table(emb_dir / "validation_predictions.csv")
    val_proto = read_table(proto_dir / "validation_scores.csv")
    val_mask = _rank1_candidate_ids(val_pred, val_proto, rare_class=rare_class)
    val_candidates = val_pred.loc[val_mask].copy().reset_index(drop=True)

    if not val_candidates.empty:
        val_cell_ids = val_candidates["cell_id"].astype(str).tolist()
        val_expr = _load_expression_for_cells(adata, val_cell_ids, genes)
        val_marker_scores = score_candidates(val_expr, val_candidates, signatures=signatures, rare_class=rare_class, gene_names=genes)
        val_scored = pd.concat([val_candidates, val_marker_scores], axis=1)
        curve = marker_threshold_curve(val_pred, val_scored, rare_class=rare_class, thresholds=default_thresholds(val_scored))
        selected_threshold = choose_threshold(curve, max_false_rescue_rate=args.max_false_rescue_rate)
    else:
        val_scored = val_candidates.copy()
        curve = pd.DataFrame()
        selected_threshold = float("inf")

    write_table(val_scored, out_dir / "validation_scored.csv")
    write_table(curve, out_dir / "marker_threshold_curve.csv")
    write_table(
        pd.DataFrame([{"selected_marker_threshold": selected_threshold, "max_false_rescue_rate": args.max_false_rescue_rate}]),
        out_dir / "selected_thresholds.csv",
    )
    print(f"  Val candidates: {len(val_candidates)}, selected threshold: {selected_threshold:.4f}")

    # ── Test: score candidates using the threshold selected on val ─────────────
    test_pred = read_table(emb_dir / "test_predictions.csv")
    test_proto = read_table(proto_dir / "test_scores.csv")
    test_mask = _rank1_candidate_ids(test_pred, test_proto, rare_class=rare_class)
    test_candidates = test_pred.loc[test_mask].copy().reset_index(drop=True)

    if not test_candidates.empty:
        test_cell_ids = test_candidates["cell_id"].astype(str).tolist()
        test_expr = _load_expression_for_cells(adata, test_cell_ids, genes)
        test_marker_scores = score_candidates(test_expr, test_candidates, signatures=signatures, rare_class=rare_class, gene_names=genes)
        test_scored = pd.concat([test_candidates, test_marker_scores], axis=1)
    else:
        test_scored = test_candidates.copy()

    write_table(test_scored, out_dir / "test_scored.csv")
    print(f"  Test candidates: {len(test_candidates)}")
    print(f"Done. Gate+marker outputs in {out_dir}")


if __name__ == "__main__":
    main()
