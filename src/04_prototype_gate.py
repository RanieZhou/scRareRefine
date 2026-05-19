"""Stage 4: Evaluate prototype gate rules on validation and test cells.

Reads:
    outputs/{dataset}/{run_id}/embeddings/
    outputs/{dataset}/{run_id}/prototype/

Writes:
    outputs/{dataset}/{run_id}/gate/
        validation_results.csv   per-gate metrics
        validation_candidates.csv
        test_results.csv
        test_candidates.csv

Usage:
    python src/04_prototype_gate.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from utils import classification_tables, load_config, make_run_dir, parse_rare_train_size, read_table, write_table


def gate_masks(predictions: pd.DataFrame, prototype: pd.DataFrame, *, rare_class: str) -> dict[str, pd.Series]:
    import pandas as pd

    rank_col = f"prototype_rank_{rare_class}"
    d_col = f"d_pred_minus_d_{rare_class}"
    base = predictions["predicted_label"].astype(str).ne(rare_class)
    rank = pd.to_numeric(prototype[rank_col], errors="coerce")
    d_score = pd.to_numeric(prototype[d_col], errors="coerce")
    margin = pd.to_numeric(predictions["margin"], errors="coerce")
    entropy = pd.to_numeric(predictions["entropy"], errors="coerce")

    def _q(s, q):
        return float(pd.to_numeric(s, errors="coerce").dropna().quantile(q))

    margin_q25 = _q(margin, 0.25)
    entropy_q50 = _q(entropy, 0.50)
    d_q90 = _q(d_score, 0.90)
    pred_counts = predictions["predicted_label"].astype(str).value_counts()
    neighbor_major = predictions["predicted_label"].astype(str).isin(
        [c for c in pred_counts.index if c != rare_class][:2]
    )
    return {
        "rank1": base & rank.le(1),
        "rank2_margin_q25": base & rank.le(2) & margin.le(margin_q25),
        "rank2_dscore_q90": base & rank.le(2) & d_score.ge(d_q90),
        "rank2_margin_q25_entropy_q50": base & rank.le(2) & margin.le(margin_q25) & entropy.ge(entropy_q50),
        "rank2_margin_q25_neighbor_major": base & rank.le(2) & margin.le(margin_q25) & neighbor_major,
    }


def evaluate_gates(
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
        n_candidates = int(mask.sum())
        rescued = int((mask & rare_errors).sum())
        false_rescues = int((mask & non_rare).sum())
        rows.append({
            "gate_name": gate_name,
            **overall,
            "n_candidates": n_candidates,
            "rescued_rare_errors": rescued,
            "false_rescues": false_rescues,
            "candidate_precision_for_rare_error": rescued / n_candidates if n_candidates else 0.0,
            "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
            "modification_rate": n_candidates / len(predictions) if len(predictions) else 0.0,
            "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
        })
        if n_candidates:
            cand = predictions.loc[mask, ["cell_id", "true_label", "predicted_label", "margin", "entropy"]].copy()
            cand.insert(0, "gate_name", gate_name)
            for col in [f"prototype_rank_{rare_class}", f"d_pred_minus_d_{rare_class}", f"distance_to_{rare_class}", "distance_to_pred"]:
                if col in prototype.columns:
                    cand[col] = prototype.loc[mask, col].to_numpy()
            candidate_rows.append(cand)

    candidates = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
    return pd.DataFrame(rows), candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: evaluate prototype gate rules")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    proto_dir = run_dir / "prototype"
    gate_dir = run_dir / "gate"

    for split_name in ["validation", "test"]:
        pred = read_table(emb_dir / f"{split_name}_predictions.csv")
        proto = read_table(proto_dir / f"{split_name}_scores.csv")
        results, candidates = evaluate_gates(pred, proto, rare_class=rare_class)
        write_table(results, gate_dir / f"{split_name}_results.csv")
        write_table(candidates, gate_dir / f"{split_name}_candidates.csv")
        print(f"  {split_name}: {len(results)} gate rules, rank1 candidates = {int(results[results['gate_name'].eq('rank1')]['n_candidates'].iloc[0]) if not results.empty else 0}")

    print(f"Done. Gate results in {gate_dir}")


if __name__ == "__main__":
    main()
