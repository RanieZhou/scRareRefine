"""Stage 6: Prototype-scANVI probability fusion with validation-driven parameter selection.

Two fusion variants are computed and saved:

  fusion       — global: blend scANVI + prototype probs for ALL cells (original method)
  fusion_gated — targeted: apply entropy-weighted fusion ONLY to prototype rank-1 candidates;
                 all other cells keep scANVI predictions unchanged.

Parameters are selected on validation, then applied to test (inductive constraint preserved).

Reads:
    outputs/{dataset}/{run_id}/embeddings/
    outputs/{dataset}/{run_id}/prototype/       (for fusion_gated candidate mask)

Writes:
    outputs/{dataset}/{run_id}/fusion/
        validation_grid.csv          global fusion val metrics
        best_params.csv              global best params
        test_results.csv             global fusion test metrics
        gated_validation_grid.csv    gated fusion val metrics
        gated_best_params.csv        gated best params
        gated_test_results.csv       gated fusion test metrics

Usage:
    python src/06_fusion.py \\
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

from utils import classification_tables, load_config, make_run_dir, parse_rare_train_size, read_table, write_table


def _latent_matrix(latent_df: pd.DataFrame) -> np.ndarray:
    return latent_df[[c for c in latent_df.columns if c.startswith("latent_")]].to_numpy()


# ── Shared: prototype probability computation ─────────────────────────────────

def prototype_probabilities(
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
        raise ValueError("No labeled reference cells for prototype probabilities")
    proto_vecs = np.vstack([
        reference_latent[reference_is_labeled & reference_labels.eq(cls).to_numpy()].mean(axis=0)
        for cls in classes
    ])
    distances = np.sqrt(((query_latent[:, None, :] - proto_vecs[None, :, :]) ** 2).sum(axis=2))
    logits = -distances / max(temperature, 1e-8)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return pd.DataFrame(probs, columns=classes)


def fusion_effect(
    y_true: pd.Series,
    baseline_pred: pd.Series,
    fused_pred: pd.Series,
    *,
    rare_class: str,
) -> dict:
    overall, _ = classification_tables(y_true, fused_pred, rare_class=rare_class)
    bl = baseline_pred.astype(str)
    fu = fused_pred.astype(str)
    yt = y_true.astype(str)
    changed = bl.ne(fu)
    rare_errors = yt.eq(rare_class) & bl.ne(rare_class)
    non_rare = yt.ne(rare_class)
    rescued = int((changed & rare_errors).sum())
    false_rescues = int((changed & non_rare & fu.eq(rare_class)).sum())
    n_changed = int(changed.sum())
    overall.update({
        "n_changed": n_changed,
        "modification_rate": n_changed / len(y_true) if len(y_true) else 0.0,
        "rescued_rare_errors": rescued,
        "false_rescues": false_rescues,
        "rare_error_recall": rescued / int(rare_errors.sum()) if int(rare_errors.sum()) else 0.0,
        "major_to_rare_false_rescue_rate": false_rescues / int(non_rare.sum()) if int(non_rare.sum()) else 0.0,
    })
    return overall


# ── Global fusion (original method) ──────────────────────────────────────────

def fuse(
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    *,
    margin: np.ndarray,
    alpha_min: float,
    beta: float,
) -> pd.Series:
    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    if not common:
        raise ValueError("No overlapping class columns between scANVI and prototype probabilities")
    alpha = np.clip(
        alpha_min + (1.0 - alpha_min) * np.clip(np.asarray(margin, dtype=float), 0.0, 1.0),
        0.0, 1.0,
    )
    if beta < 1.0:
        s_top = np.array(common)[p_scanvi[common].to_numpy(dtype=float).argmax(axis=1)]
        p_top = np.array(common)[p_proto[common].to_numpy(dtype=float).argmax(axis=1)]
        alpha[s_top != p_top] *= beta
        alpha = np.clip(alpha, 0.0, 1.0)
    a = alpha[:, None]
    fused = a * p_scanvi[common].to_numpy(dtype=float) + (1.0 - a) * p_proto[common].to_numpy(dtype=float)
    fused /= fused.sum(axis=1, keepdims=True) + 1e-12
    return pd.Series([common[i] for i in fused.argmax(axis=1)], index=p_scanvi.index)


def _fusion_grid() -> list[tuple[float, float, float]]:
    return [
        (temperature, alpha_min, beta)
        for temperature in [0.5, 1.0, 2.0]
        for alpha_min in [0.3, 0.5, 0.7]
        for beta in [0.5, 1.0]
    ]


def select_best_params(
    val_results: pd.DataFrame,
    *,
    baseline_accuracy: float,
    max_false_rescue_rate: float = 0.005,
) -> tuple[float, float, float]:
    eligible = val_results[
        val_results["overall_accuracy"].ge(baseline_accuracy - 0.005)
        & val_results["major_to_rare_false_rescue_rate"].le(max_false_rescue_rate)
    ]
    if eligible.empty:
        eligible = val_results.copy()
    best = eligible.sort_values(
        ["rare_f1", "overall_accuracy", "major_to_rare_false_rescue_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["temperature"]), float(best["alpha_min"]), float(best.get("beta", 1.0))


# ── Gated fusion (new method) ─────────────────────────────────────────────────

def _rank1_mask(predictions: pd.DataFrame, proto_scores: pd.DataFrame, *, rare_class: str) -> np.ndarray:
    """Boolean mask: cells predicted non-rare whose prototype rank for rare class == 1."""
    rank_col = f"prototype_rank_{rare_class}"
    not_rare = predictions["predicted_label"].astype(str).ne(rare_class)
    rank1 = pd.to_numeric(proto_scores[rank_col], errors="coerce").eq(1)
    return (not_rare & rank1).fillna(False).to_numpy(dtype=bool)


def gated_fuse(
    predictions: pd.DataFrame,
    p_scanvi: pd.DataFrame,
    p_proto: pd.DataFrame,
    candidate_mask: np.ndarray,
    *,
    rare_class: str,
    alpha: float,
    rare_prob_threshold: float,
) -> pd.Series:
    """Rescue rank-1 candidates whose fused rare-class probability exceeds a threshold.

    For each rank-1 candidate:
        fused_rare_prob = (1 - alpha) * p_proto[rare] + alpha * p_scanvi[rare]
        if fused_rare_prob >= rare_prob_threshold  →  relabel as rare_class
        else                                       →  keep scANVI prediction

    Non-candidates are never modified.

    alpha controls the scANVI vs prototype balance (0 = full prototype, 1 = full scANVI).
    rare_prob_threshold controls precision/recall trade-off.
    Both are selected on validation.
    """
    result = predictions["predicted_label"].astype(str).copy()

    if not candidate_mask.any():
        return result

    common = sorted(set(p_scanvi.columns) & set(p_proto.columns))
    if rare_class not in common:
        return result

    rare_idx = common.index(rare_class)
    p_s_rare = p_scanvi[common].to_numpy(dtype=float)[candidate_mask, rare_idx]
    p_p_rare = p_proto[common].to_numpy(dtype=float)[candidate_mask, rare_idx]

    fused_rare = (1.0 - alpha) * p_p_rare + alpha * p_s_rare

    cand_indices = np.where(candidate_mask)[0]
    for idx, p_rare in zip(cand_indices, fused_rare):
        if p_rare >= rare_prob_threshold:
            result.iloc[idx] = rare_class

    return result


def _gated_fusion_grid() -> list[tuple[float, float, float]]:
    return [
        (temperature, alpha, threshold)
        for temperature in [0.5, 1.0, 2.0]
        for alpha in [0.0, 0.2, 0.4]          # scANVI weight; 0 = pure prototype signal
        for threshold in [0.3, 0.5, 0.7]       # rescue if fused rare prob >= this
    ]


def select_best_gated_params(
    val_results: pd.DataFrame,
    *,
    baseline_accuracy: float,
    max_false_rescue_rate: float = 0.005,
) -> tuple[float, float, float]:
    eligible = val_results[
        val_results["overall_accuracy"].ge(baseline_accuracy - 0.005)
        & val_results["major_to_rare_false_rescue_rate"].le(max_false_rescue_rate)
    ]
    if eligible.empty:
        eligible = val_results.copy()
    best = eligible.sort_values(
        ["rare_f1", "overall_accuracy", "major_to_rare_false_rescue_rate"],
        ascending=[False, False, True],
    ).iloc[0]
    return float(best["temperature"]), float(best["alpha"]), float(best["rare_prob_threshold"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 6: prototype-scANVI fusion (global + gated)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--max_false_rescue_rate", type=float, default=0.005)
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    proto_dir = run_dir / "prototype"
    out_dir = run_dir / "fusion"

    train_pred   = read_table(emb_dir / "train_predictions.csv")
    train_latent = read_table(emb_dir / "train_latent.csv")
    val_pred     = read_table(emb_dir / "validation_predictions.csv")
    val_latent   = read_table(emb_dir / "validation_latent.csv")
    test_pred    = read_table(emb_dir / "test_predictions.csv")
    test_latent  = read_table(emb_dir / "test_latent.csv")

    prob_cols = [c for c in val_pred.columns if c.startswith("prob_")]
    if not prob_cols:
        print("No prob_ columns found in predictions — skipping fusion.")
        return

    scanvi_val  = val_pred[prob_cols].rename(columns=lambda c: c.removeprefix("prob_"))
    scanvi_test = test_pred[[c for c in test_pred.columns if c.startswith("prob_")]].rename(
        columns=lambda c: c.removeprefix("prob_")
    )

    baseline_accuracy = float(
        classification_tables(
            val_pred["true_label"], val_pred["predicted_label"], rare_class=rare_class
        )[0]["overall_accuracy"]
    )

    ref_latent     = _latent_matrix(train_latent)
    ref_labels     = train_pred["true_label"]
    ref_is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()

    # ── Global fusion (original) ──────────────────────────────────────────────
    print(f"Running global fusion grid search ({len(_fusion_grid())} combos) ...")
    val_results = []
    for temperature, alpha_min, beta in _fusion_grid():
        proto_val = prototype_probabilities(
            _latent_matrix(val_latent),
            reference_latent=ref_latent, reference_labels=ref_labels,
            reference_is_labeled=ref_is_labeled, temperature=temperature,
        )
        fused_val = fuse(
            scanvi_val, proto_val,
            margin=val_pred["margin"].to_numpy(), alpha_min=alpha_min, beta=beta,
        )
        metrics = fusion_effect(
            val_pred["true_label"], val_pred["predicted_label"], fused_val, rare_class=rare_class
        )
        val_results.append({**metrics, "temperature": temperature, "alpha_min": alpha_min, "beta": beta})

    val_df = pd.DataFrame(val_results)
    temperature, alpha_min, beta = select_best_params(
        val_df, baseline_accuracy=baseline_accuracy,
        max_false_rescue_rate=args.max_false_rescue_rate,
    )
    print(f"  Best global params: T={temperature}, alpha_min={alpha_min}, beta={beta}")

    proto_test = prototype_probabilities(
        _latent_matrix(test_latent),
        reference_latent=ref_latent, reference_labels=ref_labels,
        reference_is_labeled=ref_is_labeled, temperature=temperature,
    )
    fused_test = fuse(
        scanvi_test, proto_test,
        margin=test_pred["margin"].to_numpy(), alpha_min=alpha_min, beta=beta,
    )
    test_metrics = fusion_effect(
        test_pred["true_label"], test_pred["predicted_label"], fused_test, rare_class=rare_class
    )

    write_table(val_df, out_dir / "validation_grid.csv")
    write_table(pd.DataFrame([{"temperature": temperature, "alpha_min": alpha_min, "beta": beta}]),
                out_dir / "best_params.csv")
    write_table(pd.DataFrame([test_metrics]), out_dir / "test_results.csv")
    print(f"  Global  → rare_f1={test_metrics['rare_f1']:.4f}  acc={test_metrics['overall_accuracy']:.4f}")

    # ── Gated fusion (new method) ─────────────────────────────────────────────
    try:
        val_proto  = read_table(proto_dir / "validation_scores.csv")
        test_proto = read_table(proto_dir / "test_scores.csv")
    except FileNotFoundError:
        print("  Prototype scores not found — skipping gated fusion. Run 03_prototype.py first.")
        print(f"Done. Fusion results in {out_dir}")
        return

    val_mask  = _rank1_mask(val_pred,  val_proto,  rare_class=rare_class)
    test_mask = _rank1_mask(test_pred, test_proto, rare_class=rare_class)
    print(f"\nRunning gated fusion grid search ({len(_gated_fusion_grid())} combos) ...")
    print(f"  Val rank-1 candidates: {val_mask.sum()}  |  Test rank-1 candidates: {test_mask.sum()}")

    gated_val_results = []
    for temperature, alpha, threshold in _gated_fusion_grid():
        proto_val = prototype_probabilities(
            _latent_matrix(val_latent),
            reference_latent=ref_latent, reference_labels=ref_labels,
            reference_is_labeled=ref_is_labeled, temperature=temperature,
        )
        gated_pred_val = gated_fuse(
            val_pred, scanvi_val, proto_val, val_mask,
            rare_class=rare_class, alpha=alpha, rare_prob_threshold=threshold,
        )
        metrics = fusion_effect(
            val_pred["true_label"], val_pred["predicted_label"], gated_pred_val, rare_class=rare_class
        )
        gated_val_results.append({**metrics, "temperature": temperature, "alpha": alpha, "rare_prob_threshold": threshold})

    gated_val_df = pd.DataFrame(gated_val_results)
    g_temperature, g_alpha, g_threshold = select_best_gated_params(
        gated_val_df, baseline_accuracy=baseline_accuracy,
        max_false_rescue_rate=args.max_false_rescue_rate,
    )
    print(f"  Best gated params: T={g_temperature}, alpha={g_alpha}, threshold={g_threshold}")

    proto_test_gated = prototype_probabilities(
        _latent_matrix(test_latent),
        reference_latent=ref_latent, reference_labels=ref_labels,
        reference_is_labeled=ref_is_labeled, temperature=g_temperature,
    )
    gated_pred_test = gated_fuse(
        test_pred, scanvi_test, proto_test_gated, test_mask,
        rare_class=rare_class, alpha=g_alpha, rare_prob_threshold=g_threshold,
    )
    gated_test_metrics = fusion_effect(
        test_pred["true_label"], test_pred["predicted_label"], gated_pred_test, rare_class=rare_class
    )

    write_table(gated_val_df, out_dir / "gated_validation_grid.csv")
    write_table(pd.DataFrame([{"temperature": g_temperature, "alpha": g_alpha, "rare_prob_threshold": g_threshold}]),
                out_dir / "gated_best_params.csv")
    write_table(pd.DataFrame([gated_test_metrics]), out_dir / "gated_test_results.csv")
    print(f"  Gated   → rare_f1={gated_test_metrics['rare_f1']:.4f}  acc={gated_test_metrics['overall_accuracy']:.4f}")

    print(f"\nDone. Fusion results in {out_dir}")


if __name__ == "__main__":
    main()
