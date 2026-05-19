"""scRareRefine main pipeline entry point.

Runs the scRareRefine rescue pipeline (prototype scoring → gate → marker
verification) on pre-existing scANVI embeddings and outputs a focused
comparison between scRareRefine and the scANVI baseline.

Prerequisites (must already exist):
    outputs/{dataset}/{run_id}/embeddings/     from 02_baseline_scanvi.py
    outputs/{dataset}/{run_id}/selected_hvg_genes.csv

Intermediate results are cached (prototype/, gate_marker/). Use --force to
recompute even when cache exists.

Usage:
    python src/main.py --config configs/immune_dc.yaml --seed 42 \\
        --rare_class ASDC --rare_train_size 20
    python src/main.py --config configs/immune_dc.yaml --seed 42 \\
        --rare_class ASDC --rare_train_size 20 --force
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    classification_tables,
    load_adata,
    load_config,
    log1p_cpm,
    make_run_dir,
    parse_rare_train_size,
    read_table,
    write_table,
)

_SRC = Path(__file__).parent


def _import(filename: str):
    spec = importlib.util.spec_from_file_location(filename, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_proto_mod = _import("03_prototype.py")
_gm_mod = _import("05_prototype_gate_marker.py")

_separability_metrics = _proto_mod.separability_metrics
_prototype_scores = _proto_mod.prototype_scores
_compute_marker_signatures = _gm_mod.compute_marker_signatures
_score_candidates = _gm_mod.score_candidates
_marker_threshold_curve = _gm_mod.marker_threshold_curve
_choose_threshold = _gm_mod.choose_threshold
_default_thresholds = _gm_mod.default_thresholds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _latent(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _require(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found: {path}\n"
            f"  ({label})"
        )
    return read_table(path)


def _load_expr(adata, cell_ids: list[str], genes: list[str]) -> np.ndarray:
    from scipy import sparse
    idx = adata.obs_names.isin(cell_ids)
    sub = adata[idx]
    id_pos = {cid: i for i, cid in enumerate(sub.obs_names)}
    ordered = [id_pos[c] for c in cell_ids if c in id_pos]
    sub = sub[ordered]
    var_idx = [sub.var_names.get_loc(g) for g in genes if g in sub.var_names]
    X = sub.X
    if sparse.issparse(X):
        X = X.toarray()
    return log1p_cpm(np.asarray(X, dtype=np.float32)[:, var_idx])


# ── Stage: prototype scores + separability ────────────────────────────────────

def _run_prototype(run_dir: Path, *, rare_class: str, force: bool) -> dict:
    proto_dir = run_dir / "prototype"
    emb_dir = run_dir / "embeddings"
    cached = not force and (proto_dir / "separability.csv").exists() \
             and (proto_dir / "validation_scores.csv").exists() \
             and (proto_dir / "test_scores.csv").exists()
    if cached:
        print("  [prototype] cache hit")
        return read_table(proto_dir / "separability.csv").iloc[0].to_dict()

    train_pred = _require(emb_dir / "train_predictions.csv", "train embeddings")
    train_lat = _require(emb_dir / "train_latent.csv", "train latent")
    margin = train_pred["margin"].to_numpy() if "margin" in train_pred.columns else np.ones(len(train_pred))

    sep = _separability_metrics(
        _latent(train_lat),
        reference_labels=train_pred["true_label"],
        reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
        rare_class=rare_class,
    )
    write_table(pd.DataFrame([sep]), proto_dir / "separability.csv")
    print(f"  [prototype] sep_ratio={sep['separability_ratio']:.3f} [{sep['rescue_confidence']}]")

    for split in ("validation", "test"):
        pred = _require(emb_dir / f"{split}_predictions.csv", f"{split} embeddings")
        lat = _require(emb_dir / f"{split}_latent.csv", f"{split} latent")
        scores = _prototype_scores(
            _latent(lat),
            reference_latent=_latent(train_lat),
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            predicted_labels=pred["predicted_label"],
            rare_class=rare_class,
            margin=margin,
        )
        write_table(scores, proto_dir / f"{split}_scores.csv")
        print(f"  [prototype] {split}: {int(scores['prototype_rescue_candidate'].sum())} candidates")
    return sep


# ── Stage: gate + marker verification ────────────────────────────────────────

def _run_gate_marker(
    run_dir: Path, adata, *, rare_class: str, genes: list[str],
    max_false_rescue_rate: float, force: bool,
) -> float:
    out_dir = run_dir / "gate_marker"
    emb_dir = run_dir / "embeddings"
    proto_dir = run_dir / "prototype"
    threshold_path = out_dir / "selected_thresholds.csv"
    cached = not force and threshold_path.exists() and (out_dir / "test_scored.csv").exists()
    if cached:
        print("  [gate_marker] cache hit")
        return float(read_table(threshold_path)["selected_marker_threshold"].iloc[0])

    train_pred = _require(emb_dir / "train_predictions.csv", "train embeddings")
    train_expr = _load_expr(adata, train_pred["cell_id"].astype(str).tolist(), genes)
    signatures = _compute_marker_signatures(
        train_expr, gene_names=genes,
        labels=train_pred["true_label"],
        is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
    )
    print(f"  [gate_marker] signatures for {len(signatures)} types")
    sig_rows = [{"cell_type": ct, "gene": g, "rank": i + 1}
                for ct, gs in signatures.items() for i, g in enumerate(gs)]
    if sig_rows:
        write_table(pd.DataFrame(sig_rows), out_dir / "marker_signatures.csv")

    def _rank1_mask(pred: pd.DataFrame, proto: pd.DataFrame) -> np.ndarray:
        col = f"prototype_rank_{rare_class}"
        if "prototype_rescue_candidate" not in proto.columns:
            return np.zeros(len(pred), dtype=bool)
        rank = proto[col] if col in proto.columns else pd.Series(999, index=proto.index)
        return (proto["prototype_rescue_candidate"].fillna(False).astype(bool) & rank.eq(1)).to_numpy()

    # Validation: select threshold
    val_pred = _require(emb_dir / "validation_predictions.csv", "val embeddings")
    val_proto = _require(proto_dir / "validation_scores.csv", "val prototype scores")
    val_mask = _rank1_mask(val_pred, val_proto)
    val_cands = val_pred.loc[val_mask].copy().reset_index(drop=True)

    if not val_cands.empty:
        val_expr = _load_expr(adata, val_cands["cell_id"].astype(str).tolist(), genes)
        val_scored = pd.concat(
            [val_cands, _score_candidates(val_expr, val_cands, signatures=signatures, rare_class=rare_class, gene_names=genes)],
            axis=1,
        )
        curve = _marker_threshold_curve(val_pred, val_scored, rare_class=rare_class, thresholds=_default_thresholds(val_scored))
        threshold = _choose_threshold(curve, max_false_rescue_rate=max_false_rescue_rate)
    else:
        val_scored = val_cands.copy()
        curve = pd.DataFrame()
        threshold = float("inf")

    write_table(val_scored, out_dir / "validation_scored.csv")
    write_table(curve, out_dir / "marker_threshold_curve.csv")
    write_table(pd.DataFrame([{"selected_marker_threshold": threshold, "max_false_rescue_rate": max_false_rescue_rate}]), threshold_path)
    print(f"  [gate_marker] val candidates: {len(val_cands)}, threshold: {threshold:.4f}")

    # Test: apply threshold
    test_pred = _require(emb_dir / "test_predictions.csv", "test embeddings")
    test_proto = _require(proto_dir / "test_scores.csv", "test prototype scores")
    test_mask = _rank1_mask(test_pred, test_proto)
    test_cands = test_pred.loc[test_mask].copy().reset_index(drop=True)

    if not test_cands.empty:
        test_expr = _load_expr(adata, test_cands["cell_id"].astype(str).tolist(), genes)
        test_scored = pd.concat(
            [test_cands, _score_candidates(test_expr, test_cands, signatures=signatures, rare_class=rare_class, gene_names=genes)],
            axis=1,
        )
    else:
        test_scored = test_cands.copy()

    write_table(test_scored, out_dir / "test_scored.csv")
    print(f"  [gate_marker] test candidates: {len(test_cands)}")
    return threshold


# ── Apply rescue and compute metrics ─────────────────────────────────────────

def _apply_rescue(test_pred: pd.DataFrame, scored: pd.DataFrame, *, threshold: float, rare_class: str) -> pd.Series:
    pred = test_pred["predicted_label"].astype(str).copy()
    if scored.empty or "marker_margin" not in scored.columns or "cell_id" not in scored.columns:
        return pred
    margins = pd.to_numeric(scored["marker_margin"], errors="coerce")
    verified = set(scored.loc[margins.ge(threshold).fillna(False), "cell_id"].astype(str))
    if "cell_id" in test_pred.columns:
        pred.loc[test_pred["cell_id"].astype(str).isin(verified)] = rare_class
    return pred


def _metrics_row(y_true: pd.Series, y_pred: pd.Series, *, rare_class: str, method: str, **extra) -> dict:
    m, _ = classification_tables(y_true, y_pred, rare_class=rare_class)
    return {"method": method, "rare_class": rare_class, **m, **extra}


# ── Visualization ─────────────────────────────────────────────────────────────

def _plot_comparison(df: pd.DataFrame, out_path: Path, *, rare_class: str, sep: dict) -> None:
    methods = df["method"].tolist()
    colors = {"baseline": "#8da0cb", "scRareRefine": "#e78ac3"}
    labels = {"baseline": "scANVI\nBaseline", "scRareRefine": "scRareRefine"}

    metrics = [
        ("rare_f1",          "Rare-class F1"),
        ("rare_recall",      "Rare Recall"),
        ("rare_precision",   "Rare Precision"),
        ("overall_accuracy", "Overall Accuracy"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    seed = df["seed"].iloc[0]
    rts = df["rare_train_size"].iloc[0]
    sep_r = sep.get("separability_ratio", float("nan"))
    fig.suptitle(
        f"{rare_class}  |  seed={seed}  |  rts={rts}  |  sep_ratio={sep_r:.3f}",
        fontsize=10, fontweight="bold",
    )

    for ax, (col, title) in zip(axes, metrics):
        vals = [float(df.loc[df["method"] == m, col].iloc[0]) if col in df.columns else 0.0 for m in methods]
        bars = ax.bar(range(len(methods)), vals,
                      color=[colors.get(m, "#aaa") for m in methods],
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="scRareRefine pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified")
    parser.add_argument("--max_false_rescue_rate", type=float, default=0.001)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)

    print(f"\nscRareRefine  |  {config['dataset']['name']}  |  {rare_class}  |  rts={rare_train_size}  |  seed={args.seed}")

    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_dir}\n  Run 02_baseline_scanvi.py first.")

    # Step 1: prototype + separability
    print("\nStep 1/3: Prototype scoring ...")
    sep = _run_prototype(run_dir, rare_class=rare_class, force=args.force)

    # Step 2: gate + marker
    print("\nStep 2/3: Gate + marker verification ...")
    genes_path = run_dir / "selected_hvg_genes.csv"
    if not genes_path.exists():
        raise FileNotFoundError(f"HVG list not found: {genes_path}")
    genes = read_table(genes_path)["gene"].astype(str).tolist()
    adata = load_adata(config)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    threshold = _run_gate_marker(
        run_dir, adata, rare_class=rare_class, genes=genes,
        max_false_rescue_rate=args.max_false_rescue_rate, force=args.force,
    )

    # Step 3: metrics
    print("\nStep 3/3: Computing metrics ...")
    test_pred = _require(emb_dir / "test_predictions.csv", "test embeddings")
    y_true = test_pred["true_label"].astype(str)
    base_pred = test_pred["predicted_label"].astype(str)
    scored = read_table(run_dir / "gate_marker" / "test_scored.csv") \
        if (run_dir / "gate_marker" / "test_scored.csv").exists() else pd.DataFrame()
    rescue_pred = _apply_rescue(test_pred, scored, threshold=threshold, rare_class=rare_class)

    n_rescued = int((rescue_pred.ne(base_pred) & rescue_pred.eq(rare_class)).sum())
    n_false = int((rescue_pred.ne(base_pred) & rescue_pred.eq(rare_class) & y_true.ne(rare_class)).sum())

    common = {"seed": args.seed, "rare_train_size": str(rare_train_size),
              "split_mode": args.split_mode, "sep_ratio": sep.get("separability_ratio", float("nan"))}
    rows = [
        _metrics_row(y_true, base_pred, rare_class=rare_class, method="baseline", **common),
        _metrics_row(y_true, rescue_pred, rare_class=rare_class, method="scRareRefine",
                     n_rescued=n_rescued, false_rescues=n_false, **common),
    ]
    df = pd.DataFrame(rows)
    out_path = write_table(df, run_dir / "metrics" / "scRareRefine_metrics.csv")
    print(f"  Saved: {out_path}")

    # Print summary
    display = ["method", "rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    display = [c for c in display if c in df.columns]
    print(f"\n{'─'*55}")
    print(df[display].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    sep_r = sep.get("separability_ratio", float("nan"))
    conf = sep.get("rescue_confidence", "")
    print(f"{'─'*55}")
    print(f"sep_ratio={sep_r:.3f} [{conf}]  n_rescued={n_rescued}  false_rescues={n_false}\n")

    # Plot
    _plot_comparison(df, run_dir / "metrics" / "scRareRefine_comparison.png", rare_class=rare_class, sep=sep)


if __name__ == "__main__":
    main()
