"""Stage 11: UMAP visualization of latent space with rescue annotations.

Shows:
  - UMAP of all test+val cells colored by true label
  - Rescue candidates highlighted (gate, gate+marker, correct vs wrong)
  - Confidence (entropy) overlay

Reads:
    outputs/{dataset}/{run_id}/embeddings/
    outputs/{dataset}/{run_id}/gate/
    outputs/{dataset}/{run_id}/gate_marker/

Writes:
    outputs/{dataset}/{run_id}/figures/
        umap_celltypes.png        colored by true label
        umap_rescue.png           rescue outcomes highlighted
        umap_confidence.png       colored by scANVI entropy

Usage:
    python src/11_umap_visualize.py \\
        --config configs/immune_dc.yaml \\
        --seed 42 --rare_class ASDC --rare_train_size 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from utils import load_config, make_run_dir, parse_rare_train_size, read_table


def _latent_matrix(latent_df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in latent_df.columns if c.startswith("latent_")]
    return latent_df[cols].to_numpy(dtype=float)


def _run_umap(latent: np.ndarray, seed: int = 42) -> np.ndarray:
    try:
        from umap import UMAP
        reducer = UMAP(n_neighbors=30, min_dist=0.3, random_state=seed)
        return reducer.fit_transform(latent)
    except ImportError:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=seed, perplexity=min(30, latent.shape[0] - 1))
        return reducer.fit_transform(latent)


def fig_celltypes(embedding: np.ndarray, labels: pd.Series, rare_class: str, out_path: Path) -> None:
    unique_labels = sorted(labels.unique())
    n_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20", n_classes)
    color_map = {lab: cmap(i) for i, lab in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in unique_labels:
        mask = labels.eq(lab)
        size = 30 if lab == rare_class else 5
        alpha = 1.0 if lab == rare_class else 0.3
        zorder = 10 if lab == rare_class else 1
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=[color_map[lab]], s=size, alpha=alpha, linewidths=0, zorder=zorder,
                   label=f"{lab} (n={mask.sum()})")
    ax.set_title(f"Latent UMAP — cell types  |  rare: {rare_class}", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=6, markerscale=2, loc="best", framealpha=0.8,
              ncol=max(1, n_classes // 8))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_rescue_outcomes(
    embedding: np.ndarray,
    true_labels: pd.Series,
    baseline_labels: pd.Series,
    gate_rescued: pd.Series,
    gate_marker_rescued: pd.Series | None,
    rare_class: str,
    out_path: Path,
) -> None:
    # Categories:
    # 1. True rare, baseline correct (green circle)
    # 2. True rare, baseline wrong, NOT rescued (red X)
    # 3. True rare, baseline wrong, rescued by gate (orange triangle-up)
    # 4. True rare, baseline wrong, rescued by gate+marker (purple diamond)
    # 5. Not rare (light gray)
    # 6. False rescue (gate marks non-rare as rare) (black X)

    not_rare = true_labels.ne(rare_class)
    rare_correct_base = true_labels.eq(rare_class) & baseline_labels.eq(rare_class)
    rare_missed = true_labels.eq(rare_class) & baseline_labels.ne(rare_class)
    gate_bool = gate_rescued.fillna(False).astype(bool)
    gm_bool = gate_marker_rescued.fillna(False).astype(bool) if gate_marker_rescued is not None else pd.Series(False, index=true_labels.index)

    rescued_gate_only = rare_missed & gate_bool & ~gm_bool
    rescued_gm = rare_missed & gm_bool
    not_rescued = rare_missed & ~gate_bool & ~gm_bool
    false_rescue_gate = not_rare & gate_bool

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(embedding[not_rare, 0], embedding[not_rare, 1],
               c="#cccccc", s=4, alpha=0.25, linewidths=0, zorder=1, label="Non-rare")
    ax.scatter(embedding[rare_correct_base, 0], embedding[rare_correct_base, 1],
               c="#2ca02c", s=40, alpha=0.9, linewidths=0.5, edgecolors="white", zorder=5,
               marker="o", label=f"Rare (baseline correct, n={rare_correct_base.sum()})")
    ax.scatter(embedding[not_rescued, 0], embedding[not_rescued, 1],
               c="#d62728", s=70, alpha=0.9, linewidths=0.8, edgecolors="white", zorder=8,
               marker="X", label=f"Rare missed, not rescued (n={not_rescued.sum()})")
    ax.scatter(embedding[rescued_gate_only, 0], embedding[rescued_gate_only, 1],
               c="#ff7f0e", s=70, alpha=0.9, linewidths=0.8, edgecolors="white", zorder=9,
               marker="^", label=f"Rescued by gate only (n={rescued_gate_only.sum()})")
    ax.scatter(embedding[rescued_gm, 0], embedding[rescued_gm, 1],
               c="#9467bd", s=70, alpha=0.9, linewidths=0.8, edgecolors="white", zorder=10,
               marker="D", label=f"Rescued by gate+marker (n={rescued_gm.sum()})")
    ax.scatter(embedding[false_rescue_gate, 0], embedding[false_rescue_gate, 1],
               c="#000000", s=50, alpha=0.8, linewidths=0.5, edgecolors="red", zorder=11,
               marker="P", label=f"False rescue (n={false_rescue_gate.sum()})")

    ax.set_title(f"Rescue outcomes  |  rare: {rare_class}", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_confidence(embedding: np.ndarray, entropy: pd.Series, out_path: Path, title: str) -> None:
    ent = entropy.fillna(0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1],
                    c=ent, cmap="RdYlGn_r", s=5, alpha=0.5, linewidths=0, vmin=0)
    plt.colorbar(sc, ax=ax, label="scANVI prediction entropy (higher = uncertain)")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 11: UMAP visualization")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--split", default="test", choices=["test", "validation", "both"])
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    run_dir = make_run_dir(config, args.split_mode, args.seed, rare_class, rare_train_size)
    fig_dir = run_dir / "figures"

    splits = ["test", "validation"] if args.split == "both" else [args.split]
    for split_name in splits:
        print(f"Processing {split_name} split ...")
        pred = read_table(run_dir / "embeddings" / f"{split_name}_predictions.csv")
        latent = read_table(run_dir / "embeddings" / f"{split_name}_latent.csv")
        lat = _latent_matrix(latent)

        # Load rescue masks from gate
        gate_cands_path = run_dir / "gate" / f"{split_name}_candidates.csv"
        gm_scored_path = run_dir / "gate_marker" / f"{split_name}_scored.csv"

        gate_rescued = pd.Series(False, index=pred.index)
        gm_rescued = pd.Series(False, index=pred.index)

        if gate_cands_path.exists():
            gc = read_table(gate_cands_path)
            rank1_gc = gc[gc["gate_name"].eq("rank1")] if "gate_name" in gc.columns else gc
            if "cell_id" in rank1_gc.columns and "cell_id" in pred.columns:
                rescued_ids = set(rank1_gc["cell_id"].astype(str))
                gate_rescued = pred["cell_id"].astype(str).isin(rescued_ids)

        threshold_df_path = run_dir / "gate_marker" / "selected_thresholds.csv"
        selected_threshold = float("inf")
        if threshold_df_path.exists():
            tdf = read_table(threshold_df_path)
            if not tdf.empty and "selected_marker_threshold" in tdf.columns:
                selected_threshold = float(tdf["selected_marker_threshold"].iloc[0])

        if gm_scored_path.exists() and selected_threshold < float("inf"):
            gms = read_table(gm_scored_path)
            if "marker_margin" in gms.columns and "cell_id" in gms.columns:
                margins = pd.to_numeric(gms["marker_margin"], errors="coerce")
                verified_ids = set(gms.loc[margins.ge(selected_threshold).fillna(False), "cell_id"].astype(str))
                if "cell_id" in pred.columns:
                    gm_rescued = pred["cell_id"].astype(str).isin(verified_ids)

        print(f"  Computing UMAP ({lat.shape[0]} cells × {lat.shape[1]} dims) ...")
        emb = _run_umap(lat, seed=args.seed)

        fig_celltypes(emb, pred["true_label"].astype(str), rare_class,
                      fig_dir / f"umap_{split_name}_celltypes.png")
        fig_rescue_outcomes(emb, pred["true_label"].astype(str),
                            pred["predicted_label"].astype(str),
                            gate_rescued, gm_rescued, rare_class,
                            fig_dir / f"umap_{split_name}_rescue.png")
        if "entropy" in pred.columns:
            fig_confidence(emb, pd.to_numeric(pred["entropy"], errors="coerce"),
                           fig_dir / f"umap_{split_name}_confidence.png",
                           f"scANVI prediction entropy  |  {split_name}  |  rare: {rare_class}")

    print(f"Done. Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
