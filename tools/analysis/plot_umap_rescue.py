"""UMAP 可视化：rescue 前后稀有细胞在 scANVI latent 空间的标注变化。

在 test 集 latent 上计算 UMAP，绘制 2×2 面板：
  (a) Ground truth        — 真实稀有细胞位置
  (b) scANVI prediction   — rescue 前（漏判）
  (c) scRareRefine pred   — rescue 后（救回）
  (d) Rescue outcome      — TP救回/漏判/误救 分解

用法：
  python tools/analysis/plot_umap_rescue.py --config configs/immune_dc.yaml --seed 42 --rts 0.05
  python tools/analysis/plot_umap_rescue.py --config configs/pancreas_baron.yaml --seed 42 --rts 0.10
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.rescue import PrototypeRescuer, conformal_rescue
from src.utils import load_config, make_run_dir, parse_rare_train_size

ALPHA = 0.01


def _lat(df):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/immune_dc.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rts", default="0.05")
    args = ap.parse_args()

    config     = load_config(args.config)
    exp        = config.get("experiment", {})
    RARE       = exp.get("rare_class")
    split_mode = exp.get("split_mode", "batch_heldout")
    size       = parse_rare_train_size(args.rts)
    RUN_DIR    = make_run_dir(config, split_mode, args.seed, RARE, size)
    dataset    = config["dataset"]["name"]
    OUT        = Path(f"results/umap/umap_rescue_{dataset}.png")

    emb = RUN_DIR / "embeddings"
    pr  = {s: pd.read_csv(emb / f"{s}_predictions.csv") for s in ["train", "validation", "test"]}
    la  = {s: pd.read_csv(emb / f"{s}_latent.csv")      for s in ["train", "validation", "test"]}

    train_lat  = _lat(la["train"])
    is_labeled = pr["train"]["is_labeled_for_scanvi"].astype(bool).to_numpy()
    ref_labels = pr["train"]["true_label"].astype(str)

    proto = PrototypeRescuer(RARE)
    proto.fit(train_lat, ref_labels, is_labeled)
    sep      = proto.separability_ratio
    lab_rare = int((ref_labels[is_labeled] == RARE).sum())

    test_lat  = _lat(la["test"])
    val_lat   = _lat(la["validation"])
    y_true    = pr["test"]["true_label"].astype(str).to_numpy()
    base_pred = pr["test"]["predicted_label"].astype(str)
    val_true  = pr["validation"]["true_label"].astype(str)
    val_base  = pr["validation"]["predicted_label"].astype(str)

    # scRareRefine：与主方法**完全一致**的 conformal_rescue（sep/necessity 闸门 + val-自适应 rank + τ），
    # 不再用旧的 isotropic_rank1 简化版，确保 UMAP 的 recall/precision 与主结果对得上。
    srr_series, conf_summary = conformal_rescue(proto, base_pred, val_base, val_true, val_lat, test_lat, alpha=ALPHA)
    srr_pred = srr_series.astype(str).to_numpy()
    base_pred = base_pred.to_numpy()

    # ── UMAP on test latent ──────────────────────────────────────────────────
    print("computing UMAP on test latent ...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.3, random_state=42)
    xy = reducer.fit_transform(test_lat)

    # ── 类别掩膜 ──────────────────────────────────────────────────────────────
    true_rare = y_true == RARE
    scanvi_rare = base_pred == RARE
    srr_rare    = srr_pred == RARE
    rescued     = srr_rare & (base_pred != RARE)          # 被改判为 rare
    tp_rescue   = rescued & true_rare                      # 救对
    fp_rescue   = rescued & (~true_rare)                   # 救错
    already_ok  = scanvi_rare & true_rare                  # 本来就对
    missed      = true_rare & (~srr_rare)                  # 仍漏判

    rec_scanvi = scanvi_rare[true_rare].mean()
    rec_srr    = srr_rare[true_rare].mean()
    prec_srr   = (srr_rare & true_rare).sum() / max(srr_rare.sum(), 1)

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    GRAY    = "#D4D4D4"   # background cells
    RED     = "#B55D5A"   # rare cell marker (muted brick rose)

    def base_scatter(ax):
        ax.scatter(xy[~true_rare, 0], xy[~true_rare, 1], s=4, c=GRAY, alpha=0.45,
                   linewidths=0, rasterized=True)

    # (a) Ground truth
    ax = axes[0, 0]
    ax.scatter(xy[~true_rare, 0], xy[~true_rare, 1], s=4, c=GRAY, alpha=0.45,
               linewidths=0, label="other cells", rasterized=True)
    ax.scatter(xy[true_rare, 0], xy[true_rare, 1], s=22, c=RED, edgecolors="k",
               linewidths=0.3, label=f"{RARE} (true, n={true_rare.sum()})")
    ax.set_title("(a) Ground truth", fontsize=13, loc="left")
    ax.legend(loc="best", fontsize=9, markerscale=1.3)

    # (b) scANVI prediction
    ax = axes[0, 1]
    base_scatter(ax)
    ax.scatter(xy[scanvi_rare, 0], xy[scanvi_rare, 1], s=22, c=RED, edgecolors="k",
               linewidths=0.3, label=f"predicted {RARE} (n={scanvi_rare.sum()})")
    ax.set_title(f"(b) scANVI prediction  —  recall={rec_scanvi:.2f}", fontsize=13, loc="left")
    ax.legend(loc="best", fontsize=9, markerscale=1.3)

    # (c) scRareRefine prediction
    ax = axes[1, 0]
    base_scatter(ax)
    ax.scatter(xy[srr_rare, 0], xy[srr_rare, 1], s=22, c=RED, edgecolors="k",
               linewidths=0.3, label=f"predicted {RARE} (n={srr_rare.sum()})")
    ax.set_title(f"(c) scRareRefine  —  recall={rec_srr:.2f}, prec={prec_srr:.2f}",
                 fontsize=13, loc="left")
    ax.legend(loc="best", fontsize=9, markerscale=1.3)

    # (d) Rescue outcome
    ax = axes[1, 1]
    ax.scatter(xy[~true_rare & ~fp_rescue, 0], xy[~true_rare & ~fp_rescue, 1],
               s=4, c=GRAY, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(xy[already_ok, 0], xy[already_ok, 1], s=24, c="#5B7FA6",
               edgecolors="k", linewidths=0.3, label=f"already correct (n={already_ok.sum()})")
    ax.scatter(xy[tp_rescue, 0], xy[tp_rescue, 1], s=30, c="#1A7A4A", marker="*",
               edgecolors="k", linewidths=0.3, label=f"rescued ✓ (n={tp_rescue.sum()})")
    ax.scatter(xy[missed, 0], xy[missed, 1], s=24, c="#C97A50", marker="v",
               edgecolors="k", linewidths=0.3, label=f"still missed (n={missed.sum()})")
    ax.scatter(xy[fp_rescue, 0], xy[fp_rescue, 1], s=40, c=RED, marker="x",
               linewidths=1.4, label=f"false rescue (n={fp_rescue.sum()})")
    ax.set_title("(d) Rescue outcome", fontsize=13, loc="left")
    ax.legend(loc="best", fontsize=9, markerscale=1.2)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP-1", fontsize=10); ax.set_ylabel("UMAP-2", fontsize=10)

    fig.suptitle(
        f"scRareRefine rescue on {dataset} ({RARE}, seed={args.seed}, "
        f"{lab_rare} labeled rare, sep={sep:.2f})\n"
        f"scANVI recall {rec_scanvi:.2f} → scRareRefine recall {rec_srr:.2f}, "
        f"precision {prec_srr:.2f}",
        fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"[saved] {OUT}")
    print(f"  sep={sep:.3f}  lab_rare={lab_rare}")
    print(f"  scANVI recall={rec_scanvi:.3f}  scRareRefine recall={rec_srr:.3f} "
          f"prec={prec_srr:.3f}")
    print(f"  rescued TP={tp_rescue.sum()}  FP={fp_rescue.sum()}  "
          f"already_ok={already_ok.sum()}  missed={missed.sum()}")

    # dump npz 供对照图复用
    npz = Path(f"results/umap/umap_rescue_{dataset}.npz")
    np.savez(npz, xy=xy, true_rare=true_rare, scanvi_rare=scanvi_rare,
             srr_rare=srr_rare, tp_rescue=tp_rescue, fp_rescue=fp_rescue,
             already_ok=already_ok, missed=missed,
             sep=sep, lab_rare=lab_rare, rec_scanvi=rec_scanvi,
             rec_srr=rec_srr, prec_srr=prec_srr,
             dataset=dataset, rare=RARE, seed=args.seed)
    print(f"[saved] {npz}")


if __name__ == "__main__":
    main()
