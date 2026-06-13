"""rare_train_size 稳健性扫描：3 数据集 × 3 seed × 4 比例，看 5 方法的稳健性曲线。

实验设计：
  数据集：immune_dc / pancreas_baron / tabula_lung_endo
  seed：42 / 43 / 44（3-seed 聚合）
  rare_train_size ∈ {0.01, 0.05, 0.10, all}
    实际标注稀有数 = max(5, int(p × 训练池稀有数))
  方法：scANVI / kNN / CellTypist / scBalance / scRareRefine

目的：验证「有标签稀有细胞从极少→全部」全谱下各方法稳健性的跨数据集一致性。

依赖：复用 tools/compare_baselines.py 的方法实现（import 时自动应用 CellTypist 兼容 patch）。
前置：需先用 tools/train_cache.py 训练好各 (数据集,seed,rare_train_size) 的 scANVI 缓存。

输出：
  results/sweep_rts/sweep_rts_summary.csv  （机读，每行 dataset×seed×rts×method）
  results/sweep_rts/sweep_rts_agg.csv      （机读，3-seed 聚合 dataset×rts×method）
  results/sweep_rts/sweep_rts_log.md       （人读，每数据集一张 mean±std 透视表）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables, load_adata
from src.rescue import PrototypeRescuer

# 复用 compare_baselines 的方法实现（import 触发 CellTypist monkey-patch）
import tools.compare_baselines as cb

# ── 扫描矩阵 ─────────────────────────────────────────────────────────────────
DATASETS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
]
SEEDS    = [42, 43, 44]
RTS_LIST = ["0.01", "0.05", "0.10", "all"]

METHODS = ["scANVI", "kNN", "CellTypist", "scBalance", "scRareRefine"]


def process_run(cfg_path: str, seed: int, rts_str: str,
                adata_cache: dict[str, ad.AnnData]) -> list[dict]:
    """处理单个 (数据集, seed, rts) run，返回 5 方法的指标行列表（缓存缺失返回 []）。"""
    config     = load_config(cfg_path)
    exp        = config.get("experiment", {})
    rare_class = exp.get("rare_class")
    split_mode = exp.get("split_mode", "batch_heldout")
    size       = parse_rare_train_size(rts_str)
    run_dir    = make_run_dir(config, split_mode, seed, rare_class, size)
    emb_dir    = run_dir / "embeddings"
    dataset    = config["dataset"]["name"]

    if not (emb_dir / "test_latent.csv").exists():
        print(f"[SKIP] {dataset} seed={seed} rts={rts_str} 缓存不存在")
        return []

    cb._check_manifest(run_dir, config, seed, rts_str)
    splits = ["train", "validation", "test"]
    preds  = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
    lats   = {s: pd.read_csv(emb_dir / f"{s}_latent.csv")      for s in splits}

    train_lat  = cb._lat(lats["train"])
    is_labeled = preds["train"]["is_labeled_for_scanvi"].astype(bool).to_numpy()
    ref_labels = preds["train"]["true_label"].astype(str)
    lab_lat    = train_lat[is_labeled]
    lab_labels = ref_labels[is_labeled].to_numpy()

    proto = PrototypeRescuer(rare_class)
    proto.fit(train_lat, ref_labels, is_labeled)

    y_true    = preds["test"]["true_label"].astype(str).to_numpy()
    val_true  = preds["validation"]["true_label"].astype(str)
    base_pred = preds["test"]["predicted_label"].astype(str)

    n_lab_rare = int((lab_labels == rare_class).sum())
    print(f"\n[{dataset} seed={seed} rts={rts_str}] sep={proto.separability_ratio:.3f}  "
          f"lab_rare={n_lab_rare}  test_rare={int((y_true==rare_class).sum())}")

    # kNN：val 上 grid-search k
    val_true_arr = val_true.to_numpy()
    best_k, best_val_f1 = 15, -1.0
    for k_cand in cb.KNN_K_GRID:
        vp = cb._knn_predict(lab_lat, lab_labels, cb._lat(lats["validation"]), k_cand)
        m, _ = classification_tables(val_true_arr, vp, rare_class=rare_class)
        if m["rare_f1"] > best_val_f1:
            best_val_f1, best_k = m["rare_f1"], k_cand
    knn_pred = cb._knn_predict(lab_lat, lab_labels, cb._lat(lats["test"]), best_k)

    # scRareRefine
    srr_pred = cb._conformal_rescue(
        proto, base_pred, cb._lat(lats["validation"]), val_true, cb._lat(lats["test"]))

    # HVG 表达（CellTypist / scBalance）
    hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()
    if dataset not in adata_cache:
        print(f"  [加载原始 h5ad] {config['dataset']['path']}")
        adata_cache[dataset] = load_adata(config)
    adata_full = adata_cache[dataset]
    idx_map    = {cid: i for i, cid in enumerate(adata_full.obs_names)}

    def _get_X(cell_ids):
        rows_idx = [idx_map[cid] for cid in cell_ids if cid in idx_map]
        if len(rows_idx) != len(cell_ids):   # 不允许静默丢失 cell_id
            raise ValueError(
                f"{dataset} seed={seed}: cell_id 与 h5ad 不匹配 "
                f"(期望 {len(cell_ids)}，命中 {len(rows_idx)})，缓存可能与当前 h5ad 不一致")
        sub   = adata_full[rows_idx]
        hvg_v = [g for g in hvg_genes if g in sub.var_names]
        return cb._log1p_norm(sub[:, hvg_v].X), hvg_v

    labeled_ids = preds["train"]["cell_id"].astype(str)[is_labeled].tolist()
    test_ids    = preds["test"]["cell_id"].astype(str).tolist()
    train_X, hvg_v = _get_X(labeled_ids)
    test_X,  _     = _get_X(test_ids)

    print("  CellTypist...", end=" ", flush=True)
    ct_status = "ok"
    try:
        ct_pred = cb._run_celltypist(train_X, lab_labels, test_X, hvg_v)
    except Exception as e:
        print(f"FAILED ({e})")
        ct_pred, ct_status = None, "failed"

    print("scBalance...", end=" ", flush=True)
    sb_status = "ok"
    try:
        sb_pred = cb._run_scbalance(train_X, lab_labels, test_X)
    except Exception as e:
        print(f"FAILED ({e})")
        sb_pred, sb_status = None, "failed"
    print("done")

    out = []
    for mname, pred_arr, status in [
        ("scANVI",       base_pred.to_numpy(),  "ok"),
        ("kNN",          knn_pred,              "ok"),
        ("CellTypist",   ct_pred,               ct_status),
        ("scBalance",    sb_pred,               sb_status),
        ("scRareRefine", srr_pred.to_numpy(),   "ok"),
    ]:
        base_row = {
            "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
            "lab_rare": n_lab_rare, "rare_class": rare_class, "method": mname,
            "status": status, "sep": round(proto.separability_ratio, 4),
            "best_k": best_k if mname == "kNN" else None,
        }
        if status == "failed":
            print(f"  {mname:15s}: FAILED (excluded from aggregation)")
            out.append(base_row)
            continue
        mres  = cb._metrics(y_true, pred_arr, base_pred.to_numpy(), rare_class)
        extra = f" [k={best_k}]" if mname == "kNN" else ""
        print(f"  {mname:15s}: F1={mres['rare_f1']:.4f}  "
              f"rec={mres['rare_recall']:.4f}  prec={mres['rare_precision']:.4f}  "
              f"FP_rate={mres['rare_fp_rate']:.5f}{extra}")
        out.append({**base_row, **mres})
    return out


def main():
    adata_cache: dict[str, ad.AnnData] = {}
    rows = []

    for cfg_path in DATASETS:
        for seed in SEEDS:
            for rts_str in RTS_LIST:
                rows.extend(process_run(cfg_path, seed, rts_str, adata_cache))

    out_dir = Path("results/sweep_rts")
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_rts_summary.csv", index=False)
    print(f"\n[saved] {out_dir}/sweep_rts_summary.csv  ({len(df)} rows)")

    if df.empty:
        print("[warn] 无数据，退出")
        return

    # ── 3-seed 聚合（仅 status==ok 的 run 进入均值）──────────────────────────
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    agg_rows = []
    for dataset in df["dataset"].unique():
        for rts in RTS_LIST:
            for method in METHODS:
                sub = ok[(ok["dataset"] == dataset) & (ok["rare_train_size"] == rts)
                         & (ok["method"] == method)]
                if sub.empty:
                    continue
                f1s = sub["rare_f1"].to_numpy()
                fps = sub["rare_fp_rate"].to_numpy()
                agg_rows.append({
                    "dataset": dataset, "rare_train_size": rts, "method": method,
                    "lab_rare": int(sub["lab_rare"].iloc[0]),
                    "n_seed": len(sub),
                    "f1_mean": round(f1s.mean(), 4), "f1_std": round(f1s.std(), 4),
                    "fp_rate_max": round(fps.max(), 6),
                })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "sweep_rts_agg.csv", index=False)
    print(f"[saved] {out_dir}/sweep_rts_agg.csv  ({len(agg)} rows)")

    # ── 控制台透视（每数据集均值）─────────────────────────────────────────────
    for dataset in df["dataset"].unique():
        print(f"\n=== {dataset}  rare F1 均值±σ（行=方法，列=rts）===")
        sub = agg[agg["dataset"] == dataset]
        for method in METHODS:
            cells = []
            for rts in RTS_LIST:
                r = sub[(sub["method"] == method) & (sub["rare_train_size"] == rts)]
                cells.append(f"{r['f1_mean'].iloc[0]:.3f}±{r['f1_std'].iloc[0]:.3f}"
                             if len(r) else "   -   ")
            print(f"  {method:15s}: " + "  ".join(cells))

    # ── Markdown 报告 ─────────────────────────────────────────────────────────
    md = [
        "# rare_train_size 稳健性扫描报告（3 数据集 × 3 seed）",
        "",
        "数据集：immune_dc / pancreas_baron / tabula_lung_endo | seed：42/43/44 | 方法：5 种",
        "",
        "实际标注稀有细胞数 = max(5, int(p × 训练池稀有数))，故小比例可能撞 5 的下限。",
        "",
    ]
    for dataset in df["dataset"].unique():
        sub = agg[agg["dataset"] == dataset]
        # lab_rare 映射
        lab_map = {rts: sub[sub["rare_train_size"] == rts]["lab_rare"].iloc[0]
                   for rts in RTS_LIST if len(sub[sub["rare_train_size"] == rts])}
        md += [
            f"## {dataset}",
            "",
            "| 方法 | " + " | ".join(f"{r}({lab_map.get(r,'-')})" for r in RTS_LIST) + " |",
            "|------|" + "------|" * len(RTS_LIST),
        ]
        for method in METHODS:
            cells = []
            for rts in RTS_LIST:
                r = sub[(sub["method"] == method) & (sub["rare_train_size"] == rts)]
                cells.append(f"{r['f1_mean'].iloc[0]:.3f}±{r['f1_std'].iloc[0]:.3f}"
                             if len(r) else "-")
            md.append(f"| {method} | " + " | ".join(cells) + " |")
        md.append("")

    (out_dir / "sweep_rts_log.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[saved] {out_dir}/sweep_rts_log.md")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
