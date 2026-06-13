"""对比实验：scRareRefine vs scANVI / kNN / CellTypist / scBalance。

方法说明：
  scANVI          — scANVI 直接预测（缓存 predicted_label）
  kNN             — k=15 最近邻，在 scANVI latent embedding 上，val 上 grid-search k∈{3,5,10,15}
  CellTypist      — Logistic Regression（Celltypist 官方工具），在 HVG log1p 归一化表达上
  scBalance       — 加权采样神经网络（scBalance 官方工具），在 HVG log1p 归一化表达上
  scRareRefine    — scANVI + conformal prototype rescue（V4 full）

输入特征：
  kNN / scRareRefine：使用 scANVI latent embedding（已缓存）
  CellTypist / scBalance：使用 HVG log1p 归一化基因表达（从原始 h5ad 提取）

约束：
  - 所有方法只用 labeled train 样本训练
  - val 仅用于 kNN k 选择 和 conformal τ 校准
  - test 标签仅用于最终评估

输出：
  results/comparison/comparison_summary.csv    （机读）
  results/comparison/comparison_summary_agg.csv（机读，3-seed 聚合）
  results/comparison/comparison_log.md         （人读）
"""
from __future__ import annotations

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables, load_adata
from src.rescue import PrototypeRescuer, ConformalRescuer

# ── CellTypist 兼容性 patch（移除 sklearn 1.8 删除的 multi_class 参数）──────────
import celltypist, sys as _sys
from sklearn.linear_model import LogisticRegression as _OrigLR

_ct_mod = _sys.modules["celltypist.train"]

def _patched_LRClassifier(indata, labels, C, solver, max_iter, n_jobs, **kwargs):
    kwargs.pop("multi_class", None)
    solver   = solver   or "lbfgs"
    max_iter = max_iter or 1000
    clf = _OrigLR(C=C, solver=solver, max_iter=max_iter, n_jobs=n_jobs, **kwargs)
    clf.fit(indata, labels)
    return clf

_ct_mod._LRClassifier = _patched_LRClassifier
# ─────────────────────────────────────────────────────────────────────────────

import scBalance

# ── run 列表 ─────────────────────────────────────────────────────────────────
RUNS = [
    ("configs/immune_dc.yaml",        42, "0.05"),
    ("configs/immune_dc.yaml",        43, "0.05"),
    ("configs/immune_dc.yaml",        44, "0.05"),
    ("configs/pancreas_baron.yaml",   42, "0.10"),
    ("configs/pancreas_baron.yaml",   43, "0.10"),
    ("configs/pancreas_baron.yaml",   44, "0.10"),
    ("configs/tabula_lung_endo.yaml", 42, "0.10"),
    ("configs/tabula_lung_endo.yaml", 43, "0.10"),
    ("configs/tabula_lung_endo.yaml", 44, "0.10"),
]

KNN_K_GRID      = [3, 5, 10, 15]
CONFORMAL_ALPHA = 0.01
CONFORMAL_LOW_SEP = 1.3


def _lat(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _log1p_norm(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """log1p 归一化到 target_sum counts/cell（CellTypist 标准输入）。"""
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    total = X.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(X / total * target_sum)


def _knn_predict(train_lat: np.ndarray, train_labels: np.ndarray,
                 query_lat: np.ndarray, k: int = 15) -> np.ndarray:
    """手动 kNN（逐元素欧氏距离），规避 sklearn/BLAS Windows DLL 问题。

    小训练集保护：k_eff=min(k, n_train)，argpartition 的 kth 必须 < n_train。
    """
    train_f32 = train_lat.astype(np.float32)
    n_train   = len(train_labels)
    k_eff     = min(k, n_train)
    kth       = min(k_eff, n_train - 1)   # argpartition kth 上界 = n_train-1
    chunk = 100
    preds = []
    for i in range(0, len(query_lat), chunk):
        q  = query_lat[i:i+chunk].astype(np.float32)
        d2 = np.sum((train_f32[None, :, :] - q[:, None, :]) ** 2, axis=2)
        nn = np.argpartition(d2, kth, axis=1)[:, :k_eff]
        for j in range(len(q)):
            lbls = train_labels[nn[j]]
            vals, cnts = np.unique(lbls, return_counts=True)
            preds.append(vals[cnts.argmax()])
    return np.array(preds)


def _load_expression(config: dict, cell_ids_by_split: dict[str, list[str]],
                     hvg_genes: list[str]) -> dict[str, np.ndarray]:
    """从原始 h5ad 提取 labeled-train / val / test 子集的 HVG log1p 表达。"""
    adata_full = load_adata(config)
    all_ids    = set()
    for ids in cell_ids_by_split.values():
        all_ids.update(ids)
    idx_map = {cid: i for i, cid in enumerate(adata_full.obs_names)}

    out = {}
    for split, ids in cell_ids_by_split.items():
        rows = [idx_map[cid] for cid in ids if cid in idx_map]
        sub  = adata_full[rows]
        # HVG 基因子集（var_names 中找）
        hvg_in_var = [g for g in hvg_genes if g in sub.var_names]
        sub = sub[:, hvg_in_var]
        X   = sub.X
        out[split] = _log1p_norm(X)
    return out


def _run_celltypist(train_X: np.ndarray, train_labels: np.ndarray,
                    test_X: np.ndarray, hvg_genes: list[str]) -> np.ndarray:
    gene_names = hvg_genes[:train_X.shape[1]]
    adata_tr  = ad.AnnData(X=train_X); adata_tr.var_names = gene_names
    adata_te  = ad.AnnData(X=test_X);  adata_te.var_names = gene_names
    logging.getLogger("celltypist").setLevel(logging.CRITICAL)
    model = celltypist.train(adata_tr, labels=train_labels, check_expression=False)
    pred  = celltypist.annotate(adata_te, model=model, majority_voting=False)
    return pred.predicted_labels["predicted_labels"].to_numpy().astype(str)


def _run_scbalance(train_X: np.ndarray, train_labels: np.ndarray,
                   test_X: np.ndarray) -> np.ndarray:
    ref_df   = pd.DataFrame(train_X)
    test_df  = pd.DataFrame(test_X)
    label_df = pd.DataFrame({"Label": train_labels})
    preds = scBalance.scBalance(test=test_df, reference=ref_df, label=label_df,
                                processing_unit="cpu")
    return np.array(preds, dtype=str)


def _conformal_rescue(proto, base_pred, val_lat, val_true, test_lat):
    if proto.separability_ratio < CONFORMAL_LOW_SEP:
        return base_pred.copy()
    test_cand  = proto.isotropic_rank1(test_lat, base_pred)
    test_score = proto.rare_membership_score(test_lat)
    val_score  = proto.rare_membership_score(val_lat)
    conf = ConformalRescuer(proto.rare_class, alpha=CONFORMAL_ALPHA)
    conf.calibrate(val_score, val_true)
    return conf.relabel(base_pred, test_cand, test_score)


def _metrics(y_true, pred, base_pred, rare_class):
    """计算稀有类指标。区分两种误报率：

    - rare_fp_rate：标准稀有假阳性率 = (pred==rare & y_true!=rare) / n_nonrare。
      所有方法可比，是 baseline 应看的误报指标。
    - rescue_ffr：rescue 误救率 = 相对 base_pred 被改判为 rare 且真值非 rare 的比例。
      只对在 base_pred(=scANVI) 基础上做改判的方法（scRareRefine）有意义；
      对独立预测的 baseline（kNN/CellTypist/scBalance）不可解释，仅供参考。
    """
    pred = np.asarray(pred, dtype=str)
    bp   = np.asarray(base_pred, dtype=str)
    m, _ = classification_tables(y_true, pred, rare_class=rare_class)
    n_nrare = int((y_true != rare_class).sum())
    # 标准稀有假阳性率（所有方法可比）
    n_fp    = int(((pred == rare_class) & (y_true != rare_class)).sum())
    # rescue 语义（相对 base_pred 改判）
    n_res   = int(((pred != bp) & (pred == rare_class)).sum())
    n_false = int(((pred != bp) & (pred == rare_class) & (y_true != rare_class)).sum())
    return {
        "rare_f1":        round(m["rare_f1"],        4),
        "rare_recall":    round(m["rare_recall"],    4),
        "rare_precision": round(m["rare_precision"], 4),
        "macro_f1":       round(m["macro_f1"],       4),
        "rare_fp_rate":   round(n_fp / max(n_nrare, 1), 6),
        "n_rescued":      n_res,
        "n_false_rescue": n_false,
        "rescue_ffr":     round(n_false / max(n_nrare, 1), 6),
    }


def _check_manifest(run_dir: Path, config: dict, seed: int, rts_str: str) -> None:
    """校验缓存 manifest 与当前配置一致；缺失或不匹配仅打印警告（兼容旧缓存，不 hard fail）。"""
    mf = run_dir / "manifest.json"
    if not mf.exists():
        print("  [provenance] ⚠️ 无 manifest.json（旧缓存，无法校验 split/代码版本）")
        return
    m   = json.loads(mf.read_text(encoding="utf-8"))
    exp = config.get("experiment", {})
    checks = {
        "dataset_path":    config["dataset"]["path"],
        "label_key":       config["dataset"].get("label_key"),
        "batch_key":       config["dataset"].get("batch_key"),
        "split_mode":      exp.get("split_mode", "batch_heldout"),
        "rare_class":      exp.get("rare_class"),
        "seed":            seed,
        "rare_train_size": str(parse_rare_train_size(rts_str)),
    }
    mism = [(k, m.get(k), v) for k, v in checks.items() if str(m.get(k)) != str(v)]
    if mism:
        print(f"  [provenance] ⚠️ manifest 与当前配置不一致: {mism}")
    else:
        print(f"  [provenance] OK  split_hash={m.get('split_hash')}  git_sha={m.get('git_sha')}")


METHODS = ["scANVI", "kNN", "CellTypist", "scBalance", "scRareRefine"]


def main():
    _adata_cache: dict[str, ad.AnnData] = {}
    rows = []

    for cfg_path, seed, rts_str in RUNS:
        config     = load_config(cfg_path)
        exp        = config.get("experiment", {})
        rare_class = exp.get("rare_class")
        split_mode = exp.get("split_mode", "batch_heldout")
        size       = parse_rare_train_size(rts_str)
        run_dir    = make_run_dir(config, split_mode, seed, rare_class, size)
        emb_dir    = run_dir / "embeddings"
        dataset    = config["dataset"]["name"]

        if not (emb_dir / "test_latent.csv").exists():
            print(f"[SKIP] {run_dir} 缓存不存在")
            continue

        _check_manifest(run_dir, config, seed, rts_str)
        splits = ["train", "validation", "test"]
        preds  = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
        lats   = {s: pd.read_csv(emb_dir / f"{s}_latent.csv")      for s in splits}

        train_lat  = _lat(lats["train"])
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
        print(f"\n[{dataset} seed={seed}] sep={proto.separability_ratio:.3f}  "
              f"lab_rare={n_lab_rare}  test_rare={int((y_true==rare_class).sum())}")

        # kNN：val 上 grid-search k
        val_true_arr = val_true.to_numpy()
        best_k, best_val_f1 = 15, -1.0
        for k_cand in KNN_K_GRID:
            vp = _knn_predict(lab_lat, lab_labels, _lat(lats["validation"]), k_cand)
            m, _ = classification_tables(val_true_arr, vp, rare_class=rare_class)
            if m["rare_f1"] > best_val_f1:
                best_val_f1, best_k = m["rare_f1"], k_cand
        knn_pred = pd.Series(
            _knn_predict(lab_lat, lab_labels, _lat(lats["test"]), best_k), dtype=str)

        # scRareRefine
        srr_pred = _conformal_rescue(
            proto, base_pred, _lat(lats["validation"]), val_true, _lat(lats["test"]))

        # HVG 表达数据（CellTypist / scBalance）
        hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()
        if dataset not in _adata_cache:
            print(f"  [加载原始 h5ad] {config['dataset']['path']}")
            _adata_cache[dataset] = load_adata(config)
        adata_full = _adata_cache[dataset]
        idx_map    = {cid: i for i, cid in enumerate(adata_full.obs_names)}

        def _get_X(cell_ids: list[str]) -> tuple[np.ndarray, list[str]]:
            rows_idx = [idx_map[cid] for cid in cell_ids if cid in idx_map]
            if len(rows_idx) != len(cell_ids):   # 不允许静默丢失 cell_id
                raise ValueError(
                    f"{dataset} seed={seed}: cell_id 与 h5ad 不匹配 "
                    f"(期望 {len(cell_ids)}，命中 {len(rows_idx)})，"
                    f"缓存可能与当前 h5ad 不一致")
            sub   = adata_full[rows_idx]
            hvg_v = [g for g in hvg_genes if g in sub.var_names]
            return _log1p_norm(sub[:, hvg_v].X), hvg_v

        labeled_ids = preds["train"]["cell_id"].astype(str)[is_labeled].tolist()
        test_ids    = preds["test"]["cell_id"].astype(str).tolist()
        train_X, hvg_v = _get_X(labeled_ids)
        test_X,  _     = _get_X(test_ids)
        assert train_X.shape[0] == len(lab_labels), \
            f"shape mismatch: {train_X.shape[0]} vs {len(lab_labels)}"

        print(f"  CellTypist...", end=" ", flush=True)
        ct_status = "ok"
        try:
            ct_pred = _run_celltypist(train_X, lab_labels, test_X, hvg_v)
        except Exception as e:
            print(f"FAILED ({e})")
            ct_pred, ct_status = None, "failed"

        print(f"scBalance...", end=" ", flush=True)
        sb_status = "ok"
        try:
            sb_pred = _run_scbalance(train_X, lab_labels, test_X)
        except Exception as e:
            print(f"FAILED ({e})")
            sb_pred, sb_status = None, "failed"
        print("done")

        for mname, pred_arr, status in [
            ("scANVI",       base_pred.to_numpy(),  "ok"),
            ("kNN",          knn_pred.to_numpy(),   "ok"),
            ("CellTypist",   ct_pred,               ct_status),
            ("scBalance",    sb_pred,               sb_status),
            ("scRareRefine", srr_pred.to_numpy(),   "ok"),
        ]:
            extra = f" [k={best_k}]" if mname == "kNN" else ""
            base_row = {
                "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
                "rare_class": rare_class, "method": mname, "status": status,
                "sep": round(proto.separability_ratio, 4),
                "best_k": best_k if mname == "kNN" else None,
            }
            if status == "failed":   # 失败方法不算指标、不进聚合
                print(f"  {mname:15s}: FAILED (excluded from aggregation)")
                rows.append(base_row)
                continue
            mres = _metrics(y_true, pred_arr, base_pred.to_numpy(), rare_class)
            print(f"  {mname:15s}: F1={mres['rare_f1']:.4f}  "
                  f"rec={mres['rare_recall']:.4f}  prec={mres['rare_precision']:.4f}  "
                  f"FP_rate={mres['rare_fp_rate']:.5f}{extra}")
            rows.append({**base_row, **mres})

    # ── 保存 CSV ──────────────────────────────────────────────────────────────
    out_dir = Path("results/comparison")
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "comparison_summary.csv", index=False)
    print(f"\n[saved] {out_dir}/comparison_summary.csv")

    # ── 3-seed 聚合（仅 status==ok 的 run 进入均值）──────────────────────────
    print("\n=== 3-seed 均值 ± σ（rare_f1）===")
    ok = df[df["status"] == "ok"]
    summary_rows = []
    for dataset in ok["dataset"].unique():
        for method in METHODS:
            sub = ok[(ok["dataset"] == dataset) & (ok["method"] == method)]
            if sub.empty:
                print(f"  {dataset:25s} {method:15s}: (无成功 run，跳过)")
                continue
            f1s = sub["rare_f1"].to_numpy()
            fps = sub["rare_fp_rate"].to_numpy()
            row = {
                "dataset": dataset, "method": method, "n_ok": len(sub),
                "f1_mean": round(f1s.mean(), 4), "f1_std": round(f1s.std(), 4),
                "fp_rate_max": round(fps.max(), 6),
            }
            summary_rows.append(row)
            print(f"  {dataset:25s} {method:15s}: "
                  f"F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
                  f"FP_rate_max={row['fp_rate_max']:.5f}  (n={len(sub)})")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "comparison_summary_agg.csv", index=False)

    # ── Markdown 报告 ─────────────────────────────────────────────────────────
    md = [
        "# 对比实验报告（scRareRefine vs baselines）",
        "",
        "实验日期：2026-06-12 | 数据集：3 | seed：42/43/44 | rare_train_size：5%/10%/10%",
        "",
        "## 方法说明",
        "",
        "| 方法 | 输入特征 | 训练数据 | 核心设计 |",
        "|------|---------|---------|---------|",
        "| scANVI | scANVI latent (20d) | labeled+unlabeled | 半监督 VAE 直接预测 |",
        "| kNN (best k) | scANVI latent (20d) | labeled only | 欧氏 k 近邻，val 上选 k∈{3,5,10,15} |",
        "| CellTypist | HVG log1p 表达 (2000-3000d) | labeled only | Logistic Regression（官方工具）|",
        "| scBalance | HVG log1p 表达 (2000-3000d) | labeled only | 加权采样神经网络（官方工具）|",
        "| **scRareRefine** | scANVI latent (20d) | labeled+unlabeled | scANVI + conformal prototype rescue |",
        "",
        "注：CellTypist 和 scBalance 使用各自设计的 HVG 基因表达输入；kNN 和 scRareRefine 使用 scANVI latent。",
        "",
        "注：rare_fp_rate = (pred==rare 且 真值非rare) / 非稀有数，所有方法可比的标准假阳性率；",
        "rescue_ffr（仅逐 run 明细）= 相对 scANVI 改判的误救率，仅对 scRareRefine 可解释。",
        "失败方法（status=failed）不计入均值。",
        "",
        "## 3-seed 均值 ± σ 结果",
        "",
        "| 数据集 | 方法 | F1 均值 | F1 σ | FP_rate_max | n_ok |",
        "|-------|------|--------|------|------------|------|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['dataset']} | {r['method']} | {r['f1_mean']:.4f} | "
            f"{r['f1_std']:.4f} | {r['fp_rate_max']:.5f} | {r['n_ok']} |"
        )

    md += [
        "",
        "## 逐 run 明细",
        "",
        "| 数据集 | seed | sep | 方法 | status | F1 | recall | precision | rare_fp_rate | rescue_ffr |",
        "|-------|------|-----|------|--------|-----|--------|-----------|-------------|-----------|",
    ]
    for _, r in df.iterrows():
        if r["status"] == "failed":
            md.append(
                f"| {r['dataset']} | {r['seed']} | {r['sep']:.3f} | {r['method']} | "
                f"failed | - | - | - | - | - |")
        else:
            md.append(
                f"| {r['dataset']} | {r['seed']} | {r['sep']:.3f} | {r['method']} | ok | "
                f"{r['rare_f1']:.4f} | {r['rare_recall']:.4f} | {r['rare_precision']:.4f} | "
                f"{r['rare_fp_rate']:.5f} | {r['rescue_ffr']:.5f} |")

    md.append("")
    (out_dir / "comparison_log.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[saved] {out_dir}/comparison_log.md")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
