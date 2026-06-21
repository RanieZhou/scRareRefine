"""CellTypist baseline 对比实验 — 在 scanvi311 环境中运行。

在 HVG log1p 归一化表达（CellTypist 官方工具标准输入）上训练 Logistic
Regression，仅使用 is_labeled_for_scanvi 标注的训练子集。

运行：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/comparison/run_celltypist_comparison.py
"""
from __future__ import annotations

import sys
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables, load_adata
from tools.comparison._conda_python import conda_python

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

# ── run 列表（5 个数据集 × seed=42 × 全部 4 个比例，不含 tabula_lung_stroma）──────
RUNS = [
    # immune_dc
    ("configs/immune_dc.yaml",                42, "0.01"),
    ("configs/immune_dc.yaml",                42, "0.05"),
    ("configs/immune_dc.yaml",                42, "0.10"),
    ("configs/immune_dc.yaml",                42, "all"),
    # pancreas_baron
    ("configs/pancreas_baron.yaml",           42, "0.01"),
    ("configs/pancreas_baron.yaml",           42, "0.05"),
    ("configs/pancreas_baron.yaml",           42, "0.10"),
    ("configs/pancreas_baron.yaml",           42, "all"),
    # tabula_lung_endo
    ("configs/tabula_lung_endo.yaml",         42, "0.01"),
    ("configs/tabula_lung_endo.yaml",         42, "0.05"),
    ("configs/tabula_lung_endo.yaml",         42, "0.10"),
    ("configs/tabula_lung_endo.yaml",         42, "all"),
    # tabula_small_intestine
    ("configs/tabula_small_intestine.yaml",   42, "0.01"),
    ("configs/tabula_small_intestine.yaml",   42, "0.05"),
    ("configs/tabula_small_intestine.yaml",   42, "0.10"),
    ("configs/tabula_small_intestine.yaml",   42, "all"),
    # tabula_sapiens_stomach
    ("configs/tabula_sapiens_stomach.yaml",   42, "0.01"),
    ("configs/tabula_sapiens_stomach.yaml",   42, "0.05"),
    ("configs/tabula_sapiens_stomach.yaml",   42, "0.10"),
    ("configs/tabula_sapiens_stomach.yaml",   42, "all"),
    # pancreas_integrated（整合人胰腺, endothelial, 5 平台）
    ("configs/pancreas_integrated.yaml",      42, "0.01"),
    ("configs/pancreas_integrated.yaml",      42, "0.05"),
    ("configs/pancreas_integrated.yaml",      42, "0.10"),
    ("configs/pancreas_integrated.yaml",      42, "all"),
]

# 可选 CLI：config 子串过滤 + --seeds 多 seed 覆盖（单一来源 tools/comparison/_runs.py）
import sys as _sys
from tools.comparison._runs import resolve_runs
RUNS = resolve_runs(RUNS, _sys.argv[1:])

METHOD_NAME = "CellTypist"
ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance",
               "ProtoCloud", "HiCat", "scCAD", "TOSICA", "scRareRefine"]

OUT_DIR          = Path("results/comparison")
SUMMARY_CSV      = OUT_DIR / "comparison_summary.csv"
AGG_CSV          = OUT_DIR / "comparison_summary_agg.csv"
SCANVI311_PYTHON = conda_python("scanvi311")


def _log1p_norm(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    total = X.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(X / total * target_sum)


def _metrics(y_true, pred, base_pred, rare_class) -> dict:
    pred = np.asarray(pred, dtype=str)
    bp   = np.asarray(base_pred, dtype=str)
    m, _ = classification_tables(y_true, pred, rare_class=rare_class)
    n_nrare = int((y_true != rare_class).sum())
    n_fp    = int(((pred == rare_class) & (y_true != rare_class)).sum())
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


def _check_manifest(run_dir: Path, config: dict, seed: int, rts_str: str) -> bool:
    mf = run_dir / "manifest.json"
    if not mf.exists():
        print("  [provenance] WARNING: 无 manifest.json（旧缓存，无法校验 split/代码版本）")
        return True
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
        print(f"  [provenance] ERROR: manifest 与当前配置不匹配，跳过此 run: {mism}")
        return False
    print(f"  [provenance] OK  split_hash={m.get('split_hash')}  git_sha={m.get('git_sha')}")
    return True


def _run_celltypist(train_X: np.ndarray, train_labels: np.ndarray,
                    test_X: np.ndarray, hvg_genes: list[str]) -> np.ndarray:
    gene_names = hvg_genes[:train_X.shape[1]]
    adata_tr  = ad.AnnData(X=train_X); adata_tr.var_names = gene_names
    adata_te  = ad.AnnData(X=test_X);  adata_te.var_names = gene_names
    logging.getLogger("celltypist").setLevel(logging.CRITICAL)
    model = celltypist.train(adata_tr, labels=train_labels, check_expression=False)
    pred  = celltypist.annotate(adata_te, model=model, majority_voting=False)
    return pred.predicted_labels["predicted_labels"].to_numpy().astype(str)


def main():
    from src.rescue import PrototypeRescuer

    run_key_set = {
        (load_config(c)["dataset"]["name"], str(int(s)), str(r))
        for c, s, r in RUNS
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
        is_own  = existing["method"] == METHOD_NAME
        in_runs = existing.apply(
            lambda row: (str(row["dataset"]), str(int(float(row["seed"]))), str(row["rare_train_size"])) in run_key_set,
            axis=1
        )
        existing = existing[~(is_own & in_runs)]
    else:
        existing = pd.DataFrame()

    new_rows = []
    _adata_cache: dict[str, ad.AnnData] = {}

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

        if not _check_manifest(run_dir, config, seed, rts_str):
            new_rows.append({"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
                             "rare_class": rare_class, "method": METHOD_NAME, "status": "failed"})
            continue

        train_lat  = pd.read_csv(emb_dir / "train_latent.csv")
        train_lat  = train_lat[[c for c in train_lat.columns if c.startswith("latent_")]].to_numpy()
        train_pred = pd.read_csv(emb_dir / "train_predictions.csv")
        is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
        ref_labels = train_pred["true_label"].astype(str)
        lab_labels = ref_labels[is_labeled].to_numpy()

        proto = PrototypeRescuer(rare_class)
        proto.fit(train_lat, ref_labels, is_labeled)

        test_pred = pd.read_csv(emb_dir / "test_predictions.csv")
        y_true    = test_pred["true_label"].astype(str).to_numpy()
        base_pred = test_pred["predicted_label"].astype(str).to_numpy()

        print(f"\n[{dataset} seed={seed} rts={rts_str}] sep={proto.separability_ratio:.3f}  "
              f"lab_rare={int((lab_labels==rare_class).sum())}  test_rare={int((y_true==rare_class).sum())}")

        status = "ok"
        ct_pred = None
        try:
            hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()
            if dataset not in _adata_cache:
                print(f"  [加载原始 h5ad] {config['dataset']['path']}")
                _adata_cache[dataset] = load_adata(config)
            adata_full = _adata_cache[dataset]
            idx_map    = {cid: i for i, cid in enumerate(adata_full.obs_names)}

            def _get_X(cell_ids: list[str]) -> tuple[np.ndarray, list[str]]:
                rows_idx = [idx_map[cid] for cid in cell_ids if cid in idx_map]
                if len(rows_idx) != len(cell_ids):
                    raise ValueError(
                        f"{dataset} seed={seed}: cell_id 与 h5ad 不匹配 "
                        f"(期望 {len(cell_ids)}，命中 {len(rows_idx)})，"
                        f"缓存可能与当前 h5ad 不一致")
                sub   = adata_full[rows_idx]
                hvg_v = [g for g in hvg_genes if g in sub.var_names]
                return _log1p_norm(sub[:, hvg_v].X), hvg_v

            labeled_ids = train_pred["cell_id"].astype(str)[is_labeled].tolist()
            test_ids    = test_pred["cell_id"].astype(str).tolist()
            train_X, hvg_v = _get_X(labeled_ids)
            test_X,  _     = _get_X(test_ids)

            print("  CellTypist...", end=" ", flush=True)
            ct_pred = _run_celltypist(train_X, lab_labels, test_X, hvg_v)
            print("done")
        except Exception as e:
            print(f"FAILED ({e})")
            status = "failed"

        row = {"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
               "rare_class": rare_class, "method": METHOD_NAME, "status": status,
               "sep": round(proto.separability_ratio, 4), "best_k": None}

        if status == "ok":
            mres = _metrics(y_true, ct_pred, base_pred, rare_class)
            row.update(mres)
            print(f"  {METHOD_NAME:15s}: F1={mres['rare_f1']:.4f}  rec={mres['rare_recall']:.4f}  "
                  f"prec={mres['rare_precision']:.4f}  FP_rate={mres['rare_fp_rate']:.5f}")

        new_rows.append(row)

    full_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    col_order = ["dataset", "seed", "rare_train_size", "rare_class", "method", "status",
                 "sep", "best_k", "rare_f1", "rare_recall", "rare_precision", "macro_f1",
                 "rare_fp_rate", "n_rescued", "n_false_rescue", "rescue_ffr"]
    for c in col_order:
        if c not in full_df.columns: full_df[c] = None
    OUT_DIR.mkdir(exist_ok=True)
    full_df[col_order].to_csv(SUMMARY_CSV, index=False)
    print(f"\n[saved] {SUMMARY_CSV}  ({len(full_df)} 行)")

    ok = full_df[full_df["status"] == "ok"]
    agg_rows = []
    print("\n=== 聚合结果 ===")
    for ds in ok["dataset"].unique():
        for rts in sorted(ok[ok["dataset"] == ds]["rare_train_size"].unique()):
            for method in ALL_METHODS:
                sub = ok[(ok["dataset"] == ds) & (ok["rare_train_size"] == rts) & (ok["method"] == method)]
                if sub.empty: continue
                f1s, recs, fps = sub["rare_f1"].to_numpy(), sub["rare_recall"].to_numpy(), sub["rare_fp_rate"].to_numpy()
                agg_rows.append({
                    "dataset": ds, "rare_train_size": rts, "method": method, "n_ok": len(sub),
                    "f1_mean": round(float(f1s.mean()), 4), "f1_std": round(float(f1s.std()), 4),
                    "rec_mean": round(float(recs.mean()), 4), "rec_std": round(float(recs.std()), 4),
                    "fp_rate_max": round(float(fps.max()), 6),
                })
                print(f"  {ds:25s} rts={rts:4s}  {method:15s}: "
                      f"F1={agg_rows[-1]['f1_mean']:.4f}±{agg_rows[-1]['f1_std']:.4f}  "
                      f"rec={agg_rows[-1]['rec_mean']:.4f}  (n={len(sub)})")

    pd.DataFrame(agg_rows).to_csv(AGG_CSV, index=False)
    print(f"[saved] {AGG_CSV}")

    print("\n重绘 comparison_bars ...")
    subprocess.run([SCANVI311_PYTHON, "tools/comparison/plot_comparison.py"], check=True)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
