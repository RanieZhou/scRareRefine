"""scCAD 对比实验 — 在 scanvi311 环境中运行。

scCAD（Nature Communications 2024）：无监督稀有细胞异常检测。
在 test 细胞上运行 scCAD，将识别为稀有的细胞预测为 rare_class，
其余细胞保留 scANVI baseline 预测。

运行：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/comparison/run_scCAD_comparison.py
"""
from __future__ import annotations

import sys
import subprocess
import warnings
import tempfile
import os
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import yaml
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path("baseline/scCAD").resolve()))
import scCAD as scCAD_lib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _conda_python import conda_python

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
]

ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance",
               "ProtoCloud", "HiCat", "scCAD", "TOSICA", "scRareRefine"]

OUT_DIR     = Path("results/comparison")
SUMMARY_CSV = OUT_DIR / "comparison_summary.csv"
AGG_CSV     = OUT_DIR / "comparison_summary_agg.csv"
SCANVI311_PYTHON = conda_python("scanvi311")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_class_name(name: str) -> str:
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def parse_rare_train_size(value) -> int | float | str:
    if isinstance(value, float): return value
    if isinstance(value, int):   return value
    s = str(value).strip().lower()
    if s == "all": return "all"
    try:
        f = float(s)
        return f if (0 < f <= 1 and "." in s) else int(f)
    except ValueError:
        raise ValueError(f"无法解析的稀有类标注规格: {value!r}")


def _rts_label(rts) -> str:
    return f"{round(rts * 100)}pct" if isinstance(rts, float) else str(rts)


def make_run_dir(config: dict, seed: int, rts_str: str) -> Path:
    exp  = config["experiment"]
    size = parse_rare_train_size(rts_str)
    run_id = (f"{exp.get('split_mode','batch_heldout')}_seed{seed}_"
              f"{safe_class_name(exp['rare_class'])}_rare{_rts_label(size)}")
    return Path("outputs") / config["dataset"]["name"] / run_id


def classification_metrics(y_true, y_pred, rare_class: str) -> dict:
    y_true = np.asarray(y_true, dtype=str)
    y_pred = np.asarray(y_pred, dtype=str)
    labels = sorted(set(y_true) | set(y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    idx = labels.index(rare_class) if rare_class in labels else -1
    return {
        "rare_f1":        float(f1[idx])   if idx >= 0 else 0.0,
        "rare_recall":    float(rec[idx])  if idx >= 0 else 0.0,
        "rare_precision": float(prec[idx]) if idx >= 0 else 0.0,
        "macro_f1":       float(f1.mean()),
    }


def compute_row_metrics(y_true, pred, base_pred, rare_class: str) -> dict:
    pred      = np.asarray(pred,      dtype=str)
    base_pred = np.asarray(base_pred, dtype=str)
    m         = classification_metrics(y_true, pred, rare_class)
    n_nrare   = int((y_true != rare_class).sum())
    n_fp      = int(((pred == rare_class) & (y_true != rare_class)).sum())
    n_res     = int(((pred != base_pred) & (pred == rare_class)).sum())
    n_false   = int(((pred != base_pred) & (pred == rare_class) & (y_true != rare_class)).sum())
    return {
        **{k: round(v, 4) for k, v in m.items()},
        "rare_fp_rate":   round(n_fp    / max(n_nrare, 1), 6),
        "n_rescued":      n_res,
        "n_false_rescue": n_false,
        "rescue_ffr":     round(n_false / max(n_nrare, 1), 6),
    }


def _load_counts(adata_sub: anndata.AnnData, use_raw: bool) -> np.ndarray:
    if use_raw and adata_sub.raw is not None:
        X = adata_sub.raw[:, adata_sub.var_names].X
    elif "counts" in adata_sub.layers:
        X = adata_sub.layers["counts"]
    else:
        X = adata_sub.X
    if sp.issparse(X): X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ── scCAD 核心 ────────────────────────────────────────────────────────────────

def run_scCAD(config: dict, run_dir: Path, seed: int) -> np.ndarray:
    emb_dir = run_dir / "embeddings"
    test_df  = pd.read_csv(emb_dir / "test_predictions.csv",  dtype={"cell_id": str})
    hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()

    test_ids  = test_df["cell_id"].tolist()
    base_pred = test_df["predicted_label"].astype(str).to_numpy()
    rare_class = config["experiment"]["rare_class"]

    # 检查缓存
    cached_csv = emb_dir / "scCAD_test_predictions.csv"
    if cached_csv.exists():
        print(f"    使用缓存预测 {cached_csv.name}")
        cached  = pd.read_csv(cached_csv, dtype=str)
        id2pred = dict(zip(cached["cell_id"], cached["sc_prediction"]))
        return np.array([id2pred.get(cid, base_pred[i]) for i, cid in enumerate(test_ids)])

    # 加载 h5ad
    print(f"    加载 {config['dataset']['path']} ...")
    use_raw = config["dataset"].get("use_raw", False)
    adata_full = anndata.read_h5ad(config["dataset"]["path"])
    id2row     = {cid: i for i, cid in enumerate(adata_full.obs_names)}
    hvg_in_var = [g for g in hvg_genes if g in adata_full.var_names]
    adata_hvg  = adata_full[:, hvg_in_var]

    # 仅取 test cells
    test_rows = [id2row[cid] for cid in test_ids if cid in id2row]
    test_ids_ok = [cid for cid in test_ids if cid in id2row]
    X_test = _load_counts(adata_hvg[test_rows], use_raw)

    print(f"    test cells: {len(test_ids_ok)}  HVG: {len(hvg_in_var)}")

    # 运行 scCAD（在 tmpdir 保存中间文件）
    print(f"    运行 scCAD (seed={seed}) ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        result, score, sub_clusters, degs = scCAD_lib.scCAD(
            data=X_test,
            dataName=config["dataset"]["name"],
            cellNames=np.array(test_ids_ok),
            geneNames=np.array(hvg_in_var),
            normalization=True,
            seed=seed,
            save_full=False,
            save_path=tmpdir + "/",
        )

    # 汇总识别为稀有的细胞 ID
    rare_cell_ids: set = set()
    for cluster in result:
        rare_cell_ids.update(cluster)

    n_rare_found = len(rare_cell_ids)
    n_rare_true  = int((test_df["true_label"].astype(str) == rare_class).sum())
    print(f"    scCAD 识别稀有细胞: {n_rare_found}  true rare: {n_rare_true}")

    # 构造预测：识别为稀有 → rare_class；其余 → base_pred
    pred = base_pred.copy()
    for i, cid in enumerate(test_ids):
        if cid in rare_cell_ids:
            pred[i] = rare_class

    # 保存缓存
    pd.DataFrame({"cell_id": test_ids, "sc_prediction": pred}).to_csv(cached_csv, index=False)

    return pred


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    # 只替换本次 RUNS 覆盖的 (dataset, seed, rts) 行，其余 scCAD 行保留
    run_key_set = {
        (load_config(c)["dataset"]["name"], str(int(s)), str(r))
        for c, s, r in RUNS
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
        is_sccad = existing["method"] == "scCAD"
        in_runs  = existing.apply(
            lambda row: (str(row["dataset"]), str(int(float(row["seed"]))), str(row["rare_train_size"])) in run_key_set,
            axis=1
        )
        existing = existing[~(is_sccad & in_runs)]
    else:
        existing = pd.DataFrame()

    new_rows = []

    for cfg_path, seed, rts_str in RUNS:
        config     = load_config(cfg_path)
        dataset    = config["dataset"]["name"]
        rare_class = config["experiment"]["rare_class"]
        run_dir    = make_run_dir(config, seed, rts_str)
        emb_dir    = run_dir / "embeddings"

        if not (emb_dir / "test_predictions.csv").exists():
            print(f"[SKIP] {run_dir} 缓存不存在")
            continue

        test_df   = pd.read_csv(emb_dir / "test_predictions.csv")
        y_true    = test_df["true_label"].astype(str).to_numpy()
        base_pred = test_df["predicted_label"].astype(str).to_numpy()

        print(f"\n[{dataset}  seed={seed}  rts={rts_str}]")
        status = "ok"
        sc_pred = None
        try:
            sc_pred = run_scCAD(config, run_dir, seed)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}"); status = "failed"

        row = {"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
               "rare_class": rare_class, "method": "scCAD", "status": status}

        if status == "ok":
            mres = compute_row_metrics(y_true, sc_pred, base_pred, rare_class)
            row.update(mres)
            print(f"  scCAD: F1={mres['rare_f1']:.4f}  rec={mres['rare_recall']:.4f}  "
                  f"prec={mres['rare_precision']:.4f}  FP_rate={mres['rare_fp_rate']:.5f}")

        new_rows.append(row)

    # 合并保存
    full_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    col_order = ["dataset","seed","rare_train_size","rare_class","method","status",
                 "sep","best_k","rare_f1","rare_recall","rare_precision","macro_f1",
                 "rare_fp_rate","n_rescued","n_false_rescue","rescue_ffr"]
    for c in col_order:
        if c not in full_df.columns: full_df[c] = None
    full_df[col_order].to_csv(SUMMARY_CSV, index=False)
    print(f"\n[saved] {SUMMARY_CSV}  ({len(full_df)} 行)")

    # 重聚合（按 dataset × rare_train_size × method）
    ok = full_df[full_df["status"] == "ok"]
    agg_rows = []
    print("\n=== 聚合结果 ===")
    for ds in ok["dataset"].unique():
        for rts in sorted(ok[ok["dataset"] == ds]["rare_train_size"].unique()):
            for method in ALL_METHODS:
                sub = ok[
                    (ok["dataset"] == ds) &
                    (ok["rare_train_size"] == rts) &
                    (ok["method"] == method)
                ]
                if sub.empty: continue
                f1s  = sub["rare_f1"].to_numpy()
                recs = sub["rare_recall"].to_numpy()
                fps  = sub["rare_fp_rate"].to_numpy()
                agg_rows.append({
                    "dataset": ds, "rare_train_size": rts, "method": method, "n_ok": len(sub),
                    "f1_mean":  round(float(f1s.mean()),  4),
                    "f1_std":   round(float(f1s.std()),   4),
                    "rec_mean": round(float(recs.mean()), 4),
                    "rec_std":  round(float(recs.std()),  4),
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
    main()
