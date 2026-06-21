"""HiCat 对比实验 — 在 sandbox310 环境中运行。

纯 Python 实现 HiCat（Briefings in Bioinformatics 2025）核心算法，无需 R/rpy2：
  log-norm → PCA(50) → Harmony(batch) → UMAP(2D) → DBSCAN → CatBoost → 置信度阈值

使用与 scANVI 相同的 HVG 基因集和 train/test split，只追加 HiCat 行，不重跑其他方法。

运行：
  D:/setup/anaconda/envs/sandbox310/python.exe tools/comparison/run_hicat_comparison.py
"""
from __future__ import annotations

import subprocess
import sys
import warnings
import json
import tempfile
import os
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import yaml
import harmonypy
from catboost import CatBoostClassifier
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
import umap as umap_lib

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
    # pancreas_integrated（整合人胰腺, endothelial, 5 平台）
    ("configs/pancreas_integrated.yaml",      42, "0.01"),
    ("configs/pancreas_integrated.yaml",      42, "0.05"),
    ("configs/pancreas_integrated.yaml",      42, "0.10"),
    ("configs/pancreas_integrated.yaml",      42, "all"),
]

# 可选 CLI：config 子串过滤 + --seeds 多 seed 覆盖（单一来源 tools/comparison/_runs.py）
import sys as _sys
from _runs import resolve_runs
RUNS = resolve_runs(RUNS, _sys.argv[1:])

HICAT_N_PCS       = 50
HICAT_CB_ITER     = 500   # CatBoost iterations
HICAT_CB_DEPTH    = 6
ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance", "ProtoCloud", "HiCat", "scRareRefine"]

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
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s == "all":
        return "all"
    try:
        f = float(s)
        return f if (0 < f <= 1 and "." in s) else int(f)
    except ValueError:
        raise ValueError(f"无法解析的稀有类标注规格: {value!r}")


def _rts_label(rts) -> str:
    if isinstance(rts, float):
        return f"{round(rts * 100)}pct"
    return str(rts)


def make_run_dir(config: dict, seed: int, rts_str: str) -> Path:
    exp        = config["experiment"]
    rare_class = exp["rare_class"]
    split_mode = exp.get("split_mode", "batch_heldout")
    size       = parse_rare_train_size(rts_str)
    run_id     = f"{split_mode}_seed{seed}_{safe_class_name(rare_class)}_rare{_rts_label(size)}"
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


# ── h5ad 数据提取 fallback（immune_dc 旧格式编码问题）────────────────────────

_EXTRACT_SCRIPT = r"""
import sys, json
import numpy as np
import anndata
import scipy.sparse as sp

args = json.loads(sys.stdin.read())
path       = args["path"]
hvg        = args["hvg"]
lab_ids    = args["labeled_ids"]
test_ids   = args["test_ids"]
use_raw    = args["use_raw"]
train_npz  = args["train_npz"]
test_npz   = args["test_npz"]
batch_key  = args.get("batch_key", None)

adata = anndata.read_h5ad(path)
id2row = {cid: i for i, cid in enumerate(adata.obs_names)}
hvg_in = [g for g in hvg if g in adata.var_names]
adata_hvg = adata[:, hvg_in]

def get_X(ids):
    ok = [cid for cid in ids if cid in id2row]
    rows = [id2row[cid] for cid in ok]
    if use_raw and adata_hvg.raw is not None:
        X = adata_hvg.raw[:, adata_hvg.var_names].X[rows]
    elif "counts" in adata_hvg.layers:
        X = adata_hvg.layers["counts"][rows]
    else:
        X = adata_hvg.X[rows]
    if sp.issparse(X):
        X = X.toarray()
    batch_vals = None
    if batch_key and batch_key in adata.obs.columns:
        batch_vals = adata.obs[batch_key].iloc[rows].astype(str).tolist()
    return np.asarray(X, dtype=np.float32), ok, batch_vals

train_X, train_ok, train_batch = get_X(lab_ids)
test_X,  test_ok,  test_batch  = get_X(test_ids)
np.savez(train_npz, X=train_X, ids=np.array(train_ok), genes=np.array(hvg_in),
         batch=np.array(train_batch if train_batch else []))
np.savez(test_npz,  X=test_X,  ids=np.array(test_ok),  genes=np.array(hvg_in),
         batch=np.array(test_batch  if test_batch  else []))
print(f"extracted train{train_X.shape} test{test_X.shape} genes={len(hvg_in)}")
"""


def _extract_counts_via_subprocess(
    config: dict, hvg_genes: list, labeled_ids: list, test_ids: list, run_dir: Path
) -> tuple:
    cache_dir = run_dir / "hicat_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_npz = str(cache_dir / "train.npz")
    test_npz  = str(cache_dir / "test.npz")

    if not (Path(train_npz).exists() and Path(test_npz).exists()):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(_EXTRACT_SCRIPT)
            script_path = f.name
        args = {
            "path":        config["dataset"]["path"],
            "hvg":         hvg_genes,
            "labeled_ids": labeled_ids,
            "test_ids":    test_ids,
            "use_raw":     config["dataset"].get("use_raw", False),
            "train_npz":   train_npz,
            "test_npz":    test_npz,
            "batch_key":   config["dataset"].get("batch_key", None),
        }
        try:
            proc = subprocess.run(
                [SCANVI311_PYTHON, script_path],
                input=json.dumps(args), capture_output=True, text=True, check=True,
            )
            print(f"    scanvi311 提取: {proc.stdout.strip()}")
        finally:
            os.unlink(script_path)

    train_d = np.load(train_npz, allow_pickle=True)
    test_d  = np.load(test_npz,  allow_pickle=True)
    return (train_d["X"], train_d["ids"].tolist(),
            test_d["X"],  test_d["ids"].tolist(),
            test_d["genes"].tolist(),
            train_d["batch"].tolist() if len(train_d["batch"]) > 0 else None,
            test_d["batch"].tolist()  if len(test_d["batch"])  > 0 else None)


def _load_counts(adata_sub: anndata.AnnData, use_raw: bool) -> np.ndarray:
    if use_raw and adata_sub.raw is not None:
        X = adata_sub.raw[:, adata_sub.var_names].X
    elif "counts" in adata_sub.layers:
        X = adata_sub.layers["counts"]
    else:
        X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ── HiCat 核心算法 ────────────────────────────────────────────────────────────

def _lognorm(X: np.ndarray) -> np.ndarray:
    """log1p(X / libsize * 1e4)，等价于 Seurat NormalizeData。"""
    lib = X.sum(axis=1, keepdims=True)
    lib = np.where(lib == 0, 1.0, lib)
    return np.log1p(X / lib * 1e4).astype(np.float32)


def run_hicat(config: dict, run_dir: Path, seed: int) -> np.ndarray:
    # NOTE: HiCat is TRANSDUCTIVE — PCA/Harmony/UMAP/DBSCAN are fit on combined
    # train+test features, and the confidence threshold is derived from test cluster
    # statistics. Results are NOT directly comparable with inductive methods on equal
    # footing; treat HiCat as a transductive upper-bound reference only.
    emb_dir = run_dir / "embeddings"

    # 读 split 信息
    train_df  = pd.read_csv(emb_dir / "train_predictions.csv", dtype={"cell_id": str})
    test_df   = pd.read_csv(emb_dir / "test_predictions.csv",  dtype={"cell_id": str})
    hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()

    is_labeled   = train_df["is_labeled_for_scanvi"].astype(bool).to_numpy()
    labeled_ids  = train_df["cell_id"][is_labeled].tolist()
    labeled_lbls = train_df["true_label"][is_labeled].astype(str).tolist()
    test_ids     = test_df["cell_id"].tolist()

    # 若已有缓存预测，直接加载
    cached_csv = emb_dir / "hicat_test_predictions.csv"
    if cached_csv.exists():
        print(f"    使用缓存预测 {cached_csv.name}")
        cached  = pd.read_csv(cached_csv, dtype=str)
        id2pred = dict(zip(cached["cell_id"], cached["hc_prediction"]))
        return np.array([id2pred.get(cid, "unknown") for cid in test_ids])

    # 加载 h5ad（含 immune_dc fallback）
    print(f"    加载 {config['dataset']['path']} ...")
    batch_key = config["dataset"].get("batch_key", None)
    use_raw   = config["dataset"].get("use_raw", False)

    hvg_in_var: list[str]
    train_batch: list | None = None
    test_batch:  list | None = None

    try:
        adata_full = anndata.read_h5ad(config["dataset"]["path"])
        id2row     = {cid: i for i, cid in enumerate(adata_full.obs_names)}
        hvg_in_var = [g for g in hvg_genes if g in adata_full.var_names]
        adata_hvg  = adata_full[:, hvg_in_var]

        def _get(cell_ids: list) -> tuple:
            ok   = [cid for cid in cell_ids if cid in id2row]
            rows = [id2row[cid] for cid in ok]
            X    = _load_counts(adata_hvg[rows], use_raw)
            batch_vals = None
            if batch_key and batch_key in adata_full.obs.columns:
                batch_vals = adata_full.obs[batch_key].iloc[rows].astype(str).tolist()
            return X, ok, batch_vals

        train_X, train_ids_ok, train_batch = _get(labeled_ids)
        test_X,  test_ids_ok,  test_batch  = _get(test_ids)

    except Exception as read_err:
        if "IOSpec" in str(read_err) or "encoding_type" in str(read_err) or "null" in str(read_err).lower():
            print(f"    h5ad 格式不兼容，通过 scanvi311 提取数据...")
            (train_X, train_ids_ok, test_X, test_ids_ok, hvg_in_var,
             train_batch, test_batch) = _extract_counts_via_subprocess(
                config, hvg_genes, labeled_ids, test_ids, run_dir)
        else:
            raise

    print(f"    HVG: {len(hvg_in_var)}/{len(hvg_genes)}  "
          f"train_labeled: {len(train_ids_ok)}  test: {len(test_ids_ok)}")

    id2lbl = dict(zip(labeled_ids, labeled_lbls))
    train_labels = np.array([id2lbl[cid] for cid in train_ids_ok])

    # ── Step 1: log-norm ──────────────────────────────────────────────────────
    train_norm = _lognorm(train_X)
    test_norm  = _lognorm(test_X)
    combined   = np.vstack([train_norm, test_norm])

    # ── Step 2: PCA(50) ───────────────────────────────────────────────────────
    n_pcs = min(HICAT_N_PCS, combined.shape[0] - 1, combined.shape[1] - 1)
    print(f"    PCA({n_pcs})...")
    pca_model = PCA(n_components=n_pcs, random_state=seed, svd_solver="randomized")
    combined_pcs = pca_model.fit_transform(combined)
    train_pcs = combined_pcs[:len(train_ids_ok)]
    test_pcs  = combined_pcs[len(train_ids_ok):]

    # ── Step 3: Harmony 批次矫正 ──────────────────────────────────────────────
    if batch_key and (train_batch or test_batch):
        tb = train_batch if train_batch else ["ref"] * len(train_ids_ok)
        qb = test_batch  if test_batch  else ["qry"] * len(test_ids_ok)
        meta = pd.DataFrame({"batch": tb + qb})
        print(f"    Harmony(batch_key={batch_key}, n_batches={meta['batch'].nunique()})...")
        ho = harmonypy.run_harmony(
            combined_pcs.astype(np.float64), meta, vars_use=["batch"],
            max_iter_harmony=10, random_state=seed, verbose=False)
        combined_pcs = ho.result().astype(np.float32)
        train_pcs = combined_pcs[:len(train_ids_ok)]
        test_pcs  = combined_pcs[len(train_ids_ok):]
    else:
        print("    跳过 Harmony（无 batch_key）")

    # ── Step 4: UMAP(2D) on combined ─────────────────────────────────────────
    print("    UMAP(2D)...")
    umap_model = umap_lib.UMAP(n_components=2, random_state=seed)
    combined_umap = umap_model.fit_transform(combined_pcs)
    train_umap = combined_umap[:len(train_ids_ok)]
    test_umap  = combined_umap[len(train_ids_ok):]

    # ── Step 5: DBSCAN on combined UMAP ──────────────────────────────────────
    print("    DBSCAN...")
    dbscan = DBSCAN()
    combined_dbscan = dbscan.fit_predict(combined_umap).astype(np.float32).reshape(-1, 1)
    train_dbscan = combined_dbscan[:len(train_ids_ok)]
    test_dbscan  = combined_dbscan[len(train_ids_ok):]

    # ── Step 6: 53-dim features [50PC + 2UMAP + 1DBSCAN] ────────────────────
    X_train = np.hstack([train_pcs, train_umap, train_dbscan])
    X_test  = np.hstack([test_pcs,  test_umap,  test_dbscan])

    # ── Step 7: CatBoost ─────────────────────────────────────────────────────
    print(f"    CatBoost(iter={HICAT_CB_ITER}, seed={seed})...")
    cb = CatBoostClassifier(
        iterations=HICAT_CB_ITER,
        depth=HICAT_CB_DEPTH,
        random_seed=seed,
        verbose=0,
    )
    cb.fit(X_train, train_labels)

    y_pred_raw   = cb.predict(X_test).flatten().astype(str)
    y_pred_proba = cb.predict_proba(X_test)
    confidence   = y_pred_proba.max(axis=1)

    # ── Step 8: 置信度阈值（HiCat 原版逻辑） ─────────────────────────────────
    test_db_str = test_dbscan.flatten().astype(int).astype(str)
    cluster_conf = (
        pd.DataFrame({"cluster": test_db_str, "conf": confidence})
        .groupby("cluster")["conf"].mean()
        .sort_values()
    )
    diff = cluster_conf.diff().values
    if len(diff) > 1 and not np.all(np.isnan(diff[1:])):
        jump_idx  = int(np.nanargmax(diff[1:])) + 1
        conf_th   = float(cluster_conf.iloc[jump_idx])
    else:
        conf_th = 0.0

    y_pred_final = y_pred_raw.copy()
    low_conf_mask = confidence < conf_th
    y_pred_final[low_conf_mask] = test_db_str[low_conf_mask]  # DBSCAN 簇标签
    print(f"    conf_th={conf_th:.4f}  low_conf_cells={low_conf_mask.sum()}")

    # 保存预测缓存
    out_csv = emb_dir / "hicat_test_predictions.csv"
    pd.DataFrame({"cell_id": test_ids_ok, "hc_prediction": y_pred_final}).to_csv(out_csv, index=False)

    return y_pred_final


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    print("NOTE: HiCat is TRANSDUCTIVE (PCA/UMAP/DBSCAN fit on combined train+test).")
    print("      It serves as a transductive upper-bound reference, not an inductive baseline.")
    # 只替换本次 RUNS 覆盖的 (dataset, seed, rts) 行，其余 HiCat 行保留
    run_key_set = {
        (load_config(c)["dataset"]["name"], str(int(s)), str(r))
        for c, s, r in RUNS
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
        is_hicat = existing["method"] == "HiCat"
        in_runs  = existing.apply(
            lambda row: (str(row["dataset"]), str(int(float(row["seed"]))), str(row["rare_train_size"])) in run_key_set,
            axis=1
        )
        existing = existing[~(is_hicat & in_runs)]
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
        hc_pred = None
        try:
            hc_pred = run_hicat(config, run_dir, seed)
        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            status = "failed"

        row = {"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
               "rare_class": rare_class, "method": "HiCat", "status": status}

        if status == "ok":
            mres = compute_row_metrics(y_true, hc_pred, base_pred, rare_class)
            row.update(mres)
            print(f"  HiCat: F1={mres['rare_f1']:.4f}  "
                  f"rec={mres['rare_recall']:.4f}  prec={mres['rare_precision']:.4f}  "
                  f"FP_rate={mres['rare_fp_rate']:.5f}")
        else:
            print("  HiCat: FAILED（不计入聚合）")

        new_rows.append(row)

    # 合并并保存
    full_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    col_order = [
        "dataset", "seed", "rare_train_size", "rare_class", "method", "status",
        "sep", "best_k",
        "rare_f1", "rare_recall", "rare_precision", "macro_f1",
        "rare_fp_rate", "n_rescued", "n_false_rescue", "rescue_ffr",
    ]
    for c in col_order:
        if c not in full_df.columns:
            full_df[c] = None
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
                if sub.empty:
                    continue
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

    # 更新图例和重绘（需先更新 plot_comparison.py 的 METHODS 列表）
    print("\n重绘 comparison_bars ...")
    subprocess.run([SCANVI311_PYTHON, "tools/comparison/plot_comparison.py"], check=True)


if __name__ == "__main__":
    main()
