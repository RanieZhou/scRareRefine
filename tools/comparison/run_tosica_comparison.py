"""TOSICA 对比实验 — 在 sandbox310 环境中运行。

TOSICA（Nature Communications 2023）：Transformer + pathway token 可解释细胞类型注释。
以 train cells 为参考集训练 TOSICA，预测 test cells 细胞类型。

运行：
  D:/setup/anaconda/envs/sandbox310/python.exe tools/comparison/run_tosica_comparison.py
"""
from __future__ import annotations

import sys
import os
import json
import subprocess
import warnings
import shutil
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import yaml
from sklearn.metrics import precision_recall_fscore_support

import TOSICA

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

GMT_MAP = {
    "immune_dc":               "human_gobp",
    "pancreas_baron":          "human_gobp",
    "tabula_lung_endo":        "human_gobp",
    "tabula_lung_stroma":      "human_gobp",
    "tabula_small_intestine":  "human_gobp",
    "tabula_sapiens_stomach":  "human_gobp",
}

ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance",
               "ProtoCloud", "HiCat", "scCAD", "TOSICA", "scRareRefine"]

TOSICA_EPOCHS    = 10
TOSICA_MAX_GS    = 100   # 限制 pathway 数量：模型从 ~172 MB 缩小到 ~57 MB/epoch
OUT_DIR          = Path("results/comparison")
SUMMARY_CSV      = OUT_DIR / "comparison_summary.csv"
AGG_CSV          = OUT_DIR / "comparison_summary_agg.csv"
SCANVI311_PYTHON = conda_python("scanvi311")
SANDBOX_PYTHON   = conda_python("sandbox310")


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
        raise ValueError(f"无法解析: {value!r}")


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


def _lognorm(X: np.ndarray) -> np.ndarray:
    libsize = X.sum(axis=1, keepdims=True).clip(1)
    return np.log1p(X / libsize * 1e4).astype(np.float32)


# ── immune_dc 数据提取（sandbox310 anndata 0.11.4 无法读取旧格式 h5ad）────────

_EXTRACT_SCRIPT = r"""
import sys, json, os
import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import yaml

config_path, train_json, test_json, hvg_json, out_npz = sys.argv[1:]

with open(config_path, encoding='utf-8') as f:
    config = yaml.safe_load(f)

train_data   = json.loads(open(train_json).read())
train_ids    = train_data['ids']
train_labels = train_data['labels']
test_ids     = json.loads(open(test_json).read())
hvg_genes    = json.loads(open(hvg_json).read())

use_raw   = config['dataset'].get('use_raw', False)
use_layer = config['dataset'].get('use_layer', None)

adata = anndata.read_h5ad(config['dataset']['path'])
id2row = {cid: i for i, cid in enumerate(adata.obs_names)}
hvg_in_var = [g for g in hvg_genes if g in adata.var_names]

def get_X(adata_sub):
    if use_raw and adata_sub.raw is not None:
        X = adata_sub.raw[:, adata_sub.var_names].X
    elif use_layer and use_layer in adata_sub.layers:
        X = adata_sub.layers[use_layer]
    else:
        X = adata_sub.X
    if sp.issparse(X): X = X.toarray()
    return np.asarray(X, dtype=np.float32)

adata_hvg    = adata[:, hvg_in_var]
train_rows   = [id2row[cid] for cid in train_ids if cid in id2row]
test_rows    = [id2row[cid] for cid in test_ids  if cid in id2row]
train_ids_ok = [cid for cid in train_ids if cid in id2row]
test_ids_ok  = [cid for cid in test_ids  if cid in id2row]
labels_ok    = [train_labels[train_ids.index(cid)] for cid in train_ids_ok]

X_train = get_X(adata_hvg[train_rows])
X_test  = get_X(adata_hvg[test_rows])

os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
np.savez(out_npz,
         X_train=X_train, X_test=X_test,
         var_names=np.array(hvg_in_var),
         train_ids=np.array(train_ids_ok),
         test_ids=np.array(test_ids_ok),
         train_labels=np.array(labels_ok))
print(f'Extracted: X_train={X_train.shape}, X_test={X_test.shape}, genes={len(hvg_in_var)}')
"""


def _extract_via_subprocess(cfg_path: str, config: dict, run_dir: Path,
                             train_ids: list, train_labels: list,
                             test_ids: list, hvg_genes: list) -> tuple:
    """通过 scanvi311 子进程提取数据（针对 immune_dc 旧格式 h5ad）。"""
    cache_dir  = run_dir / "tosica_cache"
    cache_npz  = cache_dir / "data.npz"

    if cache_npz.exists():
        print(f"    使用数据缓存 {cache_npz}")
        d = np.load(cache_npz, allow_pickle=True)
        return (d["X_train"], d["X_test"],
                d["var_names"].tolist(),
                d["train_ids"].tolist(), d["test_ids"].tolist(),
                d["train_labels"].tolist())

    tmp_dir = run_dir / "_tosica_extract_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        (tmp_dir / "train.json").write_text(
            json.dumps({"ids": train_ids, "labels": train_labels}))
        (tmp_dir / "test.json").write_text(json.dumps(test_ids))
        (tmp_dir / "hvg.json").write_text(json.dumps(hvg_genes))
        script = tmp_dir / "extract.py"
        script.write_text(_EXTRACT_SCRIPT)

        cmd = [SCANVI311_PYTHON, str(script),
               cfg_path,
               str(tmp_dir / "train.json"),
               str(tmp_dir / "test.json"),
               str(tmp_dir / "hvg.json"),
               str(cache_npz)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"提取脚本失败:\n{result.stderr}")
        print(f"    {result.stdout.strip()}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    d = np.load(cache_npz, allow_pickle=True)
    return (d["X_train"], d["X_test"],
            d["var_names"].tolist(),
            d["train_ids"].tolist(), d["test_ids"].tolist(),
            d["train_labels"].tolist())


def _load_counts_direct(adata_sub: ad.AnnData, config: dict) -> np.ndarray:
    use_raw   = config["dataset"].get("use_raw", False)
    use_layer = config["dataset"].get("use_layer", None)
    if use_raw and adata_sub.raw is not None:
        X = adata_sub.raw[:, adata_sub.var_names].X
    elif use_layer and use_layer in adata_sub.layers:
        X = adata_sub.layers[use_layer]
    else:
        X = adata_sub.X
    if sp.issparse(X): X = X.toarray()
    return np.asarray(X, dtype=np.float32)


# ── TOSICA 核心 ───────────────────────────────────────────────────────────────

def run_tosica(cfg_path: str, config: dict, run_dir: Path, seed: int, rts_str: str) -> np.ndarray:
    emb_dir    = run_dir / "embeddings"
    cached_csv = emb_dir / "tosica_test_predictions.csv"

    train_df  = pd.read_csv(emb_dir / "train_predictions.csv", dtype={"cell_id": str})
    test_df   = pd.read_csv(emb_dir / "test_predictions.csv",  dtype={"cell_id": str})
    hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()

    # 使用 is_labeled_for_scanvi 而非 true_label != Unknown，
    # 保证不同 rare_train_size 下 rare cells 的 label 可见性与 scANVI 完全一致
    labeled_mask = train_df["is_labeled_for_scanvi"].astype(bool)
    train_df     = train_df[labeled_mask]

    train_ids    = train_df["cell_id"].tolist()
    train_labels = train_df["true_label"].astype(str).tolist()
    test_ids     = test_df["cell_id"].tolist()
    base_pred    = test_df["predicted_label"].astype(str).to_numpy()
    rare_class   = config["experiment"]["rare_class"]
    dataset      = config["dataset"]["name"]

    # 预测缓存
    if cached_csv.exists():
        print(f"    使用预测缓存 {cached_csv.name}")
        cached   = pd.read_csv(cached_csv, dtype=str)
        id2pred  = dict(zip(cached["cell_id"], cached["tosica_prediction"]))
        pred_arr = np.array([id2pred.get(cid, base_pred[i])
                             for i, cid in enumerate(test_ids)])
        return pred_arr

    # ── 加载数据 ──────────────────────────────────────────────────────────
    print(f"    加载数据 ({dataset}) ...")
    try:
        adata_full = ad.read_h5ad(config["dataset"]["path"])
        id2row     = {cid: i for i, cid in enumerate(adata_full.obs_names)}
        hvg_in_var = [g for g in hvg_genes if g in adata_full.var_names]
        adata_hvg  = adata_full[:, hvg_in_var]

        tr_rows     = [id2row[cid] for cid in train_ids if cid in id2row]
        te_rows     = [id2row[cid] for cid in test_ids  if cid in id2row]
        train_ids_ok = [cid for cid in train_ids if cid in id2row]
        test_ids_ok  = [cid for cid in test_ids  if cid in id2row]
        train_labels_ok = [train_labels[train_ids.index(cid)] for cid in train_ids_ok]

        X_train = _load_counts_direct(adata_hvg[tr_rows], config)
        X_test  = _load_counts_direct(adata_hvg[te_rows], config)
        var_names = hvg_in_var

    except Exception as e:
        print(f"    直接读取失败({e})，使用子进程提取...")
        (X_train, X_test, var_names,
         train_ids_ok, test_ids_ok, train_labels_ok) = _extract_via_subprocess(
            cfg_path, config, run_dir,
            train_ids, train_labels, test_ids, hvg_genes)

    print(f"    train: {len(train_ids_ok)} cells  test: {len(test_ids_ok)} cells  "
          f"genes: {len(var_names)}")

    # ── 构建 AnnData ──────────────────────────────────────────────────────
    X_train_norm = _lognorm(X_train)
    X_test_norm  = _lognorm(X_test)

    obs_train = pd.DataFrame({"celltype": train_labels_ok}, index=train_ids_ok)
    ref_adata = ad.AnnData(X=X_train_norm,
                           obs=obs_train,
                           var=pd.DataFrame(index=var_names))

    obs_test  = pd.DataFrame(index=test_ids_ok)
    qry_adata = ad.AnnData(X=X_test_norm,
                           obs=obs_test,
                           var=pd.DataFrame(index=var_names))

    # ── 训练 TOSICA ───────────────────────────────────────────────────────
    # 目录包含 rts_str，防止不同标注比例复用旧模型权重造成跨比例污染
    safe_rts      = rts_str.replace(".", "p")
    project       = f"tmp/tosica/{dataset}_seed{seed}_rts{safe_rts}"
    project_path  = Path(project)
    project_path.mkdir(parents=True, exist_ok=True)

    existing_weights = sorted(project_path.glob("model-*.pth"))
    if existing_weights:
        model_weight = str(existing_weights[-1])
        print(f"    使用缓存模型 {model_weight}")
    else:
        print(f"    训练 TOSICA (seed={seed}, epochs={TOSICA_EPOCHS}, max_gs={TOSICA_MAX_GS}) ...")
        TOSICA.train(ref_adata,
                     gmt_path=GMT_MAP[dataset],
                     project=project,
                     label_name="celltype",
                     epochs=TOSICA_EPOCHS,
                     max_gs=TOSICA_MAX_GS,
                     batch_size=16)
        existing_weights = sorted(project_path.glob("model-*.pth"))
        if not existing_weights:
            raise RuntimeError("TOSICA 训练未生成权重文件")
        # 保留最后一轮权重，删除中间 checkpoint 以节省磁盘空间
        model_weight = str(existing_weights[-1])
        for w in existing_weights[:-1]:
            w.unlink(missing_ok=True)
        print(f"    训练完成，保留权重: {Path(model_weight).name}")

    # ── 预测 ──────────────────────────────────────────────────────────────
    print(f"    预测 {len(test_ids_ok)} test cells ...")
    # laten=True：使用 latent 输出而非 attention，避免单细胞 batch 时 att 1D 的 IndexError
    result_adata = TOSICA.pre(qry_adata,
                              model_weight_path=model_weight,
                              project=project,
                              laten=True,
                              cutoff=0.05)

    tosica_pred_raw = result_adata.obs["Prediction"].astype(str).to_numpy()

    # "Unknown" → 保留 base_pred（scANVI 预测）
    id2tosica = dict(zip(test_ids_ok, tosica_pred_raw))
    pred = np.array([
        (id2tosica.get(cid, "Unknown") if id2tosica.get(cid, "Unknown") != "Unknown"
         else base_pred[i])
        for i, cid in enumerate(test_ids)
    ])

    n_unknown = int((tosica_pred_raw == "Unknown").sum())
    n_rare_true = int((test_df["true_label"].astype(str) == rare_class).sum())
    n_rare_pred = int((pred == rare_class).sum())
    print(f"    TOSICA Unknown: {n_unknown}/{len(test_ids_ok)}  "
          f"rare pred: {n_rare_pred}  true rare: {n_rare_true}")

    # 保存预测缓存
    pd.DataFrame({"cell_id": test_ids, "tosica_prediction": pred}).to_csv(
        cached_csv, index=False)

    # 清理模型目录和提取缓存（预测已保存 CSV，不再需要大文件）
    shutil.rmtree(project_path, ignore_errors=True)
    cache_dir = run_dir / "tosica_cache"
    shutil.rmtree(cache_dir, ignore_errors=True)

    return pred


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    # 只替换本次 RUNS 覆盖的 (dataset, seed, rts) 行，其余 TOSICA 行保留
    run_key_set = {
        (load_config(c)["dataset"]["name"], str(int(s)), str(r))
        for c, s, r in RUNS
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
        is_tosica = existing["method"] == "TOSICA"
        in_runs   = existing.apply(
            lambda row: (str(row["dataset"]), str(int(float(row["seed"]))), str(row["rare_train_size"])) in run_key_set,
            axis=1
        )
        existing = existing[~(is_tosica & in_runs)]
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
        status  = "ok"
        ts_pred = None
        try:
            ts_pred = run_tosica(cfg_path, config, run_dir, seed, rts_str)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  FAILED: {e}"); status = "failed"

        row = {"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
               "rare_class": rare_class, "method": "TOSICA", "status": status}

        if status == "ok":
            mres = compute_row_metrics(y_true, ts_pred, base_pred, rare_class)
            row.update(mres)
            print(f"  TOSICA: F1={mres['rare_f1']:.4f}  rec={mres['rare_recall']:.4f}  "
                  f"prec={mres['rare_precision']:.4f}  FP_rate={mres['rare_fp_rate']:.5f}")

        new_rows.append(row)

    # 合并保存
    full_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    col_order = ["dataset", "seed", "rare_train_size", "rare_class", "method", "status",
                 "sep", "best_k", "rare_f1", "rare_recall", "rare_precision", "macro_f1",
                 "rare_fp_rate", "n_rescued", "n_false_rescue", "rescue_ffr"]
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
                    "f1_mean":     round(float(f1s.mean()),  4),
                    "f1_std":      round(float(f1s.std()),   4),
                    "rec_mean":    round(float(recs.mean()), 4),
                    "rec_std":     round(float(recs.std()),  4),
                    "fp_rate_max": round(float(fps.max()), 6),
                })
                print(f"  {ds:25s} rts={rts:4s}  {method:15s}: "
                      f"F1={agg_rows[-1]['f1_mean']:.4f}±{agg_rows[-1]['f1_std']:.4f}  "
                      f"rec={agg_rows[-1]['rec_mean']:.4f}  (n={len(sub)})")

    pd.DataFrame(agg_rows).to_csv(AGG_CSV, index=False)
    print(f"[saved] {AGG_CSV}")

    print("\n重绘 comparison_bars ...")
    subprocess.run([SANDBOX_PYTHON, "tools/comparison/plot_comparison.py"], check=False)
    subprocess.run([SCANVI311_PYTHON, "tools/comparison/plot_comparison.py"], check=False)


if __name__ == "__main__":
    main()
