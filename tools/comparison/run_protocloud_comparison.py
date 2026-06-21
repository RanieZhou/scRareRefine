"""ProtoCloud 对比实验 — 在 sandbox310 环境中运行。

独立完成全部工作：训练 ProtoCloud、预测、计算指标、
append 进已有 comparison_summary.csv、重聚合、重绘图。
不重跑 scANVI / kNN / CellTypist / scBalance / scRareRefine。

运行：
  D:/setup/anaconda/envs/sandbox310/python.exe tools/comparison/run_protocloud_comparison.py
"""
from __future__ import annotations

import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
import scipy.sparse as sp
import yaml
import ProtoCloud as pc
from sklearn.metrics import precision_recall_fscore_support

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

PROTOCLOUD_EPOCHS = 100
ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance", "ProtoCloud", "scRareRefine"]

OUT_DIR     = Path("results/comparison")
SUMMARY_CSV = OUT_DIR / "comparison_summary.csv"
AGG_CSV     = OUT_DIR / "comparison_summary_agg.csv"

# 重绘图用 scanvi311（plot_comparison.py 依赖 matplotlib Agg backend，两个环境都有）
SCANVI311_PYTHON = conda_python("scanvi311")


# ── 工具函数（不依赖 src.utils）──────────────────────────────────────────────

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
    macro_f1 = f1.mean()
    rare_f1  = float(f1[idx])  if idx >= 0 else 0.0
    rare_rec = float(rec[idx]) if idx >= 0 else 0.0
    rare_pre = float(prec[idx]) if idx >= 0 else 0.0
    return {"rare_f1": rare_f1, "rare_recall": rare_rec, "rare_precision": rare_pre,
            "macro_f1": float(macro_f1)}


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


# ── ProtoCloud 训练 + 预测 ────────────────────────────────────────────────────

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


# sandbox310 (Python 3.10) 的 anndata 0.11 无法读取某些旧格式 h5ad（encoding_type='null'）。
# 对此类文件，通过 scanvi311（anndata 0.12）提取 count 矩阵并缓存为 .npz，再在本环境加载。
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
    return np.asarray(X, dtype=np.float32), ok

train_X, train_ok = get_X(lab_ids)
test_X,  test_ok  = get_X(test_ids)
np.savez(train_npz, X=train_X, ids=np.array(train_ok), genes=np.array(hvg_in))
np.savez(test_npz,  X=test_X,  ids=np.array(test_ok),  genes=np.array(hvg_in))
print(f"extracted train{train_X.shape} test{test_X.shape} genes={len(hvg_in)}")
"""


def _extract_counts_via_subprocess(
    config: dict, hvg_genes: list, labeled_ids: list, test_ids: list, run_dir: Path
) -> tuple:
    """通过 scanvi311 python 提取 count 矩阵，缓存为 .npz。"""
    import json as _json
    import tempfile
    import os

    cache_dir = run_dir / "protocloud_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_npz = str(cache_dir / "train.npz")
    test_npz  = str(cache_dir / "test.npz")

    if not (Path(train_npz).exists() and Path(test_npz).exists()):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(_EXTRACT_SCRIPT)
            script_path = f.name
        args = {
            "path":       config["dataset"]["path"],
            "hvg":        hvg_genes,
            "labeled_ids": labeled_ids,
            "test_ids":   test_ids,
            "use_raw":    config["dataset"].get("use_raw", False),
            "train_npz":  train_npz,
            "test_npz":   test_npz,
        }
        try:
            proc = subprocess.run(
                [SCANVI311_PYTHON, script_path],
                input=_json.dumps(args), capture_output=True, text=True, check=True,
            )
            print(f"    scanvi311 提取: {proc.stdout.strip()}")
            if proc.stderr:
                print(f"    (stderr) {proc.stderr[:300]}")
        finally:
            os.unlink(script_path)

    train_d = np.load(train_npz, allow_pickle=True)
    test_d  = np.load(test_npz,  allow_pickle=True)
    return (train_d["X"], train_d["ids"].tolist(),
            test_d["X"],  test_d["ids"].tolist(),
            test_d["genes"].tolist())


def run_protocloud(config: dict, run_dir: Path, seed: int) -> np.ndarray:
    emb_dir = run_dir / "embeddings"

    # 读 split 信息
    train_df = pd.read_csv(emb_dir / "train_predictions.csv", dtype={"cell_id": str})
    test_df  = pd.read_csv(emb_dir / "test_predictions.csv",  dtype={"cell_id": str})

    # 若已有预测缓存，直接加载（跳过重新训练）
    cached_csv = emb_dir / "protocloud_test_predictions.csv"
    if cached_csv.exists():
        print(f"    使用缓存预测 {cached_csv.name}")
        cached = pd.read_csv(cached_csv, dtype=str)
        id2pred = dict(zip(cached["cell_id"], cached["pc_prediction"]))
        test_ids_all = test_df["cell_id"].tolist()
        return np.array([id2pred.get(cid, "unknown") for cid in test_ids_all])
    hvg_genes = pd.read_csv(run_dir / "selected_hvg_genes.csv")["gene"].tolist()

    is_labeled   = train_df["is_labeled_for_scanvi"].astype(bool).to_numpy()
    labeled_ids  = train_df["cell_id"][is_labeled].tolist()
    labeled_lbls = train_df["true_label"][is_labeled].astype(str).tolist()
    test_ids     = test_df["cell_id"].tolist()

    # 加载 h5ad（对旧格式文件提供 scanvi311 fallback）
    print(f"    加载 {config['dataset']['path']} ...")
    hvg_in_var: list[str]
    try:
        adata_full = anndata.read_h5ad(config["dataset"]["path"])
        use_raw    = config["dataset"].get("use_raw", False)
        id2row     = {cid: i for i, cid in enumerate(adata_full.obs_names)}
        hvg_in_var = [g for g in hvg_genes if g in adata_full.var_names]
        adata_hvg  = adata_full[:, hvg_in_var]
        print(f"    HVG: {len(hvg_in_var)}/{len(hvg_genes)}  train_labeled: {len(labeled_ids)}  test: {len(test_ids)}")

        def _get(cell_ids: list) -> tuple:
            rows = [id2row[cid] for cid in cell_ids if cid in id2row]
            return _load_counts(adata_hvg[rows], use_raw), [cid for cid in cell_ids if cid in id2row]

        train_X, train_ids_ok = _get(labeled_ids)
        test_X,  test_ids_ok  = _get(test_ids)

    except Exception as read_err:
        if "IOSpec" in str(read_err) or "encoding_type" in str(read_err) or "null" in str(read_err).lower():
            print(f"    h5ad 格式不兼容 ({read_err.__class__.__name__})，通过 scanvi311 提取数据...")
            train_X, train_ids_ok, test_X, test_ids_ok, hvg_in_var = \
                _extract_counts_via_subprocess(config, hvg_genes, labeled_ids, test_ids, run_dir)
            print(f"    HVG: {len(hvg_in_var)}/{len(hvg_genes)}  train_labeled: {len(train_ids_ok)}  test: {len(test_ids_ok)}")
        else:
            raise

    id2lbl = dict(zip(labeled_ids, labeled_lbls))
    train_labels = [id2lbl[cid] for cid in train_ids_ok]

    # 构建 AnnData（ProtoCloud 需要 var['gene_name'] 和 layers['counts']）
    def _make_adata(X, obs_df, genes):
        var = pd.DataFrame({"gene_name": genes}, index=genes)
        ad  = anndata.AnnData(X=X, obs=obs_df.copy(), var=var)
        ad.layers["counts"] = X
        return ad

    train_obs = pd.DataFrame({"celltype": train_labels}, index=train_ids_ok)
    test_obs  = pd.DataFrame(index=test_ids_ok)
    train_ad  = _make_adata(train_X, train_obs, hvg_in_var)
    test_ad   = _make_adata(test_X,  test_obs,  hvg_in_var)

    # 训练
    print(f"    训练 ProtoCloud (epochs={PROTOCLOUD_EPOCHS}, seed={seed}) ...")
    model = pc.ProtoCloudModel(latent_dim=30, num_prototypes_per_class=6)
    model.fit_model(train_ad, celltype_col="celltype",
                    test_ratio=0.0, data_balance=True,
                    seed=seed, epochs=PROTOCLOUD_EPOCHS, validate=False)

    # 预测
    model.predict_model(test_ad)
    pc_pred = test_ad.obs["pc_prediction"].astype(str).to_numpy()

    # 保存预测 CSV（供复查）
    out_csv = emb_dir / "protocloud_test_predictions.csv"
    pd.DataFrame({"cell_id": test_ids_ok, "pc_prediction": pc_pred}).to_csv(out_csv, index=False)

    return pc_pred


# ── 主逻辑 ────────────────────────────────────────────────────────────────────

def main():
    # 只替换本次 RUNS 覆盖的 (dataset, seed, rts) 行，其余 ProtoCloud 行保留
    run_key_set = {
        (load_config(c)["dataset"]["name"], str(int(s)), str(r))
        for c, s, r in RUNS
    }
    if SUMMARY_CSV.exists():
        existing = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
        is_pc   = existing["method"] == "ProtoCloud"
        in_runs = existing.apply(
            lambda row: (str(row["dataset"]), str(int(float(row["seed"]))), str(row["rare_train_size"])) in run_key_set,
            axis=1
        )
        existing = existing[~(is_pc & in_runs)]
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
        pc_pred = None
        try:
            pc_pred = run_protocloud(config, run_dir, seed)
        except Exception as e:
            print(f"  FAILED: {e}")
            status = "failed"

        row = {"dataset": dataset, "seed": seed, "rare_train_size": rts_str,
               "rare_class": rare_class, "method": "ProtoCloud", "status": status}

        if status == "ok":
            mres = compute_row_metrics(y_true, pc_pred, base_pred, rare_class)
            row.update(mres)
            print(f"  ProtoCloud: F1={mres['rare_f1']:.4f}  "
                  f"rec={mres['rare_recall']:.4f}  prec={mres['rare_precision']:.4f}  "
                  f"FP_rate={mres['rare_fp_rate']:.5f}")
        else:
            print("  ProtoCloud: FAILED（不计入聚合）")

        new_rows.append(row)

    # 合并并保存 summary
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
                    "f1_mean":  round(f1s.mean(),  4), "f1_std":  round(f1s.std(),  4),
                    "rec_mean": round(recs.mean(), 4), "rec_std": round(recs.std(), 4),
                    "fp_rate_max": round(fps.max(), 6),
                })
                print(f"  {ds:25s} rts={rts:4s}  {method:15s}: "
                      f"F1={agg_rows[-1]['f1_mean']:.4f}±{agg_rows[-1]['f1_std']:.4f}  "
                      f"rec={agg_rows[-1]['rec_mean']:.4f}  (n={len(sub)})")

    pd.DataFrame(agg_rows).to_csv(AGG_CSV, index=False)
    print(f"[saved] {AGG_CSV}")

    # 重绘柱状图（plot_comparison.py 在两个环境都能跑，用 scanvi311 保持一致）
    print("\n重绘 comparison_bars ...")
    subprocess.run([SCANVI311_PYTHON, "tools/comparison/plot_comparison.py"], check=True)


if __name__ == "__main__":
    main()
