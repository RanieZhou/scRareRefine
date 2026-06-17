"""scRareRefine 对比实验 — 在 scanvi311 环境中运行。

scANVI + conformal prototype rescue（V4 full）：当原型可分性
（separability_ratio）足够时，对低 margin 候选 cell 做 conformal 校准后的改判。

运行：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/comparison/run_scrarerefine_comparison.py
"""
from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables
from src.rescue import PrototypeRescuer, ConformalRescuer, DEFAULT_CONFORMAL_ALPHA, CONFORMAL_LOW_SEP
from tools.comparison._conda_python import conda_python

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

METHOD_NAME       = "scRareRefine"
# alpha / LOW_SEP 从 src.rescue 导入（单一来源），与 run_pipeline.py 生成的主方法 metrics
# 保持同一 conformal 设置，避免本对比脚本与主流水线各自硬编码导致 scRareRefine 数值不可比。
CONFORMAL_ALPHA   = DEFAULT_CONFORMAL_ALPHA
ALL_METHODS = ["scANVI", "kNN", "CellTypist", "scBalance",
               "ProtoCloud", "HiCat", "scCAD", "TOSICA", "scRareRefine"]

OUT_DIR          = Path("results/comparison")
SUMMARY_CSV      = OUT_DIR / "comparison_summary.csv"
AGG_CSV          = OUT_DIR / "comparison_summary_agg.csv"
SCANVI311_PYTHON = conda_python("scanvi311")


def _lat(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _conformal_rescue(proto, base_pred, val_lat, val_true, test_lat):
    if proto.separability_ratio < CONFORMAL_LOW_SEP:
        return base_pred.copy()
    test_cand  = proto.isotropic_rank1(test_lat, base_pred)
    test_score = proto.rare_membership_score(test_lat)
    val_score  = proto.rare_membership_score(val_lat)
    conf = ConformalRescuer(proto.rare_class, alpha=CONFORMAL_ALPHA)
    conf.calibrate(val_score, val_true)
    return conf.relabel(base_pred, test_cand, test_score)


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


def main():
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

        train_lat  = _lat(pd.read_csv(emb_dir / "train_latent.csv"))
        train_pred = pd.read_csv(emb_dir / "train_predictions.csv")
        is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
        ref_labels = train_pred["true_label"].astype(str)
        lab_labels = ref_labels[is_labeled].to_numpy()

        proto = PrototypeRescuer(rare_class)
        proto.fit(train_lat, ref_labels, is_labeled)

        val_pred  = pd.read_csv(emb_dir / "validation_predictions.csv")
        val_lat   = _lat(pd.read_csv(emb_dir / "validation_latent.csv"))
        val_true  = val_pred["true_label"].astype(str)

        test_pred = pd.read_csv(emb_dir / "test_predictions.csv")
        test_lat  = _lat(pd.read_csv(emb_dir / "test_latent.csv"))
        y_true    = test_pred["true_label"].astype(str).to_numpy()
        base_pred = test_pred["predicted_label"].astype(str)

        print(f"\n[{dataset} seed={seed} rts={rts_str}] sep={proto.separability_ratio:.3f}  "
              f"lab_rare={int((lab_labels==rare_class).sum())}  test_rare={int((y_true==rare_class).sum())}")

        srr_pred = _conformal_rescue(proto, base_pred, val_lat, val_true, test_lat)

        mres = _metrics(y_true, srr_pred.to_numpy(), base_pred.to_numpy(), rare_class)
        print(f"  {METHOD_NAME:15s}: F1={mres['rare_f1']:.4f}  rec={mres['rare_recall']:.4f}  "
              f"prec={mres['rare_precision']:.4f}  FP_rate={mres['rare_fp_rate']:.5f}")

        new_rows.append({
            "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
            "rare_class": rare_class, "method": METHOD_NAME, "status": "ok",
            "sep": round(proto.separability_ratio, 4), "best_k": None,
            **mres,
        })

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
    main()
