"""可控 separability 扫描（第十四轮，G21）—— 验证 CONFORMAL_LOW_SEP=1.3。

半合成纠缠：把稀有细胞朝**最近多数类**的随机配对细胞混 counts，纠缠参数 t↑ → sep↓。
对全体稀有细胞（train+val+test）在划分前统一施加 → "更难的数据集"，模型仍 inductive
（rescue 决策只用 train+val，不碰 test 标签）。每个 t 跑完整 pipeline，记录：
  - sep（train 原型可分性）
  - baseline scANVI F1
  - full（带 sep 闸门，= conformal_rescue）：F1 / gain / FFR / abstain / rank
  - nogate（关 sep 闸门 low_sep=0）：F1 / FFR  —— 展示低 sep 时若强行 rescue 的 FFR 代价

这是 **scRareRefine 自身 gate 的受控诊断，不是 benchmark 战绩**，故只比 scANVI + ours。
报告整条曲线（推翻 1.3 也照实）。

用法：D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/sep_sweep.py
（幂等：已算的 t 跳过；--force 重算）
"""
from __future__ import annotations

import sys
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, load_adata, classification_tables, log1p_cpm  # noqa: E402
from src.preprocess import run_preprocessing  # noqa: E402
from src.model import run_model_training  # noqa: E402
from src.rescue import PrototypeRescuer, conformal_rescue, DEFAULT_CONFORMAL_ALPHA  # noqa: E402
from tools.analysis.ablation import _conformal_with_overrides  # noqa: E402

CONFIG = "configs/tabula_lung_endo.yaml"
SEED = 42
RTS = 0.05                       # float 比例，固定（隔离 sep 轴）
T_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,   # 预先定死的 9 点
          0.9, 0.95]   # 补点（R4 记录）：首轮 sub-1.3 仅 1 点(sep=1.15)且 rescue 仍安全，
                       # 按第十四轮预案"补 1–2 个 t 到低 sep 区"探 rescue 崩塌边界（非 cherry-pick）
OUT_DIR = ROOT / "results" / "sep_sweep"
SUMMARY = OUT_DIR / "sep_sweep_summary.csv"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def entangle(adata, label_key: str, rare_class: str, t: float, seed: int):
    """把稀有细胞朝最近多数类的随机配对细胞混 counts：x'=round((1-t)x_rare + t x_majpair)。"""
    ad = adata.copy()
    labels = ad.obs[label_key].astype(str).to_numpy()
    X = ad.X
    is_sp = sp.issparse(X)
    rare_rows = np.where(labels == rare_class)[0]
    maj_classes = [c for c in sorted(set(labels)) if c != rare_class]

    def _norm_centroid(rows):
        sub = X[rows]
        dense = sub.toarray() if is_sp else np.asarray(sub, dtype=float)
        return log1p_cpm(dense).mean(0)

    rare_c = _norm_centroid(rare_rows)
    # 最近多数类（归一化质心欧氏距离）
    best_c, best_d = None, np.inf
    for c in maj_classes:
        d = float(np.sqrt(((_norm_centroid(np.where(labels == c)[0]) - rare_c) ** 2).sum()))
        if d < best_d:
            best_d, best_c = d, c
    maj_rows = np.where(labels == best_c)[0]

    if t > 0:
        rng = np.random.default_rng(seed)
        pair = rng.choice(maj_rows, size=len(rare_rows), replace=True)
        Xr = X[rare_rows].toarray() if is_sp else np.asarray(X[rare_rows], dtype=float)
        Xm = X[pair].toarray() if is_sp else np.asarray(X[pair], dtype=float)
        mixed = np.rint((1.0 - t) * Xr + t * Xm)
        if is_sp:
            Xl = X.tolil()
            for k, i in enumerate(rare_rows):
                Xl.rows[i] = list(np.nonzero(mixed[k])[0])
                Xl.data[i] = list(mixed[k][mixed[k] != 0])
            ad.X = Xl.tocsr()
        else:
            Xnew = np.asarray(X, dtype=float).copy()
            Xnew[rare_rows] = mixed
            ad.X = Xnew
    return ad, best_c, int(len(rare_rows))


def _lat(df):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def run_one(t: float, base_adata, config, label_key, batch_key, split_mode, rare_class):
    ad_ent, maj_c, n_rare = entangle(base_adata, label_key, rare_class, t, SEED)
    adata, tr, va, te = run_preprocessing(ad_ent, label_column=label_key, batch_key=batch_key,
                                          split_mode=split_mode, seed=SEED, rare_class=rare_class)
    _, preds, lats, _ = run_model_training(adata, tr, va, te, label_column=label_key, batch_key=batch_key,
                                           rare_class=rare_class, rare_train_size=RTS, config=config, seed=SEED)
    train_lat, val_lat, test_lat = _lat(lats["train"]), _lat(lats["validation"]), _lat(lats["test"])
    ref = preds["train"]["true_label"].astype(str)
    is_lab = preds["train"]["is_labeled_for_scanvi"].astype(bool).to_numpy()
    proto = PrototypeRescuer(rare_class)
    proto.fit(train_lat, ref, is_lab)

    y = preds["test"]["true_label"].astype(str).to_numpy()
    base = preds["test"]["predicted_label"].astype(str)
    vbase = preds["validation"]["predicted_label"].astype(str)
    vtrue = preds["validation"]["true_label"].astype(str)
    n_nonrare = int((y != rare_class).sum())

    def _m(pred):
        m, _ = classification_tables(y, pred.astype(str).to_numpy(), rare_class=rare_class)
        nf = int(((pred.astype(str).to_numpy() != base.to_numpy()) & (pred.astype(str).to_numpy() == rare_class) & (y != rare_class)).sum())
        return round(m["rare_f1"], 4), round(nf / max(n_nonrare, 1), 6)

    bl_f1, _ = _m(base)
    full_pred, summ = conformal_rescue(proto, base, vbase, vtrue, val_lat, test_lat)
    full_f1, full_ffr = _m(full_pred)
    nogate_pred, _ = _conformal_with_overrides(proto, base, vbase, vtrue, val_lat, test_lat, low_sep=0.0)
    nogate_f1, nogate_ffr = _m(nogate_pred)

    n_lab_rare = int((is_lab & (ref.to_numpy() == rare_class)).sum())
    return {
        "t": t, "sep": round(proto.separability_ratio, 4), "nearest_majority": maj_c,
        "n_rare_total": n_rare, "n_labeled_rare": n_lab_rare,
        "baseline_f1": bl_f1,
        "full_f1": full_f1, "full_gain": round(full_f1 - bl_f1, 4), "full_ffr": full_ffr,
        "full_abstain": bool(summ.get("abstain", False)), "full_reason": summ.get("reason", ""),
        "full_chosen_rank": int(summ.get("chosen_rank", 0)),
        "nogate_f1": nogate_f1, "nogate_gain": round(nogate_f1 - bl_f1, 4), "nogate_ffr": nogate_ffr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config = load_config(CONFIG)
    label_key = config["dataset"]["label_key"]
    batch_key = config["dataset"]["batch_key"]
    rare_class = config["experiment"]["rare_class"]
    split_mode = config["experiment"].get("split_mode", "batch_heldout")
    base_sha = _sha(ROOT / config["dataset"]["path"])
    print(f"[provenance] base={config['dataset']['name']} sha={base_sha} rare='{rare_class}' seed={SEED} rts={RTS}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {}
    if SUMMARY.exists() and not args.force:
        ex = pd.read_csv(SUMMARY)
        done = {round(float(r["t"]), 3): r for _, r in ex.iterrows()}

    base_adata = load_adata(config)
    rows = []
    for t in T_GRID:
        if round(t, 3) in done:
            print(f"[skip] t={t}（已算 sep={done[round(t,3)]['sep']}）")
            rows.append(done[round(t, 3)].to_dict())
            continue
        print(f"\n===== t={t} 纠缠+训练+rescue =====")
        res = run_one(t, base_adata, config, label_key, batch_key, split_mode, rare_class)
        res["base_sha"] = base_sha
        print(f"  sep={res['sep']:.3f}  base_f1={res['baseline_f1']:.3f}  "
              f"full_f1={res['full_f1']:.3f}(gain{res['full_gain']:+.3f},ffr{res['full_ffr']:.4f},"
              f"{'弃权:'+res['full_reason'] if res['full_abstain'] else 'rank'+str(res['full_chosen_rank'])})  "
              f"nogate_f1={res['nogate_f1']:.3f}(ffr{res['nogate_ffr']:.4f})")
        rows.append(res)
        pd.DataFrame(rows).to_csv(SUMMARY, index=False)   # 增量落盘（可续跑）

    pd.DataFrame(rows).to_csv(SUMMARY, index=False)
    print(f"\n[saved] {SUMMARY}")


if __name__ == "__main__":
    main()
