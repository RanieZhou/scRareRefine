"""配对显著性检验（G02-A-statest）：scRareRefine vs 各 baseline。

数据：results/comparison/comparison_summary.csv（9 方法 × 6 数据集 × 4 rts × 3 seed）。
配对单元：(dataset, rare_train_size, seed) cell —— scRareRefine 与某 baseline 同 cell 配对。
  - 全集 pairs：6×4×3 = 72
  - 稀缺区 pairs（rts ∈ {0.01,0.05,0.10}）：6×3×3 = 54

检验：
  - paired Wilcoxon signed-rank（单侧 H1: scRareRefine > baseline），对 rare_f1 之差；
    零差按 scipy 默认 (wilcox) 处理（zero_method='wilcox' 丢弃零差）。
  - mean ΔF1 的 bootstrap 95% CI（10000 次重采样，配对层面重采样 cell）。
  - 报告 win / tie / loss 计数与 median ΔF1。

诚实性注记（写进论文）：72/54 个 cell 并非完全独立——同一 (dataset,rts) 的 3 seed 相关，
且小数据集多个 rts 因 max(5,·) 标注塌缩而近似重复（见 scarce_region_distinct.csv）。
故 p 值偏乐观，应作为「方向性证据」而非严格独立样本检验；HiCat 为 transductive 上界，单列。

用法：D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/significance_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_CSV = ROOT / "results" / "comparison" / "significance_test.csv"
OUR = "scRareRefine"
BASELINES = ["scANVI", "kNN", "CellTypist", "scBalance", "ProtoCloud", "HiCat", "scCAD", "TOSICA"]
TRANSDUCTIVE = {"HiCat"}
SCARCE = ["0.01", "0.05", "0.10"]
KEY = ["dataset", "rare_train_size", "seed"]


def _paired(df: pd.DataFrame, base: str) -> pd.DataFrame:
    a = df[df.method == OUR].set_index(KEY)["rare_f1"]
    b = df[df.method == base].set_index(KEY)["rare_f1"]
    j = pd.concat([a.rename("our"), b.rename("base")], axis=1).dropna()
    return j


def _boot_ci(delta: np.ndarray, n=10000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if len(delta) == 0:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(delta), size=(n, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _test_block(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    print(f"\n========== {label} ==========")
    print(f"{'baseline':12s} {'n':>3s} {'win':>3s} {'tie':>3s} {'los':>3s} "
          f"{'medianΔ':>8s} {'meanΔ':>7s} {'boot95%CI':>18s} {'Wilcoxon p(1-sided)':>20s}")
    for base in BASELINES:
        j = _paired(df, base)
        d = (j["our"] - j["base"]).to_numpy()
        n = len(d)
        win = int((d > 1e-9).sum()); tie = int((np.abs(d) <= 1e-9).sum()); los = int((d < -1e-9).sum())
        med = float(np.median(d)) if n else np.nan
        mean = float(d.mean()) if n else np.nan
        lo, hi = _boot_ci(d)
        # Wilcoxon 单侧（greater）；全零差或无非零差时不可计算
        if n and np.any(np.abs(d) > 1e-12):
            try:
                p = float(stats.wilcoxon(d, alternative="greater", zero_method="wilcox").pvalue)
            except ValueError:
                p = np.nan
        else:
            p = np.nan
        tag = " (transductive)" if base in TRANSDUCTIVE else ""
        print(f"{base:12s} {n:3d} {win:3d} {tie:3d} {los:3d} "
              f"{med:+8.3f} {mean:+7.3f} [{lo:+.3f},{hi:+.3f}] {p:20.2e}{tag}")
        rows.append({"region": label, "baseline": base, "transductive": base in TRANSDUCTIVE,
                     "n_pairs": n, "win": win, "tie": tie, "loss": los,
                     "median_delta": round(med, 4), "mean_delta": round(mean, 4),
                     "boot_ci_lo": round(lo, 4), "boot_ci_hi": round(hi, 4),
                     "wilcoxon_p_greater": p})
    return rows


def main():
    df = pd.read_csv(ROOT / "results" / "comparison" / "comparison_summary.csv", dtype={"rare_train_size": str})
    df = df[df.status == "ok"].copy()
    seeds = sorted(df.seed.dropna().astype(int).unique().tolist())
    print(f"seeds={seeds}  方法={sorted(df.method.unique())}")

    rows = []
    rows += _test_block(df, "ALL rts (6ds×4rts×3seed)")
    rows += _test_block(df[df.rare_train_size.isin(SCARCE)], "SCARCE rts<=0.10 (6ds×3rts×3seed)")

    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\n[saved] {OUT_CSV}")
    print("\n注：cell 非完全独立（3 seed 相关 + 标注塌缩近似重复），p 值偏乐观，作方向性证据；"
          "HiCat 为 transductive 上界，单列。")


if __name__ == "__main__":
    main()
