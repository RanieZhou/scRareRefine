"""全 benchmark CONFORMAL_LOW_SEP 敏感性（第十四轮 G82，codex Round 3 #2）。

在真实 6 数据集 × 4 rts × 3 seed = 648/... 缓存嵌入上，把 sep 闸门阈值 low_sep
扫 {0, 0.7, 1.0, 1.3, 1.6}（1.3=当前默认），其余组件不变（necessity / 自适应 rank / τ）。
cache-only，复用 _conformal_with_overrides，无重训。

目的（回应 codex）：证明 1.3 的 F1/FFR tradeoff 在**真实 benchmark**上不是单点碰巧——
看降低 low_sep 是否在真实数据上引入 FFR 越界或回归、抬高是否白白多弃权丢 F1。

输出：results/sep_sweep/lowsep_sensitivity{,_agg}.csv + lowsep_sensitivity.png

用法：D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/lowsep_sensitivity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables  # noqa: E402
from src.rescue import PrototypeRescuer  # noqa: E402
from tools.analysis.ablation import _conformal_with_overrides  # noqa: E402

CONFIGS = [
    "configs/immune_dc.yaml", "configs/pancreas_baron.yaml", "configs/tabula_lung_endo.yaml",
    "configs/tabula_small_intestine.yaml", "configs/tabula_sapiens_stomach.yaml", "configs/pancreas_integrated.yaml",
]
SEEDS = [42, 43, 44]
RTS = ["0.01", "0.05", "0.10", "all"]
LOW_SEP_GRID = [0.0, 0.7, 1.0, 1.3, 1.6]   # 1.3 = 当前默认
SCARCE = ["0.01", "0.05", "0.10"]
OUT = ROOT / "results" / "sep_sweep"


def _lat(df):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def main():
    rows = []
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path); exp = cfg.get("experiment", {})
        rare = exp["rare_class"]; sm = exp.get("split_mode", "batch_heldout"); ds = cfg["dataset"]["name"]
        for seed in SEEDS:
            for rts in RTS:
                rd = make_run_dir(cfg, sm, seed, rare, parse_rare_train_size(rts))
                emb = rd / "embeddings"
                if not (emb / "test_latent.csv").exists():
                    continue
                tr_p = pd.read_csv(emb / "train_predictions.csv"); tr_l = _lat(pd.read_csv(emb / "train_latent.csv"))
                va_p = pd.read_csv(emb / "validation_predictions.csv"); va_l = _lat(pd.read_csv(emb / "validation_latent.csv"))
                te_p = pd.read_csv(emb / "test_predictions.csv"); te_l = _lat(pd.read_csv(emb / "test_latent.csv"))
                is_lab = tr_p["is_labeled_for_scanvi"].astype(bool).to_numpy()
                proto = PrototypeRescuer(rare); proto.fit(tr_l, tr_p["true_label"].astype(str), is_lab)
                y = te_p["true_label"].astype(str).to_numpy()
                base = te_p["predicted_label"].astype(str)
                vbase = va_p["predicted_label"].astype(str); vtrue = va_p["true_label"].astype(str)
                n_nr = int((y != rare).sum())
                bl, _ = classification_tables(y, base.to_numpy(), rare_class=rare)
                for ls in LOW_SEP_GRID:
                    final, summ = _conformal_with_overrides(proto, base, vbase, vtrue, va_l, te_l, low_sep=ls)
                    fp = final.astype(str).to_numpy()
                    nf = int(((fp != base.to_numpy()) & (fp == rare) & (y != rare)).sum())
                    m, _ = classification_tables(y, fp, rare_class=rare)
                    rows.append({
                        "low_sep": ls, "dataset": ds, "seed": seed, "rts": rts,
                        "sep": round(proto.separability_ratio, 4),
                        "baseline_f1": round(bl["rare_f1"], 4), "f1": round(m["rare_f1"], 4),
                        "gain": round(m["rare_f1"] - bl["rare_f1"], 4),
                        "ffr": round(nf / max(n_nr, 1), 6), "n_false": nf,
                        "abstain": bool(summ.get("abstain", False)),
                    })
        print(f"[done] {ds}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "lowsep_sensitivity.csv", index=False)

    # 聚合：每 low_sep 在全集 / 稀缺区的 mean F1、max FFR、abstain 数；以及 vs 默认 1.3 的 ΔF1
    def _agg(sub, region):
        out = []
        for ls in LOW_SEP_GRID:
            s = sub[sub.low_sep == ls]
            out.append({"region": region, "low_sep": ls, "n": len(s),
                        "f1_mean": round(s.f1.mean(), 4), "ffr_max": round(s.ffr.max(), 6),
                        "n_abstain": int(s.abstain.sum()), "n_ffr_over_alpha": int((s.ffr > 0.01).sum())})
        return out
    agg = _agg(df, "ALL") + _agg(df[df.rts.isin(SCARCE)], "SCARCE")
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(OUT / "lowsep_sensitivity_agg.csv", index=False)

    print("\n=== low_sep 敏感性（ALL 648-cell 口径）===")
    print(aggdf[aggdf.region == "ALL"][["low_sep", "f1_mean", "ffr_max", "n_abstain", "n_ffr_over_alpha"]].to_string(index=False))
    print("\n=== SCARCE 口径 ===")
    print(aggdf[aggdf.region == "SCARCE"][["low_sep", "f1_mean", "ffr_max", "n_abstain", "n_ffr_over_alpha"]].to_string(index=False))

    # 哪些 (dataset,rts,seed) 在 low_sep 降低后从弃权转 rescue 且变差/FFR 越界
    print("\n=== 降低 low_sep 后受影响的 cell（默认1.3弃权 → 更低阈值改判）===")
    piv = df.pivot_table(index=["dataset", "rts", "seed", "sep"], columns="low_sep", values=["f1", "ffr"])
    for (ds, rts, seed, sep), r in piv.iterrows():
        f13 = r[("f1", 1.3)]; f07 = r[("f1", 0.7)]
        if abs(f13 - f07) > 1e-9 or r[("ffr", 0.7)] > 0.01:
            print(f"  {ds} rts={rts} seed={seed} sep={sep}: F1 @1.3={f13:.3f} @1.0={r[('f1',1.0)]:.3f} @0.7={f07:.3f} | "
                  f"FFR @0.7={r[('ffr',0.7)]:.4f}")

    print(f"\n[saved] {OUT/'lowsep_sensitivity.csv'}  {OUT/'lowsep_sensitivity_agg.csv'}")


if __name__ == "__main__":
    main()
