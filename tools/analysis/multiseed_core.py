"""核心三方法多 seed 聚合（G01 第十三轮 Phase 2）：scANVI / kNN / scRareRefine。

cache-only（读 outputs/.../embeddings），对 seed∈{42,43,44} × 6 数据集 × 4 rts 计算
三方法 rare 指标，做 3-seed mean±std 聚合，并自动判定第十三轮验收线 (b)(c)。

不重训、不碰 test 标签调参（kNN 的 k、scRareRefine 的 rank/τ 均在 val 选）；
不改任何既有 seed=42 产物，只写 results/multiseed/。

前置：先用 tools/analysis/gen_multiseed_cache.py 生成 seed 43/44 嵌入。
缺某 seed 缓存的配置自动跳过并在日志注明（不静默假装 3 seed）。

用法：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/multiseed_core.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import (
    load_config,
    make_run_dir,
    parse_rare_train_size,
    classification_tables,
)  # noqa: E402
from src.rescue import PrototypeRescuer, conformal_rescue  # noqa: E402

CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/pancreas_integrated.yaml",
]
SEEDS = [42, 43, 44]
RTS = ["0.01", "0.05", "0.10", "all"]
SCARCE = ["0.01", "0.05", "0.10"]
KNN_K_GRID = [3, 5, 10, 15]
TESTBEDS = ["immune_dc", "pancreas_baron", "tabula_sapiens_stomach"]  # 验收线 (c)

OUT = ROOT / "results" / "multiseed"


def _lat(df):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _knn(train_lat, train_lbl, q_lat, k):
    tr = train_lat.astype(np.float32)
    n = len(train_lbl)
    k_eff = min(k, n)
    kth = min(k_eff, n - 1)
    out = []
    for i in range(0, len(q_lat), 100):
        q = q_lat[i : i + 100].astype(np.float32)
        d2 = np.sum((tr[None] - q[:, None]) ** 2, axis=2)
        nn = np.argpartition(d2, kth, axis=1)[:, :k_eff]
        for j in range(len(q)):
            v, c = np.unique(train_lbl[nn[j]], return_counts=True)
            out.append(v[c.argmax()])
    return np.array(out)


def _row(y_true, pred, base, rare):
    m, _ = classification_tables(y_true, pred, rare_class=rare)
    n_nr = int((y_true != rare).sum())
    n_false = int(((pred != base) & (pred == rare) & (y_true != rare)).sum())
    n_fp = int(((pred == rare) & (y_true != rare)).sum())
    incremental_fpr = round(n_false / max(n_nr, 1), 6)
    return {
        "rare_f1": round(m["rare_f1"], 4),
        "rare_recall": round(m["rare_recall"], 4),
        "rare_precision": round(m["rare_precision"], 4),
        "rare_fp_rate": round(n_fp / max(n_nr, 1), 6),
        "incremental_fpr": incremental_fpr,
        "rescue_ffr": incremental_fpr,
    }


def main():
    rows = []
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        exp = cfg.get("experiment", {})
        rare = exp.get("rare_class")
        sm = exp.get("split_mode", "batch_heldout")
        ds = cfg["dataset"]["name"]
        for seed in SEEDS:
            for rts in RTS:
                rd = make_run_dir(cfg, sm, seed, rare, parse_rare_train_size(rts))
                emb = rd / "embeddings"
                if not (emb / "test_latent.csv").exists():
                    print(f"[skip] {ds} seed={seed} rts={rts}（无缓存）")
                    continue
                tr_p = pd.read_csv(emb / "train_predictions.csv")
                tr_l = _lat(pd.read_csv(emb / "train_latent.csv"))
                va_p = pd.read_csv(emb / "validation_predictions.csv")
                va_l = _lat(pd.read_csv(emb / "validation_latent.csv"))
                te_p = pd.read_csv(emb / "test_predictions.csv")
                te_l = _lat(pd.read_csv(emb / "test_latent.csv"))

                is_lab = tr_p["is_labeled_for_scanvi"].astype(bool).to_numpy()
                ref = tr_p["true_label"].astype(str)
                proto = PrototypeRescuer(rare)
                proto.fit(tr_l, ref, is_lab)

                y = te_p["true_label"].astype(str).to_numpy()
                base = te_p["predicted_label"].astype(str)
                vt = va_p["true_label"].astype(str)

                # scANVI
                rows.append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "rare_train_size": rts,
                        "method": "scANVI",
                        "sep": round(proto.separability_ratio, 4),
                        **_row(y, base.to_numpy(), base.to_numpy(), rare),
                    }
                )
                # kNN（val grid-search k）
                lab_l, lab_y = tr_l[is_lab], ref[is_lab].to_numpy()
                vy = vt.to_numpy()
                bk, bf = 15, -1.0
                for k in KNN_K_GRID:
                    vp = _knn(lab_l, lab_y, va_l, k)
                    mm, _ = classification_tables(vy, vp, rare_class=rare)
                    if mm["rare_f1"] > bf:
                        bf, bk = mm["rare_f1"], k
                kp = _knn(lab_l, lab_y, te_l, bk)
                rows.append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "rare_train_size": rts,
                        "method": "kNN",
                        "sep": round(proto.separability_ratio, 4),
                        **_row(y, kp, base.to_numpy(), rare),
                    }
                )
                # scRareRefine
                srr, _ = conformal_rescue(
                    proto, base, va_p["predicted_label"].astype(str), vt, va_l, te_l
                )
                rows.append(
                    {
                        "dataset": ds,
                        "seed": seed,
                        "rare_train_size": rts,
                        "method": "scRareRefine",
                        "sep": round(proto.separability_ratio, 4),
                        **_row(y, srr.to_numpy(), base.to_numpy(), rare),
                    }
                )
                print(f"[ok] {ds} seed={seed} rts={rts}  scANVI/kNN/SRR done")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "core_summary.csv", index=False)

    # 3-seed 聚合
    agg = []
    for (ds, rts, mth), g in df.groupby(["dataset", "rare_train_size", "method"]):
        agg.append(
            {
                "dataset": ds,
                "rare_train_size": rts,
                "method": mth,
                "n_seed": len(g),
                "f1_mean": round(g["rare_f1"].mean(), 4),
                "f1_std": round(g["rare_f1"].std(ddof=0), 4),
                "rec_mean": round(g["rare_recall"].mean(), 4),
                "incremental_fpr_max": round(g["incremental_fpr"].max(), 6),
                "ffr_max": round(g["rescue_ffr"].max(), 6),
                "fp_rate_max": round(g["rare_fp_rate"].max(), 6),
            }
        )
    aggdf = pd.DataFrame(agg)
    aggdf.to_csv(OUT / "core_agg.csv", index=False)

    # 验收线判定
    print("\n================ 第十三轮验收线判定 ================")
    seeds_done = sorted(df["seed"].unique().tolist())
    print(f"seeds present: {seeds_done}")
    # (b) 稀缺区 scRareRefine F1 ≥ scANVI 且 incremental FPR ≤ 0.01（逐 seed）
    sc = df[df["rare_train_size"].isin(SCARCE)]
    piv = sc.pivot_table(
        index=["dataset", "seed", "rare_train_size"], columns="method", values="rare_f1"
    )
    b_ok = bool((piv["scRareRefine"] >= piv["scANVI"] - 1e-9).all())
    ffr_ok = bool(
        (df[df["method"] == "scRareRefine"]["incremental_fpr"] <= 0.01 + 1e-9).all()
    )
    print(
        f"(b) 稀缺区 SRR F1 ≥ scANVI（逐 seed 全格）: {b_ok}；SRR incremental FPR ≤ α=0.01 全格: {ffr_ok}"
    )
    # (c) 三 testbed 稀缺区 F1 增益 mean - std > 0
    print("(c) testbed 稀缺区 F1 增益 mean+/-std (SRR - scANVI):")
    c_all = True
    for ds in TESTBEDS:
        sub = sc[sc["dataset"] == ds]
        if sub.empty:
            print(f"    {ds}: 无数据")
            c_all = False
            continue
        gains = []
        for (seed, rts), gg in sub.groupby(["seed", "rare_train_size"]):
            srrf = gg[gg.method == "scRareRefine"]["rare_f1"]
            scaf = gg[gg.method == "scANVI"]["rare_f1"]
            if len(srrf) and len(scaf):
                gains.append(float(srrf.iloc[0] - scaf.iloc[0]))
        g = np.array(gains)
        ok = (g.mean() - g.std(ddof=0)) > 0
        c_all = c_all and ok
        print(
            f"    {ds}: gain {g.mean():+.4f} +/- {g.std(ddof=0):.4f}  (mean-std>0: {ok}, n={len(g)})"
        )
    print(
        f"\n验收：(b)={b_ok and ffr_ok}  (c)={c_all}  "
        f"[需 seeds={SEEDS} 全部就绪才算最终判定，当前 {seeds_done}]"
    )
    print(f"[saved] {OUT / 'core_summary.csv'}  {OUT / 'core_agg.csv'}")


if __name__ == "__main__":
    main()
