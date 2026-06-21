"""Round 12 G65 — MIN_VAL_MISSED sensitivity sweep（k ∈ {1, 2, 3, 5}）。

对每个 k，复跑 conformal_rescue 路径（通过 ablation 的 _conformal_with_overrides），
统计：每个 (dataset, rts) 是否弃权、F1、recall、FFR。然后聚合：
  - 全局弃权数
  - 各数据集 4-rts 平均 F1
  - FFR_max
  - 是否触发 pancreas_integrated 回归（rts=0.01/0.05 V6 vs V0）

证明 k=3 不是 cherry-pick：k=1 时 pancreas_integrated 0.01/0.05 仍回归（val_missed=2/1 不触发 abstain），
k≥2 起回归消除，k=3 是稳健下限。

输出：results/ablation/min_val_missed_sensitivity.csv + 控制台聚合
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables  # noqa: E402
from src.rescue import (  # noqa: E402
    PrototypeRescuer,
    DEFAULT_CONFORMAL_ALPHA,
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
)
# 复用 ablation 里的 _conformal_with_overrides（带 Wilson + 可配 min_val_missed）
sys.path.insert(0, str(ROOT / "tools" / "analysis"))
from ablation import _conformal_with_overrides, _lat  # noqa: E402

RUNS = [
    (c, 42, r) for c in [
        "configs/immune_dc.yaml",
        "configs/pancreas_baron.yaml",
        "configs/pancreas_integrated.yaml",
        "configs/tabula_lung_endo.yaml",
        "configs/tabula_sapiens_stomach.yaml",
        "configs/tabula_small_intestine.yaml",
    ] for r in ("0.01", "0.05", "0.10", "all")
]

K_GRID = (1, 2, 3, 5)

rows = []
for cfg_path, seed, rts_str in RUNS:
    cfg = load_config(cfg_path); exp = cfg["experiment"]
    rare = exp["rare_class"]; sm = exp.get("split_mode","batch_heldout")
    sz = parse_rare_train_size(rts_str)
    rd = make_run_dir(cfg, sm, seed, rare, sz)
    emb = rd / "embeddings"
    ds = cfg["dataset"]["name"]
    if not (emb / "test_latent.csv").exists(): continue

    train_pred = pd.read_csv(emb / "train_predictions.csv", low_memory=False)
    train_lat = _lat(pd.read_csv(emb / "train_latent.csv", low_memory=False))
    val_pred = pd.read_csv(emb / "validation_predictions.csv", low_memory=False)
    val_lat = _lat(pd.read_csv(emb / "validation_latent.csv", low_memory=False))
    test_pred = pd.read_csv(emb / "test_predictions.csv", low_memory=False)
    test_lat = _lat(pd.read_csv(emb / "test_latent.csv", low_memory=False))

    proto = PrototypeRescuer(rare)
    proto.fit(train_lat, train_pred["true_label"].astype(str),
              train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy())
    base_pred = test_pred["predicted_label"].astype(str)
    val_base = val_pred["predicted_label"].astype(str)
    val_true = val_pred["true_label"].astype(str)
    y_true = test_pred["true_label"].astype(str).to_numpy()
    bl, _ = classification_tables(y_true, base_pred.to_numpy(), rare_class=rare)

    for k in K_GRID:
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=True,
            min_val_missed=k, rank_grid=CONFORMAL_RANK_GRID, use_conformal_tau=True,
        )
        fp = final.astype(str).to_numpy()
        base_arr = base_pred.to_numpy()
        n_nonrare = int((y_true != rare).sum())
        n_false = int(((fp != base_arr) & (fp == rare) & (y_true != rare)).sum())
        m, _ = classification_tables(y_true, fp, rare_class=rare)
        rows.append({
            "dataset": ds, "rts": rts_str, "k_min_val_missed": k,
            "abstain": bool(summary.get("abstain", False)),
            "abstain_reason": summary.get("reason", ""),
            "val_missed": summary.get("val_missed", -1),
            "chosen_rank": summary.get("chosen_rank", 0),
            "baseline_f1": round(bl["rare_f1"], 4),
            "rare_f1": round(m["rare_f1"], 4),
            "rare_recall": round(m["rare_recall"], 4),
            "rare_precision": round(m["rare_precision"], 4),
            "ffr": round(n_false / max(n_nonrare, 1), 6),
        })

out = ROOT / "results" / "ablation"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "min_val_missed_sensitivity.csv", index=False)

print(f"[saved] {out/'min_val_missed_sensitivity.csv'}")
print("\n=== 聚合：(dataset, k) 4-rts 平均 ===")
agg_rows = []
for ds in df["dataset"].unique():
    for k in K_GRID:
        sub = df[(df["dataset"] == ds) & (df["k_min_val_missed"] == k)]
        agg_rows.append({
            "dataset": ds, "k": k,
            "n_abstain": int(sub["abstain"].sum()),
            "f1_mean": round(sub["rare_f1"].mean(), 4),
            "f1_vs_baseline_mean": round((sub["rare_f1"] - sub["baseline_f1"]).mean(), 4),
            "ffr_max": round(sub["ffr"].max(), 6),
        })
agg = pd.DataFrame(agg_rows)
agg.to_csv(out / "min_val_missed_sensitivity_agg.csv", index=False)
print(agg.to_string(index=False))

# 重点：pancreas_integrated rts=0.01/0.05 在不同 k 下是否回归
print("\n=== 关键诊断：pancreas_integrated 回归是否消除 ===")
pi = df[(df["dataset"] == "pancreas_integrated") & (df["rts"].isin(["0.01", "0.05"]))]
print(pi[["rts","k_min_val_missed","val_missed","abstain","abstain_reason","rare_f1","baseline_f1"]].to_string(index=False))
