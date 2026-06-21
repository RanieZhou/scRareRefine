"""Round 12 G64 — Wilson 上界 rank 选择 row-level 诊断表。

对每个 (dataset, rts) × k ∈ {1,2,3}，导出 val 上的：
  n_val_nonrare / v_fire / v_false / v_ffr_point / wilson_upper_95 / val_rare_f1_if_chosen / chosen_by_rule

让审稿人能验证：
  - Wilson 选择规则是 "wilson_upper_95 ≤ α" 而不是 point estimate
  - 每个被剔除的 rank 都有具体 wilson_upper 数值证据
  - 哪些数据集 / rts 触发了 Wilson 剔除，相应代价是什么

输出：results/ablation/wilson_diagnostics.csv
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
    ConformalRescuer,
    DEFAULT_CONFORMAL_ALPHA,
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
    MIN_VAL_MISSED,
)

RUNS = [
    ("configs/immune_dc.yaml",              42, s) for s in ("0.01","0.05","0.10","all")
] + [
    ("configs/pancreas_baron.yaml",         42, s) for s in ("0.01","0.05","0.10","all")
] + [
    ("configs/pancreas_integrated.yaml",    42, s) for s in ("0.01","0.05","0.10","all")
] + [
    ("configs/tabula_lung_endo.yaml",       42, s) for s in ("0.01","0.05","0.10","all")
] + [
    ("configs/tabula_sapiens_stomach.yaml", 42, s) for s in ("0.01","0.05","0.10","all")
] + [
    ("configs/tabula_small_intestine.yaml", 42, s) for s in ("0.01","0.05","0.10","all")
]

Z = 1.96
ALPHA = DEFAULT_CONFORMAL_ALPHA


def wilson_upper(p, n, z=Z):
    n = max(n, 1)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center + half


def _lat(df): return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


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

    proto = PrototypeRescuer(rare)
    proto.fit(train_lat, train_pred["true_label"].astype(str),
              train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy())

    val_true = val_pred["true_label"].astype(str)
    val_base = val_pred["predicted_label"].astype(str)
    val_score = proto.rare_membership_score(val_lat)
    val_ranks = proto.rare_rank(val_lat)
    n_vnr = int(val_true.ne(rare).sum())
    val_missed = int((val_true.eq(rare) & val_base.ne(rare)).sum())

    # 早退原因（与 src.rescue.conformal_rescue 镜像）
    abstain_pre_rank = None
    if proto.separability_ratio < CONFORMAL_LOW_SEP:
        abstain_pre_rank = f"sep<{CONFORMAL_LOW_SEP}"
    elif val_missed < MIN_VAL_MISSED:
        abstain_pre_rank = f"val_missed={val_missed} < MIN_VAL_MISSED={MIN_VAL_MISSED}"

    conf = ConformalRescuer(rare, alpha=ALPHA)
    tau = conf.calibrate(val_score, val_true)
    if not np.isfinite(tau) and abstain_pre_rank is None:
        abstain_pre_rank = "tau=inf"

    chosen_rank = None
    candidate_keys = []
    if abstain_pre_rank is None:
        # 真跑 Wilson 选择
        best = None
        for k in CONFORMAL_RANK_GRID:
            v_cand = (val_ranks <= k) & val_base.ne(rare).to_numpy()
            v_fire = v_cand & (val_score >= tau)
            v_false = int((v_fire & val_true.ne(rare).to_numpy()).sum())
            wup = wilson_upper(v_false / max(n_vnr, 1), n_vnr)
            if wup > ALPHA: continue
            v_relabel = val_base.copy(); v_relabel[v_fire] = rare
            vf1, _ = classification_tables(val_true, v_relabel, rare_class=rare)
            key = (round(vf1["rare_f1"], 6), -k)
            candidate_keys.append((k, key))
            if best is None or key > best:
                best = key; chosen_rank = k

    for k in CONFORMAL_RANK_GRID:
        v_cand = (val_ranks <= k) & val_base.ne(rare).to_numpy()
        v_fire = v_cand & (val_score >= tau) if np.isfinite(tau) else np.zeros(len(val_score), dtype=bool)
        v_false = int((v_fire & val_true.ne(rare).to_numpy()).sum())
        v_ffr = v_false / max(n_vnr, 1)
        wup = wilson_upper(v_ffr, n_vnr)
        v_relabel = val_base.copy(); v_relabel[v_fire] = rare
        try:
            vf1, _ = classification_tables(val_true, v_relabel, rare_class=rare)
            val_rare_f1 = round(vf1["rare_f1"], 4)
        except Exception:
            val_rare_f1 = None
        if abstain_pre_rank is not None:
            disp = "abstain-pre-rank"
        elif wup > ALPHA:
            disp = "rejected (wilson>α)"
        elif k == chosen_rank:
            disp = "CHOSEN"
        else:
            disp = "feasible"
        rows.append({
            "dataset": ds, "rts": rts_str, "sep": round(proto.separability_ratio, 3),
            "n_val_nonrare": n_vnr, "val_missed": val_missed,
            "tau": round(float(tau), 4) if np.isfinite(tau) else float("inf"),
            "k": k,
            "v_fire": int(v_fire.sum()), "v_false": v_false,
            "v_ffr_point": round(v_ffr, 5), "wilson_upper_95": round(wup, 5),
            "val_rare_f1_if_chosen": val_rare_f1,
            "rule_disposition": disp,
            "abstain_pre_rank_reason": abstain_pre_rank or "",
        })

out = ROOT / "results" / "ablation"
out.mkdir(parents=True, exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out / "wilson_diagnostics.csv", index=False)
print(df.to_string(index=False))
print(f"\n[saved] {out/'wilson_diagnostics.csv'}")
print(f"\n=== 关键统计 ===")
print(f"rejected by Wilson 总数: {(df['rule_disposition']=='rejected (wilson>α)').sum()}")
print(f"abstain-pre-rank 行数: {(df['rule_disposition']=='abstain-pre-rank').sum()}")
print(f"CHOSEN 分布 (k=1/2/3):")
chosen = df[df['rule_disposition']=='CHOSEN']
for k in (1, 2, 3):
    print(f"  k={k}: {(chosen['k']==k).sum()} cells")
