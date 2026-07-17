"""Ablation（重构版 2026-06-21）—— 量化 conformal_rescue 各组件贡献，3-seed。

把消融拆成**两张性质不同的表**（解决旧 V0–V7 混表、跳号、看似 7 组件的可读性问题）：

表 1 · 组件留一法（leave-one-out）：每行只去掉 1 个组件，证明各组件该不该留。
  A0_baseline            裸 scANVI（不 rescue，参照线）
  A1_minus_sep           去 sep 安全网（LOW_SEP=0），其余=Full
  A2_minus_necessity     去 necessity 守门，其余=Full
  A3_minus_adaptive_rank 去自适应 rank（退化成固定 rank=1），其余=Full
  A4_minus_tau           去 conformal τ（候选直接全改判），其余=Full
  A5_full                完整方法（= src.rescue.conformal_rescue）

表 2 · rank 敏感性（单独子研究）：证明"自适应 ≥ 任何固定值"。
  R1_rank1 / R2_rank2 / R3_rank3   固定 rank=1/2/3（sep+necessity+τ 全开）
  R_adaptive                       val-自适应（= A5_full）

  说明：A3_minus_adaptive_rank 与 R1_rank1 是同一配置（去自适应=退回固定 rank=1），
  两表都出现以作交叉引用 + 一致性自检。A5_full 与 R_adaptive 同理。

真正可拆组件只有 **4 个**（2 弃权闸门 sep/necessity + 2 拯救机制 rank/τ）；
表 2 的 rank 1/2/3 是对其中"自适应 rank"机制的敏感性扫描，不是额外组件。

跑 6 数据集 × 4 rts × seed∈{42,43,44} = 72 配置 × 10 变体。全 inductive，复用 embeddings 缓存，不重训。

输出：
  results/ablation/ablation_summary.csv          逐 (dataset,rts,seed,variant)
  results/ablation/ablation_table1_components.csv 表1：(dataset,variant) 3-seed×4rts mean±std + Δvs_full
  results/ablation/ablation_table2_rank.csv       表2：(dataset,rank_variant) mean±std + FFR
  results/ablation/ablation_log.md                人读两张表

用法：D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/ablation.py
"""

from __future__ import annotations
import sys
import json
import hashlib
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
from src.rescue import (  # noqa: E402
    PrototypeRescuer,
    ConformalRescuer,
    conformal_rescue,
    DEFAULT_CONFORMAL_ALPHA,
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
    MIN_VAL_MISSED,
)

ALPHA = DEFAULT_CONFORMAL_ALPHA
SEEDS = [42, 43, 44]
CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
]
RTS = ["0.01", "0.05", "0.10", "all"]
RUNS = [(c, s, r) for c in CONFIGS for s in SEEDS for r in RTS]

# 变体定义：(name, group, spec)
#   group ∈ {"A"(表1 留一法), "R"(表2 rank 敏感性)}
#   spec: None=baseline | "full"=conformal_rescue | dict=_conformal_with_overrides 的 kwargs
_FULL_KW = dict(
    low_sep=CONFORMAL_LOW_SEP,
    enforce_necessity=True,
    rank_grid=CONFORMAL_RANK_GRID,
    use_conformal_tau=True,
)
VARIANTS = [
    ("A0_baseline", "A", None),
    ("A1_minus_sep", "A", {**_FULL_KW, "low_sep": 0.0}),
    ("A2_minus_necessity", "A", {**_FULL_KW, "enforce_necessity": False}),
    ("A3_minus_adaptive_rank", "A", {**_FULL_KW, "rank_grid": (1,)}),
    ("A4_minus_tau", "A", {**_FULL_KW, "use_conformal_tau": False}),
    ("A5_full", "A", "full"),
    ("R1_rank1", "R", {**_FULL_KW, "rank_grid": (1,)}),
    ("R2_rank2", "R", {**_FULL_KW, "rank_grid": (2,)}),
    ("R3_rank3", "R", {**_FULL_KW, "rank_grid": (3,)}),
    ("R_adaptive", "R", "full"),
]


def _lat(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _conformal_with_overrides(
    proto,
    base_pred_test,
    val_pred_labels,
    val_true,
    val_lat,
    test_lat,
    *,
    low_sep=CONFORMAL_LOW_SEP,
    enforce_necessity=True,
    min_val_missed=MIN_VAL_MISSED,
    rank_grid=CONFORMAL_RANK_GRID,
    use_conformal_tau=True,
):
    """conformal_rescue 的可消融版：三道闸门 + τ 独立开关。inductive 协议不变（val 选参，不碰 test 标签）。"""
    base_pred_test = pd.Series(base_pred_test).astype(str).reset_index(drop=True)
    val_pred_labels = pd.Series(val_pred_labels).astype(str).reset_index(drop=True)
    val_true = pd.Series(val_true).astype(str).reset_index(drop=True)
    rare = proto.rare_class
    summary = {
        "abstain": False,
        "reason": "",
        "chosen_rank": 0,
        "tau": float("inf"),
        "n_candidate": 0,
        "n_rescued": 0,
    }

    if proto.separability_ratio < low_sep:
        summary.update(abstain=True, reason=f"sep<{low_sep}")
        return base_pred_test.copy(), summary

    val_missed = int((val_true.eq(rare) & val_pred_labels.ne(rare)).sum())
    summary["val_missed"] = val_missed
    if (
        enforce_necessity
        and int(val_true.eq(rare).sum()) > 0
        and val_missed < min_val_missed
    ):
        summary.update(abstain=True, reason="necessity")
        return base_pred_test.copy(), summary

    val_score = proto.rare_membership_score(val_lat)
    test_score = proto.rare_membership_score(test_lat)
    if use_conformal_tau:
        tau = ConformalRescuer(rare, alpha=ALPHA).calibrate(val_score, val_true)
    else:
        tau = float("-inf")
    summary["tau"] = tau
    if use_conformal_tau and not np.isfinite(tau):
        summary.update(abstain=True, reason="tau=inf")
        return base_pred_test.copy(), summary

    if len(rank_grid) == 1:
        chosen_rank = rank_grid[0]
    else:
        val_ranks = proto.rare_rank(val_lat)
        n_val_nonrare = int(val_true.ne(rare).sum())
        best = None
        chosen_rank = None
        z = 1.96
        for k in rank_grid:
            v_cand = (val_ranks <= k) & val_pred_labels.ne(rare).to_numpy()
            v_fire = v_cand & (val_score >= tau)
            v_relabel = val_pred_labels.copy()
            v_relabel[v_fire] = rare
            v_false = int((v_fire & val_true.ne(rare).to_numpy()).sum())
            n = max(n_val_nonrare, 1)
            p = v_false / n
            denom = 1.0 + z * z / n
            center = (p + z * z / (2 * n)) / denom
            half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
            if center + half > ALPHA:
                continue
            vf1, _ = classification_tables(val_true, v_relabel, rare_class=rare)
            key = (round(vf1["rare_f1"], 6), -k)
            if best is None or key > best:
                best = key
                chosen_rank = k
        if best is None:
            summary.update(abstain=True, reason="no_feasible_rank")
            return base_pred_test.copy(), summary
    summary["chosen_rank"] = chosen_rank

    test_cand = proto.rank_candidate(test_lat, base_pred_test, max_rank=chosen_rank)
    fire = test_cand & (np.asarray(test_score) >= tau)
    final = base_pred_test.copy()
    final.iloc[np.where(fire)[0]] = rare
    summary["n_candidate"] = int(test_cand.sum())
    summary["n_rescued"] = int(final.ne(base_pred_test).sum())
    return final, summary


def run_variant(spec, proto, val_lat, test_lat, val_pred, test_pred, y_true, rare):
    base_pred = test_pred["predicted_label"].astype(str)
    val_base = val_pred["predicted_label"].astype(str)
    val_true = val_pred["true_label"].astype(str)

    if spec is None:  # baseline
        final = base_pred.copy()
        summary = {
            "abstain": False,
            "chosen_rank": 0,
            "tau": float("nan"),
            "n_rescued": 0,
        }
    elif spec == "full":  # 完整方法
        final, summary = conformal_rescue(
            proto, base_pred, val_base, val_true, val_lat, test_lat, alpha=ALPHA
        )
    else:  # 可消融版
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat, **spec
        )

    fp = final.astype(str).to_numpy()
    base_arr = base_pred.to_numpy()
    n_rescued = int(((fp != base_arr) & (fp == rare)).sum())
    n_false = int(((fp != base_arr) & (fp == rare) & (y_true != rare)).sum())
    n_nonrare = int((y_true != rare).sum())
    m, _ = classification_tables(y_true, fp, rare_class=rare)
    bl, _ = classification_tables(y_true, base_arr, rare_class=rare)
    incremental_fpr = round(n_false / max(n_nonrare, 1), 6)
    return {
        "abstain": bool(summary.get("abstain", False)),
        "chosen_rank": int(summary.get("chosen_rank", 0)),
        "baseline_f1": round(bl["rare_f1"], 4),
        "rare_f1": round(m["rare_f1"], 4),
        "rare_recall": round(m["rare_recall"], 4),
        "rare_precision": round(m["rare_precision"], 4),
        "n_rescued": n_rescued,
        "n_false_rescues": n_false,
        "incremental_fpr": incremental_fpr,
        "ffr": incremental_fpr,
    }


def _cell_id_align_hash(*paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        if not p.exists():
            continue
        try:
            ids = (
                pd.read_csv(p, usecols=["cell_id"], low_memory=False)["cell_id"]
                .astype(str)
                .tolist()
            )
        except Exception:
            ids = []
        h.update(("|".join(ids) + "\n").encode("utf-8"))
    return h.hexdigest()[:12]


def main():
    rows = []
    for cfg_path, seed, rts_str in RUNS:
        cfg = load_config(cfg_path)
        exp = cfg.get("experiment", {})
        rare = exp["rare_class"]
        sm = exp.get("split_mode", "batch_heldout")
        rd = make_run_dir(cfg, sm, seed, rare, parse_rare_train_size(rts_str))
        emb = rd / "embeddings"
        ds = cfg["dataset"]["name"]
        if not (emb / "test_latent.csv").exists():
            print(f"[SKIP] no cache: {ds} seed={seed} rts={rts_str}")
            continue

        mf_path = rd / "manifest.json"
        if mf_path.exists():
            mf = json.loads(mf_path.read_text(encoding="utf-8"))
            split_hash = mf.get("split_hash", "")
            git_sha = mf.get("git_sha", "")
            if git_sha in ("", "unknown"):
                git_sha = "legacy_pre_git_sha_recording"
        else:
            split_hash = git_sha = "no_manifest"
        cell_id_hash = _cell_id_align_hash(
            emb / "train_predictions.csv",
            emb / "validation_predictions.csv",
            emb / "test_predictions.csv",
        )

        train_pred = pd.read_csv(emb / "train_predictions.csv")
        train_lat = _lat(pd.read_csv(emb / "train_latent.csv"))
        val_pred = pd.read_csv(emb / "validation_predictions.csv")
        val_lat = _lat(pd.read_csv(emb / "validation_latent.csv"))
        test_pred = pd.read_csv(emb / "test_predictions.csv")
        test_lat = _lat(pd.read_csv(emb / "test_latent.csv"))

        ref_labels = train_pred["true_label"].astype(str)
        is_lab = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
        proto = PrototypeRescuer(rare)
        proto.fit(train_lat, ref_labels, is_lab)
        y_true = test_pred["true_label"].astype(str).to_numpy()

        print(f"\n[{ds} seed={seed} rts={rts_str}] sep={proto.separability_ratio:.3f}")
        for name, group, spec in VARIANTS:
            res = run_variant(
                spec, proto, val_lat, test_lat, val_pred, test_pred, y_true, rare
            )
            tag = " (abstain)" if res["abstain"] else ""
            print(
                f"  {name:24s}: F1={res['rare_f1']:.4f}{tag}  rec={res['rare_recall']:.4f}  "
                f"rank={res['chosen_rank']}  rescued={res['n_rescued']}  false={res['n_false_rescues']}  "
                f"incremental_fpr={res['incremental_fpr']:.5f}"
            )
            rows.append(
                {
                    "dataset": ds,
                    "rare_class": rare,
                    "seed": seed,
                    "rts": rts_str,
                    "variant": name,
                    "group": group,
                    "sep": round(proto.separability_ratio, 4),
                    "split_hash": split_hash,
                    "git_sha": git_sha,
                    "cell_id_align_hash": cell_id_hash,
                    **res,
                }
            )

    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    print(
        f"\n[saved] {out_dir / 'ablation_summary.csv'}  ({len(df)} 行, seeds={SEEDS})"
    )

    # 每 cell 的 full F1（算 Δvs_full）
    full_f1 = df[df.variant == "A5_full"].set_index(["dataset", "seed", "rts"])[
        "rare_f1"
    ]

    def _delta_vs_full(sub):
        # 约定：delta = Full - 该变体（逐 cell）。正 = 去掉该组件后 F1 掉这么多（组件对 F1 有正贡献）；
        # 负 = 去掉反而升（该组件价值在 FFR/安全而非 F1，须看 FFR_max）。
        idx = sub.set_index(["dataset", "seed", "rts"]).index
        return full_f1.reindex(idx).to_numpy() - sub["rare_f1"].to_numpy()

    # 表 1：组件留一法（group A），按 (dataset, variant) 聚合 12 cell(4rts×3seed)
    A_order = [
        "A0_baseline",
        "A1_minus_sep",
        "A2_minus_necessity",
        "A3_minus_adaptive_rank",
        "A4_minus_tau",
        "A5_full",
    ]
    t1 = []
    for ds in df["dataset"].unique():
        for v in A_order:
            sub = df[(df.dataset == ds) & (df.variant == v)]
            if sub.empty:
                continue
            dvf = _delta_vs_full(sub)
            t1.append(
                {
                    "dataset": ds,
                    "variant": v,
                    "n_cells": len(sub),
                    "f1_mean": round(sub.rare_f1.mean(), 4),
                    "f1_std": round(sub.rare_f1.std(ddof=0), 4),
                    "recall_mean": round(sub.rare_recall.mean(), 4),
                    "ffr_max": round(sub.ffr.max(), 6),
                    "gain_vs_baseline": round(
                        (sub.rare_f1 - sub.baseline_f1).mean(), 4
                    ),
                    "delta_vs_full": round(float(np.nanmean(dvf)), 4),
                    "n_abstain": int(sub.abstain.sum()),
                }
            )
    t1df = pd.DataFrame(t1)
    t1df.to_csv(out_dir / "ablation_table1_components.csv", index=False)

    # 表 2：rank 敏感性（group R），按 (dataset, variant) 聚合
    R_order = ["R1_rank1", "R2_rank2", "R3_rank3", "R_adaptive"]
    t2 = []
    for ds in df["dataset"].unique():
        for v in R_order:
            sub = df[(df.dataset == ds) & (df.variant == v)]
            if sub.empty:
                continue
            t2.append(
                {
                    "dataset": ds,
                    "variant": v,
                    "n_cells": len(sub),
                    "f1_mean": round(sub.rare_f1.mean(), 4),
                    "f1_std": round(sub.rare_f1.std(ddof=0), 4),
                    "recall_mean": round(sub.rare_recall.mean(), 4),
                    "ffr_max": round(sub.ffr.max(), 6),
                    "n_abstain": int(sub.abstain.sum()),
                }
            )
    t2df = pd.DataFrame(t2)
    t2df.to_csv(out_dir / "ablation_table2_rank.csv", index=False)
    print(
        f"[saved] {out_dir / 'ablation_table1_components.csv'}  {out_dir / 'ablation_table2_rank.csv'}"
    )

    # 跨数据集 OVERALL 行（pool 所有 cell）
    def _overall(order, group):
        out = []
        for v in order:
            sub = df[df.variant == v]
            if sub.empty:
                continue
            row = {
                "dataset": "OVERALL",
                "variant": v,
                "n_cells": len(sub),
                "f1_mean": round(sub.rare_f1.mean(), 4),
                "f1_std": round(sub.rare_f1.std(ddof=0), 4),
                "recall_mean": round(sub.rare_recall.mean(), 4),
                "ffr_max": round(sub.ffr.max(), 6),
                "n_abstain": int(sub.abstain.sum()),
            }
            if group == "A":
                row["gain_vs_baseline"] = round(
                    (sub.rare_f1 - sub.baseline_f1).mean(), 4
                )
                row["delta_vs_full"] = round(float(np.nanmean(_delta_vs_full(sub))), 4)
            out.append(row)
        return out

    ov1 = _overall(A_order, "A")
    ov2 = _overall(R_order, "R")

    # 人读 log（两张表）
    L = [
        "# Ablation Report（重构版，3-seed）",
        "",
        f"**Date**: 2026-06-21  |  **Seeds**: {SEEDS}  |  **Datasets**: {df['dataset'].nunique()}  |  **rts**: {RTS}",
        "",
        "真正可拆组件 = 4 个（sep / necessity 弃权闸门 + 自适应rank / τ 拯救机制）。",
        "表 1 答「每个组件该不该留」，表 2 答「自适应 rank 为何优于任何固定值」。",
        "聚合单元 = 每 (dataset, variant) 的 4 rts × 3 seed = 12 cell；f1_std 含 rts 轴差异（非纯 seed 方差）。",
        "",
        "## 表 1 · 组件留一法（leave-one-out）",
        "",
        "| dataset | variant | F1 mean±std | recall | gain vs baseline | **Δ=Full−变体** | FFR_max | abstain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in ov1 + t1:
        L.append(
            f"| {r['dataset']} | {r['variant']} | {r['f1_mean']:.3f}±{r['f1_std']:.3f} | {r['recall_mean']:.3f} | "
            f"{r.get('gain_vs_baseline', 0):+.3f} | {r.get('delta_vs_full', 0):+.3f} | {r['ffr_max']:.5f} | {r['n_abstain']} |"
        )
    L += [
        "",
        "> 读法：`Δ=Full−变体`（逐 cell 均值）。**正 = 去掉该组件 F1 掉这么多（对 F1 有正贡献）**；",
        "> **负 = 去掉反而升 → 该组件价值在 FFR/安全而非 F1，须看 FFR_max**（如 −sep 升 F1 但 FFR 破 α；−τ 同理）。",
        "> A5_full 的 Δ 恒为 0（自比）；A0_baseline 的 gain vs baseline 恒为 0。",
        "",
        "## 表 2 · rank 敏感性（自适应 vs 固定）",
        "",
        "| dataset | variant | F1 mean±std | recall | FFR_max | abstain |",
        "|---|---|---|---|---|---|",
    ]
    for r in ov2 + t2:
        L.append(
            f"| {r['dataset']} | {r['variant']} | {r['f1_mean']:.3f}±{r['f1_std']:.3f} | {r['recall_mean']:.3f} | "
            f"{r['ffr_max']:.5f} | {r['n_abstain']} |"
        )
    L += [
        "",
        "> R1_rank1 == A3_minus_adaptive_rank（去自适应=退回固定 rank=1）；R_adaptive == A5_full（交叉引用+一致性自检）。",
        "> 看点：固定 rank=3 时 FFR_max 是否冲破 α=0.01；自适应是否在 FFR≤α 下拿到 ≥ 任何固定值的 F1。",
    ]
    (out_dir / "ablation_log.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[saved] {out_dir / 'ablation_log.md'}")


if __name__ == "__main__":
    main()
