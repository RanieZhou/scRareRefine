"""Round 10 ablation — 量化 conformal_rescue 各组件的贡献。

对齐当前主方法 src.rescue.conformal_rescue 的真实组件（separability 安全网 + necessity 守门 +
val-自适应候选 rank ∈ {1,2,3} + conformal τ），逐一拆除。

变体（A 层 ablation，全部 inductive，复用 embeddings 缓存）：
  V0 baseline_scanvi   完全不 rescue（参照线）
  V1 no_sep_gate       关 separability 安全网 (设 LOW_SEP=0)
  V2 no_necessity      关 necessity 守门
  V3 rank1_fixed       固定 rank=1，无 val-自适应
  V4 rank2_fixed       固定 rank=2，无 val-自适应
  V5 no_conformal_tau  候选直接全 relabel（去 τ，仅保 sep + necessity + rank=1 候选筛选）
  V6 full              当前 conformal_rescue（参考方法）

跑 6 数据集 × 4 rts × seed=42 = 24 配置 × 7 变体 = 168 行。

输出：
  results/ablation/ablation_summary.csv     逐配置 × 变体
  results/ablation/ablation_summary_agg.csv （dataset, variant）F1/recall/FFR 均值
  results/ablation/ablation_log.md          人读总结

用法：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/ablation.py
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
    conformal_rescue,
    DEFAULT_CONFORMAL_ALPHA,
    CONFORMAL_LOW_SEP,
    CONFORMAL_RANK_GRID,
    MIN_VAL_MISSED,
)
import json
import hashlib

RUNS = [
    ("configs/immune_dc.yaml",              42, "0.01"),
    ("configs/immune_dc.yaml",              42, "0.05"),
    ("configs/immune_dc.yaml",              42, "0.10"),
    ("configs/immune_dc.yaml",              42, "all"),
    ("configs/pancreas_baron.yaml",         42, "0.01"),
    ("configs/pancreas_baron.yaml",         42, "0.05"),
    ("configs/pancreas_baron.yaml",         42, "0.10"),
    ("configs/pancreas_baron.yaml",         42, "all"),
    ("configs/pancreas_integrated.yaml",    42, "0.01"),
    ("configs/pancreas_integrated.yaml",    42, "0.05"),
    ("configs/pancreas_integrated.yaml",    42, "0.10"),
    ("configs/pancreas_integrated.yaml",    42, "all"),
    ("configs/tabula_lung_endo.yaml",       42, "0.01"),
    ("configs/tabula_lung_endo.yaml",       42, "0.05"),
    ("configs/tabula_lung_endo.yaml",       42, "0.10"),
    ("configs/tabula_lung_endo.yaml",       42, "all"),
    ("configs/tabula_sapiens_stomach.yaml", 42, "0.01"),
    ("configs/tabula_sapiens_stomach.yaml", 42, "0.05"),
    ("configs/tabula_sapiens_stomach.yaml", 42, "0.10"),
    ("configs/tabula_sapiens_stomach.yaml", 42, "all"),
    ("configs/tabula_small_intestine.yaml", 42, "0.01"),
    ("configs/tabula_small_intestine.yaml", 42, "0.05"),
    ("configs/tabula_small_intestine.yaml", 42, "0.10"),
    ("configs/tabula_small_intestine.yaml", 42, "all"),
]

ALPHA = DEFAULT_CONFORMAL_ALPHA

VARIANTS = [
    "V0_baseline_scanvi",
    "V1_no_sep_gate",
    "V2_no_necessity",
    "V3_rank1_fixed",
    "V4_rank2_fixed",
    "V5_no_conformal_tau",
    "V6_full",
    "V7_rank3_fixed",  # Round 11 加：rank=3 sensitivity（G62 — 验证 val 不会选 rank=3 时它的代价）
]


def _lat(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _conformal_with_overrides(
    proto: PrototypeRescuer,
    base_pred_test: pd.Series,
    val_pred_labels: pd.Series,
    val_true: pd.Series,
    val_lat: np.ndarray,
    test_lat: np.ndarray,
    *,
    low_sep: float = CONFORMAL_LOW_SEP,
    enforce_necessity: bool = True,
    min_val_missed: int = MIN_VAL_MISSED,
    rank_grid=CONFORMAL_RANK_GRID,
    use_conformal_tau: bool = True,
):
    """conformal_rescue 的可消融版本：三道闸门 + τ 独立开关，便于 ablation。

    保持与 src.rescue.conformal_rescue 同样的 inductive 协议（val 选 rank、val 校 τ，绝不碰 test 标签）。
    use_conformal_tau=False 时 τ 设 -inf，等价于「候选直接 relabel，无 FFR 控制」。
    """
    base_pred_test = pd.Series(base_pred_test).astype(str).reset_index(drop=True)
    val_pred_labels = pd.Series(val_pred_labels).astype(str).reset_index(drop=True)
    val_true = pd.Series(val_true).astype(str).reset_index(drop=True)
    rare = proto.rare_class
    summary = {"abstain": False, "reason": "", "chosen_rank": 0, "tau": float("inf"),
               "n_candidate": 0, "n_rescued": 0}

    # 道 1：separability 安全网（可关）
    if proto.separability_ratio < low_sep:
        summary.update(abstain=True, reason=f"sep<{low_sep}")
        return base_pred_test.copy(), summary

    # 道 2：necessity + split-shift 守门（可关）
    val_missed = int((val_true.eq(rare) & val_pred_labels.ne(rare)).sum())
    summary["val_missed"] = val_missed
    if enforce_necessity and int(val_true.eq(rare).sum()) > 0 and val_missed < min_val_missed:
        reason = "val baseline 零漏判稀有" if val_missed == 0 else f"val_missed={val_missed} < min_val_missed={min_val_missed}"
        summary.update(abstain=True, reason=reason)
        return base_pred_test.copy(), summary

    # conformal τ
    val_score = proto.rare_membership_score(val_lat)
    test_score = proto.rare_membership_score(test_lat)
    if use_conformal_tau:
        conf = ConformalRescuer(rare, alpha=ALPHA)
        tau = conf.calibrate(val_score, val_true)
    else:
        tau = float("-inf")
    summary["tau"] = tau
    if use_conformal_tau and not np.isfinite(tau):
        summary.update(abstain=True, reason="tau=inf")
        return base_pred_test.copy(), summary

    # 道 3：val-自适应候选 rank（Wilson 95% 上界控 FFR）
    if len(rank_grid) == 1:
        chosen_rank = rank_grid[0]
    else:
        val_ranks = proto.rare_rank(val_lat)
        n_val_nonrare = int(val_true.ne(rare).sum())
        best = None
        chosen_rank = rank_grid[0]
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
            v_ffr_upper = center + half
            if v_ffr_upper > ALPHA:
                continue
            vf1, _ = classification_tables(val_true, v_relabel, rare_class=rare)
            key = (round(vf1["rare_f1"], 6), -k)
            if best is None or key > best:
                best = key
                chosen_rank = k
    summary["chosen_rank"] = chosen_rank

    test_cand = proto.rank_candidate(test_lat, base_pred_test, max_rank=chosen_rank)
    fire = test_cand & (np.asarray(test_score) >= tau)
    final = base_pred_test.copy()
    final.iloc[np.where(fire)[0]] = rare
    summary["n_candidate"] = int(test_cand.sum())
    summary["n_rescued"] = int(final.ne(base_pred_test).sum())
    return final, summary


def run_variant(
    variant: str,
    proto: PrototypeRescuer,
    val_lat: np.ndarray,
    test_lat: np.ndarray,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    y_true: np.ndarray,
    rare_class: str,
):
    base_pred = test_pred["predicted_label"].astype(str)
    val_base = val_pred["predicted_label"].astype(str)
    val_true = val_pred["true_label"].astype(str)

    if variant == "V0_baseline_scanvi":
        final = base_pred.copy()
        summary = {"abstain": False, "reason": "baseline", "chosen_rank": 0,
                   "tau": float("nan"), "n_candidate": 0, "n_rescued": 0}
    elif variant == "V1_no_sep_gate":
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=0.0, enforce_necessity=True, rank_grid=CONFORMAL_RANK_GRID, use_conformal_tau=True,
        )
    elif variant == "V2_no_necessity":
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=False, rank_grid=CONFORMAL_RANK_GRID, use_conformal_tau=True,
        )
    elif variant == "V3_rank1_fixed":
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=True, rank_grid=(1,), use_conformal_tau=True,
        )
    elif variant == "V4_rank2_fixed":
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=True, rank_grid=(2,), use_conformal_tau=True,
        )
    elif variant == "V5_no_conformal_tau":
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=True, rank_grid=CONFORMAL_RANK_GRID, use_conformal_tau=False,
        )
    elif variant == "V6_full":
        final, s = conformal_rescue(proto, base_pred, val_base, val_true, val_lat, test_lat, alpha=ALPHA)
        summary = s
    elif variant == "V7_rank3_fixed":
        # G62 sensitivity: 强制 rank=3 看代价（验证 val-自适应剔除 rank=3 是有理由的）
        final, summary = _conformal_with_overrides(
            proto, base_pred, val_base, val_true, val_lat, test_lat,
            low_sep=CONFORMAL_LOW_SEP, enforce_necessity=True, rank_grid=(3,), use_conformal_tau=True,
        )
    else:
        raise ValueError(variant)

    fp = final.astype(str).to_numpy()
    base_arr = base_pred.to_numpy()
    n_rescued = int(((fp != base_arr) & (fp == rare_class)).sum())
    n_false = int(((fp != base_arr) & (fp == rare_class) & (y_true != rare_class)).sum())
    n_nonrare = int((y_true != rare_class).sum())
    m, _ = classification_tables(y_true, fp, rare_class=rare_class)
    bl, _ = classification_tables(y_true, base_arr, rare_class=rare_class)
    return {
        "sep": round(proto.separability_ratio, 4),
        "abstain": bool(summary.get("abstain", False)),
        "abstain_reason": summary.get("reason", ""),
        "chosen_rank": int(summary.get("chosen_rank", 0)),
        "tau": float(summary.get("tau", float("nan"))),
        "baseline_f1": round(bl["rare_f1"], 4),
        "rare_f1": round(m["rare_f1"], 4),
        "rare_recall": round(m["rare_recall"], 4),
        "rare_precision": round(m["rare_precision"], 4),
        "n_rescued": n_rescued,
        "n_false_rescues": n_false,
        "ffr": round(n_false / max(n_nonrare, 1), 6),
        "f1_gain": round(m["rare_f1"] - bl["rare_f1"], 4),
    }


def _cell_id_align_hash(*csv_paths) -> str:
    """对若干 CSV 的 cell_id 列拼接后取 sha256 前 12 位，用于跨 split 对齐校验。"""
    h = hashlib.sha256()
    for p in csv_paths:
        if not p.exists():
            continue
        try:
            ids = pd.read_csv(p, usecols=["cell_id"], low_memory=False)["cell_id"].astype(str).tolist()
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
        size = parse_rare_train_size(rts_str)
        rd = make_run_dir(cfg, sm, seed, rare, size)
        emb = rd / "embeddings"
        ds = cfg["dataset"]["name"]
        if not (emb / "test_latent.csv").exists():
            print(f"[SKIP] no cache: {ds} rts={rts_str}")
            continue

        # G63 provenance：读 manifest + 算 cell-id alignment hash
        mf_path = rd / "manifest.json"
        if mf_path.exists():
            mf = json.loads(mf_path.read_text(encoding="utf-8"))
            split_hash = mf.get("split_hash", "")
            git_sha = mf.get("git_sha", "")
            # 旧 manifest 在 git_sha 字段引入前生成（immune_dc / pancreas_baron / stomach）
            # 不重训以避免改变 evaluation 数据；改为显式标 legacy 以保持 provenance 透明
            if git_sha in ("", "unknown"):
                git_sha = "legacy_pre_git_sha_recording"
        else:
            split_hash = "no_manifest"
            git_sha = "no_manifest"
        cell_id_hash = _cell_id_align_hash(
            emb / "train_predictions.csv", emb / "validation_predictions.csv", emb / "test_predictions.csv"
        )
        cache_path = str(emb.resolve().relative_to(ROOT.resolve())) if emb.resolve().is_relative_to(ROOT.resolve()) else str(emb)

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

        print(f"\n[{ds} rts={rts_str}] sep={proto.separability_ratio:.3f}")
        for v in VARIANTS:
            res = run_variant(v, proto, val_lat, test_lat, val_pred, test_pred, y_true, rare)
            tag = " (弃权)" if res["abstain"] else ""
            print(f"  {v:22s}: F1={res['rare_f1']:.4f}{tag}  rec={res['rare_recall']:.4f}  "
                  f"prec={res['rare_precision']:.4f}  rank={res['chosen_rank']}  "
                  f"rescued={res['n_rescued']}  false={res['n_false_rescues']}  ffr={res['ffr']:.5f}")
            rows.append({
                "dataset": ds, "rare_class": rare, "seed": seed, "rts": rts_str,
                "variant": v,
                "split_hash": split_hash, "git_sha": git_sha,
                "cell_id_align_hash": cell_id_hash, "cache_path": cache_path,
                **res,
            })

    out_dir = ROOT / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ablation_summary.csv", index=False)
    print(f"\n[saved] {out_dir/'ablation_summary.csv'}")

    # 聚合：按 (dataset, variant)
    agg_rows = []
    for ds in df["dataset"].unique():
        for v in VARIANTS:
            sub = df[(df["dataset"] == ds) & (df["variant"] == v)]
            if sub.empty:
                continue
            agg_rows.append({
                "dataset": ds, "variant": v, "n": len(sub),
                "f1_mean": round(sub["rare_f1"].mean(), 4),
                "recall_mean": round(sub["rare_recall"].mean(), 4),
                "prec_mean": round(sub["rare_precision"].mean(), 4),
                "ffr_max": round(sub["ffr"].max(), 6),
                "gain_mean": round(sub["f1_gain"].mean(), 4),
                "n_abstain": int(sub["abstain"].sum()),
                "n_rescued_total": int(sub["n_rescued"].sum()),
                "n_false_total": int(sub["n_false_rescues"].sum()),
            })
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "ablation_summary_agg.csv", index=False)
    print(f"[saved] {out_dir/'ablation_summary_agg.csv'}")

    # 人读 log
    lines = [
        "# Round 10 Ablation Report",
        "",
        "**Date**: 2026-06-19  |  **Seed**: 42  |  **Datasets**: 6  |  **rts**: 0.01/0.05/0.10/all",
        "",
        "## 变体定义",
        "",
        "| 变体 | 改动 |",
        "|------|------|",
        "| V0 baseline_scanvi   | 完全不 rescue（参照线） |",
        "| V1 no_sep_gate       | 关 separability 安全网（LOW_SEP=0） |",
        "| V2 no_necessity      | 关 necessity 守门 |",
        "| V3 rank1_fixed       | 固定 rank=1，无 val-自适应 |",
        "| V4 rank2_fixed       | 固定 rank=2，无 val-自适应 |",
        "| V5 no_conformal_tau  | 候选直接全 relabel（去 τ） |",
        "| V6 full              | 当前 conformal_rescue（reference） |",
        "",
        "## 聚合表（按数据集 × 变体）",
        "",
        "| dataset | variant | n | F1 mean | recall mean | prec mean | FFR_max | gain mean | n_abstain | rescued_total | false_total |",
        "|---------|---------|---|---------|-------------|-----------|---------|-----------|-----------|---------------|-------------|",
    ]
    for r in agg_rows:
        lines.append(
            f"| {r['dataset']} | {r['variant']} | {r['n']} | {r['f1_mean']:.4f} | "
            f"{r['recall_mean']:.4f} | {r['prec_mean']:.4f} | {r['ffr_max']:.5f} | "
            f"{r['gain_mean']:+.4f} | {r['n_abstain']} | {r['n_rescued_total']} | {r['n_false_total']} |"
        )
    lines += ["", "## 逐配置明细", "",
              "| dataset | rts | variant | sep | F1 | recall | prec | rank | rescued | false | FFR | abstain |",
              "|---------|-----|---------|-----|-----|--------|------|------|---------|-------|-----|---------|"]
    for _, r in df.iterrows():
        ab = "Y" if r["abstain"] else "N"
        lines.append(
            f"| {r['dataset']} | {r['rts']} | {r['variant']} | {r['sep']:.3f} | "
            f"{r['rare_f1']:.4f} | {r['rare_recall']:.4f} | {r['rare_precision']:.4f} | "
            f"{r['chosen_rank']} | {r['n_rescued']} | {r['n_false_rescues']} | {r['ffr']:.5f} | {ab} |"
        )
    (out_dir / "ablation_log.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {out_dir/'ablation_log.md'}")


if __name__ == "__main__":
    main()
