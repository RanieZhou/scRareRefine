"""消融实验：逐组件验证 conformal 方案各部分的贡献。

4 个变体（累加组件，每一步只改一处）：
  V1 no_rank1     : 全部 predicted≠rare 为候选 + 各向异性 score + conformal τ
  V2 rank1_nofilter: rank-1 候选 + 直接全救（无 score/τ 过滤）
  V3 isotropic    : rank-1 + 各向同性 softmax score + conformal τ
  V4 full         : rank-1 + 各向异性 score + conformal τ （完整方法）

弃权逻辑：
  V1/V3/V4 使用 conformal 弃权阈值 sep < 1.3（rescue 有效下限）
  V2        使用全局弃权阈值 sep < 1.1（极低可分）

结果保存：
  results/ablation/ablation_summary.csv   （机读，每行一个 run × variant）
  results/ablation/ablation_log.md        （人读，含均值±σ 汇总表）

用法：
    python tools/ablation.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, make_run_dir, parse_rare_train_size, classification_tables
from src.rescue import PrototypeRescuer, ConformalRescuer

# ── 待评估的 run 列表（与 evaluate_all.py 保持一致）──────────────────────────
RUNS = [
    ("configs/immune_dc.yaml",        42, "0.05"),
    ("configs/immune_dc.yaml",        43, "0.05"),
    ("configs/immune_dc.yaml",        44, "0.05"),
    ("configs/pancreas_baron.yaml",   42, "0.10"),
    ("configs/pancreas_baron.yaml",   43, "0.10"),
    ("configs/pancreas_baron.yaml",   44, "0.10"),
    ("configs/tabula_lung_endo.yaml", 42, "0.10"),
    ("configs/tabula_lung_endo.yaml", 43, "0.10"),
    ("configs/tabula_lung_endo.yaml", 44, "0.10"),
]

CONFORMAL_LOW_SEP = 1.3  # conformal 弃权阈值（与 rescue.py 一致）
GATE_LOW_SEP      = 1.1  # rank-1 直接拯救弃权阈值
CONFORMAL_ALPHA   = 0.01 # 发表级 FFR 上界（与 rescue.py 默认值一致）


def _lat(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def _isotropic_membership(proto_rescuer: PrototypeRescuer, query_latent: np.ndarray) -> np.ndarray:
    """各向同性隶属度：softmax_c(-d_c)[rare]，不按类内半径归一化（V3 消融对照）。"""
    classes = proto_rescuer.classes
    P = np.vstack([proto_rescuer.prototypes[c] for c in classes])
    d = np.sqrt(((query_latent[:, None, :] - P[None]) ** 2).sum(2))
    logits = -d
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)
    return p[:, classes.index(proto_rescuer.rare_class)]


def run_variant(
    variant: str,
    proto_rescuer: PrototypeRescuer,
    val_lat: np.ndarray,
    test_lat: np.ndarray,
    val_pred: pd.DataFrame,
    test_pred: pd.DataFrame,
    y_true: np.ndarray,
    rare_class: str,
) -> dict:
    """运行单个消融变体，返回指标 dict。"""
    sep = proto_rescuer.separability_ratio
    base_pred = test_pred["predicted_label"].astype(str)
    val_true  = val_pred["true_label"].astype(str)

    final_pred = base_pred.copy()
    abstain = False

    # ── V1: 无 rank-1 约束 ──────────────────────────────────────────────────
    if variant == "v1_no_rank1":
        if sep < CONFORMAL_LOW_SEP:
            abstain = True
        else:
            # 候选：所有 predicted≠rare
            test_cand = base_pred.ne(rare_class).to_numpy()
            val_cand_all = val_pred["predicted_label"].astype(str).ne(rare_class).to_numpy()
            # 评分：各向异性（与 V4 相同）
            test_score = proto_rescuer.rare_membership_score(test_lat)
            val_score  = proto_rescuer.rare_membership_score(val_lat)
            # 阈值：conformal（在 val 全部非稀有上校准）
            conf = ConformalRescuer(rare_class, alpha=CONFORMAL_ALPHA)
            conf.calibrate(val_score, val_true)
            final_pred = conf.relabel(base_pred, test_cand, test_score)

    # ── V2: rank-1 直接全救（无 score / τ 过滤）────────────────────────────
    elif variant == "v2_rank1_nofilter":
        if sep < GATE_LOW_SEP:
            abstain = True
        else:
            test_cand = proto_rescuer.isotropic_rank1(test_lat, base_pred)
            final_pred = base_pred.copy()
            final_pred.iloc[np.where(test_cand)[0]] = rare_class

    # ── V3: rank-1 + 各向同性 score + conformal τ ───────────────────────────
    elif variant == "v3_isotropic":
        if sep < CONFORMAL_LOW_SEP:
            abstain = True
        else:
            test_cand = proto_rescuer.isotropic_rank1(test_lat, base_pred)
            test_score = _isotropic_membership(proto_rescuer, test_lat)
            val_score  = _isotropic_membership(proto_rescuer, val_lat)
            conf = ConformalRescuer(rare_class, alpha=CONFORMAL_ALPHA)
            conf.calibrate(val_score, val_true)
            final_pred = conf.relabel(base_pred, test_cand, test_score)

    # ── V4: 完整 conformal（rank-1 + 各向异性 + conformal τ）────────────────
    elif variant == "v4_full":
        if sep < CONFORMAL_LOW_SEP:
            abstain = True
        else:
            test_cand  = proto_rescuer.isotropic_rank1(test_lat, base_pred)
            test_score = proto_rescuer.rare_membership_score(test_lat)
            val_score  = proto_rescuer.rare_membership_score(val_lat)
            conf = ConformalRescuer(rare_class, alpha=CONFORMAL_ALPHA)
            conf.calibrate(val_score, val_true)
            final_pred = conf.relabel(base_pred, test_cand, test_score)

    else:
        raise ValueError(f"未知变体: {variant}")

    # ── 统计指标 ──────────────────────────────────────────────────────────────
    fp = final_pred.astype(str).to_numpy()
    n_rescued      = int(((fp != base_pred.to_numpy()) & (fp == rare_class)).sum())
    n_false        = int(((fp != base_pred.to_numpy()) & (fp == rare_class) & (y_true != rare_class)).sum())
    n_nonrare      = int((y_true != rare_class).sum())
    m, _           = classification_tables(y_true, fp, rare_class=rare_class)
    bl, _          = classification_tables(y_true, base_pred.to_numpy(), rare_class=rare_class)
    return {
        "sep":             round(sep, 4),
        "abstain":         abstain,
        "baseline_f1":     round(bl["rare_f1"],    4),
        "rare_f1":         round(m["rare_f1"],      4),
        "rare_recall":     round(m["rare_recall"],  4),
        "rare_precision":  round(m["rare_precision"], 4),
        "n_rescued":       n_rescued,
        "n_false_rescues": n_false,
        "ffr":             round(n_false / max(n_nonrare, 1), 6),
        "f1_gain":         round(m["rare_f1"] - bl["rare_f1"], 4),
    }


# ── 主循环 ─────────────────────────────────────────────────────────────────────
VARIANTS = ["v1_no_rank1", "v2_rank1_nofilter", "v3_isotropic", "v4_full"]

rows = []
for cfg_path, seed, rts_str in RUNS:
    config    = load_config(cfg_path)
    exp       = config.get("experiment", {})
    rare_class = exp.get("rare_class")
    split_mode = exp.get("split_mode", "batch_heldout")
    size       = parse_rare_train_size(rts_str)
    run_dir    = make_run_dir(config, split_mode, seed, rare_class, size)
    emb_dir    = run_dir / "embeddings"
    dataset    = config["dataset"]["name"]

    if not (emb_dir / "test_latent.csv").exists():
        print(f"[SKIP] {run_dir} 缓存不存在")
        continue

    splits = ["train", "validation", "test"]
    preds  = {s: pd.read_csv(emb_dir / f"{s}_predictions.csv") for s in splits}
    lats   = {s: pd.read_csv(emb_dir / f"{s}_latent.csv")      for s in splits}

    train_lat    = _lat(lats["train"])
    ref_labels   = preds["train"]["true_label"]
    ref_is_lab   = preds["train"]["is_labeled_for_scanvi"].astype(bool).to_numpy()

    proto = PrototypeRescuer(rare_class)
    proto.fit(train_lat, ref_labels, ref_is_lab)

    y_true = preds["test"]["true_label"].astype(str).to_numpy()
    print(f"\n[{dataset} seed={seed}] sep={proto.separability_ratio:.3f}")

    for v in VARIANTS:
        res = run_variant(v, proto, _lat(lats["validation"]), _lat(lats["test"]),
                          preds["validation"], preds["test"], y_true, rare_class)
        tag = "(弃权)" if res["abstain"] else ""
        print(f"  {v}: F1={res['rare_f1']:.4f}{tag}  recall={res['rare_recall']:.4f}  "
              f"prec={res['rare_precision']:.4f}  rescued={res['n_rescued']}  "
              f"false={res['n_false_rescues']}  ffr={res['ffr']:.5f}")
        rows.append({
            "dataset": dataset, "seed": seed, "rare_train_size": rts_str,
            "rare_class": rare_class, "variant": v, **res,
        })

# ── 保存 CSV ──────────────────────────────────────────────────────────────────
out_dir = Path("results/ablation")
out_dir.mkdir(exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(out_dir / "ablation_summary.csv", index=False)
print(f"\n[saved] {out_dir}/ablation_summary.csv")

# ── 生成汇总表（3-seed 均值±σ）────────────────────────────────────────────────
print("\n=== 3-seed 均值 ± σ（rare_f1）===")
summary_rows = []
for dataset in df["dataset"].unique():
    for v in VARIANTS:
        sub = df[(df["dataset"] == dataset) & (df["variant"] == v)]
        f1s = sub["rare_f1"].to_numpy()
        ffrs = sub["ffr"].to_numpy()
        gains = sub["f1_gain"].to_numpy()
        row = {
            "dataset": dataset, "variant": v,
            "f1_mean": round(f1s.mean(), 4), "f1_std": round(f1s.std(), 4),
            "ffr_max": round(ffrs.max(), 6),
            "gain_mean": round(gains.mean(), 4),
        }
        summary_rows.append(row)
        print(f"  {dataset:25s} {v:20s}: F1={row['f1_mean']:.4f}±{row['f1_std']:.4f}  "
              f"gain={row['gain_mean']:+.4f}  FFR_max={row['ffr_max']:.5f}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out_dir / "ablation_summary_agg.csv", index=False)

# ── 写 Markdown 报告 ──────────────────────────────────────────────────────────
md_lines = [
    "# 消融实验报告（scRareRefine Conformal 方案）",
    "",
    f"实验日期：2026-06-12 | 数据集：3 | seed：42/43/44 | rare_train_size：5%/10%/10%",
    "",
    "## 变体定义",
    "",
    "| 变体 | 候选筛选 | 评分函数 | 阈值校准 | 弃权阈值 |",
    "|------|---------|---------|---------|---------|",
    "| V1 no_rank1 | 全部 predicted≠rare | 各向异性 softmax(-d/r) | conformal (val 非稀有) | sep < 1.3 |",
    "| V2 rank1_nofilter | 各向同性 rank=1 | 无（全救） | 无 | sep < 1.1 |",
    "| V3 isotropic | 各向同性 rank=1 | 各向同性 softmax(-d) | conformal (val 非稀有) | sep < 1.3 |",
    "| V4 full（完整方法） | 各向同性 rank=1 | 各向异性 softmax(-d/r) | conformal (val 非稀有) | sep < 1.3 |",
    "",
    "## 3-seed 均值 ± σ 结果",
    "",
    "| 数据集 | 变体 | F1 均值 | F1 σ | 提升 | FFR_max |",
    "|-------|------|--------|------|------|--------|",
]
for r in summary_rows:
    md_lines.append(
        f"| {r['dataset']} | {r['variant']} | {r['f1_mean']:.4f} | "
        f"{r['f1_std']:.4f} | {r['gain_mean']:+.4f} | {r['ffr_max']:.5f} |"
    )

md_lines += [
    "",
    "## 逐 run 明细",
    "",
    "| 数据集 | seed | 变体 | sep | F1 | recall | precision | rescued | false | FFR |",
    "|-------|------|------|-----|-----|--------|-----------|---------|-------|-----|",
]
for _, r in df.iterrows():
    ab = "(弃)" if r["abstain"] else ""
    md_lines.append(
        f"| {r['dataset']} | {r['seed']} | {r['variant']} | {r['sep']:.3f} | "
        f"{r['rare_f1']:.4f}{ab} | {r['rare_recall']:.4f} | {r['rare_precision']:.4f} | "
        f"{r['n_rescued']} | {r['n_false_rescues']} | {r['ffr']:.5f} |"
    )

md_lines.append("")
(out_dir / "ablation_log.md").write_text("\n".join(md_lines), encoding="utf-8")
print(f"[saved] {out_dir}/ablation_log.md")
