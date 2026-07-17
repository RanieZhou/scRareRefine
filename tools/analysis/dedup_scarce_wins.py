"""稀缺区 win-most 计数（3-seed 版，审查勘误 2026-06-20 → 多 seed 更新 2026-06-21）。

背景（experiment_log 第七轮 line 501 已披露）：scANVI 半监督标注数
  = max(5, int(rts × 训练池稀有数))（src/model.py:make_scanvi_labels）。
训练池稀有数小的数据集，多个名义 rts 会塌缩到同一标注数（同 seed → 同样的细胞 →
同一份 scANVI 嵌入 → 逐位相同的对比结果）。**注意 batch_heldout split 随 seed 变化，
故训练池稀有数、塌缩模式可能随 seed 不同**，本脚本按 (dataset, rts, seed) 实算。

本脚本（3-seed）：
  - 读 comparison_summary.csv（status==ok，seed∈{42,43,44}）
  - 为每个 (dataset, rts, seed) 由 train_predictions.csv 算实际标注稀有数 n_labeled_rare
  - 仅取稀缺区 rts ∈ {0.01,0.05,0.10}
  - 每个 (dataset, rts) 用 **3-seed 均值** 判 win-most（scRareRefine 均值胜过过半 baseline 均值）
    / best（均值第一）
  - 标注每个 (dataset) 内 rts 的塌缩情况（按各 seed 的 n_labeled_rare 是否相同）
  - 输出 results/comparison/scarce_region_distinct_8dataset.csv

用法：D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/dedup_scarce_wins.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402

SUMMARY_CSV = ROOT / "results" / "comparison" / "comparison_summary.csv"
OUT_CSV = ROOT / "results" / "comparison" / "scarce_region_distinct_8dataset.csv"
SCARCE = ["0.01", "0.05", "0.10"]
OUR = "scRareRefine"
CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/mouse_lung_tms_10x.yaml",
    "configs/mouse_pancreas_tms_10x.yaml",
]


def _labeled_rare(cfg_path, seed, rts) -> int | None:
    cfg = load_config(cfg_path)
    exp = cfg.get("experiment", {})
    rare = exp.get("rare_class")
    rd = make_run_dir(
        cfg,
        exp.get("split_mode", "batch_heldout"),
        seed,
        rare,
        parse_rare_train_size(rts),
    )
    tp = rd / "embeddings" / "train_predictions.csv"
    if not tp.exists():
        return None
    df = pd.read_csv(tp, usecols=["true_label", "is_labeled_for_scanvi"])
    return int(
        (
            df["is_labeled_for_scanvi"].astype(bool)
            & (df["true_label"].astype(str) == str(rare))
        ).sum()
    )


def main():
    df = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
    df = df[df["status"] == "ok"].copy()
    name2cfg = {load_config(c)["dataset"]["name"]: c for c in CONFIGS}
    seeds = sorted(df["seed"].dropna().astype(int).unique().tolist())

    sc = df[df["rare_train_size"].isin(SCARCE)].copy()
    rows = []
    nominal = 0
    for ds in sorted(sc["dataset"].unique()):
        if ds not in name2cfg:
            continue
        for rts in SCARCE:
            cell = sc[(sc["dataset"] == ds) & (sc["rare_train_size"] == rts)]
            if cell.empty or OUR not in set(cell["method"]):
                continue
            nominal += 1
            # 3-seed 均值（每方法）
            means = cell.groupby("method")["rare_f1"].mean()
            our = float(means.get(OUR, np.nan))
            our_std = float(cell[cell["method"] == OUR]["rare_f1"].std(ddof=0))
            others = means.drop(index=[OUR], errors="ignore")
            n_oth = len(others)
            n_beat = int((others < our - 1e-9).sum())
            n_tie = int((others.sub(our).abs() <= 1e-9).sum())
            win_most = n_beat > n_oth / 2.0
            is_best = our >= float(others.max()) - 1e-9 if n_oth else True
            # 各 seed 标注数（看塌缩）
            labs = {s: _labeled_rare(name2cfg[ds], s, rts) for s in seeds}
            lab_str = "/".join(str(labs[s]) for s in seeds)
            rows.append(
                {
                    "dataset": ds,
                    "rts": rts,
                    "n_seed": cell["seed"].nunique(),
                    "n_labeled_rare(by_seed)": lab_str,
                    "our_f1_mean": round(our, 4),
                    "our_f1_std": round(our_std, 4),
                    "n_others": n_oth,
                    "n_beat": n_beat,
                    "n_tie": n_tie,
                    "win_most": bool(win_most),
                    "is_best": bool(is_best),
                }
            )

    out = pd.DataFrame(rows)
    # 塌缩检测：同一 dataset 内，若两 rts 在所有 seed 上 n_labeled 都相同 → 同一实验
    out["collapse_group"] = ""
    for ds in out["dataset"].unique():
        sub = out[out["dataset"] == ds]
        seen = {}
        for _, r in sub.iterrows():
            key = r["n_labeled_rare(by_seed)"]
            seen.setdefault(key, []).append(r["rts"])
        for key, rtss in seen.items():
            if len(rtss) > 1:
                out.loc[
                    (out["dataset"] == ds) & (out["rts"].isin(rtss)), "collapse_group"
                ] = "|".join(rtss)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # distinct 计数：每个 (dataset, 唯一 n_labeled 模式) 计 1
    distinct_keys = set()
    distinct_rows = []
    for _, r in out.iterrows():
        k = (r["dataset"], r["n_labeled_rare(by_seed)"])
        if k not in distinct_keys:
            distinct_keys.add(k)
            distinct_rows.append(r)
    nd = len(distinct_rows)
    nd_win = sum(1 for r in distinct_rows if r["win_most"])
    nd_best = sum(1 for r in distinct_rows if r["is_best"])

    print(out.to_string(index=False))
    print(f"\n================ 稀缺区 3-seed 计数（seeds={seeds}）================")
    print(f"名义稀缺格（dataset×rts，含塌缩）       : {nominal}")
    print(f"distinct 实验（按 by-seed 标注模式去重） : {nd}")
    print(f"  win-most（3-seed 均值胜过过半 baseline）: {nd_win}/{nd}")
    print(f"  best（3-seed 均值第一）                 : {nd_best}/{nd}")
    coll = out[out["collapse_group"] != ""][
        ["dataset", "rts", "n_labeled_rare(by_seed)", "collapse_group"]
    ].drop_duplicates()
    print("\n塌缩格（同 dataset 内 rts 在所有 seed 标注数都相同 → 同一实验）：")
    print(coll.to_string(index=False) if len(coll) else "  无")
    print(f"\n[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
