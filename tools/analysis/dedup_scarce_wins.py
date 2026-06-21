"""稀缺区 win-most 计数去重（审查勘误 2026-06-20）。

背景（见 results/experiment_log.md 第七轮 line 501 已披露的机制）：
  scANVI 半监督标注数 = max(5, int(rts × 训练池稀有数))（src/model.py:make_scanvi_labels）。
  对训练池稀有数较小的数据集，多个名义 rts 会塌缩到同一个标注数（同 seed → 同样的 5 个
  细胞 → 同一份 scANVI 嵌入 → 逐位相同的对比结果）。例如：
    - tabula_sapiens_stomach（train 仅 52 mast）：rts=0.01/0.05/0.10 全部 = 5
    - pancreas_baron（train 106 gamma）：rts=0.01/0.05 都 = 5
  因此第九轮「标注稀缺区 5 数据集 × 3 比例 = 15 格 15/15」按名义 rts 计数会重复计入塌缩格。

本脚本：
  - 读 results/comparison/comparison_summary.csv（status==ok）
  - 为每个 (dataset, rts) 由 outputs/.../train_predictions.csv 计算实际标注稀有数 n_labeled_rare
  - 仅取稀缺区 rts ∈ {0.01, 0.05, 0.10}
  - 按 (dataset, n_labeled_rare) 去重为 distinct 实验，重算 scRareRefine 的 win-most / best 计数
  - 输出 results/comparison/scarce_region_distinct.csv

用法：
  D:/setup/anaconda/envs/scanvi311/python.exe tools/analysis/dedup_scarce_wins.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402

SUMMARY_CSV = ROOT / "results" / "comparison" / "comparison_summary.csv"
OUT_CSV = ROOT / "results" / "comparison" / "scarce_region_distinct.csv"
SCARCE_RTS = ["0.01", "0.05", "0.10"]
OUR = "scRareRefine"

# config 路径与 dataset 名映射（用于定位 outputs run_dir）
CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_small_intestine.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/pancreas_integrated.yaml",
]


def _labeled_rare_count(cfg_path: str, seed: int, rts_str: str) -> int | None:
    """从缓存 train_predictions.csv 读实际被标注的稀有细胞数（is_labeled & true==rare）。"""
    cfg = load_config(cfg_path)
    exp = cfg.get("experiment", {})
    rare = exp.get("rare_class")
    sm = exp.get("split_mode", "batch_heldout")
    run_dir = make_run_dir(cfg, sm, seed, rare, parse_rare_train_size(rts_str))
    tp = run_dir / "embeddings" / "train_predictions.csv"
    if not tp.exists():
        return None
    df = pd.read_csv(tp, usecols=["true_label", "is_labeled_for_scanvi"])
    isl = df["is_labeled_for_scanvi"].astype(bool)
    return int((isl & (df["true_label"].astype(str) == str(rare))).sum())


def main():
    df = pd.read_csv(SUMMARY_CSV, dtype={"rare_train_size": str})
    df = df[df["status"] == "ok"].copy()

    name2cfg = {load_config(c)["dataset"]["name"]: c for c in CONFIGS}

    # 标注实际 labeled_rare
    key2lab: dict[tuple, int] = {}
    for ds in df["dataset"].unique():
        if ds not in name2cfg:
            continue
        for rts in df[df["dataset"] == ds]["rare_train_size"].unique():
            for seed in df[(df["dataset"] == ds) & (df["rare_train_size"] == rts)]["seed"].dropna().unique():
                n = _labeled_rare_count(name2cfg[ds], int(seed), str(rts))
                if n is not None:
                    key2lab[(ds, str(rts), int(seed))] = n

    df["n_labeled_rare"] = df.apply(
        lambda r: key2lab.get((r["dataset"], str(r["rare_train_size"]), int(r["seed"])), np.nan), axis=1)

    scarce = df[df["rare_train_size"].astype(str).isin(SCARCE_RTS)].copy()

    rows = []
    seen_distinct: set[tuple] = set()
    nominal_cells = 0
    for (ds, seed), g in scarce.groupby(["dataset", "seed"]):
        if OUR not in set(g["method"]):
            continue
        # 按 (n_labeled_rare) 分组：同一标注数 = 同一 distinct 实验
        for n_lab, gg in g.groupby("n_labeled_rare"):
            rts_collapsed = sorted(gg[gg["method"] == OUR]["rare_train_size"].astype(str).unique().tolist())
            nominal_cells += len(rts_collapsed)
            distinct_key = (ds, int(seed), int(n_lab) if pd.notna(n_lab) else -1)
            if distinct_key in seen_distinct:
                continue
            seen_distinct.add(distinct_key)

            # 取代表 rts（最小）下各方法 f1
            rep_rts = rts_collapsed[0]
            cell = gg[gg["rare_train_size"].astype(str) == rep_rts]
            our_f1 = float(cell[cell["method"] == OUR]["rare_f1"].iloc[0])
            others = cell[cell["method"] != OUR][["method", "rare_f1"]]
            n_others = len(others)
            n_beat = int((others["rare_f1"] < our_f1 - 1e-9).sum())
            n_tie = int((others["rare_f1"].sub(our_f1).abs() <= 1e-9).sum())
            win_most = n_beat > n_others / 2.0
            is_best = our_f1 >= float(others["rare_f1"].max()) - 1e-9 if n_others else True

            rows.append({
                "dataset": ds, "seed": int(seed),
                "n_labeled_rare": int(n_lab) if pd.notna(n_lab) else None,
                "rts_collapsed": "|".join(rts_collapsed),
                "n_rts_collapsed": len(rts_collapsed),
                "our_f1": round(our_f1, 4),
                "n_others": n_others, "n_beat": n_beat, "n_tie": n_tie,
                "win_most": bool(win_most), "is_best": bool(is_best),
            })

    out = pd.DataFrame(rows).sort_values(["dataset", "seed", "n_labeled_rare"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    n_distinct = len(out)
    n_winmost = int(out["win_most"].sum())
    n_best = int(out["is_best"].sum())
    print(out.to_string(index=False))
    print("\n================ 稀缺区计数勘误 ================")
    print(f"名义格数（按 rts 计，含塌缩重复）: {nominal_cells}")
    print(f"distinct 实验数（按实际标注数去重）: {n_distinct}")
    print(f"  其中 win-most（胜过过半对比方法）: {n_winmost}/{n_distinct}")
    print(f"  其中 best（F1 第一）            : {n_best}/{n_distinct}")
    print(f"塌缩说明：")
    for _, r in out[out["n_rts_collapsed"] > 1].iterrows():
        print(f"  - {r['dataset']} 标注数={r['n_labeled_rare']}: rts {r['rts_collapsed']} 为同一实验（计 1 格）")
    print(f"\n[saved] {OUT_CSV}")


if __name__ == "__main__":
    main()
