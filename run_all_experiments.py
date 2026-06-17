"""Run all comparison experiments across datasets, seeds, and train sizes.

一键执行所有数据集、种子和已知标注规模的对比实验。
流水线依次执行：
    1. scANVI Baseline（输出 embeddings 供其他对比基线消费）
    2. k-NN Baseline
    3. CellTypist Baseline（自动检测，若未安装则自动跳过）
    4. scBalance Baseline（自动检测，若未安装则自动跳过）
    5. scRareRefine 核心一键管线 (Prototype Gating / Gate+Marker / Fusion)

并在运行结束后自动聚合所有结果，输出漂亮的科学性能对比 Markdown 报表。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

# 将项目根目录插入系统搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_config, parse_rare_train_size, make_run_dir

# ==============================================================================
# 【全局填空区】 请在此处配置您想要运行的实验数据集矩阵和随机种子
# ==============================================================================
EXPERIMENTS = [
    # (配置文件, 稀有细胞类型, 标注规模列表)
    # immune_dc 已有完整结果，如需重跑取消注释
    # ("configs/immune_dc.yaml", "ASDC", ["0.01", "0.05", "0.1", "all"]),
    ("configs/pancreas_baron.yaml",         "gamma",                                  ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_lung_endo.yaml",       "endothelial cell of lymphatic vessel",   ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_lung_stroma.yaml",     "bronchial smooth muscle cell",           ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_small_intestine.yaml", "intestinal tuft cell",                   ["0.01", "0.05", "0.1", "all"]),
    ("configs/tabula_sapiens_stomach.yaml", "mast cell",                              ["0.01", "0.05", "0.1", "all"]),
]
SEEDS = [42, 43, 44]
# ==============================================================================

Job = tuple[int, str, str, str, int]


def build_jobs() -> list[Job]:
    """ 拼装所有实验超参组合任务 """
    jobs: list[Job] = []
    index = 0
    for cfg, rare_class, rts_list in EXPERIMENTS:
        for rts in rts_list:
            for seed in SEEDS:
                index += 1
                jobs.append((index, cfg, rare_class, rts, seed))
    return jobs


def select_jobs(
    jobs: list[Job],
    *,
    start_at: int = 1,
    end_at: int | None = None,
) -> list[Job]:
    """ 截取指定索引范围内的子任务 """
    if start_at < 1:
        raise ValueError("--start_at 必须大等于 1")
    if end_at is not None and end_at < start_at:
        raise ValueError("--end_at 必须大等于 --start_at")
    return [
        job for job in jobs
        if job[0] >= start_at and (end_at is None or job[0] <= end_at)
    ]


def check_library_installed(py_executable: str, lib_name: str) -> bool:
    """ 在指定的 Python 运行环境中静默检查某个包是否已安装 """
    try:
        res = subprocess.run(
            [py_executable, "-c", f"import {lib_name}"],
            capture_output=True,
            text=True
        )
        return res.returncode == 0
    except Exception:
        return False


def collect_and_print_summary(jobs: list[Job], project_root: Path) -> None:
    """ 遍历并收集所有实验的指标数据，融合成大表并输出 Markdown 对比统计 """
    print("\n" + "=" * 80)
    print("      【全实验数据自动化聚合与分析中】")
    print("=" * 80)
    
    records = []
    for _, cfg_file, rare_class, rts, seed in jobs:
        try:
            config = load_config(project_root / cfg_file)
            dataset_name = config["dataset"]["name"]
            parsed_rts = parse_rare_train_size(rts)
            
            split_mode = config.get("experiment", {}).get("split_mode", "batch_heldout")
            # 定位输出目录
            run_dir = make_run_dir(config, split_mode, seed, rare_class, parsed_rts)
            run_dir_abs = project_root / run_dir
            
            if not run_dir_abs.exists():
                continue
                
            # 1. 读取主流程指标 (baseline, gate_only, gate_marker, fusion)
            main_csv = run_dir_abs / "metrics" / "final_metrics.csv"
            if main_csv.exists():
                df = pd.read_csv(main_csv)
                for _, row in df.iterrows():
                    records.append({
                        "dataset": dataset_name,
                        "rare_class": rare_class,
                        "rare_train_size": str(parsed_rts),
                        "seed": seed,
                        "method": row["method"],
                        "overall_accuracy": row.get("overall_accuracy", np.nan),
                        "macro_f1": row.get("macro_f1", np.nan),
                        "rare_f1": row.get("rare_f1", np.nan),
                        "rare_precision": row.get("rare_precision", np.nan),
                        "rare_recall": row.get("rare_recall", np.nan),
                    })
            
            # 2. 读取 knn 指标
            knn_csv = run_dir_abs / "knn" / "test_metrics.csv"
            if knn_csv.exists():
                df = pd.read_csv(knn_csv)
                # 剔除里面的 baseline 重复行，只保留 knn 行
                df = df[df["method"] != "baseline"]
                for _, row in df.iterrows():
                    records.append({
                        "dataset": dataset_name,
                        "rare_class": rare_class,
                        "rare_train_size": str(parsed_rts),
                        "seed": seed,
                        "method": row["method"],
                        "overall_accuracy": row.get("overall_accuracy", np.nan),
                        "macro_f1": row.get("macro_f1", np.nan),
                        "rare_f1": row.get("rare_f1", np.nan),
                        "rare_precision": row.get("rare_precision", np.nan),
                        "rare_recall": row.get("rare_recall", np.nan),
                    })
                    
            # 3. 读取 celltypist 指标
            ct_csv = run_dir_abs / "celltypist" / "test_metrics.csv"
            if ct_csv.exists():
                df = pd.read_csv(ct_csv)
                df = df[df["method"] != "baseline"]
                for _, row in df.iterrows():
                    records.append({
                        "dataset": dataset_name,
                        "rare_class": rare_class,
                        "rare_train_size": str(parsed_rts),
                        "seed": seed,
                        "method": "celltypist",
                        "overall_accuracy": row.get("overall_accuracy", np.nan),
                        "macro_f1": row.get("macro_f1", np.nan),
                        "rare_f1": row.get("rare_f1", np.nan),
                        "rare_precision": row.get("rare_precision", np.nan),
                        "rare_recall": row.get("rare_recall", np.nan),
                    })
                    
            # 4. 读取 scbalance 指标
            sb_csv = run_dir_abs / "scbalance" / "test_metrics.csv"
            if sb_csv.exists():
                df = pd.read_csv(sb_csv)
                df = df[df["method"] != "baseline"]
                for _, row in df.iterrows():
                    records.append({
                        "dataset": dataset_name,
                        "rare_class": rare_class,
                        "rare_train_size": str(parsed_rts),
                        "seed": seed,
                        "method": "scbalance",
                        "overall_accuracy": row.get("overall_accuracy", np.nan),
                        "macro_f1": row.get("macro_f1", np.nan),
                        "rare_f1": row.get("rare_f1", np.nan),
                        "rare_precision": row.get("rare_precision", np.nan),
                        "rare_recall": row.get("rare_recall", np.nan),
                    })
        except Exception as e:
            print(f"  [Warning] 读取指标出错 {cfg_file}: {e}")
            continue

    if not records:
        print("未检测到任何可用的实验指标文件。请确认实验已经运行成功。")
        return

    # 汇总输出
    summary_df = pd.DataFrame(records)
    
    # 将汇总大表存盘
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(results_dir / "all_experiments_summary.csv", index=False)
    print(f"-> 成功保存所有实验的合并指标表格至: {results_dir / 'all_experiments_summary.csv'}")

    # 聚合均值（对 Seed 求均值）展示对比
    grouped = summary_df.groupby(["dataset", "rare_class", "rare_train_size", "method"])["rare_f1"].mean().reset_index()
    
    # 格式化方法名称
    method_mapping = {
        "baseline": "scANVI Baseline",
        "scRareRefine": "scRareRefine (Ours)",
        "celltypist": "CellTypist",
        "scbalance": "scBalance"
    }
    grouped["method"] = grouped["method"].map(lambda m: method_mapping.get(m, str(m).replace("knn_k15", "k-NN")))

    # 转换为宽表，展示不同方法在 F1-Score 上的表现对比
    pivot_df = grouped.pivot(index=["dataset", "rare_class", "rare_train_size"], columns="method", values="rare_f1")

    # 重新排序列，确保 baseline 第一，scRareRefine 最后，其他居中
    all_cols = pivot_df.columns.tolist()
    pref_order = ["scANVI Baseline", "k-NN", "CellTypist", "scBalance", "scRareRefine (Ours)"]
    final_cols = [c for c in pref_order if c in all_cols] + [c for c in all_cols if c not in pref_order]
    pivot_df = pivot_df[final_cols]
    
    print("\n>>> 【学术评估报表：各方法在稀有类识别上的平均 F1-Score 表现】")
    print(pivot_df.round(4).to_markdown())
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键批量执行所有对比基准与项目算法实验")
    parser.add_argument("--dry_run", action="store_true", help="只打印命令不执行具体程序")
    parser.add_argument(
        "--start_at",
        type=int,
        default=1,
        help="从第几个任务索引开始执行 (1-based)",
    )
    parser.add_argument(
        "--end_at",
        type=int,
        default=None,
        help="在第几个任务索引停止执行 (inclusive)",
    )
    args = parser.parse_args()

    py = sys.executable
    project_root = Path(__file__).resolve().parent

    # 检查库安装状态，以柔性控制对比实验
    has_celltypist = check_library_installed(py, "celltypist")
    has_scbalance = check_library_installed(py, "scBalance")

    all_jobs = build_jobs()
    total = len(all_jobs)
    jobs = select_jobs(all_jobs, start_at=args.start_at, end_at=args.end_at)
    failed: list[str] = []

    if not jobs:
        print(f"没有任务被选中（总任务数: {total}）。")
        return

    print("\n" + "="*80)
    print(f"      【scRareRefine 批量对比实验总控台】")
    print(f"  - 总共组合任务数 : {total}")
    print(f"  - 选中执行任务数 : {len(jobs)}")
    print(f"  - 依赖环境检测   : CellTypist(已安装={has_celltypist}) | scBalance(已安装={has_scbalance})")
    print("="*80 + "\n")

    for index, cfg, rare_class, rts, seed in jobs:
        label = f"[{index}/{total}] {cfg} | {rare_class} | rts={rts} | seed={seed}"
        print(f"\n{'#' * 80}")
        print(f"  正在执行组合: {label}")
        print(f"{'#' * 80}")

        # 解析 split_mode 并透传
        try:
            config = load_config(project_root / cfg)
            split_mode = config.get("experiment", {}).get("split_mode", "batch_heldout")
        except Exception:
            split_mode = "batch_heldout"

        # 构建子实验命令组
        sub_steps = []
        
        # 1. 运行 scANVI 基准模型，产生后续对比依赖的嵌入
        sub_steps.append((
            "scANVI Baseline",
            [py, "baseline/scanvi/scanvi_baseline.py", "--config", cfg, "--seed", str(seed), "--rare_class", rare_class, "--rare_train_size", str(rts), "--split_mode", split_mode]
        ))
        
        # 2. 运行 k-NN 基准
        sub_steps.append((
            "k-NN Baseline",
            [py, "baseline/knn/knn_baseline.py", "--config", cfg, "--seed", str(seed), "--rare_class", rare_class, "--rare_train_size", str(rts), "--split_mode", split_mode]
        ))
        
        # 3. 运行 CellTypist 基准 (仅在库安装时)
        if has_celltypist:
            sub_steps.append((
                "CellTypist Baseline",
                [py, "baseline/celltypist/celltypist_baseline.py", "--config", cfg, "--seed", str(seed), "--rare_class", rare_class, "--rare_train_size", str(rts), "--split_mode", split_mode]
            ))
            
        # 4. 运行 scBalance 基准 (仅在库安装时)
        if has_scbalance:
            sub_steps.append((
                "scBalance Baseline",
                [py, "baseline/scbalance/scbalance_baseline.py", "--config", cfg, "--seed", str(seed), "--rare_class", rare_class, "--rare_train_size", str(rts), "--split_mode", split_mode]
            ))
            
        # 5. 运行项目一键管线 (Prototype Gating, Gate+Marker, Fusion)
        sub_steps.append((
            "scRareRefine Pipeline",
            [py, "run_pipeline.py", "--config", cfg, "--seed", str(seed), "--rare_class", rare_class, "--rare_train_size", str(rts), "--split_mode", split_mode]
        ))

        # 执行命令组
        job_failed = False
        t_start = time.time()
        
        for step_name, cmd in sub_steps:
            print(f"\n-> [步骤] 启动 {step_name} ...")
            print(f"   命令: {' '.join(cmd)}")
            
            if args.dry_run:
                continue
                
            res = subprocess.run(cmd, cwd=project_root)
            if res.returncode != 0:
                print(f"  *** 错误: {step_name} 执行失败 (退出码: {res.returncode}) ***")
                job_failed = True
                # 主流程失败则中止当前任务的后续对比步骤
                if step_name in ["scANVI Baseline", "scRareRefine Pipeline"]:
                    break

        t_elapsed = time.time() - t_start
        if job_failed:
            print(f"  *** 任务 {label} 发生失败 - 继续下一个组合 ***")
            failed.append(label)
        elif args.dry_run:
            print(f"  [Dry-Run] 任务组合已模拟跳过，无实际执行。")
        else:
            print(f"  [完成] 组合已成功跑通，耗时: {t_elapsed / 60:.1f} 分钟")

    # 6. 数据自动汇总及对比展示
    if not args.dry_run:
        collect_and_print_summary(jobs, project_root)

    print("\n" + "=" * 80)
    if args.dry_run:
        print("批量 Dry-Run 模拟检查完成，无实际任务运行。")
    else:
        print(f"批量任务完成。共尝试执行 {len(jobs)} 个参数组合。")
        if failed:
            print(f"以下任务组合发生失败 ({len(failed)} 个):")
            for f in failed:
                print(f"  - {f}")
        else:
            print("所有选中的实验任务组合均完美成功运行！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
