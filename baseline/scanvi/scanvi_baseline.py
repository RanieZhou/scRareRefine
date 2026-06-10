"""scANVI Baseline 对比基准模块

基于重构后的 preprocess 与 model 核心模块，一键执行数据划分预处理、
双阶段 VAE 和半监督表示训练、表征提取以及测试集基线预测，保存所有评估必须的嵌入与指标文件。
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# 将项目根目录插入系统搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.preprocess import run_preprocessing
from src.model import run_model_training
from src.utils import (
    load_config,
    load_adata,
    make_run_dir,
    parse_rare_train_size,
    write_table,
    classification_tables,
    ResourceMonitor
)


def make_split_summary(
    obs: pd.DataFrame,
    split: pd.Series,
    *,
    label_key: str,
    batch_key: str,
    rare_class: str,
) -> pd.DataFrame:
    """ 汇总并统计各样本划分（训练集、验证集、测试集）中稀有细胞与批次分布情况 """
    rows = []
    for s in ["train", "validation", "test"]:
        mask = split.eq(s)
        sub = obs[mask]
        true_labels = sub[label_key].astype(str)
        n_cells = len(sub)
        n_rare = (true_labels == rare_class).sum()
        batches = sub[batch_key].astype(str)
        n_batches = batches.nunique()
        rare_batches = batches[true_labels == rare_class].nunique()
        
        row = {
            "split": s,
            "n_cells": n_cells,
            "n_rare": int(n_rare),
            "rare_ratio": round(n_rare / n_cells, 4) if n_cells > 0 else 0.0,
            "n_batches": n_batches,
            "rare_batches": int(rare_batches),
        }
        
        if s == "train" and "is_labeled_for_scanvi" in obs.columns:
            labeled_mask = mask & obs["is_labeled_for_scanvi"].astype(bool)
            labeled_rare = (obs.loc[labeled_mask, label_key].astype(str) == rare_class).sum()
            row["train_labeled_rare"] = int(labeled_rare)
            row["train_unlabeled_rare"] = int(n_rare - labeled_rare)
        else:
            row["train_labeled_rare"] = ""
            row["train_unlabeled_rare"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="scANVI Baseline 基准预测入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, required=True, help="实验随机种子")
    parser.add_argument("--split_mode", default=None, help="三路切分模式")
    parser.add_argument("--rare_class", default=None, help="稀有细胞类别名")
    parser.add_argument("--rare_train_size", required=True, help="训练集稀有细胞显式标注规模")
    parser.add_argument("--scvi_epochs", type=int, default=None, help="指定 scVI 训练 epoch 数")
    parser.add_argument("--scanvi_epochs", type=int, default=None, help="指定 scANVI 训练 epoch 数")
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    label_column = config["dataset"].get("label_key", "label")
    batch_key = config["dataset"].get("batch_key", "batch")
    
    split_mode = args.split_mode if args.split_mode is not None else config.get("experiment", {}).get("split_mode", "batch_heldout")

    # 拼装基准保存路径
    run_dir = make_run_dir(config, split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("      【scANVI Baseline 训练与预测启动】")
    print(f"  - 配置文件 : {args.config}")
    print(f"  - 稀有类型 : {rare_class}")
    print(f"  - 随机种子 : {args.seed}")
    print(f"  - 保存目录 : {run_dir}")
    print("=" * 80 + "\n")

    # 开启硬件及运行时间检测
    with ResourceMonitor(sample_interval_seconds=1.0) as monitor:
        # 1. 载入数据并预处理切分
        print(">>> [Stage 1/2] 载入表达矩阵并完成自适应体检与切分...")
        adata_raw = load_adata(config)
        adata, train_idx, val_idx, test_idx = run_preprocessing(
            adata_raw,
            label_column=label_column,
            batch_key=batch_key,
            split_mode=split_mode,
            seed=args.seed,
            rare_class=rare_class
        )

        # 2. 训练半监督 scANVI 并推理提取表示
        print(">>> [Stage 2/2] 训练无监督 scVI 与半监督 scANVI 并导出潜在表示...")
        scanvi_model, predictions_dict, latents_dict, selected_genes = run_model_training(
            adata,
            train_idx,
            val_idx,
            test_idx,
            label_column=label_column,
            batch_key=batch_key,
            rare_class=rare_class,
            rare_train_size=rare_train_size,
            config=config,
            seed=args.seed,
            scvi_epochs=args.scvi_epochs,
            scanvi_epochs=args.scanvi_epochs
        )

        # 3. 写入表征与预测文件 (train, validation, test)
        for split_name in ["train", "validation", "test"]:
            write_table(predictions_dict[split_name], emb_dir / f"{split_name}_predictions.csv")
            write_table(latents_dict[split_name], emb_dir / f"{split_name}_latent.csv")
            print(f"  [保存] 已保存 {split_name} 预测结果及低维嵌入。")

    # 4. 统计切分汇总与高变基因，以及资源占用写入
    print("\n>>> 正在写入基线配置文件与性能统计大表...")
    
    # 汇总划分分配表
    split_series = pd.Series("none", index=adata.obs_names)
    split_series.iloc[train_idx] = "train"
    split_series.iloc[val_idx] = "validation"
    split_series.iloc[test_idx] = "test"
    
    assignments = adata.obs[[label_column, "scanvi_label", "is_labeled_for_scanvi"]].copy()
    assignments.insert(0, "cell_id", adata.obs_names.astype(str))
    assignments["split"] = split_series.to_numpy()
    write_table(assignments, run_dir / "split_assignments.csv")

    # 生成 split_summary 统计
    split_summary = make_split_summary(
        adata.obs,
        split_series,
        label_key=label_column,
        batch_key=batch_key,
        rare_class=rare_class,
    )
    write_table(split_summary, run_dir / "split_summary.csv")
    print(f"\n  划分样本汇总统计 ({rare_class}):")
    print(split_summary.to_string(index=False))

    # 保存 HVG 列表
    write_table(pd.DataFrame({"gene": selected_genes}), run_dir / "selected_hvg_genes.csv")

    # 保存性能资源汇总
    usage = monitor.summary()
    write_table(
        pd.DataFrame([{
            **usage,
            "seed": args.seed,
            "rare_class": rare_class,
            "rare_train_size": str(rare_train_size),
            "split_mode": split_mode
        }]),
        run_dir / "resource_summary.csv",
    )

    # 计算测试集上的基线分类表现
    y_true_test = predictions_dict["test"]["true_label"].astype(str)
    base_pred_test = predictions_dict["test"]["predicted_label"].astype(str)
    baseline_metrics, _ = classification_tables(y_true_test, base_pred_test, rare_class=rare_class)

    print("\n" + "="*50)
    print(f"  [scANVI Baseline 运行总结] (稀有类: {rare_class})")
    print("-"*50)
    print(f"  测试集 Accuracy    : {baseline_metrics.get('overall_accuracy', 0):.4f}")
    print(f"  测试集 Macro-F1    : {baseline_metrics.get('macro_f1', 0):.4f}")
    print(f"  稀有类 F1-Score    : {baseline_metrics.get('rare_f1', 0):.4f}")
    print(f"  耗时与峰值物理内存 : {usage['wall_time_seconds']:.2f} 秒 | {usage['peak_rss_mb']:.2f} MB")
    print("="*50 + "\n")
    print(f"  [成功] scANVI 基准训练及测试预测已完成，输出保存在: {run_dir}")


if __name__ == "__main__":
    main()
