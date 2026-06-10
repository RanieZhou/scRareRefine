"""CellTypist 对比 Baseline 预测模块

使用 CellTypist (Logistic Regression, OvR 或多分类) 算法训练并识别稀有细胞，
作为模型在原始表达量空间上的线性基线分类性能参考。
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# 将项目根目录插入系统搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from src.utils import (
    classification_tables,
    load_config,
    load_adata,
    make_run_dir,
    parse_rare_train_size,
    read_table,
    write_table
)


def _plot(metrics_df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    """ 绘制 CellTypist 对比 Baseline 的分类指标对比图 """
    cols = ["rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    labels = {"baseline": "Baseline\n(scANVI)", "lr": "CellTypist"}
    colors = {"baseline": "#8da0cb", "lr": "#fc8d62"}
    methods = metrics_df["method"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"CellTypist vs Baseline  |  {rare_class}", fontsize=10, fontweight="bold")
    for ax, col in zip(axes, cols):
        vals = [float(metrics_df.loc[metrics_df["method"] == m, col].iloc[0])
                if col in metrics_df.columns else 0.0 for m in methods]
        bars = ax.bar(range(len(methods)), vals,
                      color=[colors.get(m, "#aaa") for m in methods],
                      width=0.5, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([labels.get(m, m) for m in methods], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_title(col.replace("_", " "), fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存 CellTypist 对比图至: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CellTypist 对比 Baseline 预测入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, required=True, help="实验随机种子")
    parser.add_argument("--split_mode", default=None, help="三路切分模式")
    parser.add_argument("--rare_class", default=None, help="稀有细胞类别名")
    parser.add_argument("--rare_train_size", required=True, help="训练集稀有细胞显式标注规模")
    args = parser.parse_args()

    try:
        import celltypist
    except ImportError:
        raise ImportError("未检测到 celltypist 库，请先使用 pip install celltypist 进行安装")

    # 检查 scikit-learn 版本兼容性
    import sklearn
    from packaging.version import Version
    if Version(sklearn.__version__) >= Version("1.7"):
        import importlib
        _ct_mod = importlib.import_module("celltypist.train")
        _train_file = Path(_ct_mod.__file__).read_text()
        if "multi_class" in _train_file:
            raise RuntimeError(
                f"当前安装的 celltypist (1.7.1及以下) 与 scikit-learn {sklearn.__version__} 存在冲突。 "
                "LogisticRegression 的 multi_class='ovr' 参数在 sklearn 1.7+ 中已被废除。\n"
                "请手动修改 site-packages/celltypist/train.py，删除其中 LogisticRegression 调用的 multi_class='ovr' 参数。"
            )

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    
    split_mode = args.split_mode if args.split_mode is not None else config.get("experiment", {}).get("split_mode", "batch_heldout")

    run_dir = make_run_dir(config, split_mode, args.seed, rare_class, rare_train_size)
    out_dir = run_dir / "celltypist"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 解析训练集标注的细胞子集
    assignments_path = run_dir / "split_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"未找到 Stage 2 模型输出的划分分配文件: {assignments_path}，请先运行主流程。")
    assignments = read_table(assignments_path)
    train_asgn = assignments[assignments["split"] == "train"]

    # 抽取对应规模的稀有类标注细胞
    labeled_rare = train_asgn[
        train_asgn["is_labeled_for_scanvi"].astype(str).isin(["True", "1", "true"]) &
        (train_asgn["scanvi_label"] == rare_class)
    ]
    if rare_train_size != "all":
        if isinstance(rare_train_size, float):
            n_rare = max(5, int(rare_train_size * len(labeled_rare)))
        else:
            n_rare = int(rare_train_size)
        if len(labeled_rare) > n_rare:
            labeled_rare = labeled_rare.sample(n_rare, random_state=args.seed)

    labeled_major = train_asgn[
        train_asgn["is_labeled_for_scanvi"].astype(str).isin(["True", "1", "true"]) &
        (train_asgn["scanvi_label"] != rare_class)
    ]
    labeled_ids = set(pd.concat([labeled_major["cell_id"], labeled_rare["cell_id"]]).astype(str))

    # 2. 载入原始表达数据并预归一化
    print("正在加载 AnnData 原始表达数据...")
    adata_full = load_adata(config)
    adata_full.obs_names = adata_full.obs_names.astype(str)

    # 过滤到与模型一致的高变基因 (HVG) 空间
    hvg_path = run_dir / "selected_hvg_genes.csv"
    if hvg_path.exists():
        hvg_df = pd.read_csv(hvg_path)
        hvgs = [g for g in hvg_df["gene"].tolist() if g in adata_full.var_names]
        adata_full = adata_full[:, hvgs].copy()
        print(f"  [高变基因] 成功加载 {len(hvgs)} 个 HVG 特征基因。")
    else:
        print("  [Warning] 未检测到 HVG 基因文件，使用全基因空间进行计算。")

    print("正在进行库大小标准化与 Log1p 连续值转换...")
    sc.pp.normalize_total(adata_full, target_sum=1e4)
    sc.pp.log1p(adata_full)

    # 3. 切分出训练集与测试集子集
    test_pred_path = run_dir / "embeddings" / "test_predictions.csv"
    if not test_pred_path.exists():
        raise FileNotFoundError(f"未找到 Stage 2 模型输出的测试预测文件: {test_pred_path}")
    test_meta = read_table(test_pred_path)
    test_ids = test_meta["cell_id"].astype(str).tolist()

    obs_id_set = set(adata_full.obs_names.tolist())
    train_ids_list = [i for i in labeled_ids if i in obs_id_set]
    test_ids_filtered = [i for i in test_ids if i in obs_id_set]

    adata_train = adata_full[train_ids_list].copy()
    adata_test  = adata_full[test_ids_filtered].copy()

    # CellTypist 底层对稀疏矩阵不支持，需转化为稠密数组
    if sp.issparse(adata_train.X):
        adata_train.X = np.asarray(adata_train.X.todense())
    if sp.issparse(adata_test.X):
        adata_test.X = np.asarray(adata_test.X.todense())

    train_labels = (
        assignments.set_index("cell_id")["scanvi_label"]
        .reindex(train_ids_list)
        .fillna("Unknown")
        .astype(str)
    )
    adata_train.obs["celltypist_label"] = train_labels.values

    print(f"  训练细胞数 : {len(adata_train)} (稀有类: {(train_labels == rare_class).sum()})")
    print(f"  测试细胞数 : {len(adata_test)}")

    # 4. 训练 CellTypist 模型
    print("开始训练 CellTypist 分类模型 (C=1.0, max_iter=200)...")
    new_model = celltypist.train(
        adata_train,
        labels="celltypist_label",
        check_expression=False,  # 我们已提前预处理，跳过检查
        C=1.0,
        max_iter=200,
        n_jobs=4,
    )
    print("  训练完毕。")

    # 5. 推理 (采用 best match 概率最大的类，保证稀有类预测不受 leiden Over-clustering 吞噬)
    print("正在对测试集细胞进行 annotate (best match)...")
    result = celltypist.annotate(
        adata_test,
        model=new_model,
        mode="best match",
    )
    pred_labels = result.predicted_labels["predicted_labels"].astype(str).values

    true_labels = (
        test_meta.set_index("cell_id")["true_label"]
        .reindex(test_ids_filtered)
        .astype(str)
        .values
    )

    # 6. 保存与计算分类性能
    pred_df = pd.DataFrame({
        "cell_id": test_ids_filtered,
        "true_label": true_labels,
        "predicted_label": pred_labels,
    })
    write_table(pred_df, out_dir / "test_predictions.csv")

    lr_metrics, _ = classification_tables(true_labels, pred_labels, rare_class=rare_class)

    baseline_true = (test_meta.set_index("cell_id")["true_label"]
                     .reindex(test_ids_filtered).astype(str).values)
    baseline_pred_vals = (test_meta.set_index("cell_id")["predicted_label"]
                          .reindex(test_ids_filtered).astype(str).values)
    baseline_metrics, _ = classification_tables(baseline_true, baseline_pred_vals, rare_class=rare_class)

    metrics_df = pd.DataFrame([
        {"method": "baseline", **baseline_metrics},
        {"method": "lr", **lr_metrics},
    ])
    write_table(metrics_df, out_dir / "test_metrics.csv")
    _plot(metrics_df, out_dir / "comparison.png", rare_class=rare_class)

    print("\n" + "="*50)
    print(f"  [CellTypist 评估结果] (稀有类: {rare_class})")
    print("-"*50)
    print(f"  scANVI Baseline F1 : {baseline_metrics['rare_f1']:.4f}")
    print(f"  CellTypist LR F1   : {lr_metrics['rare_f1']:.4f}")
    print("="*50 + "\n")
    print(f"  [成功] CellTypist 对比实验完成，输出保存在: {out_dir}")


if __name__ == "__main__":
    main()
