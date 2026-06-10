"""k-NN 对比 Baseline 预测模块

使用低维表示空间中已知标签的训练集细胞作为参考，
通过 Euclidean 距离的 k 最近邻多数投票预测测试集细胞标签，并进行性能对比。
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录插入系统搜索路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.utils import (
    classification_tables,
    load_config,
    make_run_dir,
    parse_rare_train_size,
    read_table,
    write_table
)


def _plot(metrics_df: pd.DataFrame, out_path: Path, *, rare_class: str) -> None:
    """ 绘制 kNN 对比 Baseline 的分类指标对比图 """
    cols = ["rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    labels = {"baseline": "Baseline\n(scANVI)"}
    colors = {"baseline": "#8da0cb"}
    for m in metrics_df["method"]:
        if m != "baseline":
            labels[m] = m.replace("_", " ")
            colors[m] = "#66c2a5"

    methods = metrics_df["method"].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    fig.suptitle(f"kNN vs Baseline  |  {rare_class}", fontsize=10, fontweight="bold")
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
    print(f"  [可视化] 已保存 kNN 对比图至: {out_path}")


def knn_predict(
    query_latent: np.ndarray,
    *,
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    reference_is_labeled: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    """ 在低维潜在表示空间中利用 k-NN 进行细胞标签预测 """
    labeled = np.asarray(reference_is_labeled, dtype=bool)
    ref = np.asarray(reference_latent, dtype=float)[labeled]
    labs = pd.Series(reference_labels).astype(str).to_numpy()[labeled]

    clf = KNeighborsClassifier(n_neighbors=min(k, len(ref)), metric="euclidean", n_jobs=1)
    clf.fit(ref, labs)
    return clf.predict(np.asarray(query_latent, dtype=float))


def main() -> None:
    parser = argparse.ArgumentParser(description="kNN 对比 Baseline 预测入口")
    parser.add_argument("--config", required=True, help="YAML 配置文件路径")
    parser.add_argument("--seed", type=int, required=True, help="实验随机种子")
    parser.add_argument("--split_mode", default=None, help="三路切分模式")
    parser.add_argument("--rare_class", default=None, help="稀有细胞类别名")
    parser.add_argument("--rare_train_size", required=True, help="训练集稀有细胞显式标注规模")
    parser.add_argument("--k", type=int, default=15, help="kNN 邻居数量 k (默认: 15)")
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    
    split_mode = args.split_mode if args.split_mode is not None else config.get("experiment", {}).get("split_mode", "batch_heldout")
    
    # 拼装唯一的输出目录
    run_dir = make_run_dir(config, split_mode, args.seed, rare_class, rare_train_size)
    emb_dir = run_dir / "embeddings"
    knn_dir = run_dir / "knn"

    # 读取 scANVI 训练的低维 latent 与监督信息
    train_pred = read_table(emb_dir / "train_predictions.csv")
    train_latent = read_table(emb_dir / "train_latent.csv")
    latent_cols = [c for c in train_latent.columns if c.startswith("latent_")]
    ref_lat = train_latent[latent_cols].to_numpy(dtype=float)

    # 针对测试集进行 kNN 推理
    for split_name in ["test"]:
        pred = read_table(emb_dir / f"{split_name}_predictions.csv")
        latent = read_table(emb_dir / f"{split_name}_latent.csv")
        query_lat = latent[latent_cols].to_numpy(dtype=float)

        knn_preds = knn_predict(
            query_lat,
            reference_latent=ref_lat,
            reference_labels=train_pred["true_label"],
            reference_is_labeled=train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy(),
            k=args.k,
        )

        out_pred = pd.DataFrame({
            "cell_id": pred["cell_id"].astype(str) if "cell_id" in pred.columns else np.arange(len(pred)),
            "true_label": pred["true_label"].astype(str),
            "baseline_predicted": pred["predicted_label"].astype(str),
            "knn_predicted": knn_preds,
        })

        knn_metrics, _ = classification_tables(
            out_pred["true_label"], out_pred["knn_predicted"], rare_class=rare_class
        )
        baseline_metrics, _ = classification_tables(
            out_pred["true_label"], out_pred["baseline_predicted"], rare_class=rare_class
        )
        metrics_df = pd.DataFrame([
            {"method": "baseline", **baseline_metrics},
            {"method": f"knn_k{args.k}", "k": args.k, **knn_metrics},
        ])

        write_table(out_pred, knn_dir / f"{split_name}_predictions.csv")
        write_table(metrics_df, knn_dir / f"{split_name}_metrics.csv")
        _plot(metrics_df, knn_dir / "comparison.png", rare_class=rare_class)

        tp = ((out_pred["true_label"] == rare_class) & (out_pred["knn_predicted"] == rare_class)).sum()
        fn = ((out_pred["true_label"] == rare_class) & (out_pred["knn_predicted"] != rare_class)).sum()
        fp = ((out_pred["true_label"] != rare_class) & (out_pred["knn_predicted"] == rare_class)).sum()
        
        print("\n" + "="*50)
        print(f"  [k-NN 评估结果] (k={args.k} | 稀有类: {rare_class})")
        print("-"*50)
        print(f"  scANVI Baseline F1 : {baseline_metrics.get('rare_f1', 0):.4f}")
        print(f"  k-NN Classifier F1 : {knn_metrics.get('rare_f1', 0):.4f}")
        print(f"  稀有类预测统计      : TP={tp} | FN={fn} | FP={fp}")
        print("="*50 + "\n")

    print(f"  [成功] kNN 对比实验完成，输出保存在: {knn_dir}")


if __name__ == "__main__":
    main()
