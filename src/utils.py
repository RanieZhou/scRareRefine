from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report


# ==========================================
# 1. 配置文件加载与解析
# ==========================================
def load_config(path: str | Path) -> dict[str, Any]:
    """ 从 YAML 文件加载并返回配置字典 """
    with Path(path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件解析错误，应为映射字典: {path}")
    return config


def load_adata(config: dict[str, Any]) -> ad.AnnData:
    """ 依据配置字典中的 use_layer 或 use_raw 配置加载 h5ad 数据 """
    dataset = config["dataset"]
    adata = ad.read_h5ad(dataset["path"])
    use_layer = dataset.get("use_layer")
    
    if use_layer:
        if use_layer not in adata.layers:
            raise ValueError(f"指定的 Layer '{use_layer}' 不存在。可用图层: {list(adata.layers.keys())}")
        return ad.AnnData(X=adata.layers[use_layer].copy(), obs=adata.obs.copy(), var=adata.var.copy())
        
    if dataset.get("use_raw", False):
        if adata.raw is None:
            raise ValueError("配置指定加载 raw.X 但 adata.raw 不存在。")
        return ad.AnnData(X=adata.raw.X.copy(), obs=adata.obs.copy(), var=adata.raw.var.copy(), uns=adata.uns.copy())
        
    return adata


# ==========================================
# 2. IO 表格读写小工具
# ==========================================
def write_table(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """ 安全创建父级目录并将 DataFrame 保存至 CSV 文件 """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """ 读取 CSV 文件为 DataFrame，若文件不存在则抛出异常 """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"未找到目标文件: {path}")
    return pd.read_csv(path)


# ==========================================
# 3. 命名与输出路径生成
# ==========================================
def safe_class_name(name: str) -> str:
    """ 将类型名称中的特殊字符（空格、加号、斜杠）转为安全的下划线与小写 """
    return name.replace("+", "pos").replace(" ", "_").replace("/", "_").lower()


def parse_rare_train_size(value: str | int | float) -> int | float | str:
    """ 统一解析稀有细胞训练大小：支持 float 比例、int 绝对数量与 'all' 字符 """
    if isinstance(value, float):
        return value
    if isinstance(value, int):
        return value
    s = str(value).strip().lower()
    if s == "all":
        return "all"
    if s.endswith("pct"):
        return int(s[:-3]) / 100.0
    try:
        f = float(s)
        if 0 < f <= 1 and "." in s:
            return f
        return int(float(s))
    except ValueError:
        raise ValueError(f"无法解析的稀有类标注规格: {value!r}")


def _rts_label(rare_train_size: float | int | str) -> str:
    """ 生成标注大小对应的规范后缀字符串 """
    if isinstance(rare_train_size, float):
        return f"{round(rare_train_size * 100)}pct"
    return str(rare_train_size)


def make_run_id(split_mode: str, seed: int, rare_class: str, rare_train_size: float | int | str) -> str:
    """ 拼接出唯一的运行任务 ID 标识 """
    return f"{split_mode}_seed{seed}_{safe_class_name(rare_class)}_rare{_rts_label(rare_train_size)}"


def make_run_dir(config: dict[str, Any], split_mode: str, seed: int, rare_class: str, rare_train_size: float | int | str) -> Path:
    """ 计算对应的输出工作根目录 """
    dataset_name = config["dataset"]["name"]
    run_id = make_run_id(split_mode, seed, rare_class, rare_train_size)
    return Path("outputs") / dataset_name / run_id


def make_split_path(config: dict[str, Any], split_mode: str, seed: int) -> Path:
    """ 依据配置计算样本三路切分结果存储路径 """
    dataset_name = config["dataset"]["name"]
    return Path("data") / "splits" / dataset_name / f"{split_mode}_seed{seed}" / "split.csv"


# ==========================================
# 3b. 缓存 provenance（manifest）：run_pipeline.py / train_cache.py /
#     run_scrarerefine_comparison.py 共用同一份实现，避免各自维护一份后产生漂移
# ==========================================
def git_sha() -> str:
    """ 当前代码版本短 SHA（不在 git 仓库或 git 不可用时返回 unknown） """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def compute_split_hash(predictions_dict: dict[str, pd.DataFrame]) -> str:
    """ 对 train/val/test 的 cell_id split 取稳定哈希（排序后），用于缓存 provenance 校验 """
    h = hashlib.sha256()
    for s in ["train", "validation", "test"]:
        ids = sorted(predictions_dict[s]["cell_id"].astype(str).tolist())
        h.update(s.encode())
        h.update("\n".join(ids).encode())
    return h.hexdigest()[:16]


def compute_cached_split_hash(run_dir: str | Path) -> str | None:
    """Compute the train/validation/test cell-id hash from cached predictions."""
    run_dir = Path(run_dir)
    emb_dir = run_dir / "embeddings"
    predictions: dict[str, pd.DataFrame] = {}
    for split in ["train", "validation", "test"]:
        path = emb_dir / f"{split}_predictions.csv"
        if not path.exists():
            return None
        predictions[split] = pd.read_csv(path, usecols=["cell_id"])
    return compute_split_hash(predictions)


def build_manifest(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    label_column: str,
    batch_key: str,
    split_mode: str,
    seed: int,
    rare_class: str,
    rare_train_size: float | int | str,
    predictions_dict: dict[str, pd.DataFrame],
    n_train: int,
    n_val: int,
    n_test: int,
) -> dict[str, Any]:
    """ 构建 provenance manifest（实验参数 + split 哈希 + 代码版本），供缓存复用前校验 """
    return {
        "config": str(config_path),
        "dataset": config["dataset"]["name"],
        "dataset_path": config["dataset"]["path"],
        "label_key": label_column,
        "batch_key": batch_key,
        "split_mode": split_mode,
        "seed": seed,
        "rare_class": rare_class,
        "rare_train_size": str(rare_train_size),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "split_hash": compute_split_hash(predictions_dict),
        "git_sha": git_sha(),
    }


def check_manifest(
    run_dir: Path,
    config: dict[str, Any],
    *,
    seed: int,
    rare_class: str,
    rare_train_size: float | int | str,
    label_column: str | None = None,
    batch_key: str | None = None,
    split_mode: str | None = None,
    validate_split_hash: bool = True,
    strict_git_sha: bool = False,
) -> bool:
    """ 校验 run_dir/manifest.json 与当前实验参数是否一致。

    缺失 manifest（旧缓存，未来都会补写）时放行并打印警告；
    manifest 存在但任意字段不匹配当前 config/seed/rare_train_size 等时拒绝（返回 False）。
    """
    mf = run_dir / "manifest.json"
    if not mf.exists():
        print("  [provenance] WARNING: 无 manifest.json（旧缓存，无法校验 split/代码版本）")
        return True
    m = json.loads(mf.read_text(encoding="utf-8"))
    exp = config.get("experiment", {})
    checks = {
        "dataset_path": config["dataset"]["path"],
        "label_key": label_column or config["dataset"].get("label_key"),
        "batch_key": batch_key or config["dataset"].get("batch_key"),
        "split_mode": split_mode or exp.get("split_mode", "batch_heldout"),
        "rare_class": rare_class,
        "seed": seed,
        "rare_train_size": str(rare_train_size),
    }
    mism = [(k, m.get(k), v) for k, v in checks.items() if str(m.get(k)) != str(v)]
    if validate_split_hash:
        cached_split_hash = compute_cached_split_hash(run_dir)
        manifest_split_hash = m.get("split_hash")
        if cached_split_hash is None:
            mism.append(("split_hash", manifest_split_hash, "cached prediction files missing"))
        elif str(manifest_split_hash) != str(cached_split_hash):
            mism.append(("split_hash", manifest_split_hash, cached_split_hash))
    if mism:
        print(f"  [provenance] ERROR: manifest 与当前配置不匹配: {mism}")
        return False
    current_git = git_sha()
    manifest_git = str(m.get("git_sha", ""))
    git_note = ""
    if manifest_git in ("", "unknown", "None"):
        git_note = "  git_sha=legacy/unknown"
        if strict_git_sha:
            print("  [provenance] ERROR: manifest git_sha missing or unknown")
            return False
    elif current_git != "unknown" and manifest_git != current_git:
        git_note = f"  git_sha differs manifest={manifest_git} current={current_git}"
        if strict_git_sha:
            print(f"  [provenance] ERROR:{git_note}")
            return False

    print(f"  [provenance] OK  split_hash={m.get('split_hash')}  git_sha={m.get('git_sha')}{git_note}")
    return True


# ==========================================
# 4. 指标计算与不确定性度量
# ==========================================
def classification_tables(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    rare_class: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """ 计算分类总体指标并提取稀有类型的精准率、召回率与 F1 指标 """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0,
    )
    per_class = pd.DataFrame(
        {"label": labels, "precision": precision, "recall": recall, "f1": f1, "support": support}
    )
    
    rare_row = per_class[per_class["label"] == rare_class]
    if rare_row.empty:
        rare_precision = rare_recall = rare_f1 = 0.0
    else:
        rare_precision = float(rare_row["precision"].iloc[0])
        rare_recall = float(rare_row["recall"].iloc[0])
        rare_f1 = float(rare_row["f1"].iloc[0])
        
    metrics = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rare_precision": rare_precision,
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
    }
    return metrics, per_class


def compute_uncertainty(probabilities: pd.DataFrame, *, rare_class: str) -> pd.DataFrame:
    """ 基于类别 Softmax 概率矩阵，计算最大置信度、信息熵以及分类 Margin (差值) """
    probs = probabilities.astype(float)
    arr = probs.to_numpy()
    classes = probs.columns.to_numpy()
    
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if arr.shape[1] > 1 else order[:, 0]
    
    top1 = arr[np.arange(arr.shape[0]), top1_idx]
    top2 = arr[np.arange(arr.shape[0]), top2_idx]
    entropy = -(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum(axis=1)
    
    return pd.DataFrame(
        {
            "max_prob": top1,
            "entropy": entropy,
            "margin": top1 - top2,
            "top1_label": classes[top1_idx],
            "top2_label": classes[top2_idx],
            f"top2_is_{rare_class}": classes[top2_idx] == rare_class,
        },
        index=probabilities.index,
    )


# ==========================================
# 5. 生信计算：CPM 对数归一化
# ==========================================
def log1p_cpm(x: Any) -> np.ndarray:
    """ 稀疏与稠密兼容的 CPM 对数转换：log(1 + CPM) """
    if sparse.issparse(x):
        row_sum = np.asarray(x.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        normalized = x.multiply(10000.0 / row_sum[:, None])
        return np.log1p(normalized.toarray()).astype(np.float32)

    arr = np.asarray(x, dtype=np.float32)
    row_sum = arr.sum(axis=1)
    row_sum[row_sum == 0] = 1.0
    return np.log1p(arr * (10000.0 / row_sum[:, None])).astype(np.float32)


def load_expression_subset(adata, cell_ids: list[str], genes: list[str]) -> np.ndarray:
    """ 依指定细胞 ID 与基因列表，从 AnnData 提取子集并计算 log1p CPM 表达值（HVG 顺序对齐）。 """
    idx = adata.obs_names.isin(cell_ids)
    sub = adata[idx]
    id_pos = {cid: i for i, cid in enumerate(sub.obs_names)}
    ordered = [id_pos[c] for c in cell_ids if c in id_pos]
    sub = sub[ordered]
    var_idx = [sub.var_names.get_loc(g) for g in genes if g in sub.var_names]
    X = sub.X
    if sparse.issparse(X):
        X = X.toarray()
    return log1p_cpm(np.asarray(X, dtype=np.float32)[:, var_idx])


# ==========================================
# 6. 随机种子初始化
# ==========================================
def seed_everything(seed: int) -> None:
    """ 保证实验与 scvi 初始化结果可复现 """
    import random
    import torch
    import scvi
    scvi.settings.seed = seed
    random.seed(seed)          # 部分依赖（如 umap / 第三方采样）会用 Python 内置 random
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# 7. 资源占用监测器
# ==========================================
@dataclass
class ResourceMonitor:
    """ 自动测算并记录代码优化循环中的 Wall-Time 与峰值物理内存占用 (Peak RSS) """
    sample_interval_seconds: float = 1.0
    _start_time: float = field(init=False, default=0.0)
    _end_time: float = field(init=False, default=0.0)
    _peak_rss_bytes: int = field(init=False, default=0)
    _stop: threading.Event = field(init=False, default_factory=threading.Event)
    _thread: threading.Thread | None = field(init=False, default=None)

    def __enter__(self) -> "ResourceMonitor":
        self._start_time = time.perf_counter()
        self._end_time = 0.0
        self._peak_rss_bytes = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._sample_once()
        self._end_time = time.perf_counter()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.1, self.sample_interval_seconds * 2))

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.sample_interval_seconds)

    def _sample_once(self) -> None:
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.Error:
                continue
        self._peak_rss_bytes = max(self._peak_rss_bytes, int(rss))

    def summary(self) -> dict[str, float]:
        end = self._end_time or time.perf_counter()
        return {
            "wall_time_seconds": float(end - self._start_time),
            "peak_rss_mb": float(self._peak_rss_bytes / (1024 * 1024)),
        }


# ==========================================
# 8. 科研报表输出与可视化画图
# ==========================================
def print_classification_report(y_true, y_pred, rare_class: str):
    """ 打印高内聚的 Markdown 分类性能评估报告，并特别高亮稀有类型 """
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    
    report = classification_report(y_true, y_pred, zero_division=0)
    print("\n" + "="*50)
    print(f"      【分类性能评估报告】 (稀有类别: {rare_class})")
    print("="*50)
    print(report)
    
    # 打印简易表格聚焦
    metrics, _ = classification_tables(y_true, y_pred, rare_class=rare_class)
    print("-"*50)
    print(f"  稀有类 F1-Score:  {metrics['rare_f1']:.4f}")
    print(f"  稀有类 Recall:     {metrics['rare_recall']:.4f}")
    print(f"  稀有类 Precision:  {metrics['rare_precision']:.4f}")
    print(f"  总体预测准确率:    {metrics['overall_accuracy']:.4f}")
    print("="*50 + "\n")


def plot_marker_violin(adata, label_column: str, marker_genes: list[str], out_path: str | Path, rare_class: str = None):
    """ 绘制所选特定差异表达 Marker 基因的小提琴分布图，展现细胞表达特异性 """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 限制最大基因画图个数，防止排版拥挤
    genes_to_plot = [g for g in marker_genes if g in adata.var_names][:6]
    if not genes_to_plot:
        print("  [Warning] 没有有效基因在 adata.var_names 中，跳过小提琴图绘制。")
        return
        
    num_genes = len(genes_to_plot)
    fig, axes = plt.subplots(1, num_genes, figsize=(3 * num_genes, 4.5), sharey=True)
    if num_genes == 1:
        axes = [axes]
        
    labels = adata.obs[label_column].astype(str)
    unique_labels = sorted(labels.unique())
    
    # 类别映射与颜色分配
    colors_palette = ["#fc8d62" if (rare_class and c == rare_class) else "#8da0cb" for c in unique_labels]
    
    for ax, gene in zip(axes, genes_to_plot):
        gene_idx = adata.var_names.get_loc(gene)
        X_data = adata.X
        if sparse.issparse(X_data):
            expr = np.asarray(X_data[:, gene_idx].toarray()).ravel()
        else:
            expr = np.asarray(X_data[:, gene_idx]).ravel()
            
        # 提取各个细胞群的数据列表
        group_data = [expr[labels == c] for c in unique_labels]
        
        # 优先导入 seaborn 绘制更漂亮的小提琴图，无 seaborn 则使用 matplotlib 替代
        try:
            import seaborn as sns
            sns.violinplot(
                x=labels, y=expr, hue=labels, legend=False,
                palette={c: colors_palette[i] for i, c in enumerate(unique_labels)},
                ax=ax, density_norm="width"
            )
        except ImportError:
            parts = ax.violinplot(group_data, showmeans=True, showextrema=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors_palette[i])
                pc.set_alpha(0.7)
                
        ax.set_title(gene, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Normalized Expression" if ax == axes[0] else "")
        ax.set_xticklabels(unique_labels, rotation=45, ha="right", fontsize=9)
        
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存 Marker 表达量小提琴图至: {out_path}")


def plot_method_comparison(metrics_df: pd.DataFrame, out_path: str | Path, rare_class: str) -> None:
    """ 绘制不同后处理策略与 Baseline 的评估对比柱状图 """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 定义标准对比方法顺序、配色与标签
    method_order = ["baseline", "scRareRefine"]
    method_labels = {
        "baseline": "scANVI Baseline",
        "scRareRefine": "scRareRefine (Ours)",
    }
    method_colors = {
        "baseline": "#8da0cb",
        "scRareRefine": "#ffd92f",
    }
    
    present_methods = [m for m in method_order if m in metrics_df["method"].values]
    if not present_methods:
        return
        
    df = metrics_df.set_index("method").loc[present_methods].reset_index()
    
    fig, axes = plt.subplots(1, 4, figsize=(13, 4))
    fig.suptitle(
        f"Performance Comparison  |  Rare Class: {rare_class}  |  Seed: {df['seed'].iloc[0]}",
        fontsize=11, fontweight="bold", y=1.02
    )
    
    metrics = [
        ("rare_f1", "Rare F1-Score", "F1"),
        ("rare_recall", "Rare Recall", "Recall"),
        ("rare_precision", "Rare Precision", "Precision"),
        ("overall_accuracy", "Overall Accuracy", "Accuracy")
    ]
    
    for ax, (col, title, ylabel) in zip(axes, metrics):
        vals = [float(df.loc[df["method"] == m, col].iloc[0]) if col in df.columns else 0.0 for m in present_methods]
        colors = [method_colors[m] for m in present_methods]
        
        bars = ax.bar(range(len(present_methods)), vals, color=colors, width=0.55, edgecolor="white")
        
        # 柱状图顶部标数
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )
            
        ax.set_xticks(range(len(present_methods)))
        ax.set_xticklabels([method_labels.get(m, m) for m in present_methods], rotation=25, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存性能对比柱状图至: {out_path}")


def plot_rescue_effect(metrics_df: pd.DataFrame, out_path: str | Path, rare_class: str) -> None:
    """ 绘制后处理校正所得的实际拯救细胞数与误拯救数对比柱状图 """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 提取后处理校正方法
    rescue_methods = [m for m in metrics_df["method"].tolist() if m != "baseline"]
    if not rescue_methods:
        return
        
    df = metrics_df[metrics_df["method"].isin(rescue_methods)].set_index("method").reindex(rescue_methods).reset_index()
    
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle(f"Post-hoc Rescue Impact  |  Rare Class: {rare_class}", fontsize=11, fontweight="bold")
    
    # 拯救数指标
    metrics = [
        ("n_rescued", "Cells Rescued", "Number of Cells"),
        ("n_false_rescues", "False Rescues", "Number of Cells"),
        ("major_to_rare_false_rescue_rate", "False Rescue Rate (%)", "%")
    ]
    
    # 颜色卡
    method_colors = {
        "scRareRefine": "#ffd92f",
    }
    
    for ax, (col, title, ylabel) in zip(axes, metrics):
        if col == "major_to_rare_false_rescue_rate":
            vals = [float(df.loc[df["method"] == m, col].iloc[0]) * 100 if col in df.columns else 0.0 for m in rescue_methods]
            fmt = ".3f"
        else:
            vals = [float(df.loc[df["method"] == m, col].iloc[0]) if col in df.columns else 0.0 for m in rescue_methods]
            fmt = ".0f"
            
        colors = [method_colors.get(m, "#aaa") for m in rescue_methods]
        bars = ax.bar(range(len(rescue_methods)), vals, color=colors, width=0.5, edgecolor="white")
        
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max(vals) * 0.02 if max(vals) > 0 else 0.01),
                f"{val:{fmt}}", ha="center", va="bottom", fontsize=8, fontweight="bold"
            )
            
        ax.set_xticks(range(len(rescue_methods)))
        ax.set_xticklabels([m.replace("_", " ").title() for m in rescue_methods], rotation=15, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        if vals and max(vals) > 0:
            ax.set_ylim(0, max(vals) * 1.2)
        else:
            ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [可视化] 已保存拯救效果指标图至: {out_path}")
