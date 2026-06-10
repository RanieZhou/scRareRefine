from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.model_selection import train_test_split

# ==========================================
# 辅助小工具：生成防崩溃的分层辅助标签
# ==========================================
def _get_safe_stratify_labels(labels, min_count=5):
    """ 将样本数极少的稀有类别临时合并为 '__minor__'，防御 sklearn 分层崩溃 """
    counts = labels.value_counts()
    rare_classes = counts[counts < min_count].index
    
    strat_labels = labels.copy()
    if len(rare_classes) > 0:
        strat_labels[labels.isin(rare_classes)] = "__minor__"
    return strat_labels.to_numpy()

# ==========================================
# 1. 生信级数据体检状态机
# ==========================================
def check_data_health(adata):
    """ 自动诊断表达量状态，智能决策预处理路径 """
    X = adata.X
    # 针对稀疏与稠密矩阵做底层安全展平抽样
    vals = X.data if sp.issparse(X) else np.asarray(X).ravel()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals != 0]
    
    if vals.size == 0:
        print("   [数据体检] 矩阵非零值为空，默认不处理。")
        return "none"
        
    min_val, max_val = float(np.min(vals)), float(np.max(vals))
    p99 = float(np.percentile(vals, 99))
    is_int = np.allclose(vals % 1, 0, atol=1e-4)
    has_log_flag = "log1p" in adata.uns
    
    # 估计每个细胞的总表达量中位数
    cell_sums = np.asarray(X.sum(axis=1)).ravel()
    median_cell_sum = float(np.median(cell_sums[np.isfinite(cell_sums)])) if cell_sums.size > 0 else 0

    print(f"\n-> [智能数据体检] Max={max_val:.2f} | P99={p99:.2f} | 纯整数={is_int} | 细胞表达量中位数={median_cell_sum:.2f}")

    # 状态机分流逻辑
    if min_val < 0:
        print("   [体检结论] 检测到负值，推断数据做过 Scale 缩放，保持原样。")
        return "none"
    if is_int and (median_cell_sum > 50 or median_cell_sum == 0):
        print("   [体检结论] 确认为原始 Count 计数，执行完整预处理 (Normalize + Log1p)。")
        return "normalize_log1p"
    if 5000 <= median_cell_sum <= 20000 and p99 > 20:
        print("   [体检结论] 确认为已归一化但未开方对数化的连续流，执行 log1p_only。")
        return "log1p_only"
    if max_val <= 25 and p99 <= 10:
        print("   [体检结论] 确认为标准且健康的 log-normalized 表达量空间，跳过预处理。")
        return "none"
    if not has_log_flag and max_val > 25:
        print("   [体检结论] 数值偏大且无对数标记，执行完整预处理 (Normalize + Log1p)。")
        return "normalize_log1p"
        
    print("   [体检结论] 状态不确定，保守保持原样。")
    return "none"

# ==========================================
# 2. 严格的 70/15/15 三路分层流切分器
# ==========================================
def stratified_three_way_split(obs_df, label_column, seed=42):
    """ 序贯二阶分层划分，严格保持与原版 01_split.py 相同的切分顺序与随机种子，确保统计一致性 """
    labels = obs_df[label_column].astype(str)
    indices = np.arange(len(obs_df))
    
    print(f"-> [三路划分] 启动 70/15/15 序贯分层切分 (seed={seed})...")
    
    # 第一步：划分出 70% 的训练集与 30% 的暂留集 (Heldout) (随机种子为 seed)
    strat_1 = _get_safe_stratify_labels(labels, min_count=5)
    try:
        train_idx, heldout_idx = train_test_split(
            indices, train_size=0.70, random_state=seed, stratify=strat_1
        )
    except ValueError:
        train_idx, heldout_idx = train_test_split(
            indices, train_size=0.70, random_state=seed, stratify=None
        )
        print("   [三路划分] [Warning] 第一次切分触发长尾安全阀，降级为普通随机划分。")
    
    # 第二步：从 30% 暂留集切出 50% 的验证集与 50% 的测试集 (随机种子为 seed + 1)
    heldout_labels = labels.iloc[heldout_idx]
    strat_2 = _get_safe_stratify_labels(heldout_labels, min_count=5)
    
    try:
        val_idx, test_idx = train_test_split(
            heldout_idx, train_size=0.50, random_state=seed + 1, stratify=strat_2
        )
        print("   [划分完成] 成功达成标准二级分层抽样对齐。")
    except ValueError:
        val_idx, test_idx = train_test_split(
            heldout_idx, train_size=0.50, random_state=seed + 1, stratify=None
        )
        print("   [划分完成] [Warning] 第二次切分触发长尾安全阀，自动降级为普通随机划分。")
        
    return train_idx, val_idx, test_idx


# ==========================================
# 3. 严格的 70/15/15 批次外泛化切分器 (Batch Heldout)
# ==========================================
def batch_heldout_split(
    obs_df: pd.DataFrame,
    *,
    label_key: str,
    batch_key: str,
    seed: int,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> pd.Series:
    """ 按照批次/供体对样本进行切分，将某些批次的细胞整批分配到验证与测试集，以测试模型的跨批次泛化性能 """
    labels = obs_df[label_key].astype(str)
    batches = obs_df[batch_key].astype(str)
    classes = sorted(labels.unique())
    
    targets = {
        "train": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * train_fraction,
        "validation": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * validation_fraction,
        "test": labels.value_counts().reindex(classes, fill_value=0).to_numpy(dtype=float) * test_fraction,
    }
    split_counts = {name: np.zeros(len(classes), dtype=float) for name in targets}
    batch_counts = pd.crosstab(batches, labels).reindex(columns=classes, fill_value=0)
    batch_counts["_n"] = batch_counts.sum(axis=1)
    
    rng = np.random.default_rng(seed)
    batch_counts["_tie"] = rng.random(len(batch_counts))
    batch_counts = batch_counts.sort_values(["_n", "_tie"], ascending=[False, True])
    ordered_batches = batch_counts.index.to_numpy()
    split_order = ["train", "validation", "test"]

    batch_to_split: dict[str, str] = {}
    for batch in ordered_batches:
        counts = batch_counts.loc[batch, classes].to_numpy(dtype=float)
        scores = []
        for name in split_order:
            new_counts = split_counts[name] + counts
            target = targets[name]
            denom = np.maximum(target, 1.0)
            score = float((((new_counts - target) / denom) ** 2).sum())
            score += float(max(new_counts.sum() - target.sum(), 0.0) / max(target.sum(), 1.0))
            scores.append(score)
        chosen = split_order[int(np.argmin(scores))]
        batch_to_split[str(batch)] = chosen
        split_counts[chosen] += counts

    def _total_score(counts_by_split: dict[str, np.ndarray]) -> float:
        score = 0.0
        for name in split_order:
            target = targets[name]
            denom = np.maximum(target, 1.0)
            new_counts = counts_by_split[name]
            score += float((((new_counts - target) / denom) ** 2).sum())
            score += float(max(new_counts.sum() - target.sum(), 0.0) / max(target.sum(), 1.0))
        return score

    for missing in [name for name in split_order if name not in set(batch_to_split.values())]:
        best_move = None
        for batch, source in batch_to_split.items():
            if source == missing:
                continue
            if sum(assigned == source for assigned in batch_to_split.values()) <= 1:
                continue
            counts = batch_counts.loc[batch, classes].to_numpy(dtype=float)
            proposed = {name: values.copy() for name, values in split_counts.items()}
            proposed[source] -= counts
            proposed[missing] += counts
            candidate = (_total_score(proposed), batch, source, counts)
            if best_move is None or candidate[0] < best_move[0]:
                best_move = candidate
        if best_move is None:
            raise ValueError("无法将至少一个批次分配给每个划分区间")
        _, batch, source, counts = best_move
        batch_to_split[batch] = missing
        split_counts[source] -= counts
        split_counts[missing] += counts

    return batches.map(batch_to_split).astype(str)


# ==========================================
# 4. 顶层扁平流水线主入口
# ==========================================
def run_preprocessing(
    adata_path,
    label_column,
    batch_key=None,
    split_mode="batch_heldout",
    seed=42,
    rare_class=None
):
    """ 数据处理与样本切分模块端到端主入口 """
    print(f"\n====== [scRareRefine 预处理中心] 维度初始化开始 ======")
    
    # 严格校验 split_mode
    valid_modes = {"batch_heldout", "cell_stratified"}
    if split_mode not in valid_modes:
        raise ValueError(f"不合法的切分模式 split_mode='{split_mode}'。仅支持 {list(valid_modes)}")

    # 加载与克隆
    if isinstance(adata_path, (str, Path)):
        adata = sc.read_h5ad(adata_path)
    else:
        adata = adata_path.copy()
    adata.obs_names_make_unique()

    # 过滤极端空细胞亚群 (细胞数少于5个的类不参与训练，避免全线崩溃)
    counts = adata.obs[label_column].value_counts()
    keep_types = counts[counts >= 5].index
    adata = adata[adata.obs[label_column].isin(keep_types)].copy()

    # 运行数据体检与动态归一化通路
    actual_mode = check_data_health(adata)
    if actual_mode == "normalize_log1p":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif actual_mode == "log1p_only":
        sc.pp.log1p(adata)

    # 强力作废过期嵌入，防下游空间错配
    for k in ["X_pca", "connectivities", "distances", "neighbors"]:
        if k in adata.obsm: adata.obsm.pop(k, None)
        if k in adata.obsp: adata.obsp.pop(k, None)
        if k in adata.uns: adata.uns.pop(k, None)
        
    # 清洗 category 残留空幽灵标签
    adata.obs[label_column] = adata.obs[label_column].astype('category').cat.remove_unused_categories()
    print(f"-> [幽灵标签清洗] 列 '{label_column}' 激活状态类别数: {len(adata.obs[label_column].cat.categories)}")

    # 触发表划分
    if split_mode == "batch_heldout" and batch_key is not None:
        unique_batches = adata.obs[batch_key].nunique()
        if unique_batches < 3:
            raise ValueError(
                f"在 batch_heldout 模式下，指定的批次标识 '{batch_key}' 唯一批次数量为 {unique_batches} (< 3)。"
                "无法进行三路跨批次泛化切分！请在 YAML 配置文件中将 split_mode 修改为 cell_stratified。"
            )
            
        print(f"-> [三路划分] 启动 batch_heldout 批次泛化划分 (batch_key={batch_key}, seed={seed})...")
        split_series = batch_heldout_split(adata.obs, label_key=label_column, batch_key=batch_key, seed=seed)
        train_idx = np.flatnonzero(split_series.eq("train").to_numpy())
        val_idx = np.flatnonzero(split_series.eq("validation").to_numpy())
        test_idx = np.flatnonzero(split_series.eq("test").to_numpy())
        
        # 校验稀有细胞在各个区间的 support 数量分布
        if rare_class is not None:
            rare_class_str = str(rare_class)
            train_rare_count = (adata.obs.iloc[train_idx][label_column].astype(str) == rare_class_str).sum()
            val_rare_count = (adata.obs.iloc[val_idx][label_column].astype(str) == rare_class_str).sum()
            test_rare_count = (adata.obs.iloc[test_idx][label_column].astype(str) == rare_class_str).sum()
            
            print(f"   [划分检查] 目标稀有类 '{rare_class}' 数量分布: Train={train_rare_count} | Val={val_rare_count} | Test={test_rare_count}")
            if train_rare_count == 0 or val_rare_count == 0 or test_rare_count == 0:
                raise ValueError(
                    f"在 batch_heldout 模式下，检测到有划分区间内稀有类细胞数（Train={train_rare_count}, Val={val_rare_count}, Test={test_rare_count}）为 0，无法进行学术评估！"
                    "请使用更均匀的数据集划分，或者在 YAML 配置文件中将 split_mode 修改为 cell_stratified。"
                )
        
        # 统计各划分中实际的批次数量
        train_batches = adata.obs.iloc[train_idx][batch_key].nunique()
        val_batches = adata.obs.iloc[val_idx][batch_key].nunique()
        test_batches = adata.obs.iloc[test_idx][batch_key].nunique()
        print(f"   [划分完成] 成功达成批次划分 (Train_Batches={train_batches} | Val_Batches={val_batches} | Test_Batches={test_batches})。")
    else:
        train_idx, val_idx, test_idx = stratified_three_way_split(adata.obs, label_column, seed=seed)
    
    print(f"====== [scRareRefine 预处理中心] 运行完毕 (Train:{len(train_idx)} | Val:{len(val_idx)} | Test:{len(test_idx)}) ======\n")
    return adata, train_idx, val_idx, test_idx