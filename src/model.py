from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import scvi
from scipy import sparse
from scvi import REGISTRY_KEYS
from src.utils import compute_uncertainty, seed_everything

# ==========================================
# 辅助函数 1：基于高方差基因选择 (HVG)
# ==========================================
def select_hvg_genes(train_adata, n_top_genes: int | None) -> list[str]:
    """ 严格保证与 baseline 一致的方差降序 HVG 特征基因选择 """
    if n_top_genes is None or n_top_genes <= 0 or n_top_genes >= train_adata.n_vars:
        return train_adata.var_names.astype(str).tolist()
    
    x = train_adata.X
    if sparse.issparse(x):
        mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = np.asarray(x.multiply(x).mean(axis=0)).ravel()
    else:
        arr = np.asarray(x)
        mean = arr.mean(axis=0)
        mean_sq = (arr * arr).mean(axis=0)
        
    variance = mean_sq - mean * mean
    top_idx = np.argsort(-variance)[:n_top_genes]
    return train_adata.var_names[np.sort(top_idx)].astype(str).tolist()

# ==========================================
# 辅助函数 2：构建半监督学习的 scANVI 标签列
# ==========================================
def make_scanvi_labels(
    obs: pd.DataFrame,
    split_series: pd.Series,
    *,
    label_key: str,
    rare_class: str,
    rare_train_size: int | float | str,
    seed: int,
    unlabeled_category: str,
) -> tuple[pd.Series, np.ndarray]:
    """ 构造训练集半监督掩膜标签：掩盖部分稀有类样本为 'Unknown'，保留其余细胞的监督信号 """
    true_labels = obs[label_key].astype(str)
    labels = pd.Series(unlabeled_category, index=obs.index, dtype=object)
    is_labeled = np.zeros(len(obs), dtype=bool)

    # 1. 对主要细胞类型（Major Cell Types）在训练集里全量赋予真实标签
    train_mask = split_series.eq("train")
    train_major = train_mask & true_labels.ne(rare_class)
    labels.loc[train_major] = true_labels.loc[train_major]
    is_labeled[train_major.to_numpy()] = True

    # 2. 对稀有细胞（Rare Class）在训练集里根据设定比例进行随机有限抽样标注
    rare_train = train_mask & true_labels.eq(rare_class)
    rare_indices = np.flatnonzero(rare_train.to_numpy())
    
    if rare_train_size == "all":
        selected = rare_indices
    else:
        rng = np.random.default_rng(seed)
        if isinstance(rare_train_size, float):
            size = max(5, int(rare_train_size * len(rare_indices)))
        else:
            size = int(rare_train_size)
        selected = rng.choice(rare_indices, size=min(size, len(rare_indices)), replace=False)
        
    labels.iloc[selected] = rare_class
    is_labeled[selected] = True
    return labels.astype(str), is_labeled

# ==========================================
# 辅助函数 3：提取加速硬件配置
# ==========================================
def _train_device_kwargs() -> dict[str, int | str]:
    """ 硬件加速自动检测，支持 MPS """
    if torch.backends.mps.is_available():
        return {"accelerator": "mps", "devices": 1}
    return {}

# ==========================================
# 辅助函数 4：主训练过程 (scVI -> scANVI)
# ==========================================
def train_scanvi(
    train_adata,
    *,
    batch_key: str,
    unlabeled_category: str,
    n_latent: int,
    batch_size: int,
    scvi_epochs: int,
    scanvi_epochs: int,
) -> scvi.model.SCANVI:
    """ 双阶段训练流水线：无监督 scVI 提取基础表征 -> 半监督 scANVI 进行表征微调 """
    device_kwargs = _train_device_kwargs()
    
    # 第一阶段：训练无监督 scVI VAE
    scvi.model.SCVI.setup_anndata(train_adata, batch_key=batch_key, labels_key="scanvi_label")
    vae = scvi.model.SCVI(train_adata, n_latent=n_latent)
    print("      [模型优化] Stage 2.1: 正在训练无监督 scVI 基础模型...")
    vae.train(max_epochs=scvi_epochs, batch_size=batch_size, enable_progress_bar=False, log_every_n_steps=10, **device_kwargs)
    
    # 第二阶段：继承权重，构建半监督 scANVI 并训练
    model = scvi.model.SCANVI.from_scvi_model(vae, unlabeled_category=unlabeled_category, labels_key="scanvi_label")
    print("      [模型优化] Stage 2.2: 正在训练半监督 scANVI 目标模型...")
    model.train(max_epochs=scanvi_epochs, batch_size=batch_size, enable_progress_bar=False, log_every_n_steps=10, **device_kwargs)
    
    return model

# ==========================================
# 辅助函数 5：测试集推理与不确定性指标获取
# ==========================================
def prediction_outputs(
    model: scvi.model.SCANVI,
    adata,
    *,
    label_key: str,
    rare_class: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ 提取模型的潜在空间表示，并推断各细胞预测标签、Softmax概率及不确定性特征 """
    pred = model.predict(adata)
    soft = model.predict(adata, soft=True)
    if isinstance(soft, tuple):
        soft = soft[0]
        
    # 读取分类名称并封装概率 DataFrame
    manager = getattr(model, "adata_manager", None)
    categories = None
    if manager is not None:
        state_registry = manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        categories = getattr(state_registry, "categorical_mapping", None)
    label_cats = [str(c) for c in categories] if categories is not None else None
    
    probabilities = soft.copy() if isinstance(soft, pd.DataFrame) else pd.DataFrame(soft, columns=label_cats)
    probabilities.index = adata.obs_names
    
    # 计算信息熵与 Margin 等不确定度指标
    uncertainty = compute_uncertainty(probabilities, rare_class=rare_class)
    latent = model.get_latent_representation(adata)

    # 封装输出 DataFrame
    predictions = adata.obs.copy()
    predictions["cell_id"] = adata.obs_names
    predictions["true_label"] = adata.obs[label_key].astype(str).to_numpy()
    predictions["predicted_label"] = np.asarray(pred).astype(str)
    predictions = predictions.reset_index(drop=True)
    predictions = pd.concat(
        [predictions, uncertainty.reset_index(drop=True), probabilities.reset_index(drop=True).add_prefix("prob_")],
        axis=1,
    )
    
    latent_df = pd.DataFrame(latent, columns=[f"latent_{i}" for i in range(latent.shape[1])])
    latent_df.insert(0, "cell_id", adata.obs_names.to_numpy())
    return predictions, latent_df

# ==========================================
# 辅助函数 6：针对 Query (验证/测试) 数据集加载模型
# ==========================================
def load_query_model(
    query_adata,
    model: scvi.model.SCANVI,
    *,
    unlabeled_category: str,
    label_categories: list[str],
) -> scvi.model.SCANVI:
    """ 以 Inductive 模式加载 Query 数据（避免发生数据泄露），将验证/测试集细胞映射到模型空间 """
    query = query_adata.copy()
    categories = list(dict.fromkeys([*label_categories, unlabeled_category]))
    query.obs["scanvi_label"] = pd.Categorical([unlabeled_category] * query.n_obs, categories=categories)
    query.obs["is_labeled_for_scanvi"] = False
    
    query_model = scvi.model.SCANVI.load_query_data(query, model)
    query_model.is_trained_ = True
    return query_model

# ==========================================
# 7. 顶层端到端半监督模型训练与推理流水线主入口
# ==========================================
def run_model_training(
    adata,
    train_idx,
    val_idx,
    test_idx,
    *,
    label_column: str,
    batch_key: str,
    rare_class: str,
    rare_train_size: int | float | str,
    config: dict,
    seed: int = 42,
    scvi_epochs: int | None = None,
    scanvi_epochs: int | None = None,
) -> tuple[scvi.model.SCANVI, dict[str, pd.DataFrame], dict[str, pd.DataFrame], list[str]]:
    """ 整合 HVG 筛选、半监督构建、双阶段训练与 Inductive 推理的端到端深度学习主接口。
    
    Args:
        adata: 已经过预处理的全局 AnnData 对象
        train_idx, val_idx, test_idx: 细胞在全局表达矩阵中的一维整数划分索引
        label_column: 对应的细胞类型真实类别列名
        batch_key: 批次效应对应的 obs 列名
        rare_class: 稀有细胞类型名称
        rare_train_size: 训练集中稀有细胞被显式标注的绝对数量 (int) 或比例 (float)，或 "all"
        config: 参数配置字典 (从 YAML 读取)
        seed: 随机种子，保障结果的一致性
        scvi_epochs: 指定 scVI 训练的 epoch 数，若为 None 则使用 config 中配置
        scanvi_epochs: 指定 scANVI 训练的 epoch 数，若为 None 则使用 config 中配置
        
    Returns:
        scanvi_model: 训练完毕的目标 scANVI 模型实例
        predictions: 包含 'train', 'validation', 'test' 为 key 的预测与不确定性 DataFrame 字典
        latents: 包含 'train', 'validation', 'test' 为 key 的潜在空间表征 DataFrame 字典
        selected_genes: 最终筛选并参与模型训练的高变基因 (HVG) 列表
    """
    seed_everything(seed)
    print(f"\n====== [scRareRefine 模型中心] 启动 scANVI 半监督表示训练 (seed={seed}) ======")
    
    model_cfg = config.get("model", {})
    unlabeled_category = config.get("experiment", {}).get("unlabeled_category", "Unknown")

    # 1. 拆分原始表达矩阵用于监督信号构建
    split_series = pd.Series("none", index=adata.obs_names)
    split_series.iloc[train_idx] = "train"
    split_series.iloc[val_idx] = "validation"
    split_series.iloc[test_idx] = "test"

    # 2. 构造半监督学习掩膜信号 (scanvi_label 与 is_labeled)
    scanvi_label, is_labeled = make_scanvi_labels(
        adata.obs,
        split_series,
        label_key=label_column,
        rare_class=rare_class,
        rare_train_size=rare_train_size,
        seed=seed,
        unlabeled_category=unlabeled_category,
    )
    
    # 构造类别集，将 'Unknown' (unlabeled_category) 添加至类别映射中
    label_cats = sorted(pd.unique(adata.obs[label_column].astype(str)).tolist())
    if unlabeled_category not in label_cats:
        label_cats = label_cats + [unlabeled_category]
        
    adata.obs["scanvi_label"] = pd.Categorical(scanvi_label.astype(str), categories=[str(c) for c in label_cats])
    adata.obs["is_labeled_for_scanvi"] = is_labeled

    # 3. 特征工程：基于训练集计算高方差基因选择 (HVG)
    train_adata_full = adata[split_series.eq("train")].copy()
    selected_genes = select_hvg_genes(train_adata_full, n_top_genes=model_cfg.get("n_top_hvg", 3000))
    print(f"-> [高变基因选择] 从 {adata.n_vars} 个基因中成功筛选前 {len(selected_genes)} 个 HVG 基因。")
    
    # 过滤非 HVG 表达空间，只保留对表征有显著作用的基因
    adata_hvg = adata[:, selected_genes].copy()
    train_adata = adata_hvg[split_series.eq("train")].copy()
    val_adata = adata_hvg[split_series.eq("validation")].copy()
    test_adata = adata_hvg[split_series.eq("test")].copy()

    # 4. 执行双阶段模型优化训练
    print(f"-> [模型训练] 开始执行 scVI 和 scANVI 监督学习优化流程...")
    scanvi_model = train_scanvi(
        train_adata,
        batch_key=batch_key,
        unlabeled_category=unlabeled_category,
        n_latent=int(model_cfg.get("n_latent", 30)),
        batch_size=int(model_cfg.get("batch_size", 256)),
        scvi_epochs=int(scvi_epochs or model_cfg.get("scvi_max_epochs", 200)),
        scanvi_epochs=int(scanvi_epochs or model_cfg.get("scanvi_max_epochs", 100)),
    )
    
    model_label_cats = getattr(scanvi_model, "adata_manager", None)
    if model_label_cats is not None:
        state_reg = model_label_cats.get_state_registry(REGISTRY_KEYS.LABELS_KEY)
        cats = getattr(state_reg, "categorical_mapping", None)
        model_label_cats = [str(c) for c in cats] if cats is not None else [str(c) for c in label_cats]
    else:
        model_label_cats = [str(c) for c in label_cats]

    # 5. 模型测试集推理：对训练集、验证集与测试集细胞产出潜在特征与预测概率
    print("-> [表征提取] 正在推算训练集表征与概率...")
    train_pred, train_latent = prediction_outputs(scanvi_model, train_adata, label_key=label_column, rare_class=rare_class)

    print("-> [表征提取] 正在以 Inductive 模式映射并推断验证集与测试集表征与概率...")
    val_query = load_query_model(val_adata, scanvi_model, unlabeled_category=unlabeled_category, label_categories=model_label_cats)
    val_pred, val_latent = prediction_outputs(val_query, val_query.adata, label_key=label_column, rare_class=rare_class)

    test_query = load_query_model(test_adata, scanvi_model, unlabeled_category=unlabeled_category, label_categories=model_label_cats)
    test_pred, test_latent = prediction_outputs(test_query, test_query.adata, label_key=label_column, rare_class=rare_class)

    # 汇总输出
    predictions = {
        "train": train_pred,
        "validation": val_pred,
        "test": test_pred
    }
    latents = {
        "train": train_latent,
        "validation": val_latent,
        "test": test_latent
    }
    
    print("====== [scRareRefine 模型中心] 双阶段训练与推理工作流全部结束 ======\n")
    return scanvi_model, predictions, latents, selected_genes
