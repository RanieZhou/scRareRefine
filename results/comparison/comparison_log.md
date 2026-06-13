# 对比实验报告（scRareRefine vs baselines）

实验日期：2026-06-12 | 数据集：3 | seed：42/43/44 | rare_train_size：5%/10%/10%

## 方法说明

| 方法 | 输入特征 | 训练数据 | 核心设计 |
|------|---------|---------|---------|
| scANVI | scANVI latent (20d) | labeled+unlabeled | 半监督 VAE 直接预测 |
| kNN (best k) | scANVI latent (20d) | labeled only | 欧氏 k 近邻，val 上选 k∈{3,5,10,15} |
| CellTypist | HVG log1p 表达 (2000-3000d) | labeled only | Logistic Regression（官方工具）|
| scBalance | HVG log1p 表达 (2000-3000d) | labeled only | 加权采样神经网络（官方工具）|
| **scRareRefine** | scANVI latent (20d) | labeled+unlabeled | scANVI + conformal prototype rescue |

注：CellTypist 和 scBalance 使用各自设计的 HVG 基因表达输入；kNN 和 scRareRefine 使用 scANVI latent。

注：rare_fp_rate = (pred==rare 且 真值非rare) / 非稀有数，所有方法可比的标准假阳性率；
rescue_ffr（仅逐 run 明细）= 相对 scANVI 改判的误救率，仅对 scRareRefine 可解释。
失败方法（status=failed）不计入均值。

## 3-seed 均值 ± σ 结果

| 数据集 | 方法 | F1 均值 | F1 σ | FP_rate_max | n_ok |
|-------|------|--------|------|------------|------|
| immune_dc | scANVI | 0.0251 | 0.0188 | 0.00000 | 3 |
| immune_dc | kNN | 0.6725 | 0.0662 | 0.00000 | 3 |
| immune_dc | CellTypist | 0.5598 | 0.0388 | 0.00000 | 3 |
| immune_dc | scBalance | 0.5574 | 0.0697 | 0.00000 | 3 |
| immune_dc | scRareRefine | 0.9394 | 0.0096 | 0.00050 | 3 |
| pancreas_baron | scANVI | 0.8246 | 0.0456 | 0.00183 | 3 |
| pancreas_baron | kNN | 0.6165 | 0.1833 | 0.00122 | 3 |
| pancreas_baron | CellTypist | 0.6277 | 0.1256 | 0.00000 | 3 |
| pancreas_baron | scBalance | 0.7091 | 0.0887 | 0.00122 | 3 |
| pancreas_baron | scRareRefine | 0.8494 | 0.0274 | 0.00672 | 3 |
| tabula_lung_endo | scANVI | 0.9694 | 0.0062 | 0.00175 | 3 |
| tabula_lung_endo | kNN | 0.9519 | 0.0065 | 0.00058 | 3 |
| tabula_lung_endo | CellTypist | 0.7751 | 0.0362 | 0.00058 | 3 |
| tabula_lung_endo | scBalance | 0.9230 | 0.0364 | 0.00117 | 3 |
| tabula_lung_endo | scRareRefine | 0.9799 | 0.0035 | 0.00175 | 3 |

## 逐 run 明细

| 数据集 | seed | sep | 方法 | status | F1 | recall | precision | rare_fp_rate | rescue_ffr |
|-------|------|-----|------|--------|-----|--------|-----------|-------------|-----------|
| immune_dc | 42 | 2.387 | scANVI | ok | 0.0000 | 0.0000 | 0.0000 | 0.00000 | 0.00000 |
| immune_dc | 42 | 2.387 | kNN | ok | 0.7255 | 0.5692 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 42 | 2.387 | CellTypist | ok | 0.5946 | 0.4231 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 42 | 2.387 | scBalance | ok | 0.5311 | 0.3615 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 42 | 2.387 | scRareRefine | ok | 0.9486 | 0.9231 | 0.9756 | 0.00050 | 0.00050 |
| immune_dc | 43 | 2.026 | scANVI | ok | 0.0303 | 0.0154 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 43 | 2.026 | kNN | ok | 0.7129 | 0.5538 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 43 | 2.026 | CellTypist | ok | 0.5057 | 0.3385 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 43 | 2.026 | scBalance | ok | 0.6528 | 0.4846 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 43 | 2.026 | scRareRefine | ok | 0.9262 | 0.8692 | 0.9912 | 0.00017 | 0.00017 |
| immune_dc | 44 | 1.808 | scANVI | ok | 0.0451 | 0.0231 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 44 | 1.808 | kNN | ok | 0.5792 | 0.4077 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 44 | 1.808 | CellTypist | ok | 0.5792 | 0.4077 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 44 | 1.808 | scBalance | ok | 0.4884 | 0.3231 | 1.0000 | 0.00000 | 0.00000 |
| immune_dc | 44 | 1.808 | scRareRefine | ok | 0.9435 | 0.9000 | 0.9915 | 0.00017 | 0.00017 |
| pancreas_baron | 42 | 1.404 | scANVI | ok | 0.8056 | 0.6744 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 42 | 1.404 | kNN | ok | 0.3619 | 0.2209 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 42 | 1.404 | CellTypist | ok | 0.4505 | 0.2907 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 42 | 1.404 | scBalance | ok | 0.7068 | 0.5465 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 42 | 1.404 | scRareRefine | ok | 0.8366 | 0.7442 | 0.9552 | 0.00183 | 0.00183 |
| pancreas_baron | 43 | 1.554 | scANVI | ok | 0.7808 | 0.6628 | 0.9500 | 0.00183 | 0.00000 |
| pancreas_baron | 43 | 1.554 | kNN | ok | 0.7015 | 0.5465 | 0.9792 | 0.00061 | 0.00000 |
| pancreas_baron | 43 | 1.554 | CellTypist | ok | 0.7259 | 0.5698 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 43 | 1.554 | scBalance | ok | 0.8188 | 0.7093 | 0.9683 | 0.00122 | 0.00061 |
| pancreas_baron | 43 | 1.554 | scRareRefine | ok | 0.8242 | 0.7907 | 0.8608 | 0.00672 | 0.00488 |
| pancreas_baron | 44 | 1.124 | scANVI | ok | 0.8875 | 0.8256 | 0.9595 | 0.00183 | 0.00000 |
| pancreas_baron | 44 | 1.124 | kNN | ok | 0.7862 | 0.6628 | 0.9661 | 0.00122 | 0.00000 |
| pancreas_baron | 44 | 1.124 | CellTypist | ok | 0.7068 | 0.5465 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 44 | 1.124 | scBalance | ok | 0.6016 | 0.4302 | 1.0000 | 0.00000 | 0.00000 |
| pancreas_baron | 44 | 1.124 | scRareRefine | ok | 0.8875 | 0.8256 | 0.9595 | 0.00183 | 0.00000 |
| tabula_lung_endo | 42 | 1.659 | scANVI | ok | 0.9771 | 0.9846 | 0.9697 | 0.00117 | 0.00000 |
| tabula_lung_endo | 42 | 1.659 | kNN | ok | 0.9600 | 0.9231 | 1.0000 | 0.00000 | 0.00000 |
| tabula_lung_endo | 42 | 1.659 | CellTypist | ok | 0.7890 | 0.6615 | 0.9773 | 0.00058 | 0.00000 |
| tabula_lung_endo | 42 | 1.659 | scBalance | ok | 0.9440 | 0.9077 | 0.9833 | 0.00058 | 0.00000 |
| tabula_lung_endo | 42 | 1.659 | scRareRefine | ok | 0.9774 | 1.0000 | 0.9559 | 0.00175 | 0.00058 |
| tabula_lung_endo | 43 | 1.734 | scANVI | ok | 0.9618 | 0.9692 | 0.9545 | 0.00175 | 0.00000 |
| tabula_lung_endo | 43 | 1.734 | kNN | ok | 0.9440 | 0.9077 | 0.9833 | 0.00058 | 0.00000 |
| tabula_lung_endo | 43 | 1.734 | CellTypist | ok | 0.8108 | 0.6923 | 0.9783 | 0.00058 | 0.00058 |
| tabula_lung_endo | 43 | 1.734 | scBalance | ok | 0.8718 | 0.7846 | 0.9808 | 0.00058 | 0.00000 |
| tabula_lung_endo | 43 | 1.734 | scRareRefine | ok | 0.9774 | 1.0000 | 0.9559 | 0.00175 | 0.00000 |
| tabula_lung_endo | 44 | 1.689 | scANVI | ok | 0.9692 | 0.9692 | 0.9692 | 0.00117 | 0.00000 |
| tabula_lung_endo | 44 | 1.689 | kNN | ok | 0.9516 | 0.9077 | 1.0000 | 0.00000 | 0.00000 |
| tabula_lung_endo | 44 | 1.689 | CellTypist | ok | 0.7255 | 0.5692 | 1.0000 | 0.00000 | 0.00000 |
| tabula_lung_endo | 44 | 1.689 | scBalance | ok | 0.9531 | 0.9385 | 0.9683 | 0.00117 | 0.00000 |
| tabula_lung_endo | 44 | 1.689 | scRareRefine | ok | 0.9848 | 1.0000 | 0.9701 | 0.00117 | 0.00000 |
