# 消融实验报告（scRareRefine Conformal 方案）

实验日期：2026-06-12 | 数据集：3 | seed：42/43/44 | rare_train_size：5%/10%/10%

## 变体定义

| 变体 | 候选筛选 | 评分函数 | 阈值校准 | 弃权阈值 |
|------|---------|---------|---------|---------|
| V1 no_rank1 | 全部 predicted≠rare | 各向异性 softmax(-d/r) | conformal (val 非稀有) | sep < 1.3 |
| V2 rank1_nofilter | 各向同性 rank=1 | 无（全救） | 无 | sep < 1.1 |
| V3 isotropic | 各向同性 rank=1 | 各向同性 softmax(-d) | conformal (val 非稀有) | sep < 1.3 |
| V4 full（完整方法） | 各向同性 rank=1 | 各向异性 softmax(-d/r) | conformal (val 非稀有) | sep < 1.3 |

## 3-seed 均值 ± σ 结果

| 数据集 | 变体 | F1 均值 | F1 σ | 提升 | FFR_max |
|-------|------|--------|------|------|--------|
| immune_dc | v1_no_rank1 | 0.8089 | 0.0260 | +0.7837 | 0.01133 |
| immune_dc | v2_rank1_nofilter | 0.9394 | 0.0096 | +0.9143 | 0.00050 |
| immune_dc | v3_isotropic | 0.9394 | 0.0096 | +0.9143 | 0.00050 |
| immune_dc | v4_full | 0.9394 | 0.0096 | +0.9143 | 0.00050 |
| pancreas_baron | v1_no_rank1 | 0.8331 | 0.0388 | +0.0084 | 0.02259 |
| pancreas_baron | v2_rank1_nofilter | 0.8012 | 0.0240 | -0.0234 | 0.01709 |
| pancreas_baron | v3_isotropic | 0.8449 | 0.0307 | +0.0203 | 0.00855 |
| pancreas_baron | v4_full | 0.8494 | 0.0274 | +0.0248 | 0.00488 |
| tabula_lung_endo | v1_no_rank1 | 0.8354 | 0.0564 | -0.1340 | 0.02215 |
| tabula_lung_endo | v2_rank1_nofilter | 0.9774 | 0.0060 | +0.0081 | 0.00058 |
| tabula_lung_endo | v3_isotropic | 0.9774 | 0.0060 | +0.0081 | 0.00058 |
| tabula_lung_endo | v4_full | 0.9799 | 0.0035 | +0.0105 | 0.00058 |

## 逐 run 明细

| 数据集 | seed | 变体 | sep | F1 | recall | precision | rescued | false | FFR |
|-------|------|------|-----|-----|--------|-----------|---------|-------|-----|
| immune_dc | 42 | v1_no_rank1 | 2.387 | 0.8013 | 0.9615 | 0.6868 | 182 | 57 | 0.00949 |
| immune_dc | 42 | v2_rank1_nofilter | 2.387 | 0.9486 | 0.9231 | 0.9756 | 123 | 3 | 0.00050 |
| immune_dc | 42 | v3_isotropic | 2.387 | 0.9486 | 0.9231 | 0.9756 | 123 | 3 | 0.00050 |
| immune_dc | 42 | v4_full | 2.387 | 0.9486 | 0.9231 | 0.9756 | 123 | 3 | 0.00050 |
| immune_dc | 43 | v1_no_rank1 | 2.026 | 0.7815 | 0.9769 | 0.6513 | 193 | 68 | 0.01133 |
| immune_dc | 43 | v2_rank1_nofilter | 2.026 | 0.9262 | 0.8692 | 0.9912 | 112 | 1 | 0.00017 |
| immune_dc | 43 | v3_isotropic | 2.026 | 0.9262 | 0.8692 | 0.9912 | 112 | 1 | 0.00017 |
| immune_dc | 43 | v4_full | 2.026 | 0.9262 | 0.8692 | 0.9912 | 112 | 1 | 0.00017 |
| immune_dc | 44 | v1_no_rank1 | 1.808 | 0.8439 | 0.9769 | 0.7427 | 168 | 44 | 0.00733 |
| immune_dc | 44 | v2_rank1_nofilter | 1.808 | 0.9435 | 0.9000 | 0.9915 | 115 | 1 | 0.00017 |
| immune_dc | 44 | v3_isotropic | 1.808 | 0.9435 | 0.9000 | 0.9915 | 115 | 1 | 0.00017 |
| immune_dc | 44 | v4_full | 1.808 | 0.9435 | 0.9000 | 0.9915 | 115 | 1 | 0.00017 |
| pancreas_baron | 42 | v1_no_rank1 | 1.404 | 0.8000 | 0.9535 | 0.6891 | 61 | 37 | 0.02259 |
| pancreas_baron | 42 | v2_rank1_nofilter | 1.404 | 0.8312 | 0.7442 | 0.9412 | 10 | 4 | 0.00244 |
| pancreas_baron | 42 | v3_isotropic | 1.404 | 0.8312 | 0.7442 | 0.9412 | 10 | 4 | 0.00244 |
| pancreas_baron | 42 | v4_full | 1.404 | 0.8366 | 0.7442 | 0.9552 | 9 | 3 | 0.00183 |
| pancreas_baron | 43 | v1_no_rank1 | 1.554 | 0.8118 | 0.8023 | 0.8214 | 24 | 12 | 0.00733 |
| pancreas_baron | 43 | v2_rank1_nofilter | 1.554 | 0.7725 | 0.8488 | 0.7087 | 43 | 27 | 0.01648 |
| pancreas_baron | 43 | v3_isotropic | 1.554 | 0.8161 | 0.8256 | 0.8068 | 28 | 14 | 0.00855 |
| pancreas_baron | 43 | v4_full | 1.554 | 0.8242 | 0.7907 | 0.8608 | 19 | 8 | 0.00488 |
| pancreas_baron | 44 | v1_no_rank1 | 1.124 | 0.8875(弃) | 0.8256 | 0.9595 | 0 | 0 | 0.00000 |
| pancreas_baron | 44 | v2_rank1_nofilter | 1.124 | 0.8000 | 0.9070 | 0.7156 | 35 | 28 | 0.01709 |
| pancreas_baron | 44 | v3_isotropic | 1.124 | 0.8875(弃) | 0.8256 | 0.9595 | 0 | 0 | 0.00000 |
| pancreas_baron | 44 | v4_full | 1.124 | 0.8875(弃) | 0.8256 | 0.9595 | 0 | 0 | 0.00000 |
| tabula_lung_endo | 42 | v1_no_rank1 | 1.659 | 0.8387 | 1.0000 | 0.7222 | 24 | 23 | 0.01340 |
| tabula_lung_endo | 42 | v2_rank1_nofilter | 1.659 | 0.9774 | 1.0000 | 0.9559 | 2 | 1 | 0.00058 |
| tabula_lung_endo | 42 | v3_isotropic | 1.659 | 0.9774 | 1.0000 | 0.9559 | 2 | 1 | 0.00058 |
| tabula_lung_endo | 42 | v4_full | 1.659 | 0.9774 | 1.0000 | 0.9559 | 2 | 1 | 0.00058 |
| tabula_lung_endo | 43 | v1_no_rank1 | 1.734 | 0.9028 | 1.0000 | 0.8228 | 13 | 11 | 0.00641 |
| tabula_lung_endo | 43 | v2_rank1_nofilter | 1.734 | 0.9701 | 1.0000 | 0.9420 | 3 | 1 | 0.00058 |
| tabula_lung_endo | 43 | v3_isotropic | 1.734 | 0.9701 | 1.0000 | 0.9420 | 3 | 1 | 0.00058 |
| tabula_lung_endo | 43 | v4_full | 1.734 | 0.9774 | 1.0000 | 0.9559 | 2 | 0 | 0.00000 |
| tabula_lung_endo | 44 | v1_no_rank1 | 1.689 | 0.7647 | 1.0000 | 0.6190 | 40 | 38 | 0.02215 |
| tabula_lung_endo | 44 | v2_rank1_nofilter | 1.689 | 0.9848 | 1.0000 | 0.9701 | 2 | 0 | 0.00000 |
| tabula_lung_endo | 44 | v3_isotropic | 1.689 | 0.9848 | 1.0000 | 0.9701 | 2 | 0 | 0.00000 |
| tabula_lung_endo | 44 | v4_full | 1.689 | 0.9848 | 1.0000 | 0.9701 | 2 | 0 | 0.00000 |
