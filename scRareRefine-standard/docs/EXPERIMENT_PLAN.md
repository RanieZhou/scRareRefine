# EXPERIMENT_PLAN.md

## 目的

本文档定义 scRareRefine 项目的实验设计、baseline、评价指标、随机种子、消融实验和结果记录要求。

## 实验目标

本项目的核心实验目标是回答：

> 在相同 scANVI 基础模型、相同数据集、相同 split 和相同 seed 下，scRareRefine 是否能比 vanilla scANVI 更好地识别稀有细胞类型？

## 当前实验规模

当前阶段采用：

```text
小规模实验
```

优先完成：

1. pancreas 数据集。
2. human immune health atlas dc 数据集。
3. 如果前两个数据集结果有希望，再加入 Tabula Sapiens。

## 随机种子

当前固定使用：

```text
42
43
44
```

所有主要实验必须使用这三个 seed，结果报告 mean ± std。

## 数据集优先级

| 优先级 | 数据集 | 文件 | 状态 | 备注 |
|---:|---|---|---|---|
| 1 | pancreas | human_pancreas_norm_complexBatch.h5ad | 已有 | 已跑过 scANVI，需要复核 |
| 2 | immune_dc | human_immune_health_atlas_dc.h5ad | 已有 | 已跑过 scANVI，需要复核 |
| 3 | Tabula Sapiens | TODO | 已有 / 待确认 | 可作为扩展验证 |

## 方法分组

### Group 0：数据检查

必须先完成，不属于论文主实验，但决定后续实验是否可信。

目标：

```text
检查 .h5ad 文件结构、label 列、batch 列、细胞类型分布、rare cell 候选。
```

输出：

```text
results/reports/data_inspection_report.md
results/tables/dataset_summary.csv
results/tables/cell_type_distribution.csv
```

### Group 1：基础模型 baseline

必须包含：

```text
vanilla scANVI
```

作用：

证明 scRareRefine 是否真的在 scANVI 基础上提升稀有细胞识别。

### Group 2：scRareRefine 主方法

主方法包含：

```text
scANVI probability
+ scANVI latent embedding
+ class prototype
+ probability-prototype fusion
+ uncertainty gate
+ rare-aware adjustment
```

### Group 3：消融实验

建议消融：

| Variant | Probability | Prototype | Uncertainty Gate | Rare Adjustment | 目的 |
|---|---|---|---|---|---|
| scANVI | yes | no | no | no | 原始 baseline |
| proto-only | no | yes | optional | no | 验证 prototype 是否有用 |
| prob+proto | yes | yes | no | no | 验证融合是否有用 |
| prob+proto+gate | yes | yes | yes | no | 验证 uncertainty gate 是否减少误修正 |
| full scRareRefine | yes | yes | yes | yes | 完整方法 |

### Group 4：常规 baseline

可选，但建议至少选择 1–2 个：

```text
scANVI embedding + MLP
scANVI embedding + Logistic Regression
scANVI embedding + SVM
scVI + classifier
CellTypist
```

作用：

证明不是随便一个 classifier 就能达到相同效果。

### Group 5：稀有细胞相关 baseline

候选：

```text
scBalance
CIARA
scSID
RaceID
```

注意：这些方法不一定和你的任务完全一致，有的偏 rare cell detection，有的偏 annotation。使用前必须判断任务是否公平。

第一版建议优先尝试：

```text
scBalance
```

如果 CIARA 或 scSID 能合理适配，也可以加入一个。

## 最低可投稿实验组合

如果时间有限，最低实验组合建议为：

```text
数据集：pancreas + immune_dc
seed：42、43、44
方法：
  1. vanilla scANVI
  2. scANVI embedding + MLP 或 Logistic Regression
  3. scBalance 或另一个 rare-cell baseline
  4. scRareRefine
消融：
  1. scANVI
  2. proto-only
  3. prob+proto
  4. full scRareRefine
指标：
  rare macro-F1
  rare recall
  rare precision
  macro-F1
  balanced accuracy
  accuracy
  per-class F1
```

## 主评价指标

由于本项目关注稀有细胞识别，不能以总体 accuracy 作为唯一指标。

主指标：

```text
Rare Macro-F1
Rare Recall
Rare Precision
Per-class F1 for rare cell types
Macro-F1
Balanced Accuracy
```

辅助指标：

```text
Accuracy
Weighted-F1
Confusion Matrix
Rare-vs-common performance gap
```

可选校准指标：

```text
ECE
Brier Score
Confidence distribution
Entropy distribution
```

## 稀有细胞指标定义

设 rare cell types 为 `R`。

### Rare Macro-F1

只在 rare cell types 上计算 macro-F1。

### Rare Recall

只统计真实标签属于 rare cell type 的样本被正确识别的比例。

### Rare Precision

统计被预测为 rare cell type 的样本中有多少是真正 rare cell。

### Rare-vs-common gap

用于衡量普通类和稀有类之间的性能差距。

## 实验公平性要求

所有方法必须尽量保持：

```text
相同数据集
相同 train/validation/test split
相同 rare cell 定义
相同 seed
相同评价指标
相同 label mapping
相同预处理规则
```

禁止：

```text
scRareRefine 用一个 split，baseline 用另一个 split
scRareRefine 使用测试集信息调参
只报告 scRareRefine 成功的 seed
不报告 baseline 失败原因
改变 rare cell 定义以获得更好结果
```

## 第一阶段实验顺序

### Step 1：数据检查

命令示例：

```bash
python scripts/01_inspect_data.py --config configs/datasets/pancreas.yaml
python scripts/01_inspect_data.py --config configs/datasets/immune_dc.yaml
```

### Step 2：复现 scANVI baseline

命令示例：

```bash
python scripts/02_train_scanvi.py --config configs/experiments/pancreas_scanvi_seed42.yaml
python scripts/02_train_scanvi.py --config configs/experiments/pancreas_scanvi_seed43.yaml
python scripts/02_train_scanvi.py --config configs/experiments/pancreas_scanvi_seed44.yaml
```

### Step 3：提取 scANVI 输出

```bash
python scripts/03_extract_scanvi_outputs.py --config configs/experiments/pancreas_scanvi_seed42.yaml
```

### Step 4：运行 prototype refinement

```bash
python scripts/04_run_refinement.py --config configs/experiments/pancreas_proto_refine_seed42.yaml
```

### Step 5：评估

```bash
python scripts/05_evaluate.py --config configs/experiments/pancreas_proto_refine_seed42.yaml
```

### Step 6：汇总结果

```bash
python scripts/06_collect_results.py --results_dir results/raw --output results/tables/main_results.csv
```

### Step 7：绘图

```bash
python scripts/07_plot_results.py --table results/tables/main_results.csv
```

## 消融实验计划

| Ablation | 说明 | 预期回答的问题 |
|---|---|---|
| scANVI only | 原始 scANVI | 基础性能是多少？ |
| Prototype only | 只使用 prototype 相似度 | embedding 的 prototype 证据是否有用？ |
| Probability + Prototype | 概率与 prototype 融合 | 融合是否优于单独使用？ |
| + Uncertainty Gate | 只修正不确定样本 | gate 是否减少过度修正？ |
| + Rare Adjustment | 加入 rare-aware 权重或阈值 | 是否进一步提升 rare cell recall/F1？ |

## 结果保存规则

每个实验独立保存：

```text
results/raw/{experiment_id}/
```

必须包含：

```text
predictions.csv
metrics.json
config.yaml
environment.txt
run.log
```

汇总表：

```text
results/tables/main_results.csv
results/tables/rare_cell_results.csv
results/tables/ablation_results.csv
```

图像：

```text
results/figures/confusion_matrices/
results/figures/latent_umap/
results/figures/rare_cell_analysis/
```

## 判断项目是否有说服力

项目有希望的情况：

1. 至少两个数据集 rare macro-F1 提升。
2. seeds 42、43、44 平均提升稳定。
3. 稀有细胞 recall 提升明显。
4. overall accuracy 没有严重下降。
5. 消融实验能说明 prototype/fusion/gate 的作用。
6. 与至少一个 rare-cell baseline 相比有竞争力。

项目风险较大的情况：

1. 只有一个数据集有效。
2. rare recall 提升但 precision 大幅下降。
3. 只靠调阈值获得提升。
4. 消融实验无法证明模块贡献。
5. 与 scANVI 原始结果差距很小。
6. baseline 选择不公平或跑不通。

## 当前 TODO

- [ ] 完成数据检查脚本。
- [ ] 确认 label 列。
- [ ] 确认 rare cell 定义。
- [ ] 标准化 scANVI baseline 结果。
- [ ] 实现 prototype 计算。
- [ ] 实现 probability-prototype fusion。
- [ ] 实现 rare metrics。
- [ ] 加入至少一个常规 baseline。
- [ ] 调研并选择一个 rare-cell baseline。
