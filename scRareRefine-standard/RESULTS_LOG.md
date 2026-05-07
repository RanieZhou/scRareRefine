# RESULTS_LOG.md

## 目的

本文档用于记录 scRareRefine 项目的所有实验。

所有有意义的实验都必须记录，包括失败实验、负结果和中途放弃的实验。

## 项目名称

**scRareRefine**：基于 scANVI 输出结果的稀有细胞类型识别 refinement 方法。

## 当前状态总结

已知已完成或部分完成的 baseline：

| 日期 | 数据集 | 方法 | Seed | 状态 | 备注 |
|---|---|---|---:|---|---|
| TODO | human_pancreas_norm_complexBatch.h5ad | scANVI | TODO | 已跑过 / 待复核 | 用户反馈 scANVI + pancreas 已跑通 |
| TODO | human_immune_health_atlas_dc.h5ad | scANVI | TODO | 已跑过 / 待复核 | 用户反馈 scANVI + human_immune_health 已跑通 |

## 标准实验记录模板

每次新实验复制下面模板。

```markdown
## Experiment: EXP-YYYYMMDD-XXX

### 基本信息

- 日期：
- 实验 ID：
- 数据集：
- 数据文件：
- 方法：
- Seed：
- 配置文件：
- Git commit：
- 环境：
- 设备 / GPU：

### 数据设置

- Label 列：
- Batch 列：
- 细胞数量：
- 基因数量：
- 细胞类型数量：
- 稀有细胞定义：
- 稀有细胞类型：
- Train/validation/test 划分：

### 方法设置

- 基础模型：
- scANVI 是重新训练还是加载已有模型：
- scANVI latent 维度：
- 是否使用 scANVI 概率输出：
- Prototype 方法：
- 距离度量：
- 融合参数 alpha：
- 置信度阈值：
- 稀有类别调整策略：
- 是否使用微调：

### 运行命令

```bash
# 在这里写命令
```

### 输出文件

- Prediction 文件：
- Metrics 文件：
- Figure 目录：
- Log 文件：

### 主要指标

| 指标 | 数值 |
|---|---:|
| Accuracy | TODO |
| Macro-F1 | TODO |
| Weighted-F1 | TODO |
| Balanced Accuracy | TODO |
| Rare Macro-F1 | TODO |
| Rare Precision | TODO |
| Rare Recall | TODO |

### 稀有细胞 per-class 指标

| Cell Type | Count | Precision | Recall | F1 | 主要混淆类别 |
|---|---:|---:|---:|---:|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

### 观察

- TODO

### 问题 / 警告

- TODO

### 初步结论

- TODO

### 下一步动作

- TODO
```

## 主结果表

| Experiment ID | Dataset | Method | Seed | Accuracy | Macro-F1 | Weighted-F1 | Balanced Acc | Rare Macro-F1 | Rare Recall | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| TODO | pancreas | scANVI | 42 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | pancreas | scANVI | 43 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | pancreas | scANVI | 44 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | pancreas | scRareRefine | 42 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | pancreas | scRareRefine | 43 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | pancreas | scRareRefine | 44 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scANVI | 42 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scANVI | 43 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scANVI | 44 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scRareRefine | 42 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scRareRefine | 43 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| TODO | immune_dc | scRareRefine | 44 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

## 消融实验表

| Experiment ID | Dataset | Variant | Seed | Probability | Prototype | Gate | Rare Adjust | Rare Macro-F1 | Macro-F1 | Notes |
|---|---|---|---:|---|---|---|---|---:|---:|---|
| TODO | TODO | scANVI | TODO | yes | no | no | no | TODO | TODO | TODO |
| TODO | TODO | prob-only | TODO | yes | no | optional | no | TODO | TODO | TODO |
| TODO | TODO | proto-only | TODO | no | yes | optional | no | TODO | TODO | TODO |
| TODO | TODO | fusion | TODO | yes | yes | no | no | TODO | TODO | TODO |
| TODO | TODO | full | TODO | yes | yes | yes | yes | TODO | TODO | TODO |

## 错误分析记录

用于总结重要错误模式。

| Dataset | Method | Error Type | Description | Possible Fix |
|---|---|---|---|---|
| TODO | TODO | rare-to-common | 稀有细胞被预测为多数类 | 增强 prototype 或 rare-aware 权重 |
| TODO | TODO | common-to-rare | 普通细胞被过度修正为稀有类 | 加 uncertainty gate |
| TODO | TODO | batch-specific error | 某些 batch 表现较差 | 检查 batch-aware split 或 batch effect |
| TODO | TODO | label ambiguity | 相似细胞类型混淆 | 检查 label hierarchy 和 marker gene |

## 失败实验记录

失败实验必须记录。

| Date | Experiment ID | Dataset | Method | Failure Reason | Action |
|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO |

## 决策日志

记录重大研究决策。

| Date | Decision | Reason | Impact |
|---|---|---|---|
| TODO | 使用 scANVI 作为基础模型 | 已有 baseline，且适合半监督单细胞注释 | 定义项目方向 |
| TODO | 使用 seeds 42、43、44 | 用户当前实验设置 | 保持实验一致性 |
| TODO | 聚焦稀有细胞识别 | 当前核心研究目标 | 决定指标和论文主张 |
| TODO | 采用标准项目目录结构 | 防止 AI agent 造成目录混乱 | 提高可维护性和可复现性 |

## 下一批实验优先级

1. 检查所有 `.h5ad` 文件，识别 label 列和 batch 列。
2. 统计每个数据集的细胞类型分布。
3. 定义 rare cell type 阈值。
4. 复现 pancreas 上 scANVI baseline，seeds 42、43、44。
5. 提取 scANVI embeddings 和 predicted probabilities。
6. 实现 prototype-only refinement。
7. 实现 probability-prototype fusion。
8. 评估 rare macro-F1 和 rare recall。
9. 做错误分析，再扩展到更多数据集。
