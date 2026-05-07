# PROJECT_BRIEF.md

## 项目名称

**scRareRefine**

## 项目一句话描述

scRareRefine 是一个基于 scANVI 输出结果的稀有细胞类型识别 refinement 方法，利用 scANVI 的预测概率和 latent embedding 中的 prototype 信息，对不确定或容易被多数类吞掉的稀有细胞预测进行修正。

## 研究背景

单细胞 RNA 测序数据中的细胞类型注释是生信分析中的关键步骤。scANVI 是一种常用的半监督单细胞注释模型，能够利用已标注和未标注细胞学习 latent representation，并进行细胞类型预测。

但是，在细胞类型分布高度不均衡的场景中，稀有细胞类型通常样本数量少、边界模糊、容易与相近多数类混淆。即使基础模型整体 accuracy 较高，也可能在 rare cell type 上表现不足。

本项目希望在 scANVI 已有输出的基础上，加入一个轻量级 refinement 模块，专门提升稀有细胞识别效果。

## 当前研究目标

本项目目标是：

1. 以 scANVI 为基础模型。
2. 使用 scANVI 输出的预测概率和 latent embedding。
3. 基于 embedding 计算类别 prototype。
4. 结合 prototype 相似度、预测概率、不确定性和 rare-aware 策略，对细胞预测进行修正。
5. 提高稀有细胞类型的识别指标，尤其是 rare macro-F1、rare recall 和 per-class F1。
6. 形成一个可用于 Q2 级别小论文的完整方法和实验闭环。
7. 作为硕士论文三个工作之一。

## 项目类型

本项目偏向：

```text
scANVI + 新模块
```

也就是说，本项目不是从零训练一个新的大模型，而是在已有 scANVI 模型的基础上设计一个面向稀有细胞识别的 refinement 方法。

## 核心方法思路

初始贡献可以描述为：

> 一种基于 scANVI 的稀有细胞 refinement 模块，结合预测概率、latent space 中到各类别 prototype 的距离或相似度，以及 rare-class-aware 的决策调整策略，以提升稀有细胞类型识别效果。

可能包含以下模块：

### 1. Prototype-based refinement

- 在 scANVI latent space 中计算每个类别的 prototype。
- 计算每个细胞与各类别 prototype 的距离或相似度。
- 用 prototype 证据修正 scANVI 原始预测概率。

### 2. Rare-class-aware calibration / adjustment

- 针对稀有类别进行概率校准或阈值调整。
- 避免模型过度偏向多数类。
- 提升少数类、稀有类的 recall 和 F1。

### 3. Uncertainty-aware correction

- 只对低置信度、高熵、概率接近或 prototype 证据冲突的细胞进行 refinement。
- 对高置信度 scANVI 预测保持不变。
- 降低把普通细胞错误修正为稀有细胞的风险。

### 4. 可选微调

- 如果 post-hoc refinement 效果不足，可以考虑对 scANVI 做轻量微调。
- 微调必须和 frozen scANVI + refinement 做公平对比。

## 研究问题

本项目关注以下问题：

1. scANVI latent embedding 是否包含可用于稀有细胞识别的 prototype-level 证据？
2. 结合 scANVI 预测概率和 latent prototype 相似度，能否提高稀有细胞 F1 或 recall？
3. 这种提升是否在多个数据集、多个随机种子上稳定？
4. 稀有细胞指标提升时，整体 annotation 性能是否保持可接受？
5. 与 scANVI 原始结果和部分稀有细胞相关方法相比，scRareRefine 是否具有竞争力？

## 项目范围

### 当前版本包含

- `.h5ad` 格式的单细胞 RNA-seq 数据。
- 基于 scANVI 的细胞类型注释。
- scANVI latent embedding 提取。
- scANVI predicted probability 提取。
- latent space prototype 计算。
- 基于细胞类型频率阈值定义 rare cell type。
- 小规模多数据集实验。
- 与 scANVI baseline 以及部分额外 baseline 对比。
- 重点评估 rare-cell metrics 和整体 annotation metrics。

### 当前版本不包含

- 从零训练基础模型。
- 大规模预训练。
- 临床部署。
- 声称适用于所有单细胞数据。
- 多组学集成，除非后续明确加入。
- 没有人工验证的生物学新发现。
- 自动修改原始数据。

## 方法边界

本方法应描述为：

> 基于 scANVI 输出结果的轻量级稀有细胞识别 refinement 模块。

不要描述为：

```text
新的 foundation model
通用细胞注释模型
scANVI 的完全替代品
临床验证方法
```

## 预期创新点

本项目的创新点可以围绕以下方向组织：

1. 面向稀有细胞类型的 scANVI post-hoc refinement。
2. 融合 probability-space 证据和 latent prototype-space 证据。
3. 面向 rare cell type 的 decision adjustment 和评价体系。
4. 在多个数据集上的可复现实验验证。
5. 利用 AI agent 辅助科研，但通过标准目录、日志和 claims 控制保证可复现。

## 成功标准

如果满足以下条件，项目可以认为有投稿潜力：

1. 在至少两个数据集上，rare-cell macro-F1 相比 vanilla scANVI 有提升。
2. 在 seeds `42`、`43`、`44` 上提升相对稳定。
3. overall accuracy / macro-F1 / weighted-F1 没有严重下降。
4. 消融实验能说明 prototype、fusion、uncertainty gate 或 rare adjustment 的贡献。
5. 结果能通过 latent-space 可视化、混淆矩阵或错误分析解释。
6. 至少有一个有说服力的稀有细胞相关 baseline。

## 当前已知数据集

1. `human_pancreas_norm_complexBatch.h5ad`
2. `Tabula Sapiens`
3. `human_immune_health_atlas_dc.h5ad`

## 当前已跑通 baseline

1. scANVI + pancreas
2. scANVI + human_immune_health

## 方法临时名称

当前名称：

```text
scRareRefine
```

可选未来名称：

```text
scRareRefine
scANVI-RareRefine
RareRefine
ProtoRare
scProtoRefine
```

## 主要风险

1. 方法可能只在一个数据集上有效。
2. 稀有细胞提升可能伴随多数类性能下降。
3. 如果 scANVI latent space 区分度不足，prototype 可能无效。
4. rare cell definition 可能被认为主观。
5. 如果只是简单后处理，创新性可能被质疑。
6. 与强 baseline 比较可能没有优势。
7. AI agent 可能产生未经验证的结论或混乱代码。

## 立即下一步

1. 检查三个 `.h5ad` 文件，识别 label 列、batch 列、donor/study 列和细胞类型分布。
2. 定义 rare cell type 阈值。
3. 复现 vanilla scANVI 在各数据集、各 seed 上的结果。
4. 提取 scANVI embeddings 和 predicted probabilities。
5. 实现第一版 prototype-based refinement。
6. 评估 rare macro-F1、rare recall、balanced accuracy 和 per-class metrics。
7. 进行消融实验。
8. 与至少一个稀有细胞相关 baseline 对比。
