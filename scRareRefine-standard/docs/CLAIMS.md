# CLAIMS.md

## 目的

本文档定义 scRareRefine 项目中允许写入论文、报告和摘要的主张，以及在证据不足时禁止使用的主张。

## 当前论文主张强度

目标强度：

```text
B：中等主张
```

含义：

> 如果实验支持，可以主张 scRareRefine 在多个评估数据集上提升了稀有细胞识别效果；但不能主张其普遍优于所有方法，也不能声称解决了稀有细胞识别问题。

## 当前允许的主张

以下主张必须在实验支持后才能使用。

### Claim 1：scRareRefine 是基于 scANVI 的轻量级 refinement 模块

允许写法：

> scRareRefine 是一个构建在 scANVI 输出结果之上的轻量级稀有细胞识别 refinement 框架。

避免写法：

> scRareRefine 是一个新的单细胞 foundation model。

### Claim 2：scRareRefine 可以提升评估数据集上的稀有细胞识别效果

允许写法：

> 在评估的数据集上，scRareRefine 相比 vanilla scANVI 提升了 rare-cell macro-F1 / recall。

需要证据：

- 至少两个数据集。
- seeds 42、43、44。
- mean ± standard deviation。
- per-class rare-cell results。

### Claim 3：scANVI latent embedding 包含可用于 rare-cell refinement 的 prototype 信息

允许写法：

> 实验结果表明，scANVI latent embedding 中的类别 prototype 结构可以为稀有细胞预测修正提供额外证据。

需要证据：

- prototype-only 或 prototype-fusion 消融实验。
- latent space 可视化。
- 距离分布或相似度分析。

### Claim 4：概率与 prototype 证据融合有助于修正不确定细胞

允许写法：

> 对于部分低置信度或接近稀有细胞 prototype 的样本，融合预测概率与 prototype 相似度可以改善预测结果。

需要证据：

- probability-only、prototype-only、fusion 三组对比。
- 被修正样本的错误分析。
- uncertainty gate 消融。

### Claim 5：scRareRefine 可以在不重新训练大模型的情况下使用

允许写法：

> scRareRefine 可以作为 post-hoc refinement 模块，使用训练好的 scANVI 模型输出进行预测修正。

需要证据：

- frozen scANVI 输出下的实验。
- 运行流程说明。
- 与微调版本区分清楚。

## 需要更多证据后才能使用的主张

### Conditional Claim 1：与稀有细胞相关方法相比有竞争力

只有在与 scBalance、CIARA、scSID、RaceID 或其他相关方法公平比较后，才能写。

允许写法：

> scRareRefine 在选定数据集上与若干稀有细胞相关 baseline 相比表现出有竞争力的 rare-cell recognition performance。

需要证据：

- baseline 设置公平。
- 尽量相同数据集、相同 split、相同 rare cell 定义。
- 明确区分 annotation 和 discovery 任务。

### Conditional Claim 2：跨数据集具有一致性

只有在至少三个数据集上有稳定提升后，才能写。

允许写法：

> scRareRefine 在多个评估数据集上显示出较一致的稀有细胞识别提升。

需要证据：

- pancreas + immune_dc + Tabula Sapiens 或其他数据集。
- 相同或合理可解释的 rare cell 定义。
- seed-level 稳定结果。

### Conditional Claim 3：微调进一步提升稀有细胞识别

只有在完成微调实验后才能写。

需要比较：

```text
frozen scANVI + refinement
fine-tuned scANVI
fine-tuned scANVI + refinement
```

## 禁止使用的主张

禁止写：

1. scRareRefine 解决了稀有细胞识别问题。
2. scRareRefine 普遍优于所有细胞注释方法。
3. scRareRefine 已经临床验证。
4. scRareRefine 适用于所有单细胞数据集。
5. scRareRefine 能自动发现新的生物学细胞类型，除非有 marker gene 和领域知识验证。
6. scRareRefine 可以替代专家注释。
7. scRareRefine 是 foundation model。
8. scRareRefine 在所有指标上都优于 scANVI，除非所有指标都支持。
9. 结果显著提升，除非做了统计显著性检验。
10. 稀有细胞提升具有明确生物学意义，除非有 marker gene 或文献支持。

## 论文写作前必须收集的证据

写论文主结论前必须有：

- [ ] 数据检查报告。
- [ ] scANVI baseline 结果。
- [ ] scRareRefine 完整结果。
- [ ] 消融实验结果。
- [ ] rare-cell per-class metrics。
- [ ] 混淆矩阵。
- [ ] latent-space 可视化。
- [ ] 错误分析。
- [ ] seed mean ± std。
- [ ] rare cell 定义说明。
- [ ] baseline 公平性说明。

## 安全摘要模板

可以使用的摘要级表述：

> We propose scRareRefine, a lightweight refinement framework for rare cell type recognition based on scANVI outputs. scRareRefine combines prediction probabilities and latent-space prototype evidence to adjust uncertain predictions for rare cell populations. Experiments on selected single-cell datasets show that scRareRefine improves rare-cell-focused metrics over vanilla scANVI while maintaining comparable overall annotation performance.

中文解释：

> 我们提出 scRareRefine，一个基于 scANVI 输出结果的轻量级稀有细胞识别 refinement 框架。scRareRefine 融合预测概率和 latent-space prototype 证据，对不确定的稀有细胞相关预测进行修正。选定数据集上的实验显示，scRareRefine 相比 vanilla scANVI 提升了稀有细胞相关指标，同时保持了相近的整体注释性能。

## 不安全摘要模板

不要写：

> We propose a new universal single-cell annotation model that solves rare cell identification and outperforms all existing methods.

中文解释：

> 我们提出一个新的通用单细胞注释模型，解决了稀有细胞识别问题，并超过所有已有方法。

这种表述过强，当前项目证据不支持。

## 结果解释规则

### 情况 1：rare-cell F1 提升，但 accuracy 小幅下降

可以写：

```text
scRareRefine improves rare-cell recognition with a moderate trade-off in overall accuracy.
```

不能写：

```text
scRareRefine 全面优于 scANVI。
```

### 情况 2：只有一个数据集提升

可以写：

```text
在某个数据集上观察到稀有细胞识别提升。
```

不能写：

```text
方法具有跨数据集泛化能力。
```

### 情况 3：只有一个稀有类别提升

需要做错误分析，不能声称广泛提升稀有细胞识别。

### 情况 4：prototype-only 没有效果

需要重新检查：

1. latent space 是否有类别结构。
2. prototype 计算是否合理。
3. distance metric 是否合适。
4. rare class 是否过少。
5. 是否需要 uncertainty gate。

### 情况 5：baseline 跑不通

不能简单删除 baseline。

必须记录：

```text
baseline 名称
失败原因
是否因为任务不匹配
是否因为依赖问题
是否有替代方案
```

写入：

```text
RESULTS_LOG.md
```
