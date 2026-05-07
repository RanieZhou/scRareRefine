# CLAUDE.md

## 给 Claude Code / Codex / ARIS 的项目指令

你正在协助一个生信科研项目：**scRareRefine**。

该项目目标是基于 scANVI 的预测概率和 latent embedding，设计稀有细胞识别 refinement 模块，提高 rare cell type 的识别效果。

## 你必须遵守的优先级

当不同指令冲突时，优先级如下：

```text
1. 不修改原始数据
2. 不伪造结果
3. 不破坏标准项目结构
4. 不无记录地改变实验设置
5. 不写未经验证的科研结论
6. 完成用户指定任务
```

## 工作前必须阅读

在进行任何修改前，先阅读：

```text
PROJECT_STRUCTURE.md
AGENTS.md
docs/PROJECT_BRIEF.md
docs/DATA_CARD.md
docs/EXPERIMENT_PLAN.md
docs/CLAIMS.md
RESULTS_LOG.md
```

## 修改前计划模板

每次修改前，先输出：

```text
本次任务目标：

计划修改文件：
1.
2.
3.

不会修改的内容：
- data/raw/
- 原始 .h5ad 文件
- 已有正式结果，除非用户明确要求

预期输出：

风险：

验证方式：
```

用户确认后再修改。

## 标准目录要求

不要把代码直接扔到项目根目录。

正确位置：

```text
src/sc_rare_refine/      核心代码
scripts/                 可执行入口脚本
configs/                 配置文件
results/                 实验结果
docs/                    科研设计文档
tests/                   测试文件
notebooks/               探索分析
logs/                    日志
checkpoints/             模型权重
tmp/                     临时文件
```

禁止随意新建：

```text
code/
new/
final/
backup/
test/
临时/
实验/
新版/
```

## 当前方法假设

scRareRefine 的基本输入包括：

1. scANVI 预测概率矩阵。
2. scANVI latent embedding。
3. 训练集或参考集中每个 cell type 的 label。
4. 稀有细胞定义。

可能的 refinement 方式包括：

1. 根据 latent embedding 计算每个类别的 prototype。
2. 计算每个细胞到各类别 prototype 的距离或相似度。
3. 将 scANVI 概率与 prototype 相似度融合。
4. 对低置信度、高熵或疑似 rare cell 的样本进行修正。
5. 对高置信度预测保持不变。

## 推荐代码模块位置

```text
src/sc_rare_refine/data/inspect.py
src/sc_rare_refine/models/scanvi_wrapper.py
src/sc_rare_refine/refinement/prototypes.py
src/sc_rare_refine/refinement/fusion.py
src/sc_rare_refine/refinement/uncertainty.py
src/sc_rare_refine/refinement/rare_adjustment.py
src/sc_rare_refine/metrics/rare_metrics.py
src/sc_rare_refine/visualization/rare_report.py
```

入口脚本：

```text
scripts/01_inspect_data.py
scripts/02_train_scanvi.py
scripts/03_extract_scanvi_outputs.py
scripts/04_run_refinement.py
scripts/05_evaluate.py
scripts/06_collect_results.py
scripts/07_plot_results.py
```

## 实验要求

默认随机种子：

```text
42
43
44
```

当前小规模实验优先级：

```text
1. human_pancreas_norm_complexBatch.h5ad
2. human_immune_health_atlas_dc.h5ad
3. Tabula Sapiens
```

正式实验前必须先完成数据检查：

```text
scripts/01_inspect_data.py
```

检查内容：

```text
1. obs 列名
2. label 列候选
3. batch/donor/study 列候选
4. 每个细胞类型数量
5. 稀有细胞类型候选
6. 是否存在缺失标签
7. 是否存在极端类别不平衡
```

## 评价指标

不能只看 accuracy。

必须重点报告：

```text
rare macro-F1
rare recall
rare precision
per-class F1
macro-F1
balanced accuracy
weighted-F1
accuracy
```

可选：

```text
ECE
Brier score
entropy
confidence distribution
rare-vs-common performance gap
```

## baseline 规则

必须包含：

```text
scANVI 原始结果
scRareRefine 完整方法
```

建议包含：

```text
scANVI embedding + MLP
scANVI embedding + Logistic Regression
scANVI embedding + SVM
scBalance
CIARA 或 scSID，取决于是否能公平运行
```

比较时必须保证：

```text
相同数据集
相同 split
相同 rare cell 定义
相同 seed
相同评价指标
```

## 论文主张限制

不要写：

```text
解决了稀有细胞识别问题
全面优于所有方法
state-of-the-art
临床可用
适用于所有单细胞数据
```

可以写：

```text
在评估的数据集上，scRareRefine 相比原始 scANVI 提升了稀有细胞相关指标。
```

前提是结果支持。

## 输出风格

你应优先输出：

1. 清晰计划。
2. 小步修改。
3. 可复现实验命令。
4. 明确文件路径。
5. 风险提醒。
6. 结果记录。

不要输出：

1. 大段泛泛解释。
2. 未验证结论。
3. 未说明路径的代码。
4. 混乱目录结构。
5. 没有 config 的实验脚本。
