# AGENTS.md

## 目的

本文档定义 ARIS、Codex、Claude Code 或其他 AI 编程/科研 agent 在 scRareRefine 项目中的工作规则。

agent 必须先阅读：

```text
PROJECT_STRUCTURE.md
docs/PROJECT_BRIEF.md
docs/DATA_CARD.md
docs/EXPERIMENT_PLAN.md
docs/CLAIMS.md
RESULTS_LOG.md
```

## 项目概要

项目名称：**scRareRefine**。

项目目标：基于 scANVI 的输出结果，提高稀有细胞类型识别效果。

scRareRefine 使用：

1. scANVI 预测概率。
2. scANVI latent embedding。
3. latent space 中的类别 prototype。
4. 面向稀有细胞的 refinement 逻辑。
5. 可选的 uncertainty gate。
6. 可选的 scANVI 微调。

项目目标是支撑一篇 Q2 级别生信小论文，并作为硕士论文三个工作之一。

## 不可违反的规则

### 1. 修改前必须先给计划

在修改任何代码、配置、实验脚本或文档前，agent 必须先输出：

```text
1. 本次修改目标
2. 准备修改的文件
3. 预期输出结果
4. 潜在风险
5. 测试或验证方式
```

除非用户明确说“直接执行”，否则不能先改后说。

### 2. 禁止修改原始数据

以下内容视为只读：

```text
data/raw/
原始 .h5ad 文件
未备份的原始输入数据
```

禁止：

```text
覆盖原始 .h5ad
重写 data/raw/ 下的文件
删除原始数据
在原始数据文件上直接做 inplace 修改
```

### 3. 禁止随意新建混乱目录

顶层目录必须符合 `PROJECT_STRUCTURE.md`。

禁止新建：

```text
new_code/
final/
backup2/
test123/
临时实验/
新版结果/
随便命名的目录/
```

临时文件必须放在：

```text
tmp/
```

实验结果必须放在：

```text
results/raw/{experiment_id}/
```

核心代码必须放在：

```text
src/sc_rare_refine/
```

可执行入口脚本必须放在：

```text
scripts/
```

### 4. 禁止无记录地改变实验设置

agent 不能悄悄改变：

- 随机种子。
- train/validation/test 划分规则。
- rare cell 定义。
- label 列。
- batch 列。
- 预处理流程。
- baseline 设置。
- 指标定义。
- scANVI 训练参数。

任何变化都必须写入：

```text
RESULTS_LOG.md
对应的 config.yaml
results/raw/{experiment_id}/run.log
```

### 5. 禁止未经验证的科研主张

没有实验支持时，agent 不得写出：

```text
显著优于
state-of-the-art
全面超过
临床可用
适用于所有单细胞数据
解决了稀有细胞识别问题
```

论文主张必须符合：

```text
docs/CLAIMS.md
```

## 允许操作

agent 可以：

1. 阅读项目文件。
2. 新建脚本。
3. 在给出计划并获得认可后修改代码。
4. 运行小规模测试。
5. 在明确计划下运行实验。
6. 生成结果总结。
7. 生成图表。
8. 更新文档。
9. 给不确定信息添加 TODO。
10. 整理目录，但不能移动原始数据。

## 受限操作

以下操作必须获得明确同意：

1. 安装新 Python 包。
2. 修改环境文件。
3. 运行长时间实验。
4. 微调 scANVI。
5. 改变数据预处理规则。
6. 使用 sudo/admin 权限。
7. git commit。
8. git push。
9. 删除文件。
10. 移动已有实验结果。

## 禁止操作

agent 不得：

1. 修改原始数据。
2. 删除大目录。
3. 伪造结果。
4. 手动编辑结果文件让指标变好。
5. 在训练中错误使用测试标签。
6. 故意削弱 baseline。
7. 只报告成功 seed。
8. 隐藏失败实验。
9. 编造论文结论。
10. 把核心代码散落到项目根目录。

## 标准目录规则

核心规则：

```text
文档：docs/
配置：configs/
原始数据：data/raw/
中间数据：data/processed/、data/splits/、data/embeddings/
核心代码：src/sc_rare_refine/
入口脚本：scripts/
探索笔记：notebooks/
测试：tests/
正式结果：results/
日志：logs/
模型权重：checkpoints/
临时文件：tmp/
```

如果 agent 不确定文件应该放在哪里，必须先询问或写入 `tmp/`，不能随意新建目录。

## 每次实验必须记录的信息

每个正式实验必须记录：

```text
experiment_id
日期
数据集
随机种子
方法名
配置文件
git commit hash
label column
batch column
rare-cell definition
评价指标
输出路径
主要观察
失败原因或异常
```

并更新：

```text
RESULTS_LOG.md
```

## 每次实验必须保存的结果文件

每个实验目录至少包含：

```text
results/raw/{experiment_id}/predictions.csv
results/raw/{experiment_id}/metrics.json
results/raw/{experiment_id}/config.yaml
results/raw/{experiment_id}/environment.txt
results/raw/{experiment_id}/run.log
```

## 代码质量规则

1. 优先写小函数。
2. 避免硬编码路径。
3. 所有实验参数必须 config 化。
4. 重要函数必须写 docstring。
5. 读取数据后必须检查 shape。
6. 必须检查 label 是否缺失。
7. 必须检查 train/test label 映射是否一致。
8. 计算指标前必须确认预测数组和标签数组长度一致。
9. 不允许把大量核心逻辑写在 notebook 中。
10. 不允许在脚本里散落重复代码。

## agent 开始工作前的固定检查

每次开始任务前，agent 应先确认：

```text
1. 当前任务属于文档、代码、实验、结果分析还是论文写作？
2. 是否需要修改 data/raw/？如果需要，必须拒绝。
3. 是否需要新增顶层目录？如果需要，必须遵守 PROJECT_STRUCTURE.md。
4. 是否会改变实验设置？如果会，必须更新 config 和 RESULTS_LOG.md。
5. 是否有测试或验证命令？如果没有，需要先补充。
```
