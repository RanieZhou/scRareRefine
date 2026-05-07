# CLAUDE.md

always reply in Chinese

## 项目身份

你正在协助 scRareRefine 项目。当前标准项目根目录是本目录，当前 Python 包名是 `scrare`，核心代码位于 `src/scrare/`。

scRareRefine 的目标是基于 scANVI 的预测概率和 latent embedding，设计稀有细胞识别 refinement 模块，提高 rare cell type 的识别效果。

## 优先级

当指令冲突时，按以下优先级执行：

1. 不修改原始数据。
2. 不伪造结果。
3. 不破坏标准项目结构。
4. 不无记录地改变实验设置。
5. 不写未经验证的科研结论。
6. 完成用户指定任务。

## 修改前计划模板

修改代码、配置、实验脚本或文档前，先输出：

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

用户确认后再修改，除非用户明确要求直接执行。

## Development commands

在本目录下运行：

```bash
python -m pip install -e .[dev]
pytest -v
pytest tests/test_inductive.py
pytest tests/test_inductive.py -k rare_train_size
python -m scrare.cli.audit --config configs/immune_dc.yaml
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

## Repository shape

```text
src/scrare/cli/            CLI 入口
src/scrare/workflows/      inductive 主流程与 posthoc 工作流
src/scrare/data/           数据读取、预处理、split 逻辑
src/scrare/models/         scANVI、prototype、fusion、marker、gate
src/scrare/evaluation/     指标、数据审计与 posthoc 评估
src/scrare/infra/          配置、I/O、路径、资源监控
src/scrare/visualization/  图表生成
configs/                   当前可运行配置
tests/                     测试
results/                   标准结果归档
data/raw/                  原始数据，只读
```

## 核心 inductive 约束

如果修改评估逻辑，必须保持 train-only reference 假设：

1. validation/test cells 不能泄漏到训练 reference。
2. HVG 选择必须仅基于训练集。
3. prototype reference 只能来自训练集或合法 reference。
4. marker signature 只能来自训练集有标签样本。
5. fusion 参数只能从 validation 选择。
6. test 标签不能用于调参、阈值选择或 marker signature 构建。

## 配置约束

当前可运行配置：

```text
configs/immune_dc.yaml
configs/pancreas_epsilon.yaml
configs/pancreas_gamma.yaml
```

重要配置键：

```text
dataset.path
dataset.label_key
dataset.batch_key
dataset.use_raw
dataset.use_layer
experiment.rare_class
experiment.secondary_rare_classes
experiment.rare_train_sizes
experiment.seeds
experiment.unlabeled_category
model.n_top_hvg
model.n_latent
model.scvi_max_epochs
model.scanvi_max_epochs
model.batch_size
```

不要无记录地改变随机种子、split、rare cell 定义、label 列、batch 列、预处理流程、baseline 设置、指标定义或 scANVI 训练参数。

## 数据保护规则

以下内容只读：

```text
data/raw/
原始 .h5ad 文件
未备份的原始输入数据
```

禁止：

```text
覆盖原始 .h5ad
删除原始数据
在原始 AnnData 对象上直接做 inplace 修改并保存回原文件
```

## 结果与文档规则

1. 实验输出优先进入 `results/`。
2. 日志进入 `logs/`。
3. 模型权重进入 `checkpoints/`。
4. 临时文件进入 `tmp/`。
5. 不要在根目录随意新建目录。
6. 不要把论文草稿、图表、备份和历史输出缓存混放到项目根目录。

## 论文主张限制

不要写：

```text
解决了稀有细胞识别问题
全面优于所有方法
state-of-the-art
临床可用
适用于所有单细胞数据
```

可以写但必须有结果支持：

```text
在评估的数据集上，scRareRefine 相比原始 scANVI 提升了稀有细胞相关指标。
```

## Git 规则

不要创建 git commit、git push、force push、reset、删除分支或清理大目录，除非用户明确要求。
