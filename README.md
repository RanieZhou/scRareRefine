# scRareRefine

scRareRefine 是一个面向单细胞注释场景的稀有细胞修正项目，核心思路是：

1. 先用 scANVI 做基础细胞类型预测；
2. 再基于训练集参考细胞的 latent prototype 进行后处理融合；
3. 对候选稀有细胞再结合 marker 规则做进一步验证。

当前仓库保留的是代码、配置和测试；数据目录与实验输出目录没有上传到 GitHub。

## 为什么本地 clone 后没有 outputs 目录？

这是当前项目的预期行为，不是仓库损坏。

- `data/` 和 `outputs/` 已在 `.gitignore` 中忽略；
- 原因是相关数据文件和实验结果体积较大，没有随 GitHub 仓库一起上传；
- 因此新 clone 的仓库默认只有代码和配置，不会自带运行结果。

如果你要复现实验，需要自行准备配置文件中引用的 `.h5ad` 数据文件；运行工作流后，`outputs/` 会在本地自动生成。

## 环境要求

- Python >= 3.10
- 主要依赖见 `pyproject.toml`，包括：`anndata`、`scanpy`、`scvi-tools`、`torch`、`pandas`、`numpy`、`scikit-learn` 等

推荐安装方式：

```bash
python -m pip install -e .[dev]
```

## 项目结构

- `src/scrare/`：主 Python 包
- `src/scrare/cli/`：当前命令行入口模块
- `src/scrare/workflows/`：inductive 主流程与 posthoc 评估编排
- `src/scrare/data/`：数据加载、预处理、split 构造
- `src/scrare/models/`：scANVI、prototype、fusion、marker 等建模逻辑
- `src/scrare/evaluation/`：指标、审计与后处理评估
- `src/scrare/infra/`：配置、I/O、路径、资源监控等基础设施
- `configs/`：数据集与实验参数配置
- `tests/`：按模块划分的测试

## 核心流程

项目当前使用的是 **inductive workflow**，核心约束是：

- 训练、验证、测试集严格分开；
- 稀有类标注预算只暴露给训练集；
- prototype、marker signature、参数选择都应基于训练集或验证集，而不是测试集。

主流程如下：

1. 从配置文件读取数据集路径、标签列、batch 列和实验参数；
2. 构造 train / validation / test split；
3. 在训练集中为 rare class 保留有限数量的标注样本；
4. 训练 SCVI，再转换为 SCANVI；
5. 对验证集和测试集做推断；
6. 基于训练集 reference latent 构造 prototype 概率；
7. 在验证集上选择 fusion 参数；
8. 在测试集上报告 fusion 效果；
9. 对保存下来的 run 结果进一步做 prototype gate 和 marker verification 分析。

## 配置文件

当前仓库包含三个示例配置：

- `configs/immune_dc.yaml`
- `configs/pancreas_epsilon.yaml`
- `configs/pancreas_gamma.yaml`

配置里最重要的字段包括：

- `dataset.path`：输入 `.h5ad` 文件路径
- `dataset.label_key`：真实标签列
- `dataset.batch_key`：批次列
- `dataset.use_raw` 或 `dataset.use_layer`：使用哪一层表达矩阵
- `experiment.rare_class`：主稀有类
- `experiment.rare_train_sizes`：训练集中稀有类标注预算
- `experiment.seeds`：随机种子列表
- `model.n_top_hvg`、`model.n_latent`、`model.scvi_max_epochs`、`model.scanvi_max_epochs`

## 如何运行

### 1. 数据审计

根据配置读取数据，并输出数据集摘要、类别分布和 batch 分布：

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
```

### 2. 运行主实验：inductive scANVI + fusion

最常见的运行方式：

```bash
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
```

### 3. 运行下游 prototype / marker 分析

该命令不会重新训练模型，而是复用主实验产生的 artifacts：

```bash
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

也可以通过安装后的 console scripts 调用：

```bash
scrare-audit --config configs/immune_dc.yaml
scrare-run-inductive --config configs/immune_dc.yaml
scrare-evaluate-posthoc --config configs/immune_dc.yaml
```

## 输出目录说明

### 数据审计输出

审计命令会把结果写到配置中的 `experiment.output_dir` 下，并在对应根目录下补充表格。

例如 `configs/immune_dc.yaml` 当前配置的是：

```yaml
experiment:
  output_dir: outputs/immune_dc/audit
```

### 主实验输出

inductive workflow 默认会按如下结构自动组织输出：

```text
outputs/<dataset>/<inductive_cell|inductive_batch>/<rare_class>/
```

其中通常包含：

- `runs/<run>/artifacts/`：每次运行的预测结果、latent、模型等中间产物
- `stages/fusion/`：fusion 汇总结果
- `tables/`：根级别汇总表

### 下游分析输出

posthoc 评估会在已有 run 目录基础上继续写入：

```text
stages/prototype_marker_validation/
```

## 测试

运行全部测试：

```bash
pytest -v
```

运行单个测试文件：

```bash
pytest tests/test_inductive.py
```

按关键字筛选测试：

```bash
pytest tests/test_inductive.py -k rare_train_size
```

`tests/test_project_state.py` 会显式检查当前仓库没有重新引入旧的脚本入口和旧包实现。

## 使用前的最小检查清单

在正式运行前，建议先确认：

1. 配置文件中的 `dataset.path` 在你本地真实存在；
2. `.h5ad` 中包含配置指定的 `label_key` 和 `batch_key`；
3. 如果配置使用 `use_raw: true`，对应数据对象里确实存在 `adata.raw`；
4. 如果配置使用 `use_layer`，对应 layer 名称存在；
5. 已正确安装 `scvi-tools` 与 `torch` 相关依赖。
