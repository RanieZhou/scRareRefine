# scRareRefine

scRareRefine 是一个面向单细胞注释场景的稀有细胞修正项目。当前实现以 scANVI 为基础模型，使用 scANVI 的预测概率和 latent embedding，结合 prototype、fusion、prototype gate 和 marker verification 来评估稀有细胞类型识别效果。

本目录是当前标准项目根目录。后续开发、测试和实验运行应优先在本目录下进行。

## 当前实现状态

- 当前 Python 包名：`scrare`
- 当前核心代码：`src/scrare/`
- 当前 CLI：`src/scrare/cli/`
- 当前测试：`tests/`
- 当前可运行配置：`configs/*.yaml`
- 原始数据位置：`data/raw/`
- 当前报告归档：`results/reports/`
- 当前报告图表归档：`results/figures/`

模板中曾出现过的 `sc_rare_refine` 包名不是当前实现包名；当前运行、测试和打包都以 `scrare` 为准。

## 安装

推荐在本目录下安装：

```bash
python -m pip install -e .[dev]
```

## 常用命令

### 数据审计

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
```

### 主 inductive 实验

```bash
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
```

常用单个 slice 示例：

```bash
python -m scrare.cli.run_inductive \
  --config configs/immune_dc.yaml \
  --rare-class cDC1 \
  --split-mode batch_heldout \
  --seed 42 \
  --rare-train-size 20
```

### posthoc 评估

```bash
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

### 测试

```bash
pytest tests/test_project_state.py tests/cli/test_cli_smoke.py -v
pytest -v
```

## 配置说明

当前保留可运行的扁平配置：

```text
configs/immune_dc.yaml
configs/pancreas_epsilon.yaml
configs/pancreas_gamma.yaml
```

标准子目录 `configs/datasets/`、`configs/experiments/`、`configs/paths/` 会保留，但当前代码入口仍使用上面的扁平配置。后续如要拆分配置，需要单独设计和验证。

## 数据规则

原始数据位于：

```text
data/raw/
```

硬规则：

1. 不修改 `data/raw/`。
2. 不覆盖原始 `.h5ad` 文件。
3. 不删除原始数据。
4. 任何处理后数据应写入 `data/processed/`、`data/splits/` 或 `data/embeddings/`。

## 结果目录

标准结果目录为：

```text
results/raw/
results/tables/
results/figures/
results/reports/
```

本次迁移已将当前实验报告归档到：

```text
results/reports/
```

并将报告图表归档到：

```text
results/figures/
```

当前代码中的部分配置仍可能写入 `outputs/`。这是兼容当前实现的过渡状态；将运行输出完全切换到 `results/` 应作为后续单独任务处理。

## 核心约束

当前 workflow 是 inductive evaluation。修改代码时必须保持：

1. held-out cells 不进入训练 reference。
2. HVG 选择只基于训练集。
3. prototype reference 只来自训练集或合法 reference。
4. fusion 参数选择只基于 validation。
5. marker signature 和阈值选择不能使用 test 标签。
6. 指标报告不能只看 accuracy，必须关注 rare macro-F1、rare recall、rare precision、macro-F1 和 balanced accuracy。

## 不在本目录内继续扩散的内容

论文草稿、论文专用图、备份目录和历史输出缓存不应随意放在项目根目录。需要归档时，应先明确放入 `docs/`、`results/`、`tmp/` 或单独设计论文产物归档任务。
