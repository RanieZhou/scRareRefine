# PROJECT_STRUCTURE.md

## 目的

本文档用于规范 scRareRefine 项目的目录结构，避免 AI agent 在自动修改、实验运行和结果保存过程中把项目目录弄乱。

当前标准项目根目录是本目录。所有后续开发和实验应优先在本目录下执行。

## 当前标准目录结构

```text
scRareRefine-standard/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── PROJECT_STRUCTURE.md
├── RESULTS_LOG.md
├── pyproject.toml
├── configs/
│   ├── immune_dc.yaml
│   ├── pancreas_epsilon.yaml
│   ├── pancreas_gamma.yaml
│   ├── datasets/
│   ├── experiments/
│   └── paths/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   ├── embeddings/
│   └── external/
├── docs/
├── notebooks/
├── scripts/
├── src/
│   └── scrare/
│       ├── cli/
│       ├── data/
│       ├── evaluation/
│       ├── infra/
│       ├── models/
│       ├── visualization/
│       └── workflows/
├── tests/
├── results/
│   ├── raw/
│   ├── tables/
│   ├── figures/
│   └── reports/
├── logs/
├── checkpoints/
└── tmp/
```

## 当前包名

当前实现包名是：

```text
scrare
```

核心代码位置是：

```text
src/scrare/
```

所有 import、CLI 和测试都应使用 `scrare`。不要把新代码写入其他包名目录。

如果仍看到 `sc_rare_refine` 模板目录，它是模板遗留目录。本次迁移保留该目录，但不作为安装包、运行入口或新代码位置；不要向其中新增逻辑。如需删除该旧模板包目录，必须单独取得用户同意。

## 各目录职责

### `src/scrare/`

核心 Python 包。主要子目录：

```text
cli/            命令行入口
workflows/      inductive 主流程和 posthoc 编排
data/           数据读取、预处理和 split
models/         scANVI、prototype、fusion、marker、gate 等方法模块
evaluation/     指标、审计和 posthoc 评估
infra/          配置、I/O、路径和资源监控
visualization/  实验图表生成
```

### `configs/`

当前正式可运行配置仍位于：

```text
configs/immune_dc.yaml
configs/pancreas_epsilon.yaml
configs/pancreas_gamma.yaml
```

标准子目录保留为后续拆分使用：

```text
configs/datasets/
configs/experiments/
configs/paths/
```

在配置系统正式重构前，不要无记录地改变 seed、split、rare cell 定义、label 列、batch 列或模型训练参数。

### `data/`

数据目录：

```text
data/raw/          原始数据，只读
data/processed/    处理后数据
data/splits/       train/validation/test split 文件
data/embeddings/   latent、概率输出和 label mapping
data/external/     外部 baseline 中间文件
```

硬规则：

1. 禁止修改 `data/raw/`。
2. 禁止覆盖原始 `.h5ad` 文件。
3. 禁止删除原始数据。
4. 保存处理后数据时必须写入非 raw 子目录。

### `results/`

标准结果目录：

```text
results/raw/       单次正式实验目录
results/tables/    汇总表
results/figures/   图表
results/reports/   报告
```

当前迁移后的历史报告归档在 `results/reports/`，报告图表归档在 `results/figures/`。

### `tests/`

测试目录。核心验证包括：

1. split 无泄漏。
2. train-only HVG。
3. prototype、fusion、marker、gate 行为。
4. CLI smoke test。
5. 项目状态和目录约束。

### `docs/`

研究设计、论文边界、开发计划和长期说明文档。不要把正式实验输出直接写入 `docs/`；实验输出优先写入 `results/`。

### `scripts/`

仅放薄脚本入口或辅助脚本。当前正式入口优先使用：

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

### `logs/`、`checkpoints/`、`tmp/`

```text
logs/         日志
checkpoints/  模型权重
tmp/          临时文件
```

临时文件不得放在项目根目录。

## 禁止新增的顶层目录

不要新建以下类型目录：

```text
new_code/
final/
backup2/
test123/
临时实验/
新版结果/
随便命名的目录/
```

确实需要新增目录时，优先放入已有标准目录。

## 实验 ID 规则

正式实验目录建议使用：

```text
results/raw/EXP-YYYYMMDD-XXX/
```

每个正式实验至少保存：

```text
predictions.csv
metrics.json
config.yaml
environment.txt
run.log
```

当前代码仍有过渡性 `outputs/` 写入逻辑。将运行输出完全切换到 `results/` 需要单独设计和测试。
