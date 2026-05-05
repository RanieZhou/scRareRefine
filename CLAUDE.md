# CLAUDE.md

本文件为 Claude Code 在本仓库中工作时提供约束与导航。

always reply in Chinese

## Development commands

- Install package + test dependencies: `python -m pip install -e .[dev]`
- Run the full test suite: `pytest -v`
- Run a single test file: `pytest tests/test_inductive.py`
- Run a single test case: `pytest tests/test_inductive.py -k rare_train_size`
- Audit a dataset from config: `python -m scrare.cli.audit --config configs/immune_dc.yaml`
- Run the main inductive pipeline: `python -m scrare.cli.run_inductive --config configs/immune_dc.yaml`
- Run the downstream posthoc evaluation over saved runs: `python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml`

## Repository shape

本仓库采用 `src/` 布局，主 Python 包位于 `src/scrare/`。

- `src/scrare/cli/`: CLI 入口模块
- `src/scrare/workflows/`: 工作流编排，包括 inductive 主流程与 posthoc 工作流
- `src/scrare/data/`: 数据读取、预处理、split 逻辑
- `src/scrare/models/`: scANVI、prototype、fusion、marker 等模型与后处理逻辑
- `src/scrare/evaluation/`: 指标、数据审计与后处理评估，其中包含 `src/scrare/evaluation/posthoc.py`
- `src/scrare/infra/`: 配置、I/O、路径、资源监控等基础设施
- `configs/`: YAML 配置
- `tests/`: 按子系统组织的测试

## High-level architecture

### 1. Config-driven data loading

- 配置文件位于 `configs/`。
- `scrare.infra.config` 负责加载 YAML 并提供输出目录辅助函数。
- `scrare.data.loading` 与 `scrare.data.preprocess` 负责把配置转成 `AnnData` 并选择合适的表达矩阵。

重要配置键包括：

- `dataset.path`, `dataset.label_key`, `dataset.batch_key`
- `dataset.use_raw` 或 `dataset.use_layer`
- `experiment.rare_class`, `experiment.secondary_rare_classes`, `experiment.rare_train_sizes`, `experiment.seeds`, `experiment.unlabeled_category`
- `model.n_top_hvg`, `model.n_latent`, `model.scvi_max_epochs`, `model.scanvi_max_epochs`, `model.batch_size`

### 2. Inductive split and partial-label construction

核心设计约束是 inductive evaluation，不能把 held-out cells 泄漏到训练参考中。

- `scrare.data.splits` 实现 train/validation/test split 与部分标注构造。
- 训练集中的 major classes 保持标注。
- rare class 通过 `rare_train_size` 控制标注预算。
- validation/test cells 保持未标注。
- HVG 选择必须仅基于训练集。

如果修改评估逻辑，必须保持 train-only reference 假设：prototype、marker signature 与调参只能基于训练集或验证集，不能依赖测试标签。

### 3. Main experiment pipeline

`python -m scrare.cli.run_inductive --config ...` 会进入 `src/scrare/cli/run_inductive.py`，再调用 `src/scrare/workflows/inductive.py`。

主流程按 `(rare_class, split_mode, seed, rare_train_size)` 组合执行，通常包括：

1. 读取并可选下采样数据集。
2. 构建 train/validation/test split。
3. 创建 `scanvi_label` 与 `is_labeled_for_scanvi` 等监督字段。
4. 仅基于训练集选择 HVGs。
5. 训练 `SCVI`，再转换为 `SCANVI`。
6. 对 validation/test 做 query inference。
7. 输出预测、latent、split、HVG、metrics、confusion tables 与资源使用摘要。
8. 基于训练集 reference latent 计算 prototype 概率并进行 validation-driven fusion 搜索。
9. 用验证集选出融合参数，再在测试集上做最终评估。

### 4. Posthoc evaluation

`python -m scrare.cli.evaluate_posthoc --config ...` 会进入 `src/scrare/cli/evaluate_posthoc.py`，再调用 `src/scrare/workflows/posthoc.py`。

- 该流程不会重新训练模型。
- 它会复用已有 run artifacts，重新读取 train/validation/test 预测与 latent。
- `src/scrare/evaluation/posthoc.py` 提供与 posthoc 评估相关的逻辑。
- prototype gate 与 marker verification 的阈值选择应基于 validation，再应用到 test。

### 5. Metrics and outputs

- `scrare.evaluation.metrics` 提供共享分类与不确定性指标。
- `scrare.infra.paths`、`scrare.infra.io` 定义输出布局与表格读写。
- `scrare.infra.resources` 提供运行资源监控实现。

## Tests

测试按子系统组织。常用锚点：

- `tests/test_inductive.py`: split 正确性、标签隐藏、train-only HVG 行为
- `tests/test_fusion.py`: prototype 概率与 fusion 权重
- `tests/test_prototype.py` 和 `tests/test_prototype_gate.py`: rescue scoring 与 gate 规则
- `tests/test_marker_verifier.py`: marker signature、阈值选择、rescue 评估
- `tests/test_project_state.py`: 防止重新引入旧脚本入口、旧包实现或历史输出根目录

修改核心工作流后，优先运行对应子系统测试，再运行 `pytest -v` 全量验证。
