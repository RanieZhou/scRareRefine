# run_inductive 恢复与单入口复用设计

## 背景

当前 `python -m scrare.cli.run_inductive` 已接入 [src/scrare/cli/run_inductive.py](src/scrare/cli/run_inductive.py)，但 [src/scrare/workflows/inductive.py](src/scrare/workflows/inductive.py) 仍是占位实现，无法产出 baseline artifacts。与此同时，主流程依赖的 `scrare.data` 子包在 src 重构中缺失，导致现有 workflow 与测试无法导入。

本设计恢复 `run_inductive` 为唯一主实验入口，并保持 train-only reference / validation-driven selection / test-only final evaluation 的约束。

## 目标

1. 恢复 `run_inductive` 为可执行主实验入口。
2. 默认在单个入口下生成 5 组方法结果：
   - `baseline`
   - `baseline_plus_prototype`
   - `baseline_plus_prototype_gate`
   - `baseline_plus_prototype_gate_plus_marker`
   - `baseline_plus_fusion`
3. 支持 `--methods` 只补算指定方法。
4. 支持复用既有 baseline artifacts，避免重复训练。
5. 保留 `evaluate_posthoc` 作为既有 artifacts 的补算/复核入口，而不是唯一生成入口。

## 非目标

1. 本次不新增第二套并行 CLI。
2. 本次不改变现有输出根路径分类方式。
3. 本次不放宽 train/validation/test 泄漏边界。

## 方法口径

主流程统一使用以下 5 种方法键：

- `baseline`
- `baseline_plus_prototype`
- `baseline_plus_prototype_gate`
- `baseline_plus_prototype_gate_plus_marker`
- `baseline_plus_fusion`

后续 `evaluate_posthoc` 与共享评估逻辑需对齐这套 taxonomy，避免主流程与补算流程口径漂移。

## CLI 设计

`src/scrare/cli/run_inductive.py` 继续只做参数解析与 workflow 调度。

新增参数：

- `--rare-class`
- `--split-mode`
- `--seed`
- `--rare-train-size`
- `--methods`
- `--reuse-baseline-only`
- `--output-dir`
- `--max-cells`
- `--scvi-epochs`
- `--scanvi-epochs`
- `--train-fraction`
- `--validation-fraction`
- `--test-fraction`
- `--max-accuracy-drop`
- `--max-false-rescue-rate`

行为：

- 默认不传 `--methods` 时，运行全部 5 种方法。
- 传 `--methods` 时，仅补算目标方法。
- 传 `--reuse-baseline-only` 时，若 baseline artifacts 不完整则直接失败，不触发训练。

## 工作流分层

### CLI 层

- 解析参数
- `load_config`
- 调用 `run_inductive_workflow(config, args)`

### Workflow 编排层

`src/scrare/workflows/inductive.py` 负责：

- 枚举 `(rare_class, split_mode, seed, rare_train_size)` slices
- 判定 baseline 训练 / baseline 复用
- 在 baseline artifacts 就绪后运行其余方法
- 汇总并写出结果表

建议内部职责划分：

- `_iter_experiment_slices(...)`
- `_normalize_methods(...)`
- `_output_root(...)`
- `_run_name(...)`
- `_load_existing_baseline_artifacts(...)`
- `_baseline_artifacts_complete(...)`
- `_run_baseline_slice(...)`
- `_run_methods_from_baseline(...)`
- `_write_method_outputs(...)`

### 数据准备层

恢复 `src/scrare/data/` 子包：

- `loading.py`
- `preprocess.py`
- `splits.py`

其中提供：

- `adata_from_config`
- `subset_cells`
- `ensure_unique_names`
- `select_train_hvg_var_names`
- `batch_heldout_split`
- `cell_stratified_split`
- `make_inductive_scanvi_labels`
- `parse_rare_train_size`

## 数据流

### 阶段 A：baseline 生成

输入：

- config
- rare_class
- split_mode
- seed
- rare_train_size

流程：

1. 加载 AnnData
2. 可选下采样
3. 构造 train/validation/test split
4. 构造 `scanvi_label` 与 `is_labeled_for_scanvi`
5. 仅基于 train 选择 HVGs
6. 训练 SCVI -> SCANVI
7. 对 train / validation / test 产出 predictions 与 latent
8. 落盘 baseline artifacts

输出 artifacts：

- `train_predictions.csv`
- `validation_predictions.csv`
- `test_predictions.csv`
- `train_latent.csv`
- `validation_latent.csv`
- `test_latent.csv`
- `selected_hvg_genes.csv`
- `split_assignments.csv`
- baseline metrics / per-class metrics / resources

### 阶段 B：基于 baseline artifacts 衍生 4 组增强方法

输入：

- train/validation/test predictions
- train/validation/test latent
- selected genes
- rare_class / split_mode / seed / rare_train_size
- adata（用于 marker expression）

行为：

- prototype：从 train reference latent 派生 prototype 分数/概率
- prototype gate：在 baseline prediction 上施加 gate 规则
- marker：用 validation 选阈值，再应用到 test
- fusion：用 validation 网格搜索参数，再应用到 test

约束：

- reference 只能来自 train
- 参数选择只能用 validation
- test 只用于最终评估

## 输出布局

延续当前路径规范：

- `outputs/<dataset>/inductive_cell/<rare_class>/`
- `outputs/<dataset>/inductive_batch/<rare_class>/`

### run 级 baseline 原子产物

放在：

- `runs/<run>/artifacts/...`

### stage 级方法结果

放在：

- `stages/inductive_methods/...`

建议文件：

- `five_method_effect_runs.csv`
- `five_method_effect_summary.csv`
- `selected_marker_thresholds.csv`
- `prototype_test_candidates.csv`
- `prototype_gate_test_candidates.csv`
- `marker_verified_test_candidates.csv`
- `fusion_validation_grid.csv`
- `selected_fusion_params.csv`

## 幂等与失败语义

- baseline 缺失：默认允许训练；若 `--reuse-baseline-only` 则直接失败。
- baseline contract 不完整：视为损坏，直接报错，列出缺失文件。
- 仅补方法时：不重训 baseline，不重写 baseline 原子产物。
- 方法结果已存在：允许针对当前 slice 精确覆盖重算。

## 测试策略

### CLI / smoke

- parser 正确暴露新增参数
- `main([])` 仍因缺 `--config` 触发 argparse 失败

### workflow 编排测试

- baseline 缺失时会触发 baseline 分支
- baseline 存在且启用复用时只加载 artifacts
- `--methods baseline` 只产出 baseline
- `--methods baseline_plus_fusion` 在已有 baseline 下不重训
- `--reuse-baseline-only` 在 baseline 缺失时报错

### data 层测试

恢复并通过：

- `tests/test_inductive.py`
- `tests/test_anndata_utils.py`

### evaluation 口径测试

扩展 `tests/evaluation/test_posthoc.py`，使其与 5-method taxonomy 对齐。

## 推荐实现顺序

1. 先恢复 `src/scrare/data/`，使现有测试与 workflow 导入恢复。
2. 再让 `run_inductive` 至少能跑 baseline。
3. 再接上 `--methods` 与 baseline 复用。
4. 再统一 five-method 口径。
5. 最后补齐 posthoc 共享逻辑与测试。
