# PROJECT_STRUCTURE.md

## 目的

本文档用于规范 scRareRefine 项目的目录结构，避免 AI agent 在自动修改、实验运行、结果保存过程中把项目目录弄乱。

所有 agent，包括 ARIS、Codex、Claude Code，都必须遵守本文档。

## 推荐目录结构

```text
scRareRefine/
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── PROJECT_STRUCTURE.md
├── RESULTS_LOG.md
│
├── docs/
│   ├── PROJECT_BRIEF.md
│   ├── DATA_CARD.md
│   ├── EXPERIMENT_PLAN.md
│   └── CLAIMS.md
│
├── configs/
│   ├── datasets/
│   │   ├── pancreas.yaml
│   │   ├── immune_dc.yaml
│   │   └── tabula_sapiens.yaml
│   ├── experiments/
│   │   ├── scanvi_baseline.yaml
│   │   ├── proto_refine.yaml
│   │   └── full_scRareRefine.yaml
│   └── paths/
│       └── local.yaml
│
├── data/
│   ├── raw/
│   │   ├── human_pancreas_norm_complexBatch.h5ad
│   │   ├── human_immune_health_atlas_dc.h5ad
│   │   └── tabula_sapiens.h5ad
│   ├── processed/
│   ├── splits/
│   ├── embeddings/
│   └── external/
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_scanvi_baseline_check.ipynb
│   └── 03_result_visualization.ipynb
│
├── scripts/
│   ├── 01_inspect_data.py
│   ├── 02_train_scanvi.py
│   ├── 03_extract_scanvi_outputs.py
│   ├── 04_run_refinement.py
│   ├── 05_evaluate.py
│   ├── 06_collect_results.py
│   └── 07_plot_results.py
│
├── src/
│   └── sc_rare_refine/
│       ├── __init__.py
│       ├── data/
│       │   ├── io.py
│       │   ├── inspect.py
│       │   ├── preprocessing.py
│       │   └── split.py
│       ├── models/
│       │   ├── scanvi_wrapper.py
│       │   └── baselines.py
│       ├── refinement/
│       │   ├── prototypes.py
│       │   ├── fusion.py
│       │   ├── uncertainty.py
│       │   └── rare_adjustment.py
│       ├── metrics/
│       │   ├── classification.py
│       │   ├── rare_metrics.py
│       │   └── calibration.py
│       ├── visualization/
│       │   ├── latent_plot.py
│       │   ├── confusion_matrix.py
│       │   └── rare_report.py
│       └── utils/
│           ├── config.py
│           ├── logger.py
│           ├── seed.py
│           └── environment.py
│
├── tests/
│   ├── test_data_inspection.py
│   ├── test_label_mapping.py
│   ├── test_metrics.py
│   ├── test_prototypes.py
│   └── test_no_raw_data_modification.py
│
├── results/
│   ├── raw/
│   │   └── EXP-YYYYMMDD-XXX/
│   │       ├── predictions.csv
│   │       ├── metrics.json
│   │       ├── config.yaml
│   │       ├── environment.txt
│   │       └── run.log
│   ├── tables/
│   │   ├── main_results.csv
│   │   ├── rare_cell_results.csv
│   │   └── ablation_results.csv
│   ├── figures/
│   │   ├── confusion_matrices/
│   │   ├── latent_umap/
│   │   └── rare_cell_analysis/
│   └── reports/
│       ├── data_inspection_report.md
│       ├── experiment_summary.md
│       └── error_analysis.md
│
├── logs/
├── checkpoints/
└── tmp/
```

## 各目录职责

### `docs/`

存放研究设计和论文边界文档。

- `PROJECT_BRIEF.md`：项目目标、研究问题、贡献点、范围边界。
- `DATA_CARD.md`：数据来源、格式、标签列、batch 列、预处理规则。
- `EXPERIMENT_PLAN.md`：实验组、baseline、指标、随机种子、消融实验。
- `CLAIMS.md`：论文中允许主张什么，不能主张什么。

### `configs/`

存放所有实验配置。正式实验不允许把路径、seed、阈值、数据集名硬编码在脚本里。

建议分三类：

```text
configs/datasets/      # 数据集配置
configs/experiments/   # 实验配置
configs/paths/         # 本地路径配置
```

### `data/`

存放数据文件。

```text
data/raw/          原始数据，只读，禁止修改
data/processed/    预处理后的数据
data/splits/       train/val/test 划分文件
data/embeddings/   scANVI latent embedding、概率输出等
data/external/     外部 baseline 需要的中间文件
```

硬规则：

```text
禁止修改 data/raw/
禁止覆盖原始 .h5ad 文件
禁止在没有记录的情况下重建 split 文件
```

### `scripts/`

只放“可执行入口脚本”，例如训练、提取 embedding、运行 refinement、评估、汇总结果。

脚本应该尽量薄，复杂逻辑放到 `src/sc_rare_refine/`。

推荐命名：

```text
01_inspect_data.py
02_train_scanvi.py
03_extract_scanvi_outputs.py
04_run_refinement.py
05_evaluate.py
06_collect_results.py
07_plot_results.py
```

### `src/sc_rare_refine/`

项目的核心 Python 包。所有主要逻辑都应放在这里，而不是散落在脚本和 notebook 里。

推荐模块：

```text
data/            数据读取、检查、预处理、划分
models/          scANVI 包装器和 baseline
refinement/      prototype、fusion、uncertainty、rare adjustment
metrics/         分类指标、稀有细胞指标、校准指标
visualization/   UMAP、混淆矩阵、稀有细胞分析图
utils/           配置读取、日志、随机种子、环境记录
```

### `notebooks/`

只用于探索和可视化，不作为正式实验入口。

规则：

1. notebook 不能成为唯一实验记录。
2. notebook 中有效的代码要迁移到 `src/` 或 `scripts/`。
3. notebook 不应直接改写 `data/raw/`。

### `results/`

存放实验输出。

每次正式实验必须创建独立目录：

```text
results/raw/EXP-YYYYMMDD-XXX/
```

每个实验目录至少包含：

```text
predictions.csv
metrics.json
config.yaml
environment.txt
run.log
```

### `tests/`

存放测试文件。至少要覆盖：

1. label 映射是否正确。
2. split 是否无泄漏。
3. rare cell 定义是否一致。
4. prototype 计算是否正确。
5. 指标计算是否正确。
6. `data/raw/` 是否被误修改。

## 文件命名规范

### 实验 ID

统一格式：

```text
EXP-YYYYMMDD-XXX
```

示例：

```text
EXP-20260507-001
```

### 配置文件命名

```text
{dataset}_{method}_{seed}.yaml
```

示例：

```text
pancreas_scanvi_seed42.yaml
pancreas_scRareRefine_seed42.yaml
immune_dc_proto_refine_seed43.yaml
```

### 结果文件命名

主结果表：

```text
results/tables/main_results.csv
results/tables/rare_cell_results.csv
results/tables/ablation_results.csv
```

图像：

```text
results/figures/{experiment_id}_{figure_type}.png
```

## Agent 必须遵守的目录规则

AI agent 不得随意新建顶层目录。新增目录必须属于以下类别之一：

```text
configs/
data/
docs/
notebooks/
scripts/
src/
tests/
results/
logs/
checkpoints/
tmp/
```

不允许出现：

```text
new_code/
final_code/
backup/
test123/
临时/
实验结果新版/
随便新建的目录/
```

如果确实需要临时文件，必须放到：

```text
tmp/
```

如果是日志，必须放到：

```text
logs/
```

如果是模型权重，必须放到：

```text
checkpoints/
```

如果是正式结果，必须放到：

```text
results/raw/{experiment_id}/
```

## 推荐的第一个任务

正式改模型前，先完成：

```text
scripts/01_inspect_data.py
src/sc_rare_refine/data/inspect.py
```

输出：

```text
results/reports/data_inspection_report.md
results/tables/dataset_summary.csv
results/tables/cell_type_distribution.csv
```

这一步必须先查清楚：

1. `.h5ad` 中有哪些 `obs` 列。
2. 哪一列是 cell type label。
3. 哪一列是 batch / donor / study。
4. 每个细胞类型有多少细胞。
5. 哪些类别可以定义为 rare cell type。
