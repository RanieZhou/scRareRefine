# scRareRefine 项目目录管理指南

本文档从项目管理视角说明 `scRareRefine-standard/` 中每个目录和关键文件的作用、使用者、输入来源、输出来源和维护规则，方便快速判断文件应该放在哪里、由谁产生、由谁消费。

## 1. 项目根目录定位

当前标准项目根目录是：

```text
scRareRefine-standard/
```

后续开发、测试、实验运行和结果归档都应优先围绕该目录进行。旧根目录中的代码、论文草稿、历史输出和备份只作为迁移来源或历史材料，不应继续扩散为新的工作入口。

## 2. 总体目录地图

```text
scRareRefine-standard/
├── README.md                 项目入口说明
├── PROJECT_STRUCTURE.md      目录结构规则与约束
├── docs/                     研究设计、项目管理和长期说明文档
├── configs/                  可运行配置和未来配置拆分目录
├── data/                     数据区，raw 只读
├── src/scrare/               当前 Python 源码包
├── tests/                    自动化测试
├── results/                  标准结果输出与归档
├── logs/                     运行日志
├── checkpoints/              模型权重和训练检查点
├── notebooks/                探索性分析笔记本
├── scripts/                  辅助脚本
└── tmp/                      临时文件
```

## 3. 顶层文件

| 路径 | 作用 | 主要使用者 | 主要来源/维护者 | 备注 |
| --- | --- | --- | --- | --- |
| `README.md` | 给新读者的项目入口，说明安装、运行命令、当前实现状态 | 用户、合作者、agent | 人工维护，必要时由 agent 更新 | 内容应简洁，避免放长篇实验细节 |
| `PROJECT_STRUCTURE.md` | 目录结构规则、禁止事项、标准目录边界 | 用户、agent | 人工维护，结构变更时更新 | 偏规则约束 |
| `docs/PROJECT_DIRECTORY_GUIDE.md` | 本文档，解释每个目录的管理意义和流向 | 用户、项目管理者、agent | 人工维护，目录职责变化时更新 | 偏管理视角 |
| `RESULTS_LOG.md` | 实验结果登记和追踪 | 用户、实验执行 agent | 实验完成后更新 | 不应伪造未跑完结果 |
| `pyproject.toml` | Python 包安装、依赖、console script 配置 | Python、pip、测试、CLI | 开发者维护 | 当前打包包名为 `scrare` |
| `CLAUDE.md` | Claude Code 在本目录下工作的规则 | Claude Code | 用户维护，agent 可按要求更新 | 包含数据保护、Git 规则、科研表述限制 |
| `AGENTS.md` | 通用 agent 工作规则 | 各类 agent | 用户维护，agent 可按要求更新 | 说明哪些操作需要人工确认 |

## 4. `configs/`：实验和运行配置

### 作用

`configs/` 存放运行 CLI 和 workflow 所需的 YAML 配置，包括数据路径、label/batch 列、rare class、seed、训练参数和输出路径。

当前正式可运行配置是：

```text
configs/immune_dc.yaml
configs/pancreas_epsilon.yaml
configs/pancreas_gamma.yaml
```

保留的标准子目录：

```text
configs/datasets/
configs/experiments/
configs/paths/
```

这些子目录用于未来把数据集信息、实验参数和路径配置拆开；在配置系统正式重构前，当前 CLI 仍以顶层 `configs/*.yaml` 为正式入口。

### 谁会使用

- 用户运行 CLI 时通过 `--config` 指定。
- `src/scrare/cli/audit.py` 读取配置做数据审计。
- `src/scrare/cli/run_inductive.py` 读取配置运行 inductive workflow。
- `src/scrare/cli/evaluate_posthoc.py` 读取配置做 posthoc 评估。
- 测试会检查配置是否仍符合目录和输出规范。

### 谁会输出/维护

- 主要由用户或 agent 手动维护。
- 未来如果有配置生成脚本，应输出到 `configs/` 的合适子目录，并保留人工可读性。

### 管理规则

- 不要无记录地改变 `seed`、split、rare class、label 列、batch 列、预处理规则或训练参数。
- 当前运行输出路径应指向 `results/`，不要重新引入根目录 `outputs/` 作为正式输出位置。
- 如果需要新增数据集配置，优先复制现有 YAML 并明确数据路径、label_key、batch_key 和 rare_class。

## 5. `data/`：数据区

```text
data/
├── raw/
├── processed/
├── splits/
├── embeddings/
└── external/
```

### 5.1 `data/raw/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放原始 `.h5ad` 数据文件 |
| 使用者 | 数据审计、训练 workflow、posthoc 评估、人工检查 |
| 输入来源 | 用户手动放入，或从可信数据源下载后放入 |
| 输出来源 | 不应由代码输出或覆盖 |
| 管理规则 | 只读；禁止修改、覆盖、删除原始 `.h5ad` |

当前已知数据集包括：

```text
data/raw/human_immune_health/
data/raw/pancreas/
data/raw/tabula/
```

### 5.2 `data/processed/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放经过预处理但仍属于数据中间态的文件 |
| 使用者 | 后续训练、审计或探索分析 |
| 输入来源 | 预处理脚本、人工确认后的数据整理流程 |
| 输出来源 | 数据处理代码或脚本 |
| 管理规则 | 不能覆盖 raw；应记录处理方法和来源配置 |

### 5.3 `data/splits/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 保存 train/validation/test split、batch split 或 label mask |
| 使用者 | inductive workflow、复现实验、错误分析 |
| 输入来源 | split 生成逻辑 |
| 输出来源 | `src/scrare/data/splits.py` 或相关 workflow |
| 管理规则 | 必须避免 held-out cells 泄漏到训练 reference |

### 5.4 `data/embeddings/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 保存 latent embedding、预测概率、label mapping 等可复用中间结果 |
| 使用者 | prototype、fusion、posthoc 评估、可视化 |
| 输入来源 | scANVI 推理、workflow 中间结果 |
| 输出来源 | 模型推理代码、posthoc 中间流程 |
| 管理规则 | 如果是正式实验结果，优先同时在 `results/` 中登记或汇总 |

### 5.5 `data/external/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放外部 baseline、第三方方法或外部工具产生的中间文件 |
| 使用者 | baseline 对比、补充实验 |
| 输入来源 | 外部工具、公开资源、人工导入 |
| 输出来源 | 通常不是本项目核心 workflow |
| 管理规则 | 应记录来源、版本和处理方式，避免混入 raw 数据 |

## 6. `src/scrare/`：当前核心代码包

当前项目的 Python 包名是：

```text
scrare
```

所有新代码、import、CLI 和测试都应围绕 `src/scrare/`。不要把新逻辑写入旧模板包名目录。

```text
src/scrare/
├── cli/
├── data/
├── workflows/
├── models/
├── evaluation/
├── infra/
└── visualization/
```

### 6.1 `src/scrare/cli/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 命令行入口 |
| 使用者 | 用户、agent、测试、console scripts |
| 输入来源 | shell 命令参数、`configs/*.yaml` |
| 输出来源 | 调用 workflow 后在 `results/` 写结果 |

当前入口：

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

### 6.2 `src/scrare/data/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 数据读取、矩阵选择、预处理、split 构建 |
| 使用者 | workflow、测试 |
| 输入来源 | `configs/`、`data/raw/` |
| 输出来源 | split assignment、HVG 选择结果、处理后的 AnnData 中间状态 |
| 关键约束 | HVG 和 reference 构建必须保持 train-only |

### 6.3 `src/scrare/workflows/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 编排完整实验流程 |
| 使用者 | CLI、测试、agent |
| 输入来源 | 配置、raw 数据、已有 run artifacts |
| 输出来源 | `results/` 下的 runs、tables、stages、figures、resource summary |

主要文件：

```text
workflows/inductive.py   主 inductive 训练、推理、fusion 和汇总流程
workflows/posthoc.py     复用已有 artifacts 的 posthoc 评估流程
```

### 6.4 `src/scrare/models/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 方法模块，包括 scANVI、prototype、fusion、marker、gate |
| 使用者 | workflows、evaluation、测试 |
| 输入来源 | 训练数据、latent、预测概率、marker signature |
| 输出来源 | 预测表、prototype 分数、融合结果、候选细胞、阈值选择结果 |
| 关键约束 | 不得使用 test 标签调参或构建 marker/prototype reference |

### 6.5 `src/scrare/evaluation/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 指标计算、数据审计、posthoc 评估逻辑 |
| 使用者 | workflows、CLI、测试 |
| 输入来源 | 预测结果、真实标签、候选细胞表、validation 结果 |
| 输出来源 | metrics、confusion table、effect summary、threshold curve |

### 6.6 `src/scrare/infra/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 配置读取、路径规则、表格 I/O、资源监控 |
| 使用者 | 全部 workflow 和 CLI |
| 输入来源 | YAML、Path、运行状态 |
| 输出来源 | 标准化路径、CSV/表格、resource summary |
| 关键约束 | 输出路径规则应保持集中，默认写入 `results/` |

### 6.7 `src/scrare/visualization/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 生成实验图表 |
| 使用者 | workflow、报告撰写者、测试 |
| 输入来源 | `results/` 中的 summary table、threshold curve、resource summary |
| 输出来源 | `results/figures/` 或单次实验 root 下的图表目录 |

## 7. `tests/`：自动化测试

```text
tests/
├── cli/
├── evaluation/
├── visualization/
├── workflows/
└── test_*.py
```

### 作用

- 防止 split 泄漏。
- 验证 train-only HVG 和 prototype reference。
- 验证 fusion、marker、gate 等方法逻辑。
- 验证 CLI 可导入、参数行为和输出路径。
- 约束项目结构，不让旧脚本、旧包名或旧输出目录重新进入正式路径。

### 谁会使用

- 开发者在修改代码后运行。
- agent 在实现或重构后运行。
- 未来 CI 可直接运行。

### 谁会输出

- pytest 产生临时缓存，如 `.pytest_cache/`、`__pycache__/`。这些不是项目结果，测试后可清理。

### 常用命令

```bash
pytest tests/test_project_state.py tests/cli/test_cli_smoke.py -v
pytest -v
```

## 8. `results/`：标准结果区

```text
results/
├── raw/
├── tables/
├── figures/
└── reports/
```

### 8.1 `results/raw/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放正式实验的原始输出目录 |
| 使用者 | 实验复现者、后续汇总脚本、报告作者 |
| 输入来源 | `run_inductive`、`evaluate_posthoc`、手动整理的正式实验 |
| 输出来源 | workflow 和实验脚本 |
| 建议命名 | `EXP-YYYYMMDD-XXX/` |

建议每个正式实验至少包含：

```text
config.yaml
environment.txt
run.log
predictions.csv
metrics.json
```

### 8.2 `results/tables/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放跨 run、跨 seed、跨数据集的汇总表 |
| 使用者 | 报告作者、论文写作、可视化代码 |
| 输入来源 | `results/raw/`、workflow stage summaries |
| 输出来源 | 汇总脚本、posthoc 评估、人工整理 |

### 8.3 `results/figures/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放正式图表和报告图 |
| 使用者 | 报告作者、论文写作、项目展示 |
| 输入来源 | `results/tables/`、workflow plot inputs |
| 输出来源 | `src/scrare/visualization/`、人工整理 |

当前迁移后的历史报告图表已归档到这里。

### 8.4 `results/reports/`

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放实验报告、阶段性结果报告、PDF/Markdown 报告 |
| 使用者 | 用户、合作者、论文写作者 |
| 输入来源 | `results/tables/`、`results/figures/`、实验日志 |
| 输出来源 | 用户或 agent 撰写 |

当前迁移后的历史实验报告已归档到这里。

### 当前运行输出规则

当前代码默认将 audit、inductive 和 posthoc 运行输出写入 `results/`。如果 CLI 参数或配置中显式指定 `output_dir`，则以显式路径为准。

不要重新使用根目录 `outputs/` 作为正式输出目录。

## 9. `docs/`：项目说明和研究文档

当前已有文档包括：

```text
docs/PROJECT_BRIEF.md
docs/DATA_CARD.md
docs/EXPERIMENT_PLAN.md
docs/CLAIMS.md
docs/PROJECT_DIRECTORY_GUIDE.md
```

### 作用

- 记录项目背景、研究目标、数据说明、实验计划、论文边界和目录管理说明。
- 帮助用户和 agent 在动手前理解项目上下文。

### 谁会使用

- 用户把握整体项目。
- agent 在规划、执行和写报告前读取。
- 合作者了解研究边界。

### 谁会输出/维护

- 用户或 agent 手动撰写。
- 不应由实验 workflow 自动把正式结果写入 `docs/`。

### 管理规则

- 研究设计、说明性文档放这里。
- 正式实验结果放 `results/`。
- 临时草稿如果不准备长期保存，应放 `tmp/`，不要长期占用 `docs/`。

## 10. `logs/`：日志区

| 项目 | 说明 |
| --- | --- |
| 作用 | 保存运行日志、错误日志、实验过程记录 |
| 使用者 | 用户、agent、debugger |
| 输入来源 | CLI、实验脚本、长时间训练任务 |
| 输出来源 | 运行命令、日志重定向、未来 logging 系统 |

建议日志命名包含日期、数据集、方法或实验 ID，例如：

```text
logs/2026-05-07-immune_dc-run_inductive.log
```

## 11. `checkpoints/`：模型权重和检查点

| 项目 | 说明 |
| --- | --- |
| 作用 | 保存训练中的模型权重、scVI/scANVI 检查点或可恢复状态 |
| 使用者 | 训练 workflow、复现实验、长任务恢复 |
| 输入来源 | 模型训练过程 |
| 输出来源 | scVI/scANVI 训练代码、显式 checkpoint 保存逻辑 |

管理规则：

- 大文件不应随意提交到 Git。
- 检查点应能通过实验 ID 或配置追溯来源。
- 如果只是临时调试检查点，优先放 `tmp/` 或明确清理。

## 12. `notebooks/`：探索性分析

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放 Jupyter notebook 或交互式探索分析 |
| 使用者 | 用户、研究人员 |
| 输入来源 | `data/`、`results/`、人工探索 |
| 输出来源 | 人工 notebook 执行 |

管理规则：

- notebook 结论不能直接视为正式结果，正式结论应迁移到 `results/reports/` 或论文文档中。
- notebook 中生成的大图、大表不应长期留在 notebook 目录，应整理到 `results/`。

## 13. `scripts/`：辅助脚本

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放薄脚本、批处理脚本、一次性辅助入口 |
| 使用者 | 用户、agent、批量运行任务 |
| 输入来源 | CLI 参数、配置文件 |
| 输出来源 | 通常调用 `src/scrare/` 后输出到 `results/`、`logs/` 或 `tmp/` |

管理规则：

- 正式业务逻辑应放在 `src/scrare/`，不要堆在 `scripts/`。
- 脚本应尽量调用现有 CLI 或 workflow。
- 旧的一次性脚本如果不再使用，应避免作为新实验入口。

## 14. `tmp/`：临时文件

| 项目 | 说明 |
| --- | --- |
| 作用 | 存放临时调试文件、短期中间产物、可删除 scratch 文件 |
| 使用者 | 用户、agent、临时脚本 |
| 输入来源 | 调试、试跑、临时转换 |
| 输出来源 | 任意临时流程 |

管理规则：

- `tmp/` 中的内容默认不作为正式结果。
- 重要结果应及时迁移到 `results/` 并登记来源。
- 不要在项目根目录直接创建临时目录。

## 15. `outputs/`：遗留目录说明

如果看到 `outputs/`，它应被视为历史迁移或旧运行产生的遗留目录，不再是标准输出位置。

| 项目 | 说明 |
| --- | --- |
| 作用 | 仅用于识别和迁移旧结果，不作为新输出目标 |
| 使用者 | 用户或 agent 做历史结果清点时可能读取 |
| 输入来源 | 旧代码或迁移前运行 |
| 输出来源 | 当前标准代码不应默认写入 |

管理规则：

- 新实验不要写入 `outputs/`。
- 有价值内容应经过筛选后迁移到 `results/`。
- 是否删除旧 `outputs/` 应单独确认，不能自动清理。

## 16. 常见任务应该放哪里

| 任务 | 推荐位置 |
| --- | --- |
| 新增数据集原始 `.h5ad` | `data/raw/<dataset>/` |
| 新增数据集配置 | `configs/<dataset>.yaml`，未来可拆到 `configs/datasets/` |
| 跑一次正式 inductive 实验 | 输出到 `results/` 下的数据集/实验目录 |
| 保存汇总指标表 | `results/tables/` |
| 保存论文或报告图 | `results/figures/` |
| 保存阶段性实验报告 | `results/reports/` |
| 写项目设计、研究边界、目录说明 | `docs/` |
| 写核心功能代码 | `src/scrare/` |
| 写测试 | `tests/` |
| 保存调试日志 | `logs/` |
| 保存模型 checkpoint | `checkpoints/` |
| 临时转换、试验文件 | `tmp/` |

## 17. 目录流向总览

```text
configs/ + data/raw/
        ↓
src/scrare/cli/ → src/scrare/workflows/
        ↓
results/ + logs/ + checkpoints/
        ↓
docs/ + reports + paper writing
```

更具体地说：

1. 用户准备 `data/raw/` 和 `configs/`。
2. CLI 读取配置并调用 workflow。
3. workflow 调用 data、models、evaluation、infra、visualization 模块。
4. 运行结果写入 `results/`，日志写入 `logs/`，权重写入 `checkpoints/`。
5. 用户或 agent 根据 `results/` 撰写 `results/reports/`、更新 `RESULTS_LOG.md`，再服务于论文或阶段汇报。

## 18. 管理红线

1. 不修改、覆盖或删除 `data/raw/` 中的原始数据。
2. 不把正式实验输出写到项目根目录。
3. 不重新把 `outputs/` 作为正式输出目录。
4. 不在 `scripts/` 中堆核心逻辑。
5. 不把 notebook 临时结论直接当作正式结果。
6. 不无记录地改变配置中的 seed、split、rare class、label 列、batch 列或模型参数。
7. 不在没有验证的情况下写“全面优于”“state-of-the-art”“临床可用”等科研结论。
