# scRare 项目结构重组设计

日期：2026-05-05

## 1. 背景

当前仓库已经收敛到 inductive workflow，但代码组织仍保留较强的历史演化痕迹：

- 主体代码不在 `src/` 布局下；
- 一部分核心逻辑放在 `scrare_refine/`，另一部分编排逻辑放在 `scripts/`；
- 模块命名混合了方法名、阶段名、入口名和实现细节，边界不够清晰；
- 四阶段方法比较（baseline、baseline+prototype、baseline+prototype+marker、baseline+fusion）已经存在，但方法组合和底层模块没有显式分层。

本设计的目标是：在不改变核心研究语义的前提下，对项目做一次中等力度的结构重组，使目录、命名、入口方式和模块边界更规范、更易维护。

## 2. 目标

本次重组要同时完成以下目标：

1. 将主体代码迁移到 `src/` 布局；
2. 将包名从 `scrare_refine` 统一为 `scrare`；
3. 将正式运行入口改为 `python -m ...`；
4. 将“命令入口”“流程编排”“方法模块”“评估逻辑”“基础设施”分层；
5. 保留并显式支持四阶段方法比较：
   - `baseline`
   - `baseline_plus_prototype`
   - `baseline_plus_prototype_plus_marker`
   - `baseline_plus_fusion`
6. 将 `prototype gate` 相关逻辑独立保留在单独模块中；
7. 在重组过程中优先保持行为一致，再逐步整理测试与文档。

## 3. 非目标

本次重组不包含以下内容：

- 不改变核心算法定义；
- 不新增新的研究方法阶段；
- 不重新设计实验输出格式；
- 不对配置系统做 schema 化重写；
- 不引入长期兼容的双包名机制；
- 不把目录重构扩大成数据、报告、论文材料的全仓库大改造。

## 4. 目标目录结构

```text
.
├── src/
│   └── scrare/
│       ├── __init__.py
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── audit.py
│       │   ├── run_inductive.py
│       │   └── evaluate_posthoc.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loading.py
│       │   ├── preprocess.py
│       │   └── splits.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── scanvi.py
│       │   ├── prototype.py
│       │   ├── prototype_gate.py
│       │   ├── fusion.py
│       │   └── marker.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   ├── audit.py
│       │   └── posthoc.py
│       ├── infra/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── io.py
│       │   ├── paths.py
│       │   └── resources.py
│       └── workflows/
│           ├── __init__.py
│           ├── inductive.py
│           └── posthoc.py
├── configs/
├── tests/
├── docs/
├── README.md
├── CLAUDE.md
└── pyproject.toml
```

## 5. 分层边界

### 5.1 `cli/`

只保留命令入口，不承载大段业务逻辑。

职责：
- 参数解析；
- 组装调用参数；
- 调用 workflow 或 evaluation 入口；
- 打印摘要日志。

约束：
- 不直接写训练主循环；
- 不直接组织批量 run 扫描；
- 不实现方法细节。

### 5.2 `data/`

负责数据读取、预处理和切分。

职责：
- 从配置读取 `AnnData`；
- `raw.X` / layer 切换；
- 名称唯一化；
- 子采样；
- train / validation / test split；
- 稀有类标注预算控制；
- train-only HVG 选择。

### 5.3 `models/`

负责 baseline 和各增量方法模块。

职责：
- scVI / scANVI baseline；
- prototype 分数；
- prototype gate；
- fusion；
- marker signature / score / threshold。

这里表示的是“方法能力”，不是“实验组合”。

### 5.4 `evaluation/`

负责指标和方法比较装配。

职责：
- 分类指标与 rare rescue 指标；
- dataset audit 统计；
- 四阶段方法装配与统一比较。

### 5.5 `infra/`

负责基础设施。

职责：
- YAML 配置读取；
- 表格读写；
- 输出路径组织；
- 资源监控。

### 5.6 `workflows/`

负责流程编排。

职责：
- 主 inductive 实验总调度；
- posthoc 批量评估总调度；
- 遍历 `(rare_class, split_mode, seed, rare_train_size)`；
- 遍历已有 run 目录并组织批量分析。

增加这一层的目的，是防止原本堆在脚本里的长流程重新回流到 `cli/`。

## 6. 四阶段方法的放置方式

四阶段方法不按目录单独拆成 4 套实现，而按“可复用能力”和“组合装配”分层。

### 6.1 baseline

- 放在 `src/scrare/models/scanvi.py`
- 负责训练 SCVI / SCANVI、生成 baseline prediction、probability 和 latent

### 6.2 prototype

- 放在 `src/scrare/models/prototype.py`
- 负责 prototype 分数、reference prototype、距离与排序逻辑

### 6.3 prototype gate

- 放在 `src/scrare/models/prototype_gate.py`
- 单独保留 gate 规则与 candidate 选择逻辑

### 6.4 marker

- 放在 `src/scrare/models/marker.py`
- 负责 marker signature、marker score、threshold selection、marker-based rescue

### 6.5 fusion

- 放在 `src/scrare/models/fusion.py`
- 负责 baseline/prototype 概率融合、权重计算和参数选择

### 6.6 四阶段组合装配

- 放在 `src/scrare/evaluation/posthoc.py`
- 统一定义四个阶段：
  - `baseline`
  - `baseline_plus_prototype`
  - `baseline_plus_prototype_plus_marker`
  - `baseline_plus_fusion`

原则：
- `models/` 放方法零件；
- `evaluation/posthoc.py` 放“如何拼成四种可比较方法”。

## 7. 旧文件到新文件的迁移映射

### 7.1 入口脚本

- `scripts/audit_dataset.py` → `src/scrare/cli/audit.py`
- `scripts/run_scanvi_inductive.py` → `src/scrare/cli/run_inductive.py`
- `scripts/evaluate_inductive_prototype_marker.py` → `src/scrare/cli/evaluate_posthoc.py`

其中后两者不是原样搬迁，而是将大部分流程拆入 `workflows/`、`models/`、`evaluation/`。

### 7.2 数据层

- `scrare_refine/anndata_utils.py`
  - `adata_from_config(...)` → `src/scrare/data/loading.py`
  - `subset_cells(...)`、`ensure_unique_names(...)` → `src/scrare/data/preprocess.py`
- `scrare_refine/inductive.py`
  - split 与 label budget 相关逻辑 → `src/scrare/data/splits.py`
  - train-only HVG 选择 → `src/scrare/data/preprocess.py`

### 7.3 模型层

- `scrare_refine/fusion.py` → `src/scrare/models/fusion.py`
- `scrare_refine/prototype.py` → `src/scrare/models/prototype.py`
- `scrare_refine/prototype_gate.py` → `src/scrare/models/prototype_gate.py`
- `scrare_refine/marker_verifier.py` → `src/scrare/models/marker.py`
- 从旧主脚本中新增抽取 → `src/scrare/models/scanvi.py`

### 7.4 评估层

- `scrare_refine/metrics.py` → `src/scrare/evaluation/metrics.py`
- `scrare_refine/audit.py` → `src/scrare/evaluation/audit.py`
- 新增：`src/scrare/evaluation/posthoc.py`

### 7.5 基础设施层

- `scrare_refine/config.py` → `src/scrare/infra/config.py`
- `scrare_refine/io.py` → `src/scrare/infra/io.py`
- `scrare_refine/output_layout.py` → `src/scrare/infra/paths.py`
- `scrare_refine/resources.py` → `src/scrare/infra/resources.py`

### 7.6 流程编排层

- 从旧主脚本抽取 → `src/scrare/workflows/inductive.py`
- 从旧后处理脚本抽取 → `src/scrare/workflows/posthoc.py`

## 8. 模块间调用关系与运行入口

### 8.1 审计入口

运行命令：

```bash
python -m scrare.cli.audit --config configs/immune_dc.yaml
```

调用链：

`cli.audit` → `evaluation.audit` → `data.loading` → `infra.io` / `infra.paths`

### 8.2 主实验入口

运行命令：

```bash
python -m scrare.cli.run_inductive --config configs/immune_dc.yaml
```

调用链：

`cli.run_inductive`
→ `workflows.inductive`
→ `data.loading` / `data.preprocess` / `data.splits`
→ `models.scanvi`
→ `models.fusion`
→ `evaluation.metrics`
→ `infra.paths` / `infra.io` / `infra.resources`

### 8.3 后处理入口

运行命令：

```bash
python -m scrare.cli.evaluate_posthoc --config configs/immune_dc.yaml
```

调用链：

`cli.evaluate_posthoc`
→ `workflows.posthoc`
→ `evaluation.posthoc`
→ `models.prototype`
→ `models.prototype_gate`
→ `models.marker`
→ `evaluation.metrics`
→ `infra.paths` / `infra.io`

## 9. 命名规范

### 9.1 包和模块名

统一采用：
- 顶层包：`scrare`
- 子包：`cli`、`data`、`models`、`evaluation`、`infra`、`workflows`
- 文件名全小写，必要时使用下划线

### 9.2 方法名

四阶段统一使用稳定的内部标识：
- `baseline`
- `baseline_plus_prototype`
- `baseline_plus_prototype_plus_marker`
- `baseline_plus_fusion`

### 9.3 函数名

建议统一采用：
- `load_*`、`build_*`
- `train_*`、`predict_*`
- `evaluate_*`
- `summarize_*`
- `select_*`
- `run_*`

## 10. 兼容策略

### 10.1 运行入口

正式文档入口统一切换到：
- `python -m scrare.cli.audit ...`
- `python -m scrare.cli.run_inductive ...`
- `python -m scrare.cli.evaluate_posthoc ...`

### 10.2 旧 `scripts/`

策略：短过渡、快收口。

- 重构初期允许短暂保留旧 `scripts/`，但只做转发；
- 不再在 `scripts/` 中承载任何核心逻辑；
- 完成测试与文档更新后删除 `scripts/`。

### 10.3 导入路径

不保留长期双包名兼容。

- 统一切换到 `from scrare.... import ...`
- 不保留 `scrare_refine` 壳层
- 不引入长期 fallback import

原因：仓库规模不大，双命名只会增加后续维护复杂度。

## 11. 测试策略

### 11.1 两阶段策略

第一阶段：行为保持不变
- 先改导入和模块归位；
- 保证数据加载、split、train-only 约束、四阶段行为不变；
- 暂不急于大改测试目录。

第二阶段：测试目录跟随新结构整理
- 再按 `data/models/evaluation/infra/workflows/cli` 重组测试目录。

### 11.2 重点保护行为

需要重点锁住：
- prototype 只使用训练集中有标签的 reference cell；
- marker signature 只来自训练集有标签样本；
- fusion 参数只从 validation 选择；
- 四阶段 method 标识稳定；
- 新 CLI 入口行为与旧实现一致。

### 11.3 建议新增测试

建议新增：
- `tests/cli/`：CLI smoke tests
- `tests/workflows/`：流程编排测试
- `tests/evaluation/test_posthoc.py`：四阶段装配与命名测试

## 12. 落地顺序

### 第 1 步：建立新骨架
- 创建 `src/scrare/` 及其子包；
- 修改 `pyproject.toml` 支持 `src/` 布局。

### 第 2 步：迁基础设施和纯函数模块
- `infra/config.py`
- `infra/io.py`
- `infra/paths.py`
- `infra/resources.py`
- `evaluation/metrics.py`

### 第 3 步：迁数据层
- `data/loading.py`
- `data/preprocess.py`
- `data/splits.py`

### 第 4 步：迁模型层
- `models/prototype.py`
- `models/prototype_gate.py`
- `models/marker.py`
- `models/fusion.py`
- `models/scanvi.py`

### 第 5 步：抽 workflow
- `workflows/inductive.py`
- `workflows/posthoc.py`
- `evaluation/posthoc.py`

### 第 6 步：切换 CLI
- `cli/audit.py`
- `cli/run_inductive.py`
- `cli/evaluate_posthoc.py`

### 第 7 步：调整测试与文档
- 修正 imports；
- 跑测试；
- 更新 README 和 CLAUDE；
- 删除旧 `scripts/` 或先转发后删除。

## 13. 风险与控制

### 风险 1：主实验长流程拆分后行为漂移
控制：
- 先抽纯函数和基础设施；
- workflow 抽取时保持输入输出字段不变；
- 用现有测试锁住关键行为。

### 风险 2：输出路径或表格字段无意变化
控制：
- `infra/paths.py` 先保持旧路径语义；
- 对 summary 表中的 `method`、路径层级和关键列名保持稳定。

### 风险 3：CLI 切换导致文档或使用方式混乱
控制：
- README 和 CLAUDE 同步更新；
- 过渡期旧脚本仅做转发；
- 尽快移除旧入口，避免双入口长期共存。

### 风险 4：四阶段逻辑继续散落在不同层
控制：
- 明确要求四阶段组合只在 `evaluation/posthoc.py` 装配；
- `models/` 不直接承担“方法组合”职责。

## 14. 验收标准

本次重组完成后，应满足：

1. 主体代码全部位于 `src/scrare/`；
2. 所有正式命令通过 `python -m scrare.cli...` 运行；
3. `scripts/` 不再承载核心逻辑；
4. baseline / prototype / prototype gate / marker / fusion 的边界清晰；
5. 四阶段方法由 `evaluation/posthoc.py` 统一装配；
6. 测试通过；
7. README 与 CLAUDE 已更新到新结构；
8. 不保留长期双包名兼容。

## 15. 结论

本设计采用“方案 B + workflows 补层”的重组方式，以 `src/scrare/` 为中心，按职责拆分为 `cli/data/models/evaluation/infra/workflows` 六层。该方案在不改变核心研究语义的前提下，解决当前仓库中入口与实现混杂、命名不统一、主体代码不在 `src/` 下、四阶段方法装配边界不清晰等问题。

该设计适合后续继续扩展新的后处理模块、统一评估表格、以及将研究代码逐步演化为结构更稳定的可维护项目。
