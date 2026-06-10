# scRareRefine

基于 scANVI 潜在表示空间的单细胞稀有细胞识别与 Post-hoc 拯救校正框架。

在已有半监督模型输出的基础上，通过 **Prototype 距离门控 → Marker 基因验证 → 自适应概率融合** 的后处理拯救机制，解决模型在极小训练样本及跨批次供体泛化（`batch-heldout`）场景下对稀有细胞类型漏检、误检的问题。

> [!NOTE]
> **本框架不替代** scANVI/CellTypist/scBalance 等细胞标注与迁移学习方法，而是在其之上作为后处理优化层（Refinement Layer）使用。

---

## 🚀 核心架构与重构设计

本项目采用高内聚、低耦合的模块化设计，学术核心组件与对比实验基线已完全隔离：

```
scRareRefine/
├── run_pipeline.py          # 单次端到端一键流水线入口（数据预处理 -> 模型训练 -> 拯救后处理 -> 绘图）
├── run_all_experiments.py   # 批量参数矩阵对比实验总控脚本（多数据集、多 seed、多标注规模，自动生成汇总大表）
├── configs/                 # 结构化科学配置文件目录（包含 8 个数据集的 YAML 配置文件）
│   └── *.yaml
├── src/                     # 核心学术包目录
│   ├── __init__.py
│   ├── preprocess.py        # 生信级数据体检与严格三路切分机制（支持 batch_heldout 与 cell_stratified）
│   ├── model.py             # HVG 筛选、双阶段 scVI/scANVI 表示微调与测试集推理
│   ├── rescue.py            # 三种拯救器组件（Prototype Gating、Marker Verification、Adaptive Fusion）
│   └── utils.py             # 科学评估指标、Seaborn/Matplotlib Fallback 绘图、GBK 兼容打印与系统监控
└── baseline/                # 独立且解耦的对比基线模型目录
    ├── scanvi/
    │   └── scanvi_baseline.py # scANVI 基线模型训练，输出对比所需的低维潜在表示
    ├── knn/
    │   └── knn_baseline.py    # 基于低维表示的最近邻多数投票基线
    ├── celltypist/
    │   └── celltypist_baseline.py # CellTypist (Logistic Regression, OvR) 线性分类基线
    └── scbalance/
        └── scbalance_baseline.py # scBalance (自适应 MLP 过采样) 深度非平衡分类基线
```

---

## 📦 环境要求与快速开始

### 依赖安装

```bash
pip install scvi-tools anndata pandas numpy scipy scikit-learn matplotlib seaborn pyyaml psutil
```

### 1. 运行单次端到端管线

通过 `run_pipeline.py`，可以直接对单个数据集的某种配置运行完整的预处理、表示学习和三种后处理拯救策略（`gate_only`, `gate_marker`, `fusion`）的评估对比：

```bash
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --split_mode batch_heldout
```

**命令行常用参数说明：**
- `--config`: YAML 配置文件路径（例如 `configs/immune_dc.yaml`）。
- `--seed`: 实验随机数种子（用于控制数据切分与模型初始化随机性）。
- `--rare_class`: 目标稀有类名称（若不传，则默认读取 YAML 配置中的 `experiment.rare_class`）。
- `--rare_train_size`: 稀有类训练标注样本的规模（支持浮点数如 `0.05` 代表 5%、绝对整数如 `10`、或者 `'all'` 代表全部。若不传，则优先读取 YAML 的 `experiment.rare_train_sizes` 首元素）。
- `--split_mode`: 样本三路切分模式。可选 `batch_heldout`（默认，批次/供体外泛化划分）或 `cell_stratified`（随机分层划分）。
- `--max_false_rescue_rate`: 后处理所允许的最大误判率阈值（默认 0.001）。

**单次运行产物：**
结果将保存在 `outputs/{dataset_name}/{run_id}/metrics/` 下：
- `final_metrics.csv`: 各拯救策略的性能指标大表（包括 Precision, Recall, F1-Score, 拯救细胞数等）。
- `method_comparison.png`: 多策略性能柱状对比图。
- `rescue_effect.png`: 实际拯救细胞数与误拯救数对比图。
- `marker_violin.png`: 稀有类型特异 Marker 基因表达量小提琴图。

---

### 2. 运行批量对比实验（推荐）

通过 `run_all_experiments.py` 启动批量评测矩阵。该总控脚本会自动遍历全部数据集、种子和已标注样本规模，顺序拉起所有对比基线与 scRareRefine 流水线。并在运行结束后自动搜集所有的 `.csv` 指标文件，聚合成大表，并在控制台打印 Markdown 格式的学术评估报表：

```bash
# 执行完整评测矩阵（总共 48 组超参任务，包含所有 baseline）
python run_all_experiments.py

# 使用 dry-run 模式预览全部生成的执行命令组而并不实际运行
python run_all_experiments.py --dry_run

# 截取评测任务区间进行测试（例如只跑前 3 组任务组合）
python run_all_experiments.py --start_at 1 --end_at 3
```

**批量运行产表与产物：**
- 合并指标表保存至：`results/all_experiments_summary.csv`。
- 控制台打印类似于下表的平均 F1-Score 表现大表：
  
| 数据集 | 稀有类 | 标注规模 | scANVI Baseline | k-NN | CellTypist | scBalance | Proto Gating | Gate + Marker | Adaptive Fusion (Ours) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| immune_dc | ASDC | 0.01 | 0.0034 | 0.0000 | 0.4567 | 0.3211 | 0.8122 | 0.9254 | **0.9856** |

---

## 🔬 学术评测设计与切分机制

本项目严格定义了两种三路（Train / Validation / Test）数据切分机制，以校验模型在不同临床实验设计下的表现：

### 1. Batch Heldout (默认模式)
- **学术语义**：跨供体 / 跨批次泛化评测。该模式会根据配置的 `batch_key` 对不同的批次/样本来源（Batch/Donor）进行聚类划分。将某些批次的细胞整批次地归入 Validation 或 Test，从而完全隔离训练集与测试集的批次分布。
- **配置方法**：在 YAML 配置文件中设置 `experiment.split_mode: batch_heldout`，并确保 `dataset.batch_key` 配置正确。

### 2. Cell Stratified
- **学术语义**：单细胞随机分层三路划分。忽略批次源，在整体细胞标签上按照 70% / 15% / 15% 进行分层随机抽样，保证每个切分区间里的各细胞类型占比一致。
- **配置方法**：在 YAML 配置文件中设置 `experiment.split_mode: cell_stratified`。

---

## 🛠 核心 Inductive 约束

为确保学术评测的绝对严谨与无数据泄漏，项目在代码层面实现了以下严格的 Inductive 约束：
1. **零测试集泄漏**：所有的参考原型（Prototypes）、分类阈值（Thresholds）、特异特征（Signatures）计算均仅源于 Train / Validation 集，Test 集细胞仅用于最终推理评估。
2. **特征独立选择**：高变基因（HVG）的计算与筛选过程仅基于训练集。
3. **主动调参约束**：后处理拯救的概率决策（如 Fusion 最佳权重等）仅通过 Validation 集的不确定性指标进行选择，避免过拟合于测试集。

---

## 📊 数据与输出管理
- **原始数据** 存放在 `data/raw/` 目录下（保持只读）。
- **运行中间产物及模型表示**（包括潜在低维 latent 嵌入、预测概率等）自动写入 `outputs/` 目录。
- **合并指标大表** 自动汇总于 `results/` 目录。
