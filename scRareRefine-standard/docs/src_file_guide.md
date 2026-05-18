# `src/` 源码文件逐文件详细说明

本文档面向第一次接触本项目的人，逐个解释 `src/` 目录下每个文件在做什么、什么时候会被调用、输入是什么、输出是什么、和其他文件有什么关系。

当前 `src/` 目录采用**阶段式脚本 pipeline**，核心实验大致按以下顺序运行：

```text
00_preprocess_pbmc.py   # 只用于 PBMC/pDC 数据预处理
01_split.py             # Stage 1: 生成 train/validation/test split
02_baseline_scanvi.py   # Stage 2: 训练 scVI/scANVI，输出 embedding 和 baseline 预测
03_prototype.py         # Stage 3: 基于 scANVI latent 计算 prototype 距离、候选、separability
03b_knn_baseline.py     # Stage 3b: kNN 对比方法
03c_celltypist_baseline.py # Stage 3c: LR/CellTypist 等效对比方法
04_prototype_gate.py    # Stage 4: 测试不同 prototype gate 规则
05_prototype_gate_marker.py # Stage 5: rank1 prototype gate + marker verification
06_fusion.py            # Stage 6: scANVI probability 与 prototype probability 融合
07_evaluate.py          # Stage 7: 汇总单个 run 的所有方法指标
08_visualize.py         # Stage 8: 单个 run 的图表
09_aggregate_plot.py    # 汇总所有 runs，生成论文图
10_paper_table.py       # 汇总所有 runs，生成论文表格
11_umap_visualize.py    # 单个 run 的 UMAP 可视化
gen_pipeline_diagram.py # 画 pipeline 架构图
utils.py                # 所有脚本共享的工具函数
```

注意：虽然 `CLAUDE.md` 中提到标准包目录 `src/scrare/`，但当前实际 `src/` 下是上述阶段脚本，没有 `src/scrare/` 包目录。

---

## 1. 总体数据流

一个完整实验 run 的核心数据流如下：

```text
AnnData .h5ad + config yaml
  ↓
01_split.py
  → data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
  ↓
02_baseline_scanvi.py
  → outputs/{dataset}/{run_id}/embeddings/*.csv
  → outputs/{dataset}/{run_id}/split_assignments.csv
  → outputs/{dataset}/{run_id}/selected_hvg_genes.csv
  ↓
03_prototype.py
  → outputs/{dataset}/{run_id}/prototype/*_scores.csv
  → outputs/{dataset}/{run_id}/prototype/separability.csv
  ↓
04_prototype_gate.py
  → outputs/{dataset}/{run_id}/gate/*_results.csv
  ↓
05_prototype_gate_marker.py
  → outputs/{dataset}/{run_id}/gate_marker/*_scored.csv
  → outputs/{dataset}/{run_id}/gate_marker/selected_thresholds.csv
  ↓
06_fusion.py
  → outputs/{dataset}/{run_id}/fusion/*_results.csv
  ↓
07_evaluate.py
  → outputs/{dataset}/{run_id}/metrics/final_metrics.csv
  ↓
08_visualize.py / 09_aggregate_plot.py / 10_paper_table.py / 11_umap_visualize.py
  → 图表和论文表格
```

其中 `{run_id}` 通常长这样：

```text
batch_heldout_seed42_ASDC_rare20
cell_stratified_seed43_endothelial_cell_rare5
```

实际代码会把 rare class 名字安全化，例如空格变成 `_`，大小写转小写。

---

## 2. 当前方法主线：哪些文件负责什么？

当前主线不是单独看 separability ratio，而是：

```text
scANVI embedding + prediction
  → prototype candidate generation
  → prototype gate / rank filtering
  → marker verification
  → refined prediction
```

对应文件：

| 方法步骤 | 文件 | 作用 |
|---|---|---|
| scANVI baseline | `02_baseline_scanvi.py` | 训练 scANVI，得到 baseline label、probability、latent embedding |
| prototype distance | `03_prototype.py` | 用训练集有标签细胞构建 prototype，计算 query 到 rare prototype 的距离和 rank |
| separability ratio | `03_prototype.py` | 计算 rare class 是否在 latent space 中几何可分，用于分析和诊断 |
| prototype gate | `04_prototype_gate.py` | 测试多种 gate 规则，如 rank1、rank2+margin 等 |
| gate + marker 主方法 | `05_prototype_gate_marker.py` | 用 rank1 candidate，再用训练集 marker signature 验证是否 rescue |
| final metrics | `07_evaluate.py` | 把 baseline、prototype、gate、gate+marker、fusion 等方法汇总到一张表 |

重要实现细节：

- `prototype_gate_marker` 目前的 candidate 逻辑是 **rank1 prototype gate + marker verification**。
- `04_prototype_gate.py` 中有多种 gate 规则，并且会单独评估为 `prototype_gate` / `prototype_gate_best`。
- `separability_ratio` 目前主要作为分析、诊断、论文解释变量；它不是 Stage 5 里的 hard if 条件。

---

# 3. 文件逐个详解

---

## 3.1 `src/.DS_Store`

### 文件定位

这是 macOS 自动生成的 Finder 元数据文件，不是项目代码。

### 有什么用？

对实验、pipeline、论文图表都没有作用。

### 输入输出

无。

### 小白理解版

可以理解成“系统垃圾文件”。它不应该参与代码逻辑，也不需要阅读。

---

## 3.2 `src/utils.py`

### 文件定位

这是整个 `src/` pipeline 的**共享工具箱**。其他几乎所有脚本都会从这里导入函数。

它本身通常不单独运行，而是被其他文件调用。

### 主要用途

提供以下能力：

1. 读取 YAML 配置
2. 读取 AnnData `.h5ad` 数据
3. 统一读写 CSV 表格
4. 统一生成输出路径
5. 计算分类指标
6. 计算 scANVI prediction uncertainty
7. 做表达矩阵标准化
8. 设置随机种子
9. 监控运行时间和内存

### 主要函数

#### `load_config(path)`

读取 YAML 配置文件。

输入：

```text
configs/xxx.yaml
```

输出：

```python
config: dict
```

小白理解：把配置文件里的数据路径、label 列、模型参数等读进 Python。

---

#### `load_adata(config)`

根据配置文件读取 `.h5ad`。

它会处理三种情况：

1. 如果 config 写了 `dataset.use_layer`，就使用指定 layer。
2. 如果 config 写了 `dataset.use_raw: true`，就使用 `adata.raw.X`。
3. 否则直接使用 `adata.X`。

输入：

```text
config["dataset"]["path"]
config["dataset"].get("use_layer")
config["dataset"].get("use_raw")
```

输出：

```python
AnnData
```

小白理解：这是所有脚本读取单细胞数据的统一入口。

---

#### `write_table(df, path)` / `read_table(path)`

统一写入和读取 CSV 表格。

输入：

```python
pandas.DataFrame
```

输出：

```text
.csv 文件
```

小白理解：pipeline 的每一步几乎都靠 CSV 文件互相传数据。

---

#### `make_run_id(...)` / `make_run_dir(...)`

根据 dataset、split mode、seed、rare class、rare train size 生成输出目录。

例如：

```text
outputs/immune_dc/batch_heldout_seed42_asdc_rare20/
```

小白理解：保证每个实验 run 的输出不会混在一起。

---

#### `make_split_path(...)`

生成 split 文件路径：

```text
data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
```

---

#### `classification_tables(y_true, y_pred, rare_class)`

计算分类指标。

输出包括：

```text
overall_accuracy
macro_f1
rare_precision
rare_recall
rare_f1
```

也会返回每个 class 的 precision/recall/F1 表。

小白理解：这是所有方法评价 rare-cell 识别效果时共用的打分函数。

---

#### `compute_uncertainty(probabilities, rare_class)`

根据 scANVI 输出的类别概率，计算不确定性：

```text
max_prob    最大类别概率
entropy     概率分布熵，越大越不确定
margin      第一名概率 - 第二名概率
top1_label  概率第一的类别
top2_label  概率第二的类别
top2_is_rare_class  rare class 是否是第二名
```

小白理解：它帮助判断 scANVI 对某个 cell 的预测有多确定。

---

#### `log1p_cpm(x)`

对表达矩阵做常见单细胞标准化：

```text
每个细胞归一化到 10000 counts
再 log1p
```

主要用于 marker verification。

---

#### `seed_everything(seed)`

设置 numpy、torch、scvi 的随机种子，让实验更可复现。

---

#### `ResourceMonitor`

上下文管理器，用于记录：

```text
wall_time_seconds
peak_rss_mb
```

也就是运行时间和峰值内存。

在 `02_baseline_scanvi.py` 中用于记录 scANVI 训练资源消耗。

### 输入输出总结

输入：配置文件路径、AnnData、DataFrame、预测标签等。

输出：工具函数返回值、CSV 文件、路径对象、指标字典。

### 小白理解版总结

`utils.py` 是公共工具箱。它不定义具体实验逻辑，但所有阶段脚本都依赖它。如果要理解路径、指标、数据读取行为，优先看这个文件。

---

## 3.3 `src/00_preprocess_pbmc.py`

### 文件定位

这是 PBMC/pDC 数据集的专用预处理脚本，不是所有数据集都会用。

### 什么时候调用？

当要把原始 PBMC COVID-19 Blood Atlas 数据处理成项目可跑的 pDC 数据集时调用：

```bash
python src/00_preprocess_pbmc.py
```

### 它解决什么问题？

原始 PBMC 数据很大，大约 83 万细胞。直接跑 scANVI 会很重。

这个脚本做三件事：

1. 过滤掉没有有效 `minor_subset` 标签的细胞。
2. 把 `minor_subset` 复制成项目统一使用的 `label` 列。
3. 保留全部 pDC，同时随机采样非 pDC，使总细胞数约为 50,000。

### 特别重要的修复

当前版本会把 `adata.X` 替换成 `adata.raw.X`：

```text
adata.X 原来是 log-normalized 表达
adata.raw.X 才是原始整数 count
```

这很重要，因为 scVI/scANVI 通常应该吃原始 count，而不是 log-normalized 值。

### 输入

硬编码输入路径：

```text
data/raw/pbmc/pbmc_pdc.h5ad
```

要求原始 AnnData 中有：

```text
obs["minor_subset"]
obs["donor_id"]
obs["sex"]
obs["disease"]
obs["assay"]
raw.X
```

### 输出

```text
data/raw/pbmc/pbmc_pdc_50k.h5ad
```

输出文件包含：

```text
X = raw integer counts
obs["label"] = minor_subset
obs["donor_id"], sex, disease, assay 等必要列
```

### 小白理解版总结

这是给 PBMC 数据“瘦身”和“整理标签”的脚本。它把 83 万细胞压到 5 万左右，同时确保所有 pDC 都保留下来，并且把表达矩阵修正为原始 count。

---

## 3.4 `src/01_split.py`

### 文件定位

Stage 1：生成 train / validation / test split。

这是正式 pipeline 的第一步。

### 什么时候调用？

例如：

```bash
python src/01_split.py --config configs/immune_dc.yaml --seed 42 --split_mode batch_heldout
```

或者：

```bash
python src/01_split.py --config configs/tabula_kidney.yaml --seed 42 --split_mode cell_stratified
```

### 它做什么？

把所有细胞分成三份：

```text
train      用于训练 scVI/scANVI，并提供 prototype / marker reference
validation 用于选择 gate、marker threshold、fusion 参数
test       最终评估，只能最后打分用
```

### 支持两种 split 模式

#### 1. `cell_stratified`

按细胞类型分层随机划分。

适合：

```text
没有可靠 batch/donor 分组的数据集
单 donor 数据集
```

逻辑：

1. 先按 70% / 30% 切 train 和 heldout。
2. 再把 heldout 按 15% / 15% 切 validation 和 test。
3. 尽量保持每个 cell type 的比例一致。

对应函数：

```python
cell_stratified_split(...)
```

---

#### 2. `batch_heldout`

按 batch/donor 整体划分。

适合：

```text
有多个 donor / batch 的数据集
希望测试跨 donor 泛化
```

逻辑：

1. 统计每个 batch 中各 cell type 数量。
2. 把整个 batch 分配到 train、validation 或 test。
3. 尽量让三个 split 的 cell type 分布接近 70/15/15。
4. 保证每个 split 至少有一个 batch。

对应函数：

```python
batch_heldout_split(...)
```

### 输入

```text
config yaml
AnnData .h5ad
config["dataset"]["label_key"]
config["dataset"]["batch_key"]
seed
split_mode
```

### 输出

```text
data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
```

表格列：

```text
cell_id
split              train / validation / test
original_label     原始 cell type 标签
```

### 与其他文件的关系

`02_baseline_scanvi.py` 必须读取这个 split 文件。

如果没有先跑 `01_split.py`，Stage 2 会报错：

```text
Split not found. Run 01_split.py first.
```

### 小白理解版总结

这个文件决定哪些细胞用于训练，哪些用于调参，哪些用于最终测试。它是防止数据泄漏的第一道关。

---

## 3.5 `src/02_baseline_scanvi.py`

### 文件定位

Stage 2：训练 scVI/scANVI baseline，并输出 latent embedding、预测结果和概率。

这是整个方法的核心输入来源。

### 什么时候调用？

例如：

```bash
python src/02_baseline_scanvi.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout
```

### 它做什么？

这个文件完成以下步骤：

1. 读取 `01_split.py` 生成的 split。
2. 读取 AnnData 数据。
3. 根据 rare_train_size 构造 scANVI 半监督标签。
4. 只在 train cells 上选择 HVG。
5. 训练 scVI。
6. 从 scVI 转成 scANVI 并继续训练。
7. 对 train / validation / test 分别输出：
   - scANVI 预测标签
   - scANVI softmax probability
   - uncertainty 指标
   - latent embedding
8. 记录资源消耗。

### 半监督标签如何构造？

对应函数：

```python
make_scanvi_labels(...)
```

规则：

```text
train 中 major classes → 保留真实标签
train 中 rare class → 只保留 rare_train_size 个标签
train 中其他 rare cells → Unknown
validation/test 全部 → Unknown
```

这保证了 inductive evaluation：

```text
validation/test label 不进入训练
rare class 只有指定数量的 labeled examples
```

### HVG 选择

对应函数：

```python
select_hvg_genes(train_adata, n_top_genes)
```

只使用 train cells 计算方差最高的基因。

这点非常重要：

```text
HVG 不能用 validation/test cells，否则会泄漏测试数据分布
```

### scANVI 训练

对应函数：

```python
train_scanvi(...)
```

它先训练：

```text
SCVI
```

再转成：

```text
SCANVI
```

训练参数来自 config：

```text
model.n_latent
model.batch_size
model.scvi_max_epochs
model.scanvi_max_epochs
```

### 输出预测和 embedding

对应函数：

```python
prediction_outputs(...)
```

输出两个表：

1. predictions 表：

```text
cell_id
true_label
predicted_label
max_prob
entropy
margin
top1_label
top2_label
top2_is_{rare_class}
prob_{class1}
prob_{class2}
...
```

2. latent 表：

```text
cell_id
latent_0
latent_1
...
latent_{d-1}
```

### 输入

```text
config yaml
data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
AnnData .h5ad
rare_class
rare_train_size
seed
split_mode
```

### 输出

输出目录：

```text
outputs/{dataset}/{run_id}/
```

主要文件：

```text
split_assignments.csv
selected_hvg_genes.csv
resource_summary.csv
embeddings/train_predictions.csv
embeddings/train_latent.csv
embeddings/validation_predictions.csv
embeddings/validation_latent.csv
embeddings/test_predictions.csv
embeddings/test_latent.csv
```

### 与其他文件的关系

后续几乎所有文件都依赖它：

- `03_prototype.py` 读 latent 和 predictions
- `03b_knn_baseline.py` 读 latent
- `03c_celltypist_baseline.py` 读 split_assignments 和 selected_hvg_genes
- `04_prototype_gate.py` 读 predictions
- `05_prototype_gate_marker.py` 读 predictions、HVG、表达矩阵
- `06_fusion.py` 读 probability 和 latent
- `07_evaluate.py` 读 test predictions

### 小白理解版总结

这个文件训练 scANVI，并把 scANVI 对每个细胞的“判断结果”和“隐藏空间坐标”保存下来。后续所有 rescue 方法都是在这些输出基础上做的。

---

## 3.6 `src/03_prototype.py`

### 文件定位

Stage 3：在 scANVI latent space 里做 prototype 计算。

### 什么时候调用？

```bash
python src/03_prototype.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

它使用 Stage 2 的 latent embedding：

```text
train_latent.csv
validation_latent.csv
test_latent.csv
```

然后用 train 中有标签细胞构建每个 cell type 的 prototype：

```text
prototype = 该类 labeled training cells 的 latent 均值
```

之后对 validation/test 每个 cell 计算：

```text
到 rare prototype 的距离
到 scANVI predicted class prototype 的距离
rare prototype 的距离排名
predicted-class distance - rare distance
是否是 prototype rescue candidate
```

### prototype candidate 如何定义？

当前代码中的核心条件：

```text
scANVI predicted_label != rare_class
rare prototype rank <= 2
scANVI prediction margin <= margin 的 25% 分位数
```

对应代码：

```python
candidates = (predicted_labels != rare_class) & (ranks <= 2) & (margin <= threshold)
```

直观解释：

```text
scANVI 没判成 rare，但 latent 上 rare prototype 很靠前，并且 scANVI 自己也不太自信
```

### separability ratio 如何计算？

文件中还有函数：

```python
separability_metrics(...)
```

它计算：

```text
rare intra-class radius = rare training cells 到 rare prototype 的平均距离
nearest-majority distance = rare prototype 到最近 major prototype 的距离
separability ratio = nearest-majority distance / intra-class radius
```

输出包括：

```text
separability_ratio
nearest_majority_class
intra_rare_radius
dist_to_nearest_majority
rescue_confidence
```

`rescue_confidence` 当前规则：

```text
sep >= 1.5 → HIGH
sep >= 1.0 → MEDIUM
else       → LOW
```

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/train_predictions.csv
outputs/{dataset}/{run_id}/embeddings/train_latent.csv
outputs/{dataset}/{run_id}/embeddings/validation_predictions.csv
outputs/{dataset}/{run_id}/embeddings/validation_latent.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_latent.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/prototype/separability.csv
outputs/{dataset}/{run_id}/prototype/validation_scores.csv
outputs/{dataset}/{run_id}/prototype/test_scores.csv
```

`*_scores.csv` 里通常有：

```text
distance_to_{rare_class}
distance_to_pred
prototype_rank_{rare_class}
d_pred_minus_d_{rare_class}
prototype_rescue_candidate
```

### 与其他文件的关系

- `04_prototype_gate.py` 读取 `prototype/test_scores.csv`
- `05_prototype_gate_marker.py` 读取 `prototype/*_scores.csv`
- `06_fusion.py` 读取 prototype scores 并重新计算 prototype probabilities
- `10_paper_table.py` 读取 `separability.csv`

### 小白理解版总结

这个文件问的是：在 scANVI 的 latent 空间里，一个 test cell 是不是更像 rare class？它也是 separability ratio 的来源。

---

## 3.7 `src/03b_knn_baseline.py`

### 文件定位

Stage 3b：kNN 对比方法。

它不是 scRareRefine 主方法，而是 baseline comparison。

### 什么时候调用？

```bash
python src/03b_knn_baseline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

它使用 Stage 2 的 latent embedding。

训练集 reference：

```text
labeled training cells
```

测试集：

```text
test cells
```

方法：

```text
在 latent space 中找 k 个最近邻，按多数投票预测 label
```

默认：

```text
k = 15
```

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/train_predictions.csv
outputs/{dataset}/{run_id}/embeddings/train_latent.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_latent.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/knn/test_predictions.csv
outputs/{dataset}/{run_id}/knn/test_metrics.csv
```

### 与其他文件的关系

`07_evaluate.py` 会检查：

```text
knn/test_metrics.csv
```

如果存在，就把 kNN 结果加入 `final_metrics.csv`。

### 小白理解版总结

这是一个“最近邻投票”对比方法，用来证明 scRareRefine 不只是比 scANVI 好，也要和简单 latent-space baseline 比。

---

## 3.8 `src/03c_celltypist_baseline.py`

### 文件定位

Stage 3c：CellTypist 等效的逻辑回归 baseline。

它也是对比方法，不是 scRareRefine 主方法。

### 什么时候调用？

```bash
python src/03c_celltypist_baseline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它为什么叫 CellTypist 等效？

CellTypist 的核心思想是：

```text
表达矩阵标准化
→ 标准化特征
→ logistic regression 分类
```

这个脚本没有直接依赖 CellTypist 包，而是用 sklearn 实现同类算法，避免 CellTypist 与 sklearn 版本不兼容。

### 它做什么？

1. 读取 `split_assignments.csv`，找到 labeled training cells。
2. 读取 Stage 2 的 HVG 基因列表。
3. 从 AnnData 中取这些基因的表达矩阵。
4. 做 library-size normalize + log1p。
5. 用 `StandardScaler` 标准化。
6. 用 `LogisticRegression` 训练分类器。
7. 对 test cells 预测 label。
8. 计算 rare F1 等指标。

### 输入

```text
outputs/{dataset}/{run_id}/split_assignments.csv
outputs/{dataset}/{run_id}/selected_hvg_genes.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
AnnData .h5ad
```

### 输出

```text
outputs/{dataset}/{run_id}/celltypist/test_predictions.csv
outputs/{dataset}/{run_id}/celltypist/test_metrics.csv
```

### 与其他文件的关系

`07_evaluate.py` 会自动读取：

```text
celltypist/test_metrics.csv
```

如果存在，就加入最终指标表。

`10_paper_table.py` 也会直接收集这些 CellTypist 结果，避免必须重跑 Stage 7。

### 小白理解版总结

这是“传统表达空间分类器”的对比方法。它回答：如果不用 scRareRefine，只用 normalized expression + logistic regression，效果如何？

---

## 3.9 `src/04_prototype_gate.py`

### 文件定位

Stage 4：评估不同 prototype gate 规则。

### 什么时候调用？

```bash
python src/04_prototype_gate.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

它读取 Stage 3 的 prototype scores，然后尝试多种 gate 规则。

核心问题是：

```text
哪些 test cells 应该被当作 rare rescue candidates？
```

### 当前 gate 规则

定义在：

```python
gate_masks(...)
```

包括：

```text
rank1
rank2_margin_q25
rank2_dscore_q90
rank2_margin_q25_entropy_q50
rank2_margin_q25_neighbor_major
```

它们大致含义：

- `rank1`：rare prototype 是最近的 prototype。
- `rank2_margin_q25`：rare prototype 排名前 2，且 scANVI 预测 margin 很低。
- `rank2_dscore_q90`：rare prototype 排名前 2，且 rare 比 predicted class 更接近的程度足够强。
- `rank2_margin_q25_entropy_q50`：同时要求低 margin 和高 entropy，也就是 scANVI 不确定。
- `rank2_margin_q25_neighbor_major`：额外限制 predicted label 是主要竞争类别之一。

### 它如何评估 gate？

对每个 gate：

```text
把 gate 选中的 cells 改成 rare_class
计算 rare_f1、false_rescue、overall_accuracy 等
```

注意：这里是为了评估 gate 本身，不包含 marker verification。

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/validation_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/prototype/validation_scores.csv
outputs/{dataset}/{run_id}/prototype/test_scores.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/gate/validation_results.csv
outputs/{dataset}/{run_id}/gate/validation_candidates.csv
outputs/{dataset}/{run_id}/gate/test_results.csv
outputs/{dataset}/{run_id}/gate/test_candidates.csv
```

### 与其他文件的关系

`07_evaluate.py` 会读取 gate 结果，并产生：

```text
prototype_gate
prototype_gate_best
```

其中 `prototype_gate_best` 是根据 validation 上的表现选择 best gate，再看 test 结果。

### 小白理解版总结

这个文件是在测试“哪些 prototype candidate 过滤规则比较好”。它主要解决 false rescue 问题：不能看到像 rare 就全改成 rare。

---

## 3.10 `src/05_prototype_gate_marker.py`

### 文件定位

Stage 5：当前主线方法 `prototype_gate_marker` 的核心文件。

它做的是：

```text
rank1 prototype candidates
→ marker verification
→ validation 选择 threshold
→ test 应用 threshold
```

### 什么时候调用？

```bash
python src/05_prototype_gate_marker.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它解决什么问题？

Stage 3/4 只看 latent geometry，容易误把 major cells rescue 成 rare。

这个文件加入表达层面的证据：

```text
一个 candidate 不仅要在 latent 上像 rare
还要在表达上有 rare marker signature
```

### 关键步骤

#### Step 1：读取 HVG 和表达矩阵

输入：

```text
selected_hvg_genes.csv
AnnData .h5ad
```

表达矩阵会经过：

```text
log1p_cpm
```

即每个细胞 normalize 到 10000 counts 后 log1p。

---

#### Step 2：用训练集构建 marker signatures

对应函数：

```python
compute_marker_signatures(...)
```

它只使用：

```text
train 中 is_labeled_for_scanvi=True 的 cells
```

对每个 cell type：

```text
计算该类平均表达 - 其他类平均表达
取 top 25 个正向差异基因
```

默认要求：

```text
每个类至少 5 个 labeled cells
```

输出 marker signature：

```text
cell_type → marker gene list
```

---

#### Step 3：选 rank1 prototype candidates

对应函数：

```python
_rank1_candidate_ids(...)
```

当前规则是：

```text
scANVI predicted_label != rare_class
prototype_rank_{rare_class} <= 1
```

这就是当前 `gate_marker` 里的 gate。

注意：它没有直接复用 `04_prototype_gate.py` 中所有 gate rules，而是使用 rank1 gate。

---

#### Step 4：计算 marker margin

对应函数：

```python
score_candidates(...)
```

对每个 candidate：

```text
rare_score = rare marker genes 的平均表达
pred_score = scANVI predicted class marker genes 的平均表达
marker_margin = rare_score - pred_score
```

如果 `marker_margin` 越高，说明这个 cell 表达上越支持 rare class。

---

#### Step 5：validation 选择 threshold

对应函数：

```python
marker_threshold_curve(...)
choose_threshold(...)
```

它会尝试一系列 marker threshold，并在 validation 上选择一个阈值。

默认约束：

```text
major_to_rare_false_rescue_rate <= 0.001
```

如果没有满足约束的 threshold，就退而选所有 threshold 中表现最好的。

排序目标：

```text
rare_f1 高
rare_recall 高
rare_precision 高
threshold 尽量低
```

---

#### Step 6：test 应用 threshold

对 test candidates：

```text
marker_margin >= selected_threshold → 可以被 rescue 成 rare
否则保持 scANVI prediction
```

实际 relabel 在 `07_evaluate.py` 中完成。

### 输入

```text
AnnData .h5ad
outputs/{dataset}/{run_id}/selected_hvg_genes.csv
outputs/{dataset}/{run_id}/embeddings/train_predictions.csv
outputs/{dataset}/{run_id}/embeddings/validation_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/prototype/validation_scores.csv
outputs/{dataset}/{run_id}/prototype/test_scores.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/gate_marker/marker_signatures.csv
outputs/{dataset}/{run_id}/gate_marker/validation_scored.csv
outputs/{dataset}/{run_id}/gate_marker/marker_threshold_curve.csv
outputs/{dataset}/{run_id}/gate_marker/selected_thresholds.csv
outputs/{dataset}/{run_id}/gate_marker/test_scored.csv
```

### 与其他文件的关系

`07_evaluate.py` 读取：

```text
gate_marker/test_scored.csv
gate_marker/selected_thresholds.csv
```

然后生成最终方法：

```text
prototype_gate_marker
```

### 小白理解版总结

这个文件是主方法的关键：它先找“latent 上最像 rare 的漏判细胞”，再看这些细胞是否真的表达 rare marker。只有两种证据都支持时，才把 scANVI 的预测改成 rare。

---

## 3.11 `src/06_fusion.py`

### 文件定位

Stage 6：prototype probability 和 scANVI probability 融合。

这是主方法之外的另一种 refinement 变体。

### 什么时候调用？

```bash
python src/06_fusion.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

它不是直接用 marker，而是把两种概率混合：

```text
scANVI softmax probability
prototype-derived probability
```

### 两种 fusion

#### 1. Global fusion

对所有 cells 都融合概率。

核心函数：

```python
fuse(...)
```

大意：

```text
fused_prob = alpha * scanvi_prob + (1 - alpha) * prototype_prob
```

其中 alpha 会根据 scANVI margin 动态调整。

参数网格：

```text
temperature ∈ {0.5, 1.0, 2.0}
alpha_min ∈ {0.3, 0.5, 0.7}
beta ∈ {0.5, 1.0}
```

---

#### 2. Gated fusion

只对 rank1 candidates 做 rescue，其他 cells 保持 scANVI 预测。

核心函数：

```python
gated_fuse(...)
```

逻辑：

```text
如果 cell 是 rank1 candidate：
    fused_rare_prob = (1 - alpha) * prototype_rare_prob + alpha * scanvi_rare_prob
    如果 fused_rare_prob >= threshold：
        改成 rare
    否则保持 scANVI prediction
非 candidate 永远不改
```

参数网格：

```text
temperature ∈ {0.5, 1.0, 2.0}
alpha ∈ {0.0, 0.2, 0.4}
rare_prob_threshold ∈ {0.3, 0.5, 0.7}
```

### 参数如何选择？

在 validation 上选。

约束：

```text
overall_accuracy >= baseline_accuracy - 0.005
major_to_rare_false_rescue_rate <= 0.005
```

目标：

```text
rare_f1 最大
然后 overall_accuracy 高
然后 false rescue 低
```

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/train_predictions.csv
outputs/{dataset}/{run_id}/embeddings/train_latent.csv
outputs/{dataset}/{run_id}/embeddings/validation_predictions.csv
outputs/{dataset}/{run_id}/embeddings/validation_latent.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_latent.csv
outputs/{dataset}/{run_id}/prototype/validation_scores.csv
outputs/{dataset}/{run_id}/prototype/test_scores.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/fusion/validation_grid.csv
outputs/{dataset}/{run_id}/fusion/best_params.csv
outputs/{dataset}/{run_id}/fusion/test_results.csv
outputs/{dataset}/{run_id}/fusion/gated_validation_grid.csv
outputs/{dataset}/{run_id}/fusion/gated_best_params.csv
outputs/{dataset}/{run_id}/fusion/gated_test_results.csv
```

### 与其他文件的关系

`07_evaluate.py` 读取 fusion 结果，并加入：

```text
fusion
fusion_gated
```

### 小白理解版总结

这个文件在试另一种后处理思路：不是硬改 label，而是把 scANVI 的概率和 prototype 的概率混起来，再看 rare class 概率是否足够高。

---

## 3.12 `src/07_evaluate.py`

### 文件定位

Stage 7：把一个 run 的所有方法结果汇总成最终指标表。

### 什么时候调用？

```bash
python src/07_evaluate.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

它读取前面各阶段的输出，生成：

```text
final_metrics.csv
```

每一行是一个方法。

当前可能包含：

```text
baseline
knn_k15
celltypist
prototype
prototype_gate
prototype_gate_best
prototype_gate_marker
fusion
fusion_gated
```

其中 kNN 和 CellTypist 是可选的：

```text
如果对应文件存在，就加入；不存在就跳过
```

### 各方法如何计算？

#### `baseline`

直接用 scANVI 的 `predicted_label`。

#### `prototype`

把 `prototype_rescue_candidate=True` 的 cells 改成 rare。

#### `prototype_gate`

使用 Stage 4 的 test gate 结果。

#### `prototype_gate_best`

在 validation 上选最好的 gate，然后报告对应 test 结果。

#### `prototype_gate_marker`

读取 Stage 5 的：

```text
test_scored.csv
selected_thresholds.csv
```

对 marker_margin 超过阈值的 cells 改成 rare。

#### `fusion` / `fusion_gated`

直接读取 Stage 6 计算好的 test results。

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/prototype/test_scores.csv
outputs/{dataset}/{run_id}/gate/test_results.csv
outputs/{dataset}/{run_id}/gate/validation_results.csv
outputs/{dataset}/{run_id}/gate_marker/test_scored.csv
outputs/{dataset}/{run_id}/gate_marker/selected_thresholds.csv
outputs/{dataset}/{run_id}/fusion/test_results.csv
outputs/{dataset}/{run_id}/fusion/gated_test_results.csv
outputs/{dataset}/{run_id}/knn/test_metrics.csv       # 可选
outputs/{dataset}/{run_id}/celltypist/test_metrics.csv # 可选
```

### 输出

```text
outputs/{dataset}/{run_id}/metrics/final_metrics.csv
```

列包括：

```text
method
seed
rare_class
rare_train_size
split_mode
overall_accuracy
macro_f1
rare_precision
rare_recall
rare_f1
n_candidates
n_marker_verified
rescued_rare_errors
false_rescues
modification_rate
major_to_rare_false_rescue_rate
...
```

### 与其他文件的关系

- `08_visualize.py` 读取这个文件做单 run 图。
- `09_aggregate_plot.py` 读取所有 runs 的这个文件做汇总图。
- `10_paper_table.py` 读取所有 runs 的这个文件做论文表。

### 小白理解版总结

这个文件是“裁判”。它把每种方法在 test set 上的表现放进同一张表，方便比较。

---

## 3.13 `src/08_visualize.py`

### 文件定位

Stage 8：给单个 run 画结果图。

### 什么时候调用？

```bash
python src/08_visualize.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

读取单个 run 的：

```text
metrics/final_metrics.csv
```

然后画图展示各方法的表现。

### 主要图

#### `method_comparison.png`

比较不同方法的：

```text
rare_f1
rare_precision
rare_recall
overall_accuracy
```

#### `rescue_effect.png`

展示 rescue 行为：

```text
n_candidates
rescued_rare_errors
false_rescues
major_to_rare_false_rescue_rate
```

#### heatmap 类图

把不同方法和不同指标放在热图里。

### 输入

```text
outputs/{dataset}/{run_id}/metrics/final_metrics.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/metrics/method_comparison.png
outputs/{dataset}/{run_id}/metrics/rescue_effect.png
outputs/{dataset}/{run_id}/metrics/metrics_heatmap.png
```

### 小白理解版总结

这是单次实验的可视化脚本。你想看某一个 seed、某一个 rare class 的详细方法对比，就看它生成的图。

---

## 3.14 `src/09_aggregate_plot.py`

### 文件定位

论文级汇总图生成脚本。

它不是针对一个 run，而是扫描所有：

```text
outputs/*/*/metrics/final_metrics.csv
```

### 什么时候调用？

```bash
python src/09_aggregate_plot.py --out_dir figures/paper
```

### 它做什么？

1. 收集所有 run 的 final metrics。
2. 合并 separability ratio。
3. 生成多种汇总图。
4. 保存到 `figures/paper/`。

### 主要函数和图

#### `collect_metrics(outputs_dir)`

扫描所有：

```text
outputs/{dataset}/{run_id}/metrics/final_metrics.csv
```

并合并：

```text
outputs/{dataset}/{run_id}/prototype/separability.csv
```

输出一个大 DataFrame。

---

#### `fig_dataset_comparison(...)`

生成：

```text
fig_dataset_comparison.png
```

内容：

```text
每个 dataset/rare_class × 每个 method 的 rare_f1 heatmap
```

---

#### `fig_trainsize_ablation(...)`

生成：

```text
fig_trainsize_ablation.png
```

内容：

```text
rare_train_size 变化时，baseline / gate+marker / fusion_gated 的 rare_f1 曲线
```

---

#### `fig_separability(...)`

生成：

```text
fig_separability.png
```

内容：

```text
x = separability ratio
y = Gate+Marker F1 gain over baseline
```

用于说明 sep ratio 与 rescue 成功之间的关系。

---

#### `fig_all_methods_summary(...)`

生成：

```text
fig_all_methods_summary.png
```

内容：

```text
每个 dataset/rare_class 一个 panel，显示各方法 rare_f1 的 seed 分布
```

---

#### `fig_main_comparison(...)`

生成：

```text
fig_main_comparison.png
```

内容：

```text
baseline、kNN、Gate+Marker、Fusion-gated 的主比较柱状图
```

---

#### `fig_data_efficiency(...)`

生成：

```text
fig_data_efficiency.png
```

内容：

```text
Immune DC ASDC/cDC1 在不同 rare_train_size 下的数据效率曲线
```

---

#### `fig_headline_bar(...)`

生成：

```text
fig_headline_bar.png
```

内容：

```text
rare_train_size=5 时 ASDC/cDC1 的 headline 对比图
```

### 输入

```text
outputs/**/metrics/final_metrics.csv
outputs/**/prototype/separability.csv
```

### 输出

```text
figures/paper/aggregate_metrics.csv
figures/paper/fig_main_comparison.png
figures/paper/fig_dataset_comparison.png
figures/paper/fig_trainsize_ablation.png
figures/paper/fig_data_efficiency.png
figures/paper/fig_separability.png
figures/paper/fig_all_methods_summary.png
figures/paper/fig_headline_bar.png
```

### 小白理解版总结

这个文件把所有实验结果合起来，画论文用的总览图。

---

## 3.15 `src/10_paper_table.py`

### 文件定位

论文表格生成脚本。

### 什么时候调用？

```bash
python src/10_paper_table.py --out_dir figures/paper
```

### 它做什么？

从所有 outputs 中收集结果，生成 CSV 和 LaTeX 表。

### 主要函数

#### `collect_metrics(outputs_dir)`

读取所有：

```text
outputs/{dataset}/{run_id}/metrics/final_metrics.csv
```

还会额外读取：

```text
outputs/{dataset}/{run_id}/celltypist/test_metrics.csv
```

这样即使某些 run 的 Stage 7 没重跑，CellTypist 结果也能进入汇总表。

---

#### `collect_separability(outputs_dir)`

读取所有：

```text
outputs/{dataset}/{run_id}/prototype/separability.csv
```

并从 run_id 中解析 seed。

---

#### `table_main_results(...)`

生成主结果表。

关注：

```text
rare_train_size = 20
```

按：

```text
dataset × rare_class × method
```

统计：

```text
rare_f1_mean
rare_f1_std
rare_f1_n
rare_precision_mean
rare_recall_mean
...
```

---

#### `table_trainsize_ablation(...)`

生成训练样本量消融表。

关注方法：

```text
baseline
prototype_gate_marker
fusion_gated
```

输出不同 rare_train_size 下的 F1。

---

#### `table_separability_results(...)`

把 separability ratio 和 baseline/gate_marker 的 F1 合起来。

输出：

```text
separability_ratio
nearest_majority_class
baseline_f1
gate_marker_f1
f1_gain
```

---

#### `latex_main_table(...)`

生成 LaTeX 格式主表。

当前重点方法：

```text
baseline
knn_k15
prototype_gate_marker
fusion_gated
```

### 输入

```text
outputs/**/metrics/final_metrics.csv
outputs/**/celltypist/test_metrics.csv
outputs/**/prototype/separability.csv
```

### 输出

```text
figures/paper/table_main_results.csv
figures/paper/table_trainsize.csv
figures/paper/table_separability.csv
figures/paper/table_main_results.tex
```

### 小白理解版总结

这个文件负责把实验结果变成论文里的表格。如果要更新主表、数据效率表、separability 表，就跑它。

---

## 3.16 `src/11_umap_visualize.py`

### 文件定位

UMAP 可视化脚本，用于查看 latent space 和 rescue 行为。

### 什么时候调用？

```bash
python src/11_umap_visualize.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20
```

### 它做什么？

读取 validation + test 的 latent embedding，然后跑 UMAP，把高维 latent 压到二维图上。

它画三类图。

### 图 1：cell type UMAP

函数：

```python
fig_celltypes(...)
```

输出：

```text
umap_celltypes.png
```

展示：

```text
每个 cell 在 UMAP 上的位置，按 true_label 上色
```

---

### 图 2：rescue outcome UMAP

函数：

```python
fig_rescue_outcomes(...)
```

输出：

```text
umap_rescue.png
```

展示：

```text
哪些 cell 是 gate candidate
哪些 cell 被 gate+marker rescue
哪些 rescue 是正确或错误
```

---

### 图 3：confidence UMAP

函数：

```python
fig_confidence(...)
```

输出：

```text
umap_confidence.png
```

展示：

```text
每个 cell 的 scANVI entropy / confidence
```

### 输入

```text
outputs/{dataset}/{run_id}/embeddings/validation_predictions.csv
outputs/{dataset}/{run_id}/embeddings/validation_latent.csv
outputs/{dataset}/{run_id}/embeddings/test_predictions.csv
outputs/{dataset}/{run_id}/embeddings/test_latent.csv
outputs/{dataset}/{run_id}/gate/test_candidates.csv
outputs/{dataset}/{run_id}/gate_marker/test_scored.csv
outputs/{dataset}/{run_id}/gate_marker/selected_thresholds.csv
```

### 输出

```text
outputs/{dataset}/{run_id}/figures/umap_celltypes.png
outputs/{dataset}/{run_id}/figures/umap_rescue.png
outputs/{dataset}/{run_id}/figures/umap_confidence.png
```

### 小白理解版总结

这个文件帮你“看见”latent space：rare cells 有没有聚在一起、rescue candidates 在哪里、scANVI 不确定的区域在哪里。

---

## 3.17 `src/gen_pipeline_diagram.py`

### 文件定位

生成 pipeline 架构图的脚本。

它不参与实验计算，只负责画图。

### 什么时候调用？

```bash
python src/gen_pipeline_diagram.py --out figures/paper/fig_pipeline_diagram.png
```

### 它做什么？

用 matplotlib 手动画一个流程图，包括：

```text
输入数据
scANVI baseline
prototype rescue
gate/marker/fusion
输出指标
```

主要函数：

```python
rbox(...)          # 画圆角矩形
diamond(...)      # 画菱形判断框
arr(...)          # 画箭头
segs(...)         # 画折线箭头
draw_pipeline(...) # 画主流程
draw_bottom(...)   # 画底部说明/结果/legend
```

### 输入

没有实验数据输入，只需要输出路径参数：

```text
--out figures/paper/fig_pipeline_diagram.png
```

### 输出

```text
figures/paper/fig_pipeline_diagram.png
```

### 小白理解版总结

这是专门画方法框架图的脚本，不改变任何实验结果。

---

# 4. 从头跑一次实验时，每个文件的调用顺序

假设要跑一个数据集，比如 Immune DC 的 ASDC，流程如下：

```bash
# 1. 生成 split
python src/01_split.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --split_mode batch_heldout

# 2. 训练 scANVI baseline
python src/02_baseline_scanvi.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 3. prototype + separability
python src/03_prototype.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 3b. kNN baseline，可选
python src/03b_knn_baseline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 3c. LR/CellTypist baseline，可选
python src/03c_celltypist_baseline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 4. prototype gate
python src/04_prototype_gate.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 5. prototype gate + marker
python src/05_prototype_gate_marker.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 6. fusion
python src/06_fusion.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 7. 汇总单 run 指标
python src/07_evaluate.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout

# 8. 单 run 可视化，可选
python src/08_visualize.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout
```

如果所有 runs 都已经跑完，论文图表使用：

```bash
python src/09_aggregate_plot.py --out_dir figures/paper
python src/10_paper_table.py --out_dir figures/paper
```

如果要画 UMAP：

```bash
python src/11_umap_visualize.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_class ASDC \
  --rare_train_size 20 \
  --split_mode batch_heldout
```

---

# 5. 常见修改点：想改某个逻辑应该看哪里？

## 想改数据 split 方式

看：

```text
src/01_split.py
```

重点函数：

```text
cell_stratified_split
batch_heldout_split
```

---

## 想改 rare_train_size 如何影响标签

看：

```text
src/02_baseline_scanvi.py
```

重点函数：

```text
make_scanvi_labels
```

---

## 想改 HVG 选择

看：

```text
src/02_baseline_scanvi.py
```

重点函数：

```text
select_hvg_genes
```

---

## 想改 scANVI 训练参数

优先改 config：

```text
configs/*.yaml
```

如果要改训练逻辑，看：

```text
src/02_baseline_scanvi.py
```

重点函数：

```text
train_scanvi
```

---

## 想改 prototype candidate 条件

看：

```text
src/03_prototype.py
```

重点函数：

```text
prototype_scores
```

当前条件：

```text
predicted_label != rare_class
rare rank <= 2
scANVI margin <= 25% quantile
```

---

## 想改 separability ratio 定义

看：

```text
src/03_prototype.py
```

重点函数：

```text
separability_metrics
```

---

## 想改 prototype gate 规则

看：

```text
src/04_prototype_gate.py
```

重点函数：

```text
gate_masks
```

---

## 想改 marker signature 或 marker threshold

看：

```text
src/05_prototype_gate_marker.py
```

重点函数：

```text
compute_marker_signatures
score_candidates
marker_threshold_curve
choose_threshold
```

---

## 想让 separability ratio 真正进入 hard gate

目前 separability ratio 主要用于分析。若要把它接入主方法，可以改：

```text
src/05_prototype_gate_marker.py
```

可能位置：

```text
读取 prototype/separability.csv
如果 separability_ratio < 某阈值，则清空 val/test candidates
```

这样代码会更符合 “separability-aware gate” 的论文叙事。

---

## 想改 fusion 规则

看：

```text
src/06_fusion.py
```

重点函数：

```text
fuse
gated_fuse
_fusion_grid
_gated_fusion_grid
select_best_params
select_best_gated_params
```

---

## 想改最终指标表

看：

```text
src/07_evaluate.py
```

重点函数：

```text
_baseline_row
_prototype_row
_gate_row
_marker_row
main
```

---

## 想改论文图

看：

```text
src/09_aggregate_plot.py
figures/paper/fig_separability_gain.py
figures/paper/fig_data_efficiency_v2.py
```

其中 `src/09_aggregate_plot.py` 生成汇总 PNG，`figures/paper/` 下脚本生成更精修的论文版 SVG/PDF/TIFF。

---

## 想改论文表

看：

```text
src/10_paper_table.py
```

---

# 6. 每个文件一句话速查

| 文件 | 一句话说明 |
|---|---|
| `.DS_Store` | macOS 系统文件，无代码作用 |
| `utils.py` | 全局工具箱：读配置、读数据、路径、指标、标准化、资源监控 |
| `00_preprocess_pbmc.py` | 把 PBMC 原始数据处理成 50k pDC screening 数据集 |
| `01_split.py` | 生成 train/validation/test split |
| `02_baseline_scanvi.py` | 训练 scVI/scANVI，输出 baseline prediction 和 latent embedding |
| `03_prototype.py` | 基于 latent prototype 计算 rare candidate 和 separability ratio |
| `03b_knn_baseline.py` | kNN latent-space 对比方法 |
| `03c_celltypist_baseline.py` | LR/CellTypist 等效表达空间对比方法 |
| `04_prototype_gate.py` | 评估不同 prototype gate 规则 |
| `05_prototype_gate_marker.py` | 主方法：rank1 prototype candidate + marker verification |
| `06_fusion.py` | scANVI probability 与 prototype probability 融合方法 |
| `07_evaluate.py` | 汇总单个 run 的所有方法 test 指标 |
| `08_visualize.py` | 单个 run 的方法比较图和 rescue 效果图 |
| `09_aggregate_plot.py` | 汇总所有 runs 生成论文图 |
| `10_paper_table.py` | 汇总所有 runs 生成论文表格和 LaTeX 表 |
| `11_umap_visualize.py` | 画 latent UMAP、rescue UMAP、confidence UMAP |
| `gen_pipeline_diagram.py` | 画 pipeline 框架图 |

---

# 7. 最重要的理解点

1. `02_baseline_scanvi.py` 是后续所有方法的基础，因为它输出 scANVI prediction、probability 和 latent embedding。
2. `03_prototype.py` 负责把 latent embedding 转成 prototype 距离、candidate 和 separability ratio。
3. `05_prototype_gate_marker.py` 是当前主线 rescue 方法的核心：先用 rank1 prototype gate 找候选，再用 marker 验证。
4. `07_evaluate.py` 是单 run 的最终裁判。
5. `09_aggregate_plot.py` 和 `10_paper_table.py` 是论文层面的汇总工具。
6. 所有方法都必须遵守 inductive 约束：训练 reference、prototype、marker signature、threshold 不能使用 test label。
