# DATA_CARD.md

## 目的

本文档用于记录 scRareRefine 项目的数据集信息、数据格式、标签列、batch 列、预处理规则和稀有细胞定义。

正式实验前，必须先补全本文档中的 TODO。

## 当前数据格式

所有当前数据均为：

```text
.h5ad
```

推荐原始数据存放位置：

```text
data/raw/
```

## 当前已知数据集

| 数据集名称 | 文件名 | 格式 | 当前状态 | 备注 |
|---|---|---|---|---|
| pancreas | human_pancreas_norm_complexBatch.h5ad | .h5ad | 已有 | 已跑过 scANVI baseline，需要复核 label/batch 列 |
| immune_dc | human_immune_health_atlas_dc.h5ad | .h5ad | 已有 | 已跑过 scANVI baseline，需要复核 label/batch 列 |
| Tabula Sapiens | TODO | .h5ad | 已有 / 待确认 | 文件名、子集、label 列待检查 |

## 原始数据保护规则

以下目录只读：

```text
data/raw/
```

禁止操作：

```text
覆盖原始 .h5ad
在原始 AnnData 对象上做 inplace 修改并保存回原文件
删除原始数据
改变原始 obs/var/X/layers
```

如果需要保存处理后的文件，必须写入：

```text
data/processed/
```

如果需要保存 split，必须写入：

```text
data/splits/
```

如果需要保存 scANVI 输出，必须写入：

```text
data/embeddings/
```

## 每个数据集必须检查的信息

每个 `.h5ad` 文件必须检查：

```text
adata.shape
adata.obs.columns
adata.var.columns
adata.obsm.keys()
adata.layers.keys()
adata.uns.keys()
```

重点确认：

1. 哪一列是 cell type label。
2. 哪一列是 batch。
3. 是否有 donor / study / sample / patient 列。
4. 是否有 train/test split 信息。
5. 是否有 unlabeled cells。
6. 每个 cell type 的样本数。
7. 是否存在极小类或可疑标签。
8. scANVI 需要使用的表达矩阵来自 `X`、`raw` 还是某个 layer。

## Label 列候选

当前用户不确定 label 列名称。需要通过数据检查确认。

常见候选名：

```text
cell_type
celltype
cell type
label
labels
annotation
cell_annotation
cell_ontology_class
cell_type_label
celltype_label
final_annotation
```

实际确认后填写：

| 数据集 | Label 列 | 是否确认 | 备注 |
|---|---|---|---|
| pancreas | TODO | 否 | 待检查 |
| immune_dc | TODO | 否 | 待检查 |
| Tabula Sapiens | TODO | 否 | 待检查 |

## Batch / Donor / Study 列候选

常见候选名：

```text
batch
batch_id
study
dataset
donor
sample
sample_id
patient
individual
platform
technology
```

实际确认后填写：

| 数据集 | Batch 列 | Donor/Study 列 | 是否确认 | 备注 |
|---|---|---|---|---|
| pancreas | TODO | TODO | 否 | complexBatch 数据集可能存在明显 batch 信息 |
| immune_dc | TODO | TODO | 否 | 待检查 |
| Tabula Sapiens | TODO | TODO | 否 | 待检查 |

## 稀有细胞定义

当前建议至少测试两种定义：

### 定义 A：固定比例阈值

某个 cell type 在数据集中占比低于阈值，则定义为 rare cell type。

候选阈值：

```text
1%
3%
5%
```

### 定义 B：固定数量阈值

某个 cell type 的细胞数低于阈值，则定义为 rare cell type。

候选阈值：

```text
n < 50
n < 100
n < 200
```

### 当前初始建议

第一版小规模实验建议使用：

```text
rare cell type = cell type frequency < 5%
```

同时在消融或敏感性分析中补充：

```text
1%
3%
5%
```

最终定义必须写入实验配置，不允许在代码中硬编码。

## 数据 split 规则

当前随机种子：

```text
42
43
44
```

建议 split 原则：

1. 同一个实验必须使用固定 split。
2. scANVI baseline 和 scRareRefine 必须使用相同 split。
3. 不同 baseline 必须使用相同 train/test 划分。
4. 如果存在 donor / batch，需要考虑是否按 donor 或 batch 做更严格的划分。
5. 任何 split 文件必须保存到 `data/splits/`。

推荐文件名：

```text
data/splits/{dataset}_seed{seed}_split.npz
```

示例：

```text
data/splits/pancreas_seed42_split.npz
```

## 预处理规则

正式实验前需要确认：

1. scANVI 输入是否为 count matrix。
2. 是否使用 raw counts。
3. 是否需要 normalization。
4. 是否需要 highly variable genes。
5. 是否已有预处理版本。
6. 是否不同数据集预处理一致。

当前暂定规则：

```text
优先复用 scANVI 官方或已有 pipeline 中合理的预处理方式。
禁止为了提升结果而针对某个方法单独修改预处理。
```

## scANVI 输出文件规划

scANVI 输出应保存到：

```text
data/embeddings/{dataset}/seed{seed}/
```

每个 seed 至少保存：

```text
latent.npy
probabilities.npy
predicted_labels.csv
true_labels.csv
cell_ids.csv
label_mapping.json
```

示例：

```text
data/embeddings/pancreas/seed42/latent.npy
data/embeddings/pancreas/seed42/probabilities.npy
data/embeddings/pancreas/seed42/predicted_labels.csv
```

## 数据检查输出

`01_inspect_data.py` 应输出：

```text
results/reports/data_inspection_report.md
results/tables/dataset_summary.csv
results/tables/cell_type_distribution.csv
```

`dataset_summary.csv` 建议字段：

```text
dataset
file_path
n_cells
n_genes
label_column
batch_column
n_cell_types
n_batches
n_missing_labels
```

`cell_type_distribution.csv` 建议字段：

```text
dataset
cell_type
count
frequency
is_rare_1pct
is_rare_3pct
is_rare_5pct
```

## 当前 TODO

- [ ] 检查 `human_pancreas_norm_complexBatch.h5ad` 的 `obs` 列。
- [ ] 确认 pancreas 的 label 列。
- [ ] 确认 pancreas 的 batch 列。
- [ ] 检查 `human_immune_health_atlas_dc.h5ad` 的 `obs` 列。
- [ ] 确认 immune_dc 的 label 列。
- [ ] 确认 immune_dc 的 batch/donor/study 列。
- [ ] 确认 Tabula Sapiens 使用哪个文件或子集。
- [ ] 统计每个数据集的 cell type 分布。
- [ ] 确定最终 rare cell 定义。
