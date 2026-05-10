# scRareRefine

基于 scANVI 预测概率和 latent embedding 的稀有细胞识别 refinement 项目。

通过 prototype、gate、marker verification、probability fusion 四个可独立组合的模块，提升 scANVI 对稀有细胞类型的识别效果。

---

## 目录结构

```
scRareRefine-standard/
├── src/
│   ├── utils.py                      # 共享工具：IO、metrics、seed、路径
│   ├── 01_split.py                   # Stage 1：生成 train/val/test split
│   ├── 02_baseline_scanvi.py         # Stage 2：训练 scANVI，输出 embeddings（支持 --force）
│   ├── 03_prototype.py               # Stage 3：计算 prototype 距离得分
│   ├── 04_prototype_gate.py          # Stage 4：应用 prototype gate 规则
│   ├── 05_prototype_gate_marker.py   # Stage 5：marker 验证（threshold 在 val 选）
│   ├── 06_fusion.py                  # Stage 6：概率融合（参数在 val 选）
│   ├── 07_evaluate.py                # Stage 7：汇总各方法 test 指标
│   └── 08_visualize.py               # Stage 8：生成方法对比可视化图表
├── configs/
│   ├── immune_dc.yaml
│   ├── pancreas_epsilon.yaml
│   └── pancreas_gamma.yaml
├── data/
│   ├── raw/                          # 原始数据（只读）
│   └── splits/                       # Stage 1 输出的 split CSV
├── outputs/                          # Stage 2-8 的所有输出
│   └── {dataset}/{run_id}/
│       ├── split_assignments.csv
│       ├── selected_hvg_genes.csv
│       ├── resource_summary.csv
│       ├── embeddings/               # Stage 2 输出
│       ├── prototype/                # Stage 3 输出
│       ├── gate/                     # Stage 4 输出
│       ├── gate_marker/              # Stage 5 输出
│       ├── fusion/                   # Stage 6 输出
│       └── metrics/                  # Stage 7-8 输出（含图表）
├── notebooks/
├── logs/
├── _legacy/                          # 旧版 scrare 包代码备份（不再使用）
└── CLAUDE.md
```

`run_id` 格式：`{split_mode}_seed{seed}_{rare_class}_rare{rare_train_size}`
例：`batch_heldout_seed42_asdc_rare20`

---

## 安装依赖

```bash
pip install scvi-tools anndata pandas numpy scipy scikit-learn pyyaml psutil
```

---

## 运行流程

各阶段独立运行，每个阶段读取上一阶段的输出。Stage 3-6 是四个并列的 refinement 模块，可单独与 baseline 组合使用。

### Stage 1：生成 split（每个 seed 只需跑一次）

```bash
python src/01_split.py --config configs/immune_dc.yaml --seed 42
# 支持 --split_mode batch_heldout（默认）或 cell_stratified
```

输出：`data/splits/immune_dc/batch_heldout_seed42/split.csv`

### Stage 2：训练 scANVI baseline

```bash
python src/02_baseline_scanvi.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/immune_dc/batch_heldout_seed42_asdc_rare20/embeddings/`

> **Embedding 复用**：若 `embeddings/train_predictions.csv` 已存在，Stage 2 会自动跳过训练，直接复用已有结果。  
> 需要强制重新训练时，加 `--force` 参数：
> ```bash
> python src/02_baseline_scanvi.py ... --force
> ```
> Stage 3-8 每次都会重新计算（运行很快，无需缓存）。

### Stage 3：Prototype 得分

```bash
python src/03_prototype.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../prototype/`

### Stage 4：Prototype Gate 规则

```bash
python src/04_prototype_gate.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../gate/`（5 种 gate 规则的 val/test 指标）

### Stage 5：Prototype Gate + Marker 验证

```bash
python src/05_prototype_gate_marker.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../gate_marker/`（threshold 在 val 选，应用到 test）

### Stage 6：Probability Fusion

```bash
python src/06_fusion.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../fusion/`（参数在 val 格点搜索，最优参数应用到 test）

### Stage 7：汇总评估

```bash
python src/07_evaluate.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../metrics/final_metrics.csv`，包含每个方法的 rare_F1 / precision / recall 等指标。

### Stage 8：可视化结果对比

```bash
python src/08_visualize.py \
    --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20
```

输出：`outputs/.../metrics/` 下三张图：

| 文件 | 内容 |
|---|---|
| `method_comparison.png` | 4 格子图：Rare F1 / Recall / Precision / Overall Acc，虚线标 baseline |
| `rescue_effect.png` | Rescued 细胞数 / False Rescue 数 / False Rescue Rate |
| `metrics_heatmap.png` | 全指标热力图（绿=高，红=低） |

---

## 完整示例（immune_dc，seed=42，rare_train_size=20）

> `--rare_class` 不传时自动使用 config 中的默认值；传了则以命令行参数为准。

```bash
python src/01_split.py                  --config configs/immune_dc.yaml --seed 42
python src/02_baseline_scanvi.py        --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/03_prototype.py              --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/04_prototype_gate.py         --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/05_prototype_gate_marker.py  --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/06_fusion.py                 --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/07_evaluate.py               --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
python src/08_visualize.py              --config configs/immune_dc.yaml --seed 42 --rare_train_size 20
```

重跑时，Stage 2 自动跳过（embedding 已存在）；如需重新训练：

```bash
python src/02_baseline_scanvi.py ... --force
```

---

## 各方法说明

| 方法 | 来源阶段 | 说明 |
|---|---|---|
| baseline | Stage 2 | 原始 scANVI 预测，不做任何改动 |
| prototype | Stage 3 | prototype_rescue_candidate 候选强制改标为 rare_class |
| prototype_gate | Stage 4 | rank1 gate（prototype rank ≤ 1）候选改标 |
| prototype_gate_marker | Stage 5 | rank1 候选经 marker 得分过滤后改标，precision 最高 |
| fusion | Stage 6 | scANVI 概率与 prototype 概率加权融合，rare_F1 综合最优 |

---

## 核心 inductive 约束

以下约束必须保持，确保评估不泄漏测试信息：

1. val/test cell 不进入训练 reference
2. HVG 仅基于训练集选择
3. prototype reference 仅来自训练集标注 cell
4. marker signature 仅由训练集有标注 cell 计算
5. 所有调参（fusion 参数、marker threshold）仅基于 validation 集
6. test 标签不用于任何调参或阈值选择

---

## 数据规则

- `data/raw/` 只读，禁止修改
- split 结果写入 `data/splits/`
- 所有模型输出写入 `outputs/`
- 旧版 scrare 包代码备份在 `_legacy/`（不再使用）
