# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

always reply in Chinese

## 项目目标

scRareRefine 基于 scANVI 的预测概率和 latent embedding，设计稀有细胞识别 refinement 模块，提高 rare cell type 的识别效果。核心方法是 prototype 距离评分 + marker gene 验证的两阶段 rescue 流程。

## 常用命令

```bash
# 安装依赖
pip install -e .[dev]

# 运行完整管道（必须显式指定 seed 和 rare_train_size）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 20

# 强制重新训练（忽略缓存）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 20 --force

# 单独运行某一 Stage（如调试 Stage 5）
python src/05_prototype_gate_marker.py --config configs/immune_dc.yaml \
    --seed 42 --rare_class ASDC --rare_train_size 20

# 简化版管道（仅 baseline vs scRareRefine 对比）
python src/main.py --config configs/immune_dc.yaml --seed 42 --rare_class ASDC --rare_train_size 20

# 运行测试
pytest -v
pytest tests/test_prototype.py
pytest tests/test_prototype.py -k "test_separability"
```

## 目录结构

```
src/
├── main.py                       # 简化版管道入口（baseline vs scRareRefine）
├── utils.py                      # 共享工具：I/O、路径、指标、资源监控
├── 01_split.py                   # 生成 train/val/test split
├── 02_baseline_scanvi.py         # 训练 scANVI，输出 embeddings
├── 03_prototype.py               # 原型距离评分 + 可分性指标
├── 03b_knn_baseline.py           # kNN baseline（k=15）
├── 03c_celltypist_baseline.py    # Logistic regression baseline
├── 04_prototype_gate.py          # Prototype ranking 候选筛选
├── 05_prototype_gate_marker.py   # Marker gene 验证 + validation 阈值选择
├── 06_fusion.py                  # 概率融合（可选扩展）
├── 07_evaluate.py                # 多 seed 汇总与对比
└── 08_visualize.py               # UMAP 可视化
configs/                          # YAML 配置（immune_dc、pancreas_epsilon、pancreas_gamma 等）
data/raw/                         # 只读原始数据（.h5ad）
data/splits/                      # 生成的 split 索引
outputs/                          # 实验输出（按数据集 + run_id 组织）
```

## 架构要点

### Stage 管道与文件依赖

```
run_pipeline.py
├── Stage 1 (01_split.py) → data/splits/{dataset}/{split_mode}_seed{seed}/split.csv
├── Stage 2 (02_baseline_scanvi.py) → outputs/.../embeddings/{train|val|test}_{predictions,latent}.csv
├── Stage 3 (03_prototype.py) → outputs/.../prototype/{separability,val|test_scores}.csv
├── Stage 5 (05_prototype_gate_marker.py) → outputs/.../gate_marker/{*_scored,selected_thresholds}.csv
└── Stage 7 (07_evaluate.py) → outputs/.../all_seeds_metrics.csv
```

参数优先级：CLI 参数 > 配置文件（通过 `utils.resolve_param()` 实现）。

### 关键算法

**原型评分**（`03_prototype.py`）：基于标注训练集计算各类均值原型，对 query cells 计算欧氏距离。`prototype_rescue_candidate = rank≤2 & margin低分位`。`separability_ratio = dist_to_nearest_majority / intra_rare_radius`：≥1.3 时 rescue 有效，<1.1 时自动 abstain。

**Marker 验证**（`05_prototype_gate_marker.py`）：从训练集有标签样本计算 top-25 marker genes（按 in_class_mean - out_class_mean 排序），在 validation 集上用 `max_false_rescue_rate ≤ 0.001` 约束选择 marker_margin 阈值，再应用于 test。

### 输出目录结构

Run ID 格式：`{split_mode}_seed{seed}_{safe_rare_class}_rare{rare_train_size}`（例：`batch_heldout_seed42_asdc_rare20`）

```
outputs/{dataset}/{run_id}/
├── embeddings/    # predictions + latent（train/val/test）
├── prototype/     # separability.csv + val|test_scores.csv
├── gate_marker/   # marker_signatures.csv + scored + selected_thresholds.csv
├── knn/           # (Stage 3b)
├── celltypist/    # (Stage 3c)
└── metrics/       # scRareRefine_metrics.csv + 对比图
```

## 优先级与约束

当指令冲突时：

1. 不修改原始数据（`data/raw/`、原始 `.h5ad` 文件）
2. 不伪造结果
3. 不破坏标准项目结构
4. 不无记录地改变实验设置
5. 不写未经验证的科研结论

### Inductive 约束（修改评估逻辑时必须遵守）

- Prototype reference、HVG 选择、marker signature 均只能来自训练集
- Fusion 参数只能从 validation 选择
- Test 标签仅用于最终评估，不用于调参或阈值选择
- validation/test cells 不能泄漏到训练 reference

### 配置约束

不要无记录地改变：随机种子、split、rare cell 定义、label/batch 列、预处理流程、baseline 设置、指标定义、scANVI 训练参数。

当前可运行配置：`configs/immune_dc.yaml`、`configs/pancreas_epsilon.yaml`、`configs/pancreas_gamma.yaml`

重要配置键：`dataset.{path,label_key,batch_key,use_raw,use_layer}`、`experiment.{rare_class,secondary_rare_classes,rare_train_sizes,seeds,unlabeled_category}`、`model.{n_top_hvg,n_latent,scvi_max_epochs,scanvi_max_epochs,batch_size}`

## 修改前计划模板

修改代码、配置、实验脚本或文档前，先输出：

```text
本次任务目标：

计划修改文件：
1.
2.

不会修改的内容：
- data/raw/
- 原始 .h5ad 文件
- 已有正式结果（除非用户明确要求）

预期输出：

风险：

验证方式：
```

用户确认后再修改，除非用户明确要求直接执行。

## 结果与文档规则

实验输出 → `results/`；日志 → `logs/`；模型权重 → `checkpoints/`；临时文件 → `tmp/`。不要在根目录随意新建目录。

## 论文主张限制

不可写：解决了稀有细胞识别问题、全面优于所有方法、state-of-the-art、临床可用、适用于所有单细胞数据。

可写（需结果支持）：在评估的数据集上，scRareRefine 相比原始 scANVI 提升了稀有细胞相关指标。

## Git 规则

不要创建 git commit、git push、force push、reset、删除分支或清理大目录，除非用户明确要求。
