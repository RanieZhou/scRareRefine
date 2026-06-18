# scRareRefine

> 基于 scANVI 的稀有细胞类型识别 post-hoc refinement 模块。
> 用 prototype 距离 + conformal 阈值，在 FFR 受控（≤ 1%）的前提下显著提升 rare cell type 的 F1 / recall。

## 方法概览

scANVI 在标注稀缺时常对稀有细胞「整类漏判」。scRareRefine 在 scANVI 之后接一个 **inductive 的 post-hoc 拯救** 流程：

1. **PrototypeRescuer**：从训练集 labeled 样本算各类均值原型；候选 = 「预测不是 rare」且「到 rare 原型距离 rank ≤ k」。
2. **Conformal τ 校准**：用 validation 集上**非稀有细胞**的 rare membership score，取有限样本 (1-α) 顺序统计量作为 τ；候选中 score ≥ τ 的细胞被 relabel 为稀有类。
3. **三道闸门（全 inductive，绝不碰 test 标签）**：
   - separability 安全网（`sep < 1.3` 弃权）
   - necessity 守门（val baseline rare recall = 1.0 时弃权）
   - val-自适应候选 rank ∈ {1, 2}（高可分自动 rank=1，边界纠缠自动 rank=2）

发表级 FFR 上界 `α = 0.01`，跨数据集固定**不调参**。

## 项目结构

```
run_pipeline.py                 # 端到端主入口
src/
├── preprocess.py               # 预处理 + 三路划分（batch_heldout / cell_stratified）
├── model.py                    # scVI + scANVI 半监督训练
├── rescue.py                   # 4 个 Rescuer + conformal_rescue() 单一入口
└── utils.py                    # config / metrics / 缓存 manifest / 可视化
configs/                        # 7 个数据集 YAML
tools/
├── comparison/                 # 9 个 baseline 对比脚本 + 汇总绘图
├── analysis/                   # ablation / UMAP / rts 扫描
└── extract/                    # Tabula Sapiens 子集抽取
outputs/{dataset}/{run_id}/     # per-run embeddings + metrics + manifest
results/                        # 汇总产物（comparison / sweep_rts / umap / experiment_log.md）
tests/                          # pytest
```

## 数据集

| 配置 | 数据集 | 稀有类 |
|------|--------|--------|
| [immune_dc](configs/immune_dc.yaml) | Human Immune Health Atlas (DC) | ASDC |
| [pancreas_baron](configs/pancreas_baron.yaml) | Baron pancreas | gamma / epsilon |
| [pancreas_integrated](configs/pancreas_integrated.yaml) | 多源整合 pancreas | — |
| [tabula_lung_endo](configs/tabula_lung_endo.yaml) | Tabula Sapiens lung | endothelial 子型 |
| [tabula_lung_stroma](configs/tabula_lung_stroma.yaml) | Tabula Sapiens lung | stromal 子型 |
| [tabula_sapiens_stomach](configs/tabula_sapiens_stomach.yaml) | Tabula Sapiens stomach | mast cell |
| [tabula_small_intestine](configs/tabula_small_intestine.yaml) | Tabula Sapiens small intestine | — |

## 环境

两套 conda 虚拟环境分工：

- `scanvi311` — 主流水线（PyTorch / scvi-tools / scANVI）
- `sandbox310` — 部分对比方法（CellTypist / TOSICA / scBalance / ProtoCloud / HiCat / scCAD）

对比脚本通过 [tools/comparison/_conda_python.py](tools/comparison/_conda_python.py) 自动调度对应环境，无需手动切换。

```bash
pip install -e .[dev]
```

## 快速开始

```bash
# 单次运行（rare_train_size 支持 float 比例 / int 计数 / "all"）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05

# 强制重训（忽略 embeddings 缓存）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --force
```

输出落在 `outputs/{dataset}/batch_heldout_seed42_asdc_rare0.05/`：
- `embeddings/` — scANVI predictions + latent（train / val / test）
- `selected_hvg_genes.csv`
- `manifest.json` — provenance，下次运行自动复用
- `metrics/final_metrics.csv` — baseline vs scRareRefine
- `metrics/method_comparison.png`、`rescue_effect.png`、`marker_violin.png`

## 对比实验

九个方法（scANVI / kNN / CellTypist / scBalance / ProtoCloud / HiCat / scCAD / TOSICA / scRareRefine）共用同一份 embeddings 缓存与 manifest，避免对比偏差。

```bash
# 单方法 × 单 split
python tools/comparison/run_scrarerefine_comparison.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
python tools/comparison/run_scanvi_comparison.py       --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
# ...

# 汇总 + 绘图
python tools/comparison/plot_comparison.py
python tools/comparison/plot_comparison_grid.py
python tools/analysis/plot_sweep_rts_from_comparison.py
```

汇总产物：
- [results/comparison/comparison_summary.csv](results/comparison/comparison_summary.csv) / `_agg.csv`
- [results/comparison/comparison_bars.png](results/comparison/comparison_bars.png) / `comparison_bars_grid.png`
- [results/sweep_rts/sweep_rts_curves.png](results/sweep_rts/sweep_rts_curves.png)

## 最新结果（第九轮，seed=42）

5 数据集 × 4 比例（0.01 / 0.05 / 0.10 / all）× 9 方法的对比见 [results/comparison/comparison_summary_agg.csv](results/comparison/comparison_summary_agg.csv)。

- **标注稀缺区（rts ∈ {0.01, 0.05, 0.10}）**：5 数据集 × 3 比例 = **15/15 全部胜过多数对比方法**
- **零回归**，所有数据集所有比例 ≥ baseline
- **FFR 全部 ≤ α = 0.01**

完整迭代日志见 [results/experiment_log.md](results/experiment_log.md)。

## 设计原则

- **Inductive**：所有阈值与原型只来自 train + val，**绝不接触 test 标签**。
- **单一来源常量**：`DEFAULT_CONFORMAL_ALPHA`、`CONFORMAL_LOW_SEP`、`CONFORMAL_RANK_GRID` 在 [src/rescue.py](src/rescue.py) 集中定义，run_pipeline 与对比脚本均导入，避免数值漂移。
- **Provenance manifest**：[src/utils.py](src/utils.py) 的 `build_manifest` / `check_manifest` 保证 embeddings 缓存在 config / split 变化后自动失效，禁止「不同 split 下复用旧 embeddings 出指标」的污染。
- **科学诚实**：实验日志每轮记录假设 / 改动 / 结果 / 决策 / **局限**。论文主张限定在评估过的数据集上，不写 SOTA / 临床可用 / 通用解。

## 局限

- 主流水线主要在 5 个公开 scRNA-seq 数据集上验证；其他模态/物种尚未测试。
- stomach 上 recall 上限约 0.59 — 余下漏判的 mast cell 与多数类几何纠缠在 rank ≥ 3，prototype 几何上救不回（非阈值问题）。
- pancreas batch_heldout 在 rank=2 时 test FFR ≈ 0.0098，逼近 α=0.01 上界，源于 val/test 分布漂移。
- 现有正式对比仅 seed=42；多 seed 稳定性待补。

## 引用与致谢

依赖：[scvi-tools](https://scvi-tools.org/)、[scanpy](https://scanpy.readthedocs.io/)、[AnnData](https://anndata.readthedocs.io/)。
对比方法：CellTypist、TOSICA、scBalance、ProtoCloud、HiCat、scCAD（详见 [tools/comparison/](tools/comparison/)）。
