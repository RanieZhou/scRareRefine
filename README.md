# scRareRefine

> 基于 scANVI 的稀有细胞类型识别 post-hoc refinement 模块。
> 用 prototype 距离 + conformal 阈值，在 FFR 受控（≤ α = 1%）的前提下，于评估的数据集、标注稀缺区（rts ≤ 0.10）内提升 rare cell type 的 F1 / recall。当前主结果覆盖 6 个 human 数据集与 2 个 mouse TMS add-on 数据集。

## 方法概览

scANVI 在标注稀缺时常对稀有细胞「整类漏判」。scRareRefine 在 scANVI 之后接一个 **inductive 的 post-hoc 拯救** 流程：

1. **PrototypeRescuer**：从训练集 labeled 样本算各类均值原型；候选 = 「预测不是 rare」且「到 rare 原型距离 rank ≤ k」。
2. **Conformal τ 校准**：用 validation 集上**非稀有细胞**的 rare membership score，取有限样本 (1-α) 顺序统计量作为 τ；候选中 score ≥ τ 的细胞被 relabel 为稀有类。
3. **闸门（全 inductive，绝不碰 test 标签）**：两道弃权闸门 + 两道拯救机制
   - separability 安全网（`sep < CONFORMAL_LOW_SEP = 1.3` 弃权）
   - necessity + split-shift 守门（val baseline 漏判稀有数 < `MIN_VAL_MISSED = 3` 时弃权，覆盖「已全召回无需 rescue」与「val 漏少但 test 已 saturated」两种情形）
   - val-自适应候选 rank ∈ {1, 2, 3}（在 Wilson 95% 上界控 FFR ≤ α 约束下选 val rare F1 最高的 rank，平手取更小 rank；高可分自动 rank=1，边界纠缠自动 rank=2/3）
   - conformal τ 控 FFR（如上）

发表级 FFR 上界 `α = 0.01`，跨数据集固定**不调参**。

## 项目结构

```
run_pipeline.py                 # 端到端主入口
src/
├── preprocess.py               # 预处理 + 三路划分（batch_heldout / cell_stratified）
├── model.py                    # scVI + scANVI 半监督训练
├── rescue.py                   # 4 个 Rescuer + conformal_rescue() 单一入口
└── utils.py                    # config / metrics / 缓存 manifest / 可视化
configs/                        # 9 个数据集 YAML（8 个纳入主 comparison，1 个 lung stroma 备用）
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
| [pancreas_integrated](configs/pancreas_integrated.yaml) | 多源整合 pancreas | endothelial |
| [tabula_lung_endo](configs/tabula_lung_endo.yaml) | Tabula Sapiens lung | endothelial 子型 |
| [tabula_lung_stroma](configs/tabula_lung_stroma.yaml) | Tabula Sapiens lung | bronchial smooth muscle cell（备用，未纳入当前主 comparison） |
| [tabula_sapiens_stomach](configs/tabula_sapiens_stomach.yaml) | Tabula Sapiens stomach | mast cell |
| [tabula_small_intestine](configs/tabula_small_intestine.yaml) | Tabula Sapiens small intestine | intestinal tuft cell |
| [mouse_lung_tms_10x](configs/mouse_lung_tms_10x.yaml) | Tabula Muris Senis lung, 10x | vein endothelial cell |
| [mouse_pancreas_tms_10x](configs/mouse_pancreas_tms_10x.yaml) | Tabula Muris Senis pancreas, 10x | pancreatic D cell |

## 环境

两套 conda 虚拟环境分工：

- `scanvi311` — 主流水线与大部分 inductive 对比方法：scANVI / kNN / CellTypist / scBalance / scCAD / scRareRefine
- `sandbox310` — 依赖旧包栈的对比方法：ProtoCloud / HiCat / TOSICA

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

# 2 个 mouse TMS add-on 数据集：cache + 9-method comparison
python tools/comparison/run_mouse_tms_comparison.py --stage all

# 只补跑指定方法 / 数据集 / seed / rare_train_size，不覆盖其它组合
python tools/comparison/run_mouse_tms_comparison.py --stage methods --methods scBalance --configs configs/mouse_lung_tms_10x.yaml --seeds 42 43 44 --rts 0.01

# 汇总 + 绘图
python tools/comparison/plot_comparison.py
python tools/comparison/plot_comparison_grid.py
python tools/analysis/plot_sweep_rts_from_comparison.py
```

`run_mouse_tms_comparison.py` 会把结果日志写到 `results/mouse_tms_comparison/logs/`。方法脚本按 `(dataset, seed, rare_train_size, method)` 精确替换对应行，因此补跑单个方法不会覆盖其它方法结果。

scBalance 官方实现内部固定 `batch_size=128` 且使用 BatchNorm；当 labeled reference 数量 `mod 128 == 1` 时最后一个 batch 会只有 1 个样本并报错。本仓库的 [run_scbalance_comparison.py](tools/comparison/run_scbalance_comparison.py) 在这个边界条件下只把 weighted sampler 的 `num_samples` 减 1，避免 BatchNorm 单样本 batch，同时保留 weighted sampling 设定。

汇总产物：
- [results/comparison/comparison_summary.csv](results/comparison/comparison_summary.csv) / `_agg.csv`
- [results/comparison/comparison_bars.png](results/comparison/comparison_bars.png) / `comparison_bars_grid.png`
- [results/sweep_rts/sweep_rts_curves.png](results/sweep_rts/sweep_rts_curves.png)

## 最新结果（8 数据集，3-seed）

8 数据集（6 human + 2 mouse TMS）× 4 比例（0.01 / 0.05 / 0.10 / all）× 9 方法 × 3 seed（42/43/44，共 **864/864 全部完成、0 失败**）的对比见 [results/comparison/comparison_summary.csv](results/comparison/comparison_summary.csv)。按 dataset × rare_train_size × method 聚合后为 **288/288** 个组合，见 [results/comparison/comparison_summary_agg.csv](results/comparison/comparison_summary_agg.csv)（3-seed mean±std）。

- **标注稀缺区（rts ∈ {0.01, 0.05, 0.10}）**：24 个 dataset×rts 聚合格中，scRareRefine 为 best/tied-best **23/24**；唯一非 best 为 `tabula_small_intestine, rts=0.10`，ProtoCloud F1=0.9848，scRareRefine F1=0.9823，差距 0.0025。
- **vs scANVI（稀缺区，72 个 dataset×rts×seed 配对单元）**：41 胜 / 30 平 / **1 负**，平均 ΔF1 = +0.1545；大量平局来自 necessity 弃权或 baseline 已 saturated。
- **mouse add-on 覆盖跨物种 / 跨组织验证**：`mouse_lung_tms_10x` 与 `mouse_pancreas_tms_10x` 使用 raw-count CELLxGENE Census 导出的 Tabula Muris Senis 10x 子集；与原 6 个 human 数据集共同用于 8 数据集 comparison 图。
- **scRareRefine FFR 仍受控**：聚合表中 scRareRefine 的 `fp_rate_max` 最大为 0.009878（`mouse_lung_tms_10x, rts=all`），低于 α=0.01。

[results/comparison/significance_test.csv](results/comparison/significance_test.csv) 已基于当前 864 行 8-dataset 结果重算：ALL rts 为 96 个配对单元，稀缺区为 72 个配对单元；vs scANVI 稀缺区 41 胜 / 30 平 / 1 负，平均 ΔF1=+0.1545，bootstrap 95% CI [+0.0956,+0.2199]，one-sided Wilcoxon p=8.82e-09。旧 6-human 统计已备份为 [results/comparison/significance_test_6human_backup_20260708.csv](results/comparison/significance_test_6human_backup_20260708.csv)。完整迭代日志见 [results/experiment_log.md](results/experiment_log.md)。

## 设计原则

- **Inductive**：所有阈值与原型只来自 train + val，**绝不接触 test 标签**。
- **单一来源常量**：`DEFAULT_CONFORMAL_ALPHA`、`CONFORMAL_LOW_SEP`、`CONFORMAL_RANK_GRID` 在 [src/rescue.py](src/rescue.py) 集中定义，run_pipeline 与对比脚本均导入，避免数值漂移。
- **Provenance manifest**：[src/utils.py](src/utils.py) 的 `build_manifest` / `check_manifest` 保证 embeddings 缓存在 config / split 变化后自动失效，禁止「不同 split 下复用旧 embeddings 出指标」的污染。
- **科学诚实**：实验日志每轮记录假设 / 改动 / 结果 / 决策 / **局限**。论文主张限定在评估过的数据集上，不写 SOTA / 临床可用 / 通用解。

## 局限

- 主流水线目前在 6 个 human scRNA-seq 数据集和 2 个 mouse TMS add-on 数据集上验证；其他模态、疾病状态和更远物种尚未测试。
- stomach 上 recall 上限约 0.59 — 余下漏判的 mast cell 与多数类几何纠缠在 rank ≥ 3，prototype 几何上救不回（非阈值问题）。
- pancreas_baron batch_heldout 在 rank=2 时 test FFR ≈ 0.0098，逼近 α=0.01 上界，源于 val/test 分布漂移。
- pancreas_baron 在极端稀缺点（≤5 标注，rts=0.01/0.05）seed 敏感：gain 均值仍正（+0.18）但被 seed 方差淹没（±0.26），不宜宣称为「稳定提升」；rts≥0.10 即稳定。
- `CONFORMAL_LOW_SEP = 1.3` 是 pre-specified 保守弃权先验，非精确危险边界：合成 sep 扫描（单数据集/单方向）显示 sep 低至 ~1.15 仍可安全 rescue，而真实 pancreas_baron 在 sep≈1.22 就破 FFR — sep→风险是数据集相关的，1.3 是跨异质性的保守折中。

## 引用与致谢

依赖：[scvi-tools](https://scvi-tools.org/)、[scanpy](https://scanpy.readthedocs.io/)、[AnnData](https://anndata.readthedocs.io/)。
对比方法：CellTypist、TOSICA、scBalance、ProtoCloud、HiCat、scCAD（详见 [tools/comparison/](tools/comparison/)）。
