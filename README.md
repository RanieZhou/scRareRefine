# scRareRefine

> 基于 scANVI 的稀有细胞类型识别 post-hoc refinement 模块。
> 用 prototype 距离 + conformal 阈值，在 FFR 受控（≤ α = 1%）的前提下，于评估的数据集、标注稀缺区（rts ≤ 0.10）内提升 rare cell type 的 F1 / recall（配对检验显著，从不伤害 baseline）。

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

## 最新结果（第十四轮，3-seed）

6 数据集 × 4 比例（0.01 / 0.05 / 0.10 / all）× 9 方法 × 3 seed（42/43/44，共 **648/648 全部完成、0 失败**）的对比见 [results/comparison/comparison_summary_agg.csv](results/comparison/comparison_summary_agg.csv)（3-seed mean±std）。配对显著性检验（paired Wilcoxon + bootstrap 95% CI，配对单元 = dataset×rts×seed）见 [results/comparison/significance_test.csv](results/comparison/significance_test.csv)。

- **标注稀缺区（rts ∈ {0.01, 0.05, 0.10}）**：去重后 15 格，**win-most 15/15、best 14/15**（唯一非 best：small_intestine rts=0.10，baseline 已 saturated）
- **vs scANVI（稀缺区）**：29 胜 / 25 平 / **0 负**，ΔF1 = +0.160（bootstrap 95% CI [+0.085, +0.244]，Wilcoxon p = 1.3e-6）；平局对应 necessity 弃权（不伤害 baseline）
- **对全部 8 个 baseline 的 ΔF1 CI 均 > 0**（HiCat† 为 transductive 上界，单列）
- **FFR 全部 ≤ α = 0.01**（跨全部行最大 0.0098，pancreas_baron）

> p 值偏乐观：同 (dataset, rts) 的 3 seed 非完全独立，论文作「方向性证据」而非严格独立检验。完整迭代日志见 [results/experiment_log.md](results/experiment_log.md)。

## 设计原则

- **Inductive**：所有阈值与原型只来自 train + val，**绝不接触 test 标签**。
- **单一来源常量**：`DEFAULT_CONFORMAL_ALPHA`、`CONFORMAL_LOW_SEP`、`CONFORMAL_RANK_GRID` 在 [src/rescue.py](src/rescue.py) 集中定义，run_pipeline 与对比脚本均导入，避免数值漂移。
- **Provenance manifest**：[src/utils.py](src/utils.py) 的 `build_manifest` / `check_manifest` 保证 embeddings 缓存在 config / split 变化后自动失效，禁止「不同 split 下复用旧 embeddings 出指标」的污染。
- **科学诚实**：实验日志每轮记录假设 / 改动 / 结果 / 决策 / **局限**。论文主张限定在评估过的数据集上，不写 SOTA / 临床可用 / 通用解。

## 局限

- 主流水线主要在 6 个公开 scRNA-seq 数据集上验证；其他模态/物种尚未测试。
- stomach 上 recall 上限约 0.59 — 余下漏判的 mast cell 与多数类几何纠缠在 rank ≥ 3，prototype 几何上救不回（非阈值问题）。
- pancreas_baron batch_heldout 在 rank=2 时 test FFR ≈ 0.0098，逼近 α=0.01 上界，源于 val/test 分布漂移。
- pancreas_baron 在极端稀缺点（≤5 标注，rts=0.01/0.05）seed 敏感：gain 均值仍正（+0.18）但被 seed 方差淹没（±0.26），不宜宣称为「稳定提升」；rts≥0.10 即稳定。
- `CONFORMAL_LOW_SEP = 1.3` 是 pre-specified 保守弃权先验，非精确危险边界：合成 sep 扫描（单数据集/单方向）显示 sep 低至 ~1.15 仍可安全 rescue，而真实 pancreas_baron 在 sep≈1.22 就破 FFR — sep→风险是数据集相关的，1.3 是跨异质性的保守折中。

## 引用与致谢

依赖：[scvi-tools](https://scvi-tools.org/)、[scanpy](https://scanpy.readthedocs.io/)、[AnnData](https://anndata.readthedocs.io/)。
对比方法：CellTypist、TOSICA、scBalance、ProtoCloud、HiCat、scCAD（详见 [tools/comparison/](tools/comparison/)）。
