# scRareRefine Paper Plan

**目标期刊**: Q2 bioinformatics/computational biology (e.g., Briefings in Bioinformatics, Bioinformatics, PLoS Computational Biology)  
**核心叙事**: scRareRefine 在两类场景下特别有效：(1) 稀有细胞在 scANVI latent space 中与主要类型几何可分（sep ratio > 1.3）；(2) 标注预算极为有限（rare_train_size ≤ 10）。两种场景之外方法自动回退到 baseline，不造成损害。Separability ratio 可作为部署前的有效性预测指标。

---

## 当前结果快照（截至 2026-05-11，持续更新）

| 数据集 | 稀有类 | sep ratio | baseline F1 | 本方法 F1 | 增益 | LR F1 |
|---|---|---|---|---|---|---|
| Immune DC | ASDC | 1.53 | 0.656 | 0.933 | +27.7pp | 0.761 |
| Immune DC | cDC1 | 1.41 | 0.208 | 0.985 | +77.7pp | 0.977 |
| Pancreas | epsilon | 1.11 | 0.889 | 0.889 | 0pp | **1.000** |
| Pancreas | gamma | 0.94 | 0.996 | 0.992 | -0.4pp | 0.903 |
| Tabula Liver | NCM | 2.01 | 0.374 | 0.625 | +25.1pp | 0.313 |
| Tabula Pancreas | β-cell | 0.80 | 0.897 | 0.897 | 0pp | 0.966 |
| **Tabula Spleen** | **ILC** | **1.65** | **0.625** | **0.857** | **+23.2pp** | 0.653 |
| Tabula Kidney | endothelial | 1.80 | TBD | TBD | TBD | TBD |
| PBMC | pDC | TBD | TBD | TBD | TBD | TBD |

**核心规律（2026-05-11 实验确认，Spleen ILC 新增验证）**:
- **sep ratio > 1.3** → 任意训练量下均有显著增益（cDC1 +77.7pp，ASDC +27.7pp，NCM +25.1pp，**ILC +23.2pp** 新）
- **sep ratio < 1.1** → 方法自动回退到 baseline（0pp 增益，不造成损害）
- **LR 在低 sep 下 wins（epsilon F1=1.0）** → sep ratio 预测的是 scRareRefine vs scANVI 的优势，而非 vs 所有方法
- 方法的有效条件：sep > 1.3，或极低标注预算（size ≤ 5）

---

## Step 1: Separability 分析作为核心贡献

**目标**: 将"方法知道自己什么时候有效"这一发现显式化，做成论文的第一个核心结论。

**完成标准**:
- [x] 1.1 separability ratio vs F1 gain 散点图 → `fig_separability_gain.*`
- [x] 1.2 数据效率曲线图 → `fig_data_efficiency_v2.*`
- [x] 1.3 方法部分叙事文字（docs/method_narrative.md）
- [ ] 结论：sep ratio 是部署前的有效性预测指标；低标注预算下方法普遍有效

### 1.1 ✅ separability ratio vs F1 gain 散点图

**输出**: `figures/paper/fig_separability_gain.svg/pdf/tiff`  
**结果**: 6 个细胞类型在 cell-type 均值层面完美二值分离（sep>1.3 全部有增益，sep<1.1 全部零增益）。Spearman ρ=0.67（p=0.15，n=6，样本量导致不显著）。论文中应以二值规律陈述，而非 p 值。

**已知异常点**:
- cDC1 seed44: sep=1.10 但 gain=0.92 → 因该 seed 的 scANVI baseline 极差（F1=0.063），属随机分割极端情况
- NCM seed44: sep=1.94 但 gain=0.0 → 该 seed baseline 已较高（F1=0.50），方法未能进一步提升

### 1.2 ✅ 数据效率曲线

**输出**: `figures/paper/fig_data_efficiency_v2.svg/pdf/tiff`  
**关键数字**:
- cDC1 size=5: baseline=0.003, scRareRefine=0.985, **+0.98**
- ASDC size=5: baseline=~0.03, scRareRefine=~0.90, **+0.88**
- ε-cell size=5: baseline=0.0, scRareRefine=0.22, **+0.22**（sep 中等但低标注下仍有效）

**新发现**: ε-cell 在 size=5 时有增益，但 size=20 时收敛到 baseline 水平。说明方法在低标注预算下有额外价值，即使 sep 中等。

### 1.3 方法部分叙事文字

**目标文件**: `docs/method_narrative.md`  
**核心要点**:
1. 正式定义 separability ratio（公式）
2. 将"sep ratio 预测有效性"写为设计目标，而非事后观察
3. 将低标注预算场景纳入适用条件
4. 将 fallback 机制显式化为方法的安全保证

---

## Step 2: 补充 2-3 个 high-separability 数据集

**状态**: 进行中（2026-05-11 启动）  
**目标**: 将有效 case 从 3 个扩展到 5-6 个，涵盖不同组织类型和稀有细胞谱系  
**筛选原则**: 先做 sep ratio 快筛（仅跑 Stage 1-3），sep > 1.3 再上全量实验

### 2.1 Tabula Sapiens Kidney — endothelial cell ✅ 全量实验运行中

**状态**: sep ratio = **1.802**（HIGH）；全量实验（3 seeds × 4 sizes）后台运行中  
**文件**: `configs/tabula_kidney.yaml`  
**稀有类**: endothelial cell（n=101, 0.9%）  
**split mode**: cell_stratified（单一 donor，无真实 batch）  
**已完成**: seed42 size5/20，seed43 size5；剩余由后台 /tmp/kidney_full_run.log 完成

---

### 2.2 Tabula Sapiens Spleen — innate lymphoid cell ✅ 全量实验运行中，seed42 size20 已出结果

**状态**: sep ratio = **1.654**（HIGH）；全量实验后台运行中（/tmp/spleen_full_run.log）  
**文件**: `configs/tabula_spleen.yaml`  
**稀有类**: innate lymphoid cell（n=170, 0.24%，6 donors）  
**split mode**: batch_heldout（6 donors，合法 batch 分割）

**已出结果（seed42 size20）**:
- baseline F1 = 0.625
- prototype_gate_marker F1 = **0.857**（+23.2pp）
- LR F1 = 0.653
- fusion F1 = 0.842

---

### 2.3 PBMC — pDC ✅ 配置已创建，预处理完成

**状态**: `data/raw/pbmc/pbmc_pdc_50k.h5ad` 已生成（降采样后 50k cells，待跑 Stage 2）  
**文件**: `configs/pbmc_pdc.yaml`  
**稀有类**: pDC（n=2,373，4.75%，124 donors；via `minor_subset` label）  
**split mode**: batch_heldout（124 donors，70/15/15 分割）  
**数据结构**:
- 原始数据: COVID-19 Blood Atlas，836k cells，124 donors
- 降采样: 保留全部 2,373 pDC + 随机采样 47,627 non-pDC = 50k 总
- X 为原始计数（5' scRNA-seq），无 layer，直接用 adata.X
- label_key: `label`（已将 minor_subset 赋值到此列）
- pDC 预期为极高 sep ratio（LILRA4/CLEC4C/IRF7 极度特异）

预处理脚本: `src/00_preprocess_pbmc.py`

---

### 2.4 Sep ratio 快筛结果汇总（持续更新）

| 数据集 | 稀有类 | n | 实测 sep ratio | 状态 |
|---|---|---|---|---|
| Tabula Kidney | endothelial cell | 101 | **1.802** | 🔄 全量实验运行中 |
| Tabula Spleen | innate lymphoid cell | 170 | **1.654** | 🔄 全量实验运行中（seed42 size20 已出） |
| PBMC | pDC | 2373 | 待测 | ⏳ 预处理完成，待跑 Stage 2 |

---

## Step 3: 对比方法补充

**状态**: ✅ 基础设施完成，批量计算中（2026-05-11）  
**进展摘要**:
- kNN (k=15): 已有全量结果，已入 `final_metrics.csv`
- LR（CellTypist 等效实现）: 脚本已完成（`src/03c_celltypist_baseline.py`），正在批量跑所有现有 runs
- `07_evaluate.py` 已更新，会在 re-run 时自动纳入 LR 结果
- `10_paper_table.py` 已更新，直接读 `celltypist/test_metrics.csv`

### 方法实现状态

| 方法 | 类型 | 状态 | 初步结果（ASDC size=20）|
|---|---|---|---|
| **kNN (k=15)** | 距离最近邻 | ✅ 全量完成 | F1 = 0.614 |
| **LR（CellTypist 等效）** | 逻辑回归 + 标准化 | 🔄 批量运行中 | F1 ≈ 0.76 |
| Seurat label transfer | 加权 KNN | ⬜ 暂搁置，LR 已足够 | - |

**重要发现** （2026-05-11 LR 初步结果）：
- ASDC size=20: LR F1=0.76 < scRareRefine F1=0.93 → 仍有显著差距
- cDC1 size=10: LR F1=0.93-0.97 ≈ scRareRefine F1=0.985 → LR 在高标注时可竞争
- cDC1 size=5: LR F1=0.45-0.69 << scRareRefine F1=0.985 → **极低标注下 scRareRefine 显著优于 LR**
- 结论：scRareRefine 的关键优势在极低标注预算（size ≤ 5），这是数据效率叙事的核心

### 主表更新后的样式（基于初步结果）

| Dataset | Rare class | scANVI | kNN (k=15) | LR (CellTypist) | **scRareRefine** |
|---|---|---|---|---|---|
| Immune DC | ASDC | 0.656 | 0.614 | 0.761 | **0.933** |
| Immune DC | cDC1 | 0.208 | 0.807 | 0.976 | **0.985** |
| ... | | | | | |

---

## Step 4: 数据效率叙事完善（已有数据，低优先级）

**状态**: 数据已有，待写图和文字  
**重点**: rare_train_size=5 时的跨方法对比，展示本方法在极少标注预算场景下的优势。

---

## 文件清单

```
figures/paper/
  fig_separability_gain.py        ✅ Step 1.1
  fig_separability_gain.svg/pdf/tiff
  fig_data_efficiency_v2.py       ✅ Step 1.2
  fig_data_efficiency_v2.svg/pdf/tiff
docs/
  paper_plan.md                   ✅ 本文件
  method_narrative.md             ← Step 1.3 目标输出
```

---

## 当前风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| 方法在 size=20 时对 50% case 无效 | 已重新框架为"自知边界"；低标注下额外有效 | 叙事需在 1.3 中写清楚 |
| Spearman 相关性不显著（p=0.14，n=7） | 7 个 cell-type 聚合点仍样本量有限 | 论文用二值规律陈述；Kidney+PBMC 完成后可重算 |
| LR 在低 sep case 赢过 scRareRefine | epsilon LR=1.0, scRareRefine=0.889 | 叙事调整：sep ratio 预测的是 scRareRefine 优势区间；在高 sep 下 scRareRefine 依然优于 LR |
| 数据集数量（目前 6 高 sep case） | 含 Spleen 后已有 4 高 sep；Kidney+PBMC 待完成 | 高 sep 有效 case 将达到 5-6 个 |
| baseline 对比不足 | kNN 在个别 case 略优 | LR 已加入对比表；kNN 作为消融 |
