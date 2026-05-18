# 后续执行计划（2026-05-11 收尾整理）

本文件记录截至 2026-05-11 的进展状态与后续待执行步骤，便于下次会话接续工作。

---

## 当前已完成

### 数据集实验

| 数据集 | 稀有类 | sep ratio | 状态 | 增益 |
|---|---|---|---|---|
| Immune DC | ASDC, cDC1 | 1.53 / 1.41 | ✅ 全量完成（3 seeds × 4 sizes） | +27.7 / +77.7 pp |
| Pancreas | epsilon, gamma | 1.11 / 0.94 | ✅ 全量完成（3 seeds × 4 sizes） | 0 / -0.4 pp |
| Tabula Liver | NCM | 2.01 | ✅ 全量完成（3 seeds × 4 sizes） | +25.1 pp |
| Tabula Pancreas | β-cell | 0.80 | ✅ 全量完成（3 seeds × 4 sizes） | 0 pp |
| **Tabula Spleen** | **ILC** | **1.65** | 🔄 seed42 size20 完成；全量实验运行中 | **+23.2 pp** |
| **Tabula Kidney** | **EC** | **1.80** | 🔄 seed42+43 全量完成；seed44 运行中 | **≈-0.015 pp（高基线）** |
| **PBMC** | **pDC** | TBD | 🔄 seed42 size20 Stage 2 运行中 | TBD |

### 代码更新

- `src/03c_celltypist_baseline.py` ← 新增 LR（CellTypist 等效实现）
- `src/07_evaluate.py` ← 自动纳入 LR 结果
- `src/09_aggregate_plot.py` ← 新增 Spleen/Kidney/PBMC 数据集标签
- `src/10_paper_table.py` ← 新增新数据集支持；LaTeX 表格更新
- `figures/paper/fig_separability_gain.py` ← 新增 ILC/EC 散点；NaN 处理
- `src/00_preprocess_pbmc.py` ← **修复**：使用 raw.X（原始计数）而非 X（log-normalized）
- `run_celltypist_all.sh` ← 批量跑所有已有 runs 的 LR baseline

### 文档

- `docs/paper_plan.md` ← 更新结果快照、Step 2 状态、风险表
- `docs/method_narrative.md` ← 更新 Regime 1 定义（双条件框架：sep>1.3 + baseline<0.75）

---

## 后台进程状态（收尾时）

```bash
# 查看当前运行状态：
tail -5 /tmp/spleen_full_run.log   # Spleen 全量实验
tail -5 /tmp/kidney_full_run.log   # Kidney 全量实验
tail -5 /tmp/pbmc_sep_screening.log  # PBMC sep ratio 快筛
```

预计完成时间：
- Kidney seed44 全量：~2-3 小时
- Spleen 全量（11 runs）：~6-8 小时（70k cell 数据集，每 run ~35 min）
- PBMC sep ratio 快筛：~45-60 分钟

---

## 下次会话需执行的步骤

### Step A：检查后台任务结果（优先）

```bash
# 检查所有完成情况
ls outputs/tabula_spleen/
ls outputs/tabula_kidney/
ls outputs/pbmc_pdc/ 2>/dev/null || echo "PBMC not done"
tail -5 /tmp/spleen_full_run.log
tail -5 /tmp/kidney_full_run.log
tail -5 /tmp/pbmc_sep_screening.log
```

### Step B：PBMC sep ratio 结果检查

若 sep ratio 快筛完成，检查结果：
```bash
cat outputs/pbmc_pdc/batch_heldout_seed42_pdc_rare20/prototype/separability.csv
```

**判断**:
- sep > 1.3 → 立即启动全量 PBMC 实验：
  ```bash
  nohup bash -c '
  for seed in 42 43 44; do
    for size in 5 10 20 50; do
      [ "$seed" = "42" ] && [ "$size" = "20" ] && continue
      python3 run_pipeline.py --config configs/pbmc_pdc.yaml --seed $seed \
        --rare_class pDC --rare_train_size $size --split_mode batch_heldout --skip_visualize
      python3 src/03c_celltypist_baseline.py --config configs/pbmc_pdc.yaml \
        --seed $seed --rare_class pDC --rare_train_size $size --split_mode batch_heldout
    done
  done
  echo "ALL PBMC DONE"
  ' > /tmp/pbmc_full_run.log 2>&1 &
  ```
- sep < 1.3 → 不做全量实验；记录到 paper_plan.md

### Step C：Kidney seed44 CellTypist 补跑

若 Kidney seed44 全量完成，补跑 CellTypist：
```bash
for size in 5 10 20 50; do
  python3 src/03c_celltypist_baseline.py --config configs/tabula_kidney.yaml \
    --seed 44 --rare_class "endothelial cell" --rare_train_size $size --split_mode cell_stratified
done
```

### Step D：Spleen 全量完成后补跑 CellTypist

Spleen 的 `run_pipeline.py` 循环已包含 celltypist，无需额外操作。

### Step E：更新论文表格和图表

```bash
python3 src/10_paper_table.py          # 更新所有表格
python3 figures/paper/fig_separability_gain.py  # 更新散点图
python3 figures/paper/fig_data_efficiency_v2.py # 添加 ILC 和 PBMC 面板（待决定）
python3 src/09_aggregate_plot.py       # 更新汇总图
```

### Step F：图表决策

**fig_separability_gain.py**（当前 Spearman ρ=0.287，n=8）:
- 等 Kidney seed44 完成后，重新运行：若 Kidney 3-seed 平均接近零，ρ 应该改善
- PBMC 若 sep>1.3 且 gain 大，将显著提升 ρ
- 论文中重点陈述二值规律（sep<1.3→零增益），而非 Spearman p 值

**fig_data_efficiency_v2.py**（当前 4 面板）:
- 建议添加 ILC 面板（当 Spleen 全量完成后）
- 或扩展为 3×2 布局（6 panels）添加 PBMC pDC（若 sep 高且 gain 大）

### Step G：LaTeX 论文主表更新

当前 `table_main_results.tex` 只有 4 个数据集。完整版本需要等所有实验完成后重新生成：
```bash
python3 src/10_paper_table.py && cat figures/paper/table_main_results.tex
```

---

## 关键发现（需写入论文）

### 核心规律（双条件框架）

scRareRefine 显著改善的条件：
1. **sep ratio > 1.3**：稀有细胞在 scANVI latent space 中几何可分
2. **scANVI baseline F1 < 0.75**：scANVI 对该稀有细胞类型存在显著误分

两个条件缺一均不会有大增益：
- sep > 1.3 + baseline > 0.85（Kidney EC）→ scANVI 已做好，无需 rescue
- sep < 1.3 + any baseline（epsilon/gamma/β-cell）→ 几何结构不支持 rescue

| 条件 | 增益 | 案例 |
|---|---|---|
| sep>1.3 + baseline<0.75 | +23 to +78 pp | ASDC, cDC1, NCM, ILC |
| sep>1.3 + baseline>0.85 | ≈0 pp | Kidney EC |
| sep<1.3 | ≈0 pp | epsilon, gamma, β-cell |

### LR vs scRareRefine

- LR 在低 sep 数据集（epsilon：LR=1.0 > scRareRefine=0.889）胜出
- LR 在高 sep 数据集（ASDC：LR=0.761 < scRareRefine=0.933）落败
- sep ratio 预测的是 scRareRefine 相对 scANVI 的优势，而非相对所有方法

---

## 文件列表

```
docs/
  paper_plan.md              ✅ 更新
  method_narrative.md        ✅ 更新（双条件框架，LR 对比，摘要关键句）
  followup_plan_20260511.md  ✅ 本文件
figures/paper/
  fig_separability_gain.*    ✅ 更新（含 ILC；Kidney 1-2 seeds；Spearman ρ=0.287）
  fig_data_efficiency_v2.*   ✅ 保持不变（等 Spleen 全量后更新）
  table_main_results.csv     ✅ 更新（含 Spleen seed42 size20 + Kidney seed42-43）
  table_main_results.tex     ✅ 更新（新增数据集支持）
  table_separability.csv     ✅ 更新
outputs/
  tabula_spleen/             🔄 全量实验进行中
  tabula_kidney/             🔄 seed44 进行中（seed42+43 全量完成）
  pbmc_pdc/                  🔄 sep ratio 快筛运行中
```
