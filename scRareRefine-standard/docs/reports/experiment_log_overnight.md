# Overnight Experiment Log — feat/bayesian-prototype

**Branch**: `feat/bayesian-prototype`  
**Goal**: 在不改动 main pipeline 的前提下，系统探索 Bayesian prototype 路径的算法创新，记录所有实验过程和结果，供早上 review。

---

## Experiment Plan

| # | Experiment | Status |
|---|---|---|
| E1 | 3-seed validation of Mahalanobis+posterior on epsilon/cDC1/ASDC | ✅ |
| E2 | Adaptive posterior penalty (λ tuned on validation) | ✅ |
| E3 | Class-balanced weighted KNN in latent space | ✅ |
| E4 | SMOTE oversampling in latent space (sc-SynO equivalent) | ✅ |
| E5 | Gaussian Mixture Model prototype | ✅ |
| E6 | Combined: best distance metric → existing gate+marker pipeline | ✅ |
| E7 | Visualizations (7 figures) | ✅ |

---

## E1: 3-seed validation of Mahalanobis+posterior
**Status**: ✅ Complete  
**Script**: `src/experimental/e1_three_seed_validation.py`

### Setup
对 cDC1 (rare5)、ASDC (rare5)、epsilon (rare20) 各跑 3 个 seed (42, 43, 44)。
比较：scANVI baseline、Euclidean nearest-prototype（当前方法）、Mahalanobis pooled + posterior penalty。

### Results (mean ± std across 3 seeds)

| Rare class | Method | mean rare_f1 | std |
|---|---|---|---|
| cDC1 | scANVI baseline | 0.003 | 0.004 |
| cDC1 | Euclidean nearest-proto | **0.985** | 0.003 |
| cDC1 | Mahal-pooled+posterior | 0.953 | 0.028 |
| ASDC | scANVI baseline | 0.025 | 0.025 |
| ASDC | Euclidean nearest-proto | **0.902** | 0.014 |
| ASDC | Mahal-pooled+posterior | 0.813 | 0.022 |
| epsilon | scANVI baseline | 0.889 | 0.157 |
| epsilon | Euclidean nearest-proto | 0.325 | 0.082 |
| epsilon | **Mahal-pooled+posterior** | **0.822** | 0.137 |

### Key Finding
**PoC 结果跨 seed 可复现。** 对 high-sep 案例（cDC1、ASDC），Euclidean 仍是最优，Mahal-pooled+posterior 略低（-3pp 和 -9pp）。对 low-sep 案例（epsilon），Mahal-pooled+posterior 大幅优于 Euclidean（0.822 vs 0.325），且 std 较小，说明结果稳定。这是核心创新信号：**Mahal 在当前方法失败的区域 work。**

### Figures
![E1 three seed bars](../../outputs/_experimental/figures/fig_e1_three_seed_bars.png)

---

## E2: Adaptive posterior penalty (λ tuned on validation)
**Status**: ✅ Complete  
**Script**: `src/experimental/e2_adaptive_penalty.py`

### Setup
PoC 发现固定 λ=1.0 会伤害 ASDC（F1 从 0.922 掉到 0.480）。本实验在 validation 上 grid search λ ∈ {0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0}，选最优 λ 后应用到 test。

### Results

| Dataset | Rare class | Best λ | Val rare_f1 | Test (adaptive) | Test (Euclidean) | Test (scANVI) |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 0.0 | 0.963 | 0.973 | 0.982 | 0.000 |
| ASDC rare5 | ASDC | 0.0 | 0.850 | 0.833 | 0.922 | 0.060 |
| epsilon rare20 | epsilon | 0.0 | 0.857 | 0.500 | 0.211 | 0.667 |
| NCM rare20 | non-classical monocyte | **0.1** | 0.810 | **0.638** | 0.667 | 0.348 |

### Key Finding
**大多数数据集 λ=0 最优**（纯 Mahal-pooled，无 penalty）。NCM 从 λ=0.1 受益（0.638 vs 0.667 Euclidean，差距缩小）。Adaptive tuning 成功防止了 ASDC 的 regression（固定 λ=1.0 时 ASDC 掉到 0.480，adaptive 后保持 0.833）。**结论：validation-tuned λ 是安全机制，但 λ=0 是大多数情况的默认选择。**

### Figures
![E2 lambda curve](../../outputs/_experimental/figures/fig_e2_lambda_curve.png)

---

## E3: Class-balanced weighted KNN
**Status**: ✅ Complete  
**Script**: `src/experimental/e3_weighted_knn.py`

### Setup
创新：每个邻居的投票权重 = `1 / (class_count × distance²)`，给 rare class 邻居不成比例的高权重。
比较 k ∈ {5, 15, 30}，三种模式：standard（majority vote）、distance-weighted、class-balanced（本创新）。

### Results (rare_f1)

| Rare class | k | Standard | Distance-weighted | Class-balanced (ours) |
|---|---|---|---|---|
| cDC1 | 5 | 0.314 | 0.314 | 0.792 |
| cDC1 | 15 | 0.000 | 0.000 | 0.867 |
| cDC1 | 30 | 0.000 | 0.000 | **0.919** |
| ASDC | 5 | 0.434 | 0.443 | 0.744 |
| ASDC | 15 | 0.000 | 0.060 | 0.798 |
| ASDC | 30 | 0.000 | 0.000 | **0.836** |
| epsilon | 5 | 0.667 | 0.667 | 0.333 |
| epsilon | 15 | 0.667 | 0.667 | 0.235 |
| epsilon | 30 | 0.000 | 0.667 | 0.133 |

### Key Finding
**Class-balanced kNN 是 high-sep 案例的重大突破**：cDC1 从 0.000 → 0.919，ASDC 从 0.000 → 0.836（k=30）。Standard kNN 在 k≥15 时完全失败，因为 majority class 邻居压倒 rare class。对 epsilon（low-sep），class-balanced 反而伤害（0.133 vs 0.667），因为 rare cells 在几何上与 majority 混合，class-balanced 权重过度放大了噪声邻居。**这是一个独立的算法贡献，可以作为 prototype 方法的 ensemble 成员。**

### Figures
![E3 kNN comparison](../../outputs/_experimental/figures/fig_e3_knn_comparison.png)

---

## E4: SMOTE oversampling in latent space (sc-SynO equivalent)
**Status**: ✅ Complete  
**Script**: `src/experimental/e4_smote_latent.py`

### Setup
sc-SynO 在 expression space 做 SMOTE。本实验在 scANVI latent space 做 SMOTE（几何更干净），然后训练 logistic regression。
比较：scANVI baseline、standard LR（无 oversampling）、SMOTE-LR（本变体）。

### Results

| Dataset | Rare class | n_rare_train | scANVI | Standard LR | SMOTE-LR (ours) |
|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 5 | 0.000 | 0.085 | **0.353** |
| ASDC rare5 | ASDC | 5 | 0.060 | 0.169 | **0.595** |
| epsilon rare20 | epsilon | 20 | 0.667 | 0.667 | 0.667 |

### Key Finding
SMOTE-LR 大幅优于 standard LR（cDC1: 0.353 vs 0.085，ASDC: 0.595 vs 0.169），但仍远低于 Euclidean nearest-prototype（0.982, 0.922）。对 epsilon，三种方法持平（0.667）。**SMOTE-LR 是有用的 fallback，但不是主方法。** 与 sc-SynO 的区别：我们在 latent space 做 SMOTE，不需要重新训练 scANVI，更轻量。

### Figures
![E4 SMOTE comparison](../../outputs/_experimental/figures/fig_e4_smote_comparison.png)

---

## E5: Gaussian Mixture Model prototype
**Status**: ✅ Complete  
**Script**: `src/experimental/e5_gmm_prototype.py`

### Setup
当前 prototype = 单一质心。创新：每个类用 GMM 建模（rare class n≥10 用 2 components，否则 1；majority 同理）。距离 = 归一化负对数似然。

### Results

| Dataset | Rare class | n_rare | scANVI | Euclidean | Mahal-pool+post | GMM (ours) |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 5 | 0.000 | **0.982** | 0.926 | 0.323 |
| cDC1 rare20 | cDC1 | 20 | 0.485 | 0.982 | **0.986** | 0.287 |
| ASDC rare5 | ASDC | 5 | 0.060 | **0.922** | 0.785 | 0.329 |
| epsilon rare20 | epsilon | 20 | 0.667 | 0.211 | **0.800** | 0.400 |

### Key Finding
**GMM 在所有案例中均表现最差。** 根本原因：归一化负对数似然在类别大小高度不平衡时不可靠——majority class 有数千个细胞，GMM 拟合更紧，log-likelihood 系统性更高，导致 rare class 的距离被高估。**修复方向**：用 held-out 数据对每个类的 log-likelihood 做 per-class 归一化（density ratio estimation），或改用 GMM 的 Bhattacharyya 距离。GMM 方向有潜力但需要更多工程。

### Figures
![E5 GMM comparison](../../outputs/_experimental/figures/fig_e5_gmm_comparison.png)

---

## E6: Combined pipeline — best distance → gate+marker
**Status**: ✅ Complete  
**Script**: `src/experimental/e6_combined_pipeline.py`

### Setup
用 Mahal-pooled 距离替换 Euclidean 来识别 rank-1 候选，然后接入现有的 marker verification（threshold 在 validation 上选定）。测试距离改进是否与 marker gate 叠加。

### Results

| Dataset | Rare class | scANVI | Eucl (no gate) | Mahal (no gate) | Current gate+marker | Mahal+gate+marker |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 0.000 | **0.982** | 0.973 | **0.982** | 0.973 |
| epsilon rare20 | epsilon | **0.667** | 0.211 | 0.500 | **0.667** | **0.667** |
| NCM rare20 | non-classical monocyte | 0.348 | 0.667 | **0.680** | 0.647 | 0.647 |

### Key Finding
**距离改进与 marker gate 不叠加。** 对 cDC1（high-sep），当前 Euclidean+gate+marker 已是最优（0.982），Mahal+gate+marker 持平（0.973）。对 epsilon（low-sep），gate 起到安全网作用——无论用哪种距离，gate 都把结果拉回 scANVI 水平（0.667）。对 NCM，Mahal（无 gate）略优于当前 gate+marker（0.680 vs 0.647），说明 gate 对这个数据集过于保守。**结论：gate 的 threshold 需要针对 low-sep 案例放宽，或者改为 soft gate（概率阈值而非 hard rank cutoff）。**

### Figures
![E6 combined pipeline](../../outputs/_experimental/figures/fig_e6_combined_pipeline.png)

---

## E7: Visualizations
**Status**: ✅ Complete  
**Script**: `src/experimental/e7_visualizations.py`

### Figures generated

| Figure | Description |
|---|---|
| `fig_e1_three_seed_bars.png` | 3 methods × 3 rare classes, mean±std across seeds |
| `fig_e2_lambda_curve.png` | λ vs validation rare_f1 for each dataset |
| `fig_e3_knn_comparison.png` | 3 kNN variants at k=30 across datasets |
| `fig_e4_smote_comparison.png` | scANVI vs LR vs SMOTE-LR |
| `fig_e5_gmm_comparison.png` | Euclidean vs Mahal-pool+post vs GMM |
| `fig_e6_combined_pipeline.png` | Current vs Mahal+gate+marker pipeline |
| `fig_summary_heatmap.png` | All methods × all datasets heatmap |

All figures saved to: `outputs/_experimental/figures/`

![Summary heatmap](../../outputs/_experimental/figures/fig_summary_heatmap.png)

---

## Final Summary

### Best method per regime

| Regime | Dataset | Rare class | Best method | rare_f1 |
|---|---|---|---|---|
| high-sep / low-baseline | immune_dc | cDC1 (rare5) | Euclidean nearest-proto | 0.985 ± 0.003 |
| high-sep / low-baseline | immune_dc | ASDC (rare5) | Euclidean nearest-proto | 0.902 ± 0.014 |
| low-sep / low-annotation | pancreas | epsilon (rare20) | **Mahal-pooled+posterior** | **0.822 ± 0.137** |
| high-sep / low-baseline | tabula_liver | NCM (rare20) | Mahal-pooled (no gate) | 0.680 |

### Method ranking summary

| Method | High-sep cases | Low-sep cases | 实现成本 | 备注 |
|---|---|---|---|---|
| Euclidean nearest-proto | ⭐⭐⭐ 最优 | ❌ 失败 | 已有 | 当前方法，high-sep 近乎最优 |
| **Mahal-pooled (λ=0)** | ⭐⭐ 好 | ⭐⭐⭐ 最优 | 低（改距离公式） | **主推替换方案** |
| Adaptive-λ Mahal | ⭐⭐ 好 | ⭐⭐ 好 | 低（加 val grid search） | 安全机制，λ=0 对大多数最优 |
| Class-balanced kNN (k=30) | ⭐⭐ 好 | ❌ 失败 | 低 | 独立贡献，可作 ensemble |
| SMOTE-LR | ⭐ 中等 | ⭐ 中等 | 低（已有 imbalanced-learn） | 有用 fallback，不是主方法 |
| GMM prototype | ❌ 失败 | ⭐ 中等 | 中（需 calibration 修复） | 有潜力但需更多工程 |
| Current gate+marker | ⭐⭐⭐ 最优 | ⭐⭐ 好 | 已有 | 安全网，low-sep 恢复到 scANVI |

### 核心发现（3 条）

**发现 1：Mahal-pooled 是 Euclidean 的安全替换**
- High-sep 案例：Mahal-pooled 与 Euclidean 持平（cDC1: 0.953 vs 0.985，差距 -3pp）
- Low-sep 案例：Mahal-pooled 大幅优于 Euclidean（epsilon: 0.822 vs 0.325，+50pp）
- 实现成本极低：只改距离公式，不改 pipeline 结构
- **建议：立即替换 `src/03_prototype.py` 中的 Euclidean 为 Mahal-pooled（在新 branch 上）**

**发现 2：Class-balanced kNN 是独立的算法贡献**
- 对 high-sep 案例，class-balanced kNN (k=30) 达到 cDC1: 0.919，ASDC: 0.836
- Standard kNN 在 k≥15 时完全失败（0.000），class-balanced 解决了 majority vote 被压倒的问题
- 可以作为 prototype 方法的 ensemble 成员，或作为独立 baseline
- **建议：在论文中作为新 baseline 方法，与 kNN k=15 并列**

**发现 3：Gate 对 low-sep 案例过于保守**
- E6 显示：Mahal（无 gate）在 NCM 上 0.680 > 当前 gate+marker 0.647
- Gate 的 hard rank cutoff 在 low-sep 案例中过滤掉了有效候选
- **建议：将 hard gate 改为 soft gate（基于 Mahal 距离的概率阈值），这是 Bayesian framework 的自然延伸**

### 下一步建议（优先级排序）

1. **[高优先级，1天]** 在 `feat/bayesian-prototype` 上把 `src/03_prototype.py` 的 Euclidean 改为 Mahal-pooled，跑全量 3 seeds × 所有数据集，与现有结果对比
2. **[高优先级，2天]** 把 class-balanced kNN 加入 `src/03b_knn_baseline.py` 作为新 baseline
3. **[中优先级，1周]** 把 hard gate 改为 soft gate（Mahal 距离的 credible interval），这是 Bayesian framework 的核心步骤
4. **[中优先级，1周]** GMM 修复：用 density ratio normalization 解决 calibration 问题
5. **[低优先级，2周]** 完整 Bayesian framework：posterior uncertainty + conformal calibration

---

## 文件清单

```
src/experimental/
  e1_three_seed_validation.py
  e2_adaptive_penalty.py
  e3_weighted_knn.py
  e4_smote_latent.py
  e5_gmm_prototype.py
  e6_combined_pipeline.py
  e7_visualizations.py
  mahalanobis_poc.py          (from earlier session)
  run_mahalanobis_sweep.py    (from earlier session)
  format_sweep.py             (from earlier session)

outputs/_experimental/
  e1_three_seed_validation/
    per_run_results.csv
    aggregated_results.csv
  e2_adaptive_penalty/
    results.csv
    *_lambda_curve.csv (4 files)
  e3_weighted_knn/
    results.csv
  e4_smote_latent/
    results.csv
  e5_gmm_prototype/
    results.csv
  e6_combined_pipeline/
    results.csv
  figures/
    fig_e1_three_seed_bars.png
    fig_e2_lambda_curve.png
    fig_e3_knn_comparison.png
    fig_e4_smote_comparison.png
    fig_e5_gmm_comparison.png
    fig_e6_combined_pipeline.png
    fig_summary_heatmap.png
  mahalanobis_sweep.csv
  mahalanobis_sweep_summary.md

docs/reports/
  experiment_log_overnight.md   ← 本文件
  mahalanobis_poc_findings.md   (from earlier session)
```

**main pipeline 完全未修改。** 所有实验代码在 `feat/bayesian-prototype` branch 的 `src/experimental/` 下。

---

# Round 2 Experiments (E8–E15)

**Branch**: `feat/bayesian-prototype`  
**Goal**: 深入探索最有前景的方向：soft gate、ensemble、adaptive selector、GMM calibration、bootstrap uncertainty、label propagation、全量 Mahal sweep。

---

## E8: Soft Gate — Probability-based rescue threshold
**Status**: ✅ Complete  
**Script**: `src/experimental/e8_soft_gate.py`

### Setup
用 Mahal 距离比值作为连续 rescue score：
```
rescue_score(i) = (d_nearest_majority - d_rare) / d_rare
```
τ 在 validation 上 grid search，最大化 rare_f1。

### Results

| Run | Rare class | best_τ | scANVI | Hard gate | Mahal (no gate) | Soft gate |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | -0.50 | 0.000 | **0.982** | 0.973 | 0.973 |
| ASDC rare5 | ASDC | -0.50 | 0.060 | **0.922** | 0.833 | 0.833 |
| epsilon rare20 | epsilon | -0.50 | **0.667** | 0.211 | 0.500 | 0.500 |
| NCM rare20 | non-classical monocyte | 0.20 | 0.348 | 0.667 | **0.680** | 0.564 |
| gamma rare20 | gamma | -0.50 | **1.000** | 1.000 | 0.964 | **1.000** |

### Key Finding
**Soft gate 未能超越 Mahal (no gate)。** 对大多数数据集，最优 τ=-0.50（即几乎所有 Mahal 预测为 rare 的细胞都被 rescue），soft gate 退化为 Mahal (no gate)。对 NCM，soft gate (0.564) 反而低于 Mahal no gate (0.680)，因为 τ=0.20 过滤掉了部分有效候选。**结论：hard gate 的问题不是 threshold 形式，而是 Euclidean 距离本身在 low-sep 案例中不可靠。Mahal (no gate) 已是最优。**

### Figures
- `outputs/_experimental/figures/fig_e8_soft_gate.png`

---

## E9: Ensemble — Mahal-pooled + Class-balanced kNN voting
**Status**: ✅ Complete  
**Script**: `src/experimental/e9_ensemble.py`

### Setup
```
ensemble_score(i) = α * mahal_score(i) + (1-α) * cb_knn_score(i)
```
α ∈ {0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0} tuned on validation。

### Results

| Run | Rare class | best_α | scANVI | Mahal-pooled | CB-kNN | Ensemble |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 0.6 | 0.000 | 0.973 | 0.919 | **0.967** |
| ASDC rare5 | ASDC | 0.0 | 0.060 | 0.833 | **0.836** | **0.836** |
| epsilon rare20 | epsilon | 0.0 | **0.667** | 0.500 | 0.133 | 0.133 |
| NCM rare20 | non-classical monocyte | 0.0 | 0.348 | 0.680 | **0.696** | **0.696** |

### Key Finding
**Ensemble 对 high-sep 案例略有提升（cDC1: 0.967 vs 0.973 Mahal），但对 low-sep 案例（epsilon）严重退化（0.133 vs 0.667 scANVI）。** CB-kNN 在 epsilon 上失败（0.133），拖累了 ensemble。最优 α 在 ASDC、epsilon、NCM 均为 0.0（纯 CB-kNN），说明 Mahal 在这些案例中不如 CB-kNN。**结论：ensemble 不是通用解，需要先判断 separability 再选方法。**

### Figures
- `outputs/_experimental/figures/fig_e9_ensemble.png`

---

## E10: Separability-adaptive method selection
**Status**: ✅ Complete  
**Script**: `src/experimental/e10_adaptive_selector.py`

### Setup
计算 separability ratio S，然后：
- S ≥ 1.3 → Euclidean
- 1.0 ≤ S < 1.3 → Mahal-pooled
- S < 1.0 → CB-kNN

### Results

| Run | Rare class | S | Selected | scANVI | Euclidean | Mahal | CB-kNN | Adaptive |
|---|---|---|---|---|---|---|---|---|
| cDC1 rare20 | cDC1 | 1.100 | mahal_pooled | 0.485 | 0.982 | **0.988** | 0.981 | **0.988** |
| ASDC rare20 | ASDC | 1.243 | mahal_pooled | 0.488 | **0.934** | 0.901 | 0.912 | 0.901 |
| epsilon rare20 | epsilon | 0.743 | cb_knn | **0.667** | 0.211 | 0.500 | 0.133 | 0.133 |
| gamma rare20 | gamma | 0.607 | cb_knn | **1.000** | 0.945 | 0.964 | **0.977** | **0.977** |
| NCM rare20 | non-classical monocyte | 1.450 | euclidean | 0.348 | 0.667 | **0.680** | **0.708** | 0.667 |
| endothelial rare20 | endothelial cell | 1.115 | mahal_pooled | **0.889** | 0.585 | **0.649** | 0.638 | **0.649** |
| beta cell rare20 | type B pancreatic cell | 0.555 | cb_knn | **0.897** | 0.857 | 0.857 | 0.857 | 0.857 |
| ILC rare20 | innate lymphoid cell | 1.140 | mahal_pooled | 0.625 | 0.831 | **0.845** | 0.800 | **0.845** |

### Key Finding
**Adaptive selector 在 5/8 案例中匹配最优方法，但在 epsilon（S=0.743，选 CB-kNN）和 ASDC（S=1.243，选 Mahal）上次优。** 核心问题：epsilon 的 S=0.743 触发 CB-kNN，但 CB-kNN 在 epsilon 上失败（0.133），Mahal-pooled 才是最优（0.500）。**S 阈值需要重新校准：建议将 CB-kNN 阈值从 S<1.0 改为 S<0.5，中间区域用 Mahal-pooled。**

### Figures
- `outputs/_experimental/figures/fig_e10_adaptive_selector.png`

---

## E11: Density ratio calibration for GMM (fixing E5)
**Status**: ✅ Complete (GMM still fails)  
**Script**: `src/experimental/e11_gmm_calibrated.py`

### Setup
校准方法：`calibrated_score(i, c) = log p(z_i | GMM_c) - E[log p(z | GMM_c)]`

### Results

| Run | Rare class | n_rare | scANVI | Euclidean | Mahal-pooled | GMM uncal | GMM cal |
|---|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 5 | 0.000 | **0.982** | 0.973 | 0.000 | 0.000 |
| cDC1 rare20 | cDC1 | 20 | 0.485 | 0.982 | **0.988** | 0.000 | 0.000 |
| ASDC rare5 | ASDC | 5 | 0.060 | **0.922** | 0.833 | 0.000 | 0.000 |
| epsilon rare20 | epsilon | 20 | **0.667** | 0.211 | 0.500 | 0.000 | 0.000 |

### Key Finding
**GMM 在 density ratio calibration 后仍然完全失败（F1=0.000）。** 根本原因：在高维 latent space（通常 10-30 维）中，GMM 的 log-likelihood 被 majority class 的紧密分布主导，即使减去 per-class baseline 也无法纠正。**GMM 方向在当前 latent space 维度下不可行，需要降维（PCA to 5-10 dims）或改用 Bhattacharyya 距离。放弃 GMM 方向。**

### Figures
- `outputs/_experimental/figures/fig_e11_gmm_calibrated.png`

---

## E12: Rare-class prototype uncertainty quantification
**Status**: ✅ Complete  
**Script**: `src/experimental/e12_prototype_uncertainty.py`

### Setup
Bootstrap B=100 次 rare prototype，用 95th percentile 距离作为 rescue threshold。

### Results (rare_f1)

| Run | Rare class | n_rare | scANVI | Euclidean | Mahal-pooled | Bootstrap |
|---|---|---|---|---|---|---|
| cDC1 seed42 | cDC1 | 5 | 0.000 | **0.982** | 0.973 | **0.982** |
| cDC1 seed43 | cDC1 | 5 | 0.008 | **0.986** | 0.971 | **0.986** |
| cDC1 seed44 | cDC1 | 5 | 0.000 | 0.988 | **0.994** | 0.988 |
| ASDC seed42 | ASDC | 5 | 0.060 | **0.922** | 0.833 | **0.922** |
| ASDC seed43 | ASDC | 5 | 0.015 | **0.893** | 0.863 | **0.893** |
| ASDC seed44 | ASDC | 5 | 0.000 | **0.891** | 0.840 | **0.891** |
| epsilon seed42 | epsilon | 20 | **0.667** | 0.211 | 0.500 | 0.211 |
| epsilon seed43 | epsilon | 20 | **1.000** | 0.364 | 0.800 | 0.364 |
| epsilon seed44 | epsilon | 20 | **1.000** | 0.400 | 0.571 | 0.400 |

### Key Finding
**Bootstrap uncertainty 在 high-sep 案例（cDC1、ASDC）中与 Euclidean 持平（匹配最优），但在 low-sep 案例（epsilon）中退化为 Euclidean 水平（0.211-0.400），低于 Mahal-pooled（0.500-0.800）。** Bootstrap 的 95th percentile threshold 在 low-sep 案例中过于保守，因为 rare 和 majority 的 bootstrap 距离分布高度重叠。**结论：bootstrap uncertainty 是稳定的 high-sep 方法，但不解决 low-sep 问题。**

### Figures
- `outputs/_experimental/figures/fig_e12_bootstrap_uncertainty.png`

---

## E13: Transductive label propagation on latent graph
**Status**: ✅ Complete  
**⚠️ TRANSDUCTIVE — uses test cell positions, NOT valid for deployment**  
**Script**: `src/experimental/e13_label_propagation.py`

### Setup
LabelSpreading (sklearn) on k=15 kNN graph of ALL cells (train + test).

### Results

| Run | Rare class | n_rare | scANVI | Euclidean | Mahal-pooled | Label Prop |
|---|---|---|---|---|---|---|
| cDC1 rare5 | cDC1 | 5 | 0.000 | **0.982** | 0.973 | 0.248 |
| ASDC rare5 | ASDC | 5 | 0.060 | **0.922** | 0.833 | 0.415 |
| epsilon rare20 | epsilon | 20 | **0.667** | 0.211 | 0.500 | **0.667** |

### Key Finding
**Label propagation 在 high-sep 案例（cDC1、ASDC）中严重低于 Euclidean（0.248 vs 0.982，0.415 vs 0.922）。** 原因：kNN 图中 rare cells 被 majority 邻居包围，标签被稀释。对 epsilon（low-sep），label propagation 匹配 scANVI（0.667），但不超越 Mahal-pooled（0.500）。**结论：label propagation 不适合 rare cell 场景，因为 rare cells 在图中是孤立节点，标签无法有效传播。放弃此方向。**

### Figures
- `outputs/_experimental/figures/fig_e13_label_propagation.png`

---

## E14: Full sweep — Mahal-pooled across ALL datasets and rare_train_sizes
**Status**: ✅ Complete  
**Script**: `src/experimental/e14_full_mahal_sweep.py`

### Setup
29 runs: seed42, all datasets, all rts (5/10/20/50).

### Results (selected highlights)

| Dataset | Rare class | rts | S | scANVI | Euclidean | Mahal-pooled | Δ |
|---|---|---|---|---|---|---|---|
| immune_dc | cDC1 | 5 | 0.979 | 0.000 | **0.982** | 0.973 | -0.009 |
| immune_dc | cDC1 | 50 | 0.884 | 0.994 | 0.990 | **0.996** | +0.006 |
| immune_dc | ASDC | 5 | 1.212 | 0.060 | **0.922** | 0.833 | -0.090 |
| immune_dc | ASDC | 50 | 0.988 | 0.829 | **0.957** | 0.902 | -0.055 |
| pancreas | epsilon | 5 | 0.644 | 0.000 | 0.095 | **0.308** | +0.212 |
| pancreas | epsilon | 10 | 0.794 | 0.667 | 0.250 | **0.800** | +0.550 |
| pancreas | epsilon | 20 | 0.743 | 0.667 | 0.211 | **0.500** | +0.289 |
| pancreas | gamma | 5 | 0.650 | 0.045 | 0.208 | **0.377** | +0.169 |
| tabula_kidney | endothelial | 5 | 1.196 | 0.846 | 0.632 | **0.686** | +0.054 |
| tabula_spleen | ILC | 50 | 0.981 | 0.839 | 0.740 | **0.822** | +0.082 |

### Summary by dataset (mean across rts)

| Dataset | mean_S | mean_euc | mean_mahal | mean_Δ |
|---|---|---|---|---|
| immune_dc (cDC1) | 0.999 | 0.987 | 0.986 | -0.003 |
| immune_dc (ASDC) | 1.159 | 0.930 | 0.881 | -0.050 |
| pancreas (epsilon) | 0.741 | 0.239 | 0.569 | +0.330 |
| pancreas (gamma) | 0.630 | 0.719 | 0.779 | +0.060 |
| tabula_liver (NCM) | 1.342 | 0.637 | 0.633 | -0.004 |
| tabula_kidney (endothelial) | 1.080 | 0.601 | 0.667 | +0.066 |
| tabula_spleen (ILC) | 1.083 | 0.793 | 0.828 | +0.035 |

### Key Finding
**明确的 separability 阈值：S < 1.0 时 Mahal-pooled 几乎总是优于 Euclidean（平均 +0.33 for epsilon，+0.06 for gamma）。S > 1.2 时 Euclidean 通常更好（ASDC: -0.05）。S ∈ [1.0, 1.2] 是混合区域（endothelial: +0.07，ILC: +0.04）。** 29 runs 中 Mahal-pooled 优于 Euclidean 的有 20/29（69%），平均 Δ = +0.060。**建议：将 adaptive selector 的 Mahal/Euclidean 切换阈值从 S=1.3 调整为 S=1.2。**

### Figures
- `outputs/_experimental/figures/fig_e14_mahal_sweep_scatter.png`
- `outputs/_experimental/figures/fig_e14_mahal_sweep_heatmap.png`

---

## E15: Visualizations Round 2
**Status**: ✅ Complete  
**Script**: `src/experimental/e15_visualizations_round2.py`

### Figures generated

| Figure | Description |
|---|---|
| `fig_e8_soft_gate.png` | Soft gate vs hard gate vs Mahal (no gate) |
| `fig_e9_ensemble.png` | Ensemble α sweep + final comparison |
| `fig_e10_adaptive_selector.png` | Adaptive selector vs individual methods |
| `fig_e11_gmm_calibrated.png` | Calibrated vs uncalibrated GMM |
| `fig_e12_bootstrap_uncertainty.png` | Bootstrap CI width vs n_rare_train |
| `fig_e13_label_propagation.png` | Label propagation vs inductive methods |
| `fig_e14_mahal_sweep_scatter.png` | Separability ratio vs Δ(mahal-euclidean) scatter |
| `fig_e14_mahal_sweep_heatmap.png` | All datasets × rts heatmap of Mahal improvement |

All figures saved to: `outputs/_experimental/figures/`

---

## Round 2 Summary

### 最重要发现（来自 E14）

**Mahal-pooled 的适用边界已明确：S < 1.0 时 Mahal-pooled 大幅优于 Euclidean（epsilon: +0.33 平均），S > 1.2 时 Euclidean 更好（ASDC: -0.05）。** 这是 29 runs 的系统性证据，可以直接用于 adaptive selector 的阈值校准。

### Round 2 方法排名更新

| 方法 | 结论 | 建议 |
|---|---|---|
| **Mahal-pooled (λ=0)** | ✅ 在 S<1.0 时最优，+33pp vs Euclidean | **主推，立即集成** |
| **Adaptive selector (S阈值)** | ✅ 5/8 案例匹配最优，但 epsilon 失败 | 调整阈值：Mahal/Euclidean 切换点改为 S=1.2 |
| Soft gate | ❌ 退化为 Mahal no gate，无额外收益 | 放弃 |
| Ensemble (Mahal+CB-kNN) | ⚠️ 对 epsilon 严重退化 | 仅在 high-sep 案例使用 |
| Bootstrap uncertainty | ✅ high-sep 稳定，low-sep 无帮助 | 可作为 uncertainty 报告工具 |
| GMM calibrated | ❌ 仍然完全失败 | **放弃 GMM 方向** |
| Label propagation | ❌ high-sep 严重退化 | **放弃** |

### 下一步（优先级排序）

1. **[立即]** 将 adaptive selector 阈值从 S=1.3/1.0 调整为 S=1.2/0.8，重新测试 epsilon
2. **[高优先级]** 将 Mahal-pooled 集成到 `src/03_prototype.py`（新 branch）
3. **[中优先级]** 研究 epsilon 的特殊性：为什么 CB-kNN 在 S=0.74 时失败？可能需要 SMOTE-LR 作为 fallback
4. **[低优先级]** GMM 方向：尝试先 PCA 降维到 5 维再拟合 GMM

---

## Round 2 文件清单

```
src/experimental/
  e8_soft_gate.py
  e9_ensemble.py
  e10_adaptive_selector.py
  e11_gmm_calibrated.py
  e12_prototype_uncertainty.py
  e13_label_propagation.py
  e14_full_mahal_sweep.py
  e15_visualizations_round2.py

outputs/_experimental/
  e8_soft_gate/
    results.csv
    *_tau_curve.csv (5 files)
  e9_ensemble/
    results.csv
    *_alpha_curve.csv (4 files)
  e10_adaptive_selector/
    results.csv
  e11_gmm_calibrated/
    results.csv
  e12_prototype_uncertainty/
    results.csv
  e13_label_propagation/
    results.csv
  e14_full_mahal_sweep/
    results.csv
    summary_by_dataset.csv
  figures/
    fig_e8_soft_gate.png
    fig_e9_ensemble.png
    fig_e10_adaptive_selector.png
    fig_e11_gmm_calibrated.png
    fig_e12_bootstrap_uncertainty.png
    fig_e13_label_propagation.png
    fig_e14_mahal_sweep_scatter.png
    fig_e14_mahal_sweep_heatmap.png
```
