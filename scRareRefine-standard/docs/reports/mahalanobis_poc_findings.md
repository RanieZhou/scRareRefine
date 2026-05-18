# Mahalanobis Prototype PoC — findings (branch: `feat/bayesian-prototype`)

**目的**：在不改动 main pipeline 的前提下，快速验证 "Bayesian prototype" 路径的算法信号。
仅测试距离度量本身（no gate, no marker, no threshold）：把每个 test cell 直接分配到最近的 prototype。

**测试变体**（所有 covariance 使用 Ledoit-Wolf 收缩）：

1. `euclidean`：当前 main method 的距离 = `||z - mu_c||_2`
2. `mahalanobis (per-class)`：每个类用自己的协方差 `Sigma_c`
3. `mahalanobis (pooled / LDA-style)`：所有类共用一个 pooled within-class covariance
4. `mahalanobis per-class + posterior penalty`：加上 `tr(Sigma_c^-1) / n_c`（贝叶斯后验方差惩罚，n 越小越保守）
5. `mahalanobis pooled + posterior penalty`：pooled + posterior penalty

---

## Sweep 结果（10 个代表性 runs，seed=42）

| regime                    | dataset           | rare_class    | rts    | scANVI | Eucl(cur) | Mahal-pc | Mahal-pool | Mahal-pc+post | Mahal-pool+post |
|---------------------------|-------------------|---------------|--------|--------|-----------|----------|------------|---------------|-----------------|
| high-sep / low-baseline   | immune_dc         | ASDC          | rare5  | 0.060  | **0.922** | 0.687    | 0.833      | 0.480         | 0.785           |
| high-sep / low-baseline   | immune_dc         | ASDC          | rare20 | 0.488  | **0.934** | 0.903    | 0.901      | 0.884         | 0.868           |
| high-sep / low-baseline   | immune_dc         | cDC1          | rare5  | 0.000  | **0.982** | 0.951    | 0.973      | 0.888         | 0.926           |
| high-sep / low-baseline   | immune_dc         | cDC1          | rare20 | 0.485  | 0.982     | **0.992**| 0.988      | 0.984         | 0.986           |
| high-sep / low-baseline   | tabula_liver      | NCM           | rare20 | 0.348  | 0.667     | 0.286    | **0.680**  | 0.000         | 0.638           |
| high-sep / high-baseline  | tabula_kidney     | endothelial   | rare20 | **0.889** | 0.585  | 0.571    | 0.649      | 0.622         | 0.649           |
| low-sep / high-baseline   | pancreas          | gamma         | rare20 | **1.000** | 0.945  | 0.989    | 0.964      | 0.989         | 0.951           |
| low-sep / high-baseline   | tabula_pancreas   | β-cell        | rare20 | 0.897  | 0.857     | 0.857    | 0.857      | 0.857         | **0.968**       |
| low-sep / low-annotation  | pancreas          | epsilon       | rare20 | 0.667  | 0.211     | 0.800    | 0.500      | **1.000**     | 0.800           |
| low-sep / low-annotation  | pancreas          | epsilon       | rare5  | 0.000  | 0.095     | 0.000    | **0.308**  | 0.000         | 0.000           |

---

## 关键观察（对组会最重要）

### 1) epsilon rts20：后验惩罚变体完胜所有方法

- scANVI baseline：F1 = 0.667
- 当前 main method（Euclidean）：F1 = 0.211 （比 baseline 掉了 **-45pp**）
- Mahal-pc + posterior penalty：F1 = **1.000**
- 这是**当前 main method 的明确失败案例**（它在 low-sep 下不但没 rescue，反而伤害了 baseline）。
- 贝叶斯后验惩罚**正好扭转了这个失败**：n_rare=20、posterior penalty 让 epsilon prototype 变得"没那么自信"，避免了错误吸入 delta/gamma cells。
- 这是**"Bayesian framework 能突破当前方法只在 high-sep 下 work 的 niche"的第一个证据**。

### 2) β-cell：低 sep 下 Mahal-pool+post 优于 scANVI

- scANVI：0.897，当前方法：0.857（0 增益）
- Mahal-pool + post：**0.968**（+7pp vs scANVI）
- 另一个当前方法不能提升、但贝叶斯路径可以的案例。

### 3) Per-class covariance 在小 n 下不稳定，pooled 是更安全的选择

- NCM rts20：per-class Sigma 崩溃到 0.286；pooled Sigma 恢复到 0.680
- ASDC rts5：per-class 0.687 < pooled 0.833
- 启示：**论文路径应主推 pooled + shrinkage**，per-class 放作 ablation

### 4) Posterior penalty 不总是好

- ASDC rts5：per-class + post 掉到 0.480，说明在 high-sep / 大 intra-spread 下 penalty 过重
- NCM rts20：per-class + post = 0.000（完全失败）
- 提示：**应让 penalty 强度自适应于数据**，例如按 d/n 比例缩放，或在 validation 上调标量权重

### 5) 纯距离度量（无 gate + marker）不能完全替代当前方法

- cDC1 rts5：Euclidean 0.982 ≈ Mahal-pc 0.951；只是距离度量的差异不大
- 但当前方法在 cDC1 rts5 上用 gate+marker 能做到 0.986，说明 **gate+marker 有独立价值**
- 下一步：把 Mahalanobis + posterior penalty 作为新"距离信号"送进现有的 gate+marker 里，看能否在 low-sep 案例上继续出 gain

---

## 对路径 A 的判断

**信号足够，值得投入三个月。** 三个证据点：

1. **epsilon rts20：0.667 → 1.000** 是目前方法完全做不到的
2. **β-cell rts20：0.897 → 0.968** 是另一个 low-sep 提升
3. **pooled + shrinkage 在小 n 场景稳定**，说明 ML 里的标准工具直接迁移有效

**下一步（组会后 1-2 周）**：
- 把 Mahalanobis-posterior 的距离信号接入现有 gate+marker pipeline，跑 3 seeds
- 在 validation 上调 posterior penalty 的标量权重 λ
- 消融：pooled vs per-class，shrinkage strength，penalty scaling

**3 个月路线图**：
- Month 1: Mahalanobis + posterior penalty 完整实现，10 个 rare class × 3 seeds
- Month 2: 概率建模 + abstention（credible interval），conformal calibration
- Month 3: 竞品 benchmark (scBalance, sc-SynO)，写作

---

## 如何复现

```bash
git checkout feat/bayesian-prototype
python3 src/experimental/run_mahalanobis_sweep.py
python3 src/experimental/format_sweep.py
# 输出：
#   outputs/_experimental/mahalanobis_sweep.csv
#   outputs/_experimental/mahalanobis_sweep_summary.md
```

单独跑一个 run：

```bash
python3 src/experimental/mahalanobis_poc.py \
    --run_dir outputs/pancreas/batch_heldout_seed42_epsilon_rare20 \
    --rare_class epsilon
```

---

## 源代码（新增，仅在本 branch）

- `src/experimental/mahalanobis_poc.py` — 距离度量实现 + 单 run 评估
- `src/experimental/run_mahalanobis_sweep.py` — 10 个代表性 runs 的批处理
- `src/experimental/format_sweep.py` — 结果整理成 markdown 表

**未修改**：main pipeline 所有文件（`src/main.py`, `src/03_prototype.py`, `src/05_*.py` 等）。
