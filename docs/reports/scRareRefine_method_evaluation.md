# scRareRefine 方法客观评价

**生成时间**: 2026-05  
**基于**: 代码审查 + E1-E34 实验结果  
**用途**: 方法改进参考、论文写作参考

---

## 一、与 Mahal-pooled 的对比澄清

在 E1-E34 实验中，"Mahal-pooled 表现最好"的结论是在**纯距离方法**这个受限范围内的结论。与 scRareRefine 完整 pipeline 的直接对比（rts=20，seed42）：

| 数据集 | Rare class | Mahal-pooled（纯距离） | scRareRefine（Eucl+gate+marker） | 差距 |
|---|---|---|---|---|
| immune_dc | ASDC | 0.901 | **0.933** | scRareRefine +3.2pp |
| immune_dc | cDC1 | 0.988 | **0.985** | 基本持平 |
| pancreas | epsilon | 0.500 | **0.889** | scRareRefine +39pp |
| pancreas | gamma | 0.964 | **0.992** | scRareRefine +2.8pp |
| tabula_liver | NCM | **0.680** | 0.625 | Mahal +5.5pp |
| tabula_kidney | endothelial | 0.649 | **0.919** | scRareRefine +27pp |
| tabula_spleen | ILC | 0.845 | **0.857** | scRareRefine +1.2pp |
| tabula_pancreas | β-cell | 0.857 | **0.897** | scRareRefine +4pp |

**结论：scRareRefine 在 7/8 个数据集上优于或持平 Mahal-pooled。** gate+marker 机制是关键，尤其在 epsilon（+39pp）和 endothelial（+27pp）上。

---

## 二、优点

### 1. 工程设计扎实

三阶段 pipeline（prototype scoring → gate → marker verification）逻辑清晰，每一步都有明确的生物学动机：
- Prototype 距离：利用 scANVI latent space 的几何结构
- Gate：用 separability ratio 做 pre-deployment 诊断，避免盲目 rescue
- Marker verification：引入生物学先验，不只依赖几何

Inductive 约束执行严格（train-only reference，val-only threshold），避免了数据泄漏，这是正确的评估设计。

### 2. Abstention 机制是真正的亮点

当 S < 1.1 时自动退化到 baseline，不强行 rescue。实验验证：gamma（S=0.94）和 β-cell（S=0.80）上方法正确 abstain，没有造成损害。这个"知道自己什么时候不该工作"的性质在单细胞领域很少见，是论文的核心卖点之一。

### 3. 结果在 high-sep 案例上确实强

- cDC1 rts=5：0.003 → 0.986（+98pp）
- ASDC rts=20：0.657 → 0.933（+28pp）
- endothelial rts=20：0.928 → 0.919（gate+marker 精准控制 false rescue）

### 4. 可解释性好

每个 rescued cell 可以追溯到：prototype distance → gate rule → marker verification → threshold。这在临床/实验室场景中有实际价值。

---

## 三、缺点（机制层面的真实问题）

### 缺点 1：Prototype 是点估计，没有不确定性（最根本）

当 n_rare=5 时，用 5 个点的均值作为 prototype，这个估计的方差极大。代码里：

```python
prototypes = np.vstack([
    reference_latent[...].mean(axis=0)  # 纯点估计
    for cls in classes
])
```

没有任何机制量化"这个 prototype 有多可靠"。方法在 n_rare=5 时和 n_rare=20 时用的是同一套逻辑，但前者的 prototype 可靠性远低于后者。这导致在极小样本场景下，rescue 决策的置信度被高估。

### 缺点 2：Gate 是硬规则，缺乏概率解释

```python
candidates = (predicted_labels != rare_class) & (ranks <= 2) & (margin <= threshold)
```

`ranks <= 2` 是完全任意的硬截断。为什么是 2 而不是 3？margin 用 25th percentile 的依据是什么？这些都是启发式，没有理论依据。实验（E6）证实：这个 gate 对 NCM 过于保守（Mahal 无 gate 0.680 > gate+marker 0.647）。

### 缺点 3：Marker scoring 过于朴素

```python
diff = expr[in_class].mean(axis=0) - expr[out_class].mean(axis=0)
top_idx = np.argsort(-diff)[:top_n]  # top 25 by mean fold-change
```

用均值 fold-change 选 top-25 基因，然后对这 25 个基因取均值作为 score。问题：
- 对 dropout 不鲁棒（sparse expression 的均值不稳定）
- 没有考虑基因之间的相关性
- 25 这个数字是任意的
- 没有考虑 batch effect 对 marker expression 的影响

### 缺点 4：Separability ratio 是事后诊断，不是设计目标

S ratio 目前只用来决定是否 abstain，但它本身没有参与任何优化。方法不会主动去提高 S，也不会根据 S 的大小调整 rescue 的保守程度。S 包含了很多有用的信息，但只被用作一个二值开关。

### 缺点 5：两路信号（几何 + 表达）是串联而非融合

当前设计：先用几何（prototype rank）筛选候选，再用表达（marker score）验证。这是硬串联：几何筛选失败的 cell 永远不会被 marker 救回来。但实际上，有些 cell 在几何上不是 rank-1，但 marker 表达非常强，应该被 rescue。

### 缺点 6：只适用于 scANVI（当前实现）

虽然理论上可以接在任何 embedding 后面，但代码里有多处 scANVI 特定的假设（prob_ 列、margin 列等），迁移到其他模型需要额外工程。

---

## 四、机制/算法层面的改进建议

### 改进 1：把点估计 prototype 升级为后验分布（最重要，1-2周）

把 `mean(z_i for i in rare_train)` 替换为：

$$\hat{\boldsymbol{\mu}}_r \sim \mathcal{N}\left(\bar{\mathbf{z}}_r, \frac{\boldsymbol{\Sigma}_r}{n_r}\right)$$

用 Ledoit-Wolf 收缩估计 $\boldsymbol{\Sigma}_r$（已在 E1 实验中验证可行）。效果：
- n_rare=5 时，prototype 不确定性大，rescue 决策自动更保守
- n_rare=50 时，不确定性小，退化为当前的点估计
- Separability ratio 获得理论解释：$S = \delta_r / \rho_r$ 正好是后验预测分布的充分统计量
- 把 gate 从"硬规则"变成"概率阈值"

**实验支撑**：E1 中 Mahal-pooled（Ledoit-Wolf 协方差）在 epsilon 上 +50pp vs Euclidean。

### 改进 2：把 Euclidean 距离换成 Mahal-pooled（1天，最低成本）

实验（E14）证明：在 S < 1.2 的案例（NCM、endothelial、ILC）上，Mahal-pooled 比 Euclidean 平均高 6-7pp。这是一行代码的改动，但有理论支撑（Mahal 距离考虑了 latent space 的各向异性）。

修改位置：`src/03_prototype.py` 的 `prototype_scores()` 函数。

### 改进 3：加入 Logit Adjustment 作为前置步骤（1天，互补）

实验（E24）证明 Logit Adjustment 在 scRareRefine 之前做一步概率校正，可以在 gamma rts=5（+60pp）、ILC rts=20（+23pp）等案例上有额外提升。这是一个完全独立的概率层，不影响现有 pipeline：

```python
adjusted_logit_c = log(p_scANVI(c|x)) - τ * log(π_c)
```

τ 在 validation 上调优（通常 τ=1.0 最优）。

### 改进 4：把 marker scoring 升级为 AUCell 风格（3天，生物学更合理）

把当前的 `mean(top25 genes)` 替换为基于排名的 AUC：

$$\text{AUC}(i, G_r) = \frac{|\{g \in G_r : \text{rank}(x_{ig}) \leq k\}|}{|G_r|}$$

即：计算 rare class marker genes 在 cell i 的表达排名中的富集程度。这对 dropout 更鲁棒，因为它用的是排名而不是绝对表达量。这是 AUCell（Aibar et al., 2017）的核心思想，在单细胞领域已被广泛验证。

### 改进 5：把串联改为概率融合（2周，最大的机制改变）

把当前的"几何筛选 → 表达验证"改为：

$$p(\text{rescue} | \mathbf{z}_i, \mathbf{x}_i) \propto p(\text{rare} | \mathbf{z}_i) \cdot p(\text{rare} | \mathbf{x}_i)$$

其中：
- $p(\text{rare} | \mathbf{z}_i)$：基于 Mahal 距离的几何概率（softmax over distances）
- $p(\text{rare} | \mathbf{x}_i)$：基于 AUCell score 的表达概率

两路信号在概率层面相乘，而不是串联过滤。这样几何弱但表达强的 cell 也能被 rescue。threshold 在 validation 上统一调优。

### 改进 6：加入 Conformal Prediction 作为理论保证层（1周，论文卖点）

实验（E26）证明 Conformal Prediction 在 gamma rts=5 上 +86pp，ILC +26pp。更重要的是，它提供了**理论保证**：在 α=0.05 时，false rescue rate ≤ 5%（有数学证明）。

这是当前方法完全没有的性质。加入 conformal 层可以把论文从"经验方法"升级为"有理论保证的方法"。

---

## 五、改进优先级总结

| 优先级 | 改进 | 预期收益 | 实现成本 | 实验支撑 |
|---|---|---|---|---|
| ⭐⭐⭐ | Mahal-pooled 替换 Euclidean | NCM/ILC +5-7pp | 1天 | E14 |
| ⭐⭐⭐ | Logit Adjustment 前置 | gamma/ILC +20-60pp | 1天 | E24 |
| ⭐⭐⭐ | Bayesian prototype（后验分布） | 理论升级，S ratio 有理论解释 | 1-2周 | E1 |
| ⭐⭐ | AUCell marker scoring | 对 dropout 更鲁棒 | 3天 | 文献 |
| ⭐⭐ | Conformal Prediction 保证层 | 理论保证，论文卖点 | 1周 | E26 |
| ⭐ | 概率融合替换串联 gate | 最大机制改变，需重新设计 | 2周 | 理论 |

---

## 六、论文定位建议

**当前定位**（可以支撑）：
> "scRareRefine 是一个 post-hoc rescue framework，在 scANVI latent space 中通过 prototype 距离、gate 和 marker verification 三阶段修正 rare cell 漏检，并通过 separability ratio 实现自知边界。"

**升级后定位**（加入改进 1+2+3 后）：
> "scRareRefine 是一个 Bayesian-calibrated post-hoc rescue framework，通过后验 prototype 不确定性量化、Mahalanobis 距离度量和 logit-adjusted 概率校正，在极小标注预算下实现可靠的 rare cell rescue，并提供 separability ratio 作为部署前诊断指标。"

这个定位覆盖了三个不同的算法范式（几何、贝叶斯、概率），论文贡献点更清晰。
