# Codex Round 3 review（第十四轮 G21 sep sweep 外审，2026-06-21）

**Codex thread**: `019f04ab-5f4f-7fa3-b51d-c3b4d41074d6`
**Model**: gpt-5.5 / reasoning_effort xhigh / sandbox read-only
**Submitted**: 第十三轮 multi-seed(G01/G02)+消融重构(G03) 进展 + 第十四轮 G21 可控 sep 扫描 + 我把 1.3 证伪重定性为"保守裕度"的解释
**注**: codex 本轮**无法读仓库文件**（shell sandbox 启动失败 `ShellExecuteExW ... 1223`），是基于我提交证据的外审，非 CSV/log 核验审计。

---

## 响应原文（verbatim）

Important limitation: I attempted to read the repo files, but the local shell could not launch under the current sandbox (`ShellExecuteExW ... 1223`). So this is an external review of the evidence you supplied, not a verified CSV/log audit of the checkout.

1. **Score**

**7.7 / 10**

2. **Verdict**

**almost**

The multi-seed and ablation additions move the work substantially closer to a bioinformatics Q2 submission. The current blocker is no longer "n=1 evidence"; it is claim calibration: do not oversell the separability sweep or conformal control as more formal than the evidence supports.

3. **重定性审查**

把"1.3 是精确危险边界"改成"1.3 是保守安全裕度"基本是诚实科学，不是纯找台阶。你承认了证伪点：`sep=1.152` 时 no-gate 仍安全且有明显 F1 gain，这直接否定"1.3 精确定位风险边界"。

但你的第 4 点"升级成有受控实验定位崩塌点的保守先验"有过度包装。当前数据最多支持：

> 1.3 is a pre-fixed conservative abstention threshold; the stress test found one marginal no-gate FFR violation at the lowest observed separability, while also revealing that the gate sacrifices recoverable F1 in some low-sep but empirically safe cases.

不要写成"定位了崩塌点 sep≈0.7"。`0.0105` 只是略高于 `0.01`，需要 raw counts 和 CI；而且 `sep=0.761` 又安全，说明这不是稳定单调边界。

保留 1.3 不改是站得住的，理由是：单数据集、单扰动方向、单 seed 不足以支持把全局阈值改到 0.7；贸然改阈值反而像 test-time tuning。最小可接受处理是：保留 1.3，但降格表述为"conservative pre-specified guard"，并公开报告 below-gate safe-but-abstained cases 作为代价。

4. **Hypothesis 检验 + Confirmation Bias**

支持的结论：

- "1.3 不是精确危险边界"：强支持。
- "1.3 偏保守，会牺牲部分可恢复 F1"：强支持。
- "sep gate 有安全价值"：中等支持，来自 full benchmark no-sep FFR 破 α + stress point `sep=0.686` 破 α。
- "经验崩塌边界在 sep≈0.7"：弱支持，不应主张。
- "sep 是单调风险轴"：不支持；你自己给出的 t→sep 非单调，加上 0.686 破、0.761 安全，说明风险不是 sep 单变量决定。

最大 confirmation bias 风险是把一个"对原假设不利的发现"重新讲成"更强证据"。它确实让故事更诚实，但不是更强；它让 claim 更窄、更可信。

5. **遗漏检查**

单数据集 + 单纠缠方向 + 非单调 sep，结论强度不够支撑"跨数据集安全裕度定位"。但不一定必须补第二数据集，取决于论文怎么写。

如果只写成 exploratory stress test：不必须补第二数据集。

如果写成 threshold validation / collapse boundary localization：必须补，至少再加一个几何不同的数据集和多个 seeds。

应补报的不利点：

- 每个点的 raw FFR counts：false rescues / denominator。`0.0105` 是否只是离散误差下的一两个细胞差异很关键。
- full 与 nogate 的完整 FFR、rescued count、rank、τ、necessity 是否触发。
- below-gate but safe 的损失：例如 `sep=1.152` 被 1.3 gate 放弃了 `+0.235` F1。
- 明确说明 synthetic construction 使用 labels/centroids 是 stress-test generation，不是方法流程。
- 不要只按 sep 排序，也要报告原始 t 顺序，因为 t→sep 非单调本身是不利现象。

6. **薄弱点**

1. **G21 过度表述**
   Minimum fix: 把"定位崩塌点 sep≈0.7"改成"observed one marginal violation at the lowest-sep condition"。

2. **FFR near-boundary 仍是经验值**
   Minimum fix: 报 raw counts + Wilson/binomial CI；不要说 strict guarantee。

3. **单 stress setting 不足以校准全局阈值**
   Minimum fix: either 降格为 exploratory，or 补一个第二数据集/第二 perturbation direction。

4. **necessity 仍主要是 safety guard，不是 performance driver**
   Minimum fix: 用 per-dataset counterfactual 表证明它防了哪些 regressions；不要说它普遍提升 F1。

5. **rank cherry-pick 怀疑降低但未完全消失**
   Minimum fix: 导出 row-level rank selection diagnostics：candidate ranks、Wilson upper bounds、chosen rank、val/test FFR。

6. **multi-seed p 值仍偏乐观**
   Minimum fix: 主文强调 effect size 和 paired wins；p 值降为方向性证据，最好补 dataset×rts 层级 bootstrap。

7. **下一轮方向**

1. **改论文表述 + 补 raw diagnostics 表** — 成本低，收益最高。尤其 sep sweep raw counts、rank、τ、rescued counts、abstain reason。

2. **全 benchmark 做 global low_sep sensitivity** — 例如 `low_sep ∈ {0, 0.7, 1.0, 1.3, 1.6}`，不用改主方法，只证明 1.3 的 F1/FFR tradeoff 不是单点碰巧。

3. **第二 stress dataset 或第二 perturbation direction** — 选非 lung_endo 的几何环境。目标不是重新调阈值，而是证明"1.3 保守"不是 lung_endo 特例。

4. **把 statistical claim 降温** — 多 seed 已解决最大 blocker，但 p 值不要写成独立样本强证据；写成 robust paired improvement。

5. **补 provenance audit** — 本轮无法核 repo；若还有 `git_sha=unknown` 或 cache provenance 不全会继续被抓。

8. **Memory update**

- 新增怀疑：G21 的"sep≈0.7 崩塌边界"可能是对单个 near-boundary FFR 点的过度解释。
- 未解决疑点：`nogate_ffr=0.0105` 的 raw false-rescue count 和 CI；是否只是离散小幅越界。
- 未解决疑点：sep sweep 是否导出了 full/nogate 的 rank、τ、rescued count、abstain reason。
- 未解决疑点：multi-seed significance 是否以 dataset×rts×seed 为单位，而不是把相关 cell-level observations 当独立。
- 跨轮 pattern：凡是 hard threshold 被"验证"，必须同时报告 safe-but-blocked cases 和 unsafe-passed cases，不能只报支持安全叙事的点。
- 跨轮 pattern：把失败假设重定性可以接受，但必须让 claim 变窄，而不是说证据更强。
