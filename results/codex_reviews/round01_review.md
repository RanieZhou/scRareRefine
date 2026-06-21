# Round 01 — codex 评审原文（Round 10 / G03 ablation）

**Date**: 2026-06-19
**threadId**: `019edb7d-fede-7622-9b61-e2b66beb4439`
**Model / config**: `gpt-5.5` + `{"model_reasoning_effort": "xhigh"}`
**Reviewer Difficulty**: hard (Reviewer Memory enabled)
**Score**: 6.8/10
**Verdict**: almost

---

## 原文响应（verbatim, 不裁剪）

我核对了当前 checkout 的 `results/ablation/*.csv`、`tools/analysis/ablation.py`、`tmp/round10_dataset_adequacy.py` 和 `src/rescue.py`。结论如下。

**1. Score**：6.8/10。Round 10 把机制故事补强了，但还不到 Bioinformatics/BIB 级别的"可投即稳"。

**2. Verdict**：almost，但按现在写法不是 ready。主要卡点是单 seed、dataset adequacy 不够、pancreas_integrated 负回归、以及 conformal 不能被写成强理论保证。

**3. Hypothesis 检验**
- H1 necessity：支持，但证据偏窄。主要来自 small_intestine 这种 baseline 已能全召回的负控场景；不要写成普适性能增益，应写成 safety/abstention guard。
- H2 adaptive rank：支持最强。pancreas_baron/stomach 需要 rank=2，immune/lung_endo 需要避免 rank=2 假阳性。这是本轮最有价值的 ablation。
- H3 sep gate：基本支持，但这是机制性 no-op，不是强实验发现。真正证据是 pancreas_baron all 的 sep<1.3 场景，关 gate 会轻微伤害。
- H4 conformal τ：支持，但只支持"经验上降低 FFR 风险"。pancreas_baron V6 FFR=0.009768 已贴着 α=0.01，不能写成跨 split 的严格保证。
- H5 dataset adequacy：支持，而且比你写得更严重。small_intestine 是负控/弃权验证；pancreas_integrated 是低价值 testbed 加 split-shift failure，不应继续当主性能证据。

**4. 遗漏检查**
(a) 不算掩盖，如果你明确把它们定义为 negative-control / inadequate benchmark；如果仍把它们混进"6 数据集平均提升"就是掩盖。small_intestine 不应说成单纯 baseline-saturated，它在 CSV regime 是 abstain-necessity；pancreas_integrated 0.01/0.05 是 test saturated 但 val 未 saturated，正好暴露 necessity 的 split shift 风险。

(b) 单 seed 会被审稿人直接打：没有方差、没有 split 稳定性、α=0.01 的边界合规也不稳。最低限度要 3 seeds；理想是 5 seeds，报告 mean±sd、win/tie/loss、FFR violation count。

(c) V6 在 pancreas_integrated 比 V0 低 0.0206，算回归。更严重的是逐 rts：0.01 从 1.0000 掉到 0.9388，0.05 从 1.0000 掉到 0.9787；这是"test 已无漏判但 rescue 仍误救"的负控失败。处理方式：主文诚实报告，主结果分 testbed vs inadequate/negative-control 两层；方法上考虑加入 validation missed-rare 最小支持数或 baseline-as-option 的 abstain 选择。

**5. 薄弱点**
1. 单 seed 无稳定性。Minimum fix：先补 3 seeds，至少覆盖 6 数据集 × 4 rts × V0/V3/V4/V5/V6。
2. pancreas_integrated 负回归。Minimum fix：单独标为 split-shift negative control，并在方法里加"低 val rare support / 高方差 necessity"弃权诊断。
3. dataset adequacy 过弱。Minimum fix：替换 pancreas_integrated；small_intestine 可留作 safety negative control，但别算主 testbed。
4. conformal claim 过强。Minimum fix：改成 empirical FFR control under validation calibration，并报告 raw false rescues 和 binomial CI。
5. benchmark overfitting 风险。代码注释里 rank≤2 的理由含 test FFR 经验，论文需避免"看 test 选设计"的印象。Minimum fix：补 rank=3/3+ sensitivity，并说明 rank grid 是预注册或由 train/val/开发集确定。
6. cache provenance 未形成证据链。Minimum fix：为每个 ablation row 记录 config hash、cache path、seed、rts、cell-id alignment check。

**6. 下一轮方向**
1. Multi-seed 稳定性：边际贡献最高，成本中等。没有这个，二区审稿很容易一票打穿。
2. 换/补 testbed 数据集：优先替换 pancreas_integrated；small_intestine 保留为负控。新增数据集必须预先定义 inclusion criteria，避免 cherry-picking。
3. 处理 pancreas_integrated 类负回归：加低支持数/necessity uncertainty guard，或让 k=0 abstain 进入 val 选择。
4. Conformal section 重写：补经验 FFR、raw false rescues、CI、α 固定声明；不要承诺无条件 coverage。
5. Stomach ceiling 攻坚：作为下一阶段创新点，但成本高；rank≥3 会带 FFR 风险，最好引入 expression-side marker/orthogonal evidence，而不是单纯放宽 rank。
6. Batch-conditional τ：可作为针对 pancreas_baron 贴 α 的专门改进，但先别排第一，容易变成小修小补。

**7. Memory update**
- Round 10 strongest evidence is adaptive rank: rank=2 helps pancreas_baron/stomach, rank=1 protects immune/lung_endo.
- Necessity guard evidence is real but mostly negative-control/saturated-regime evidence, not broad performance evidence.
- pancreas_integrated is not merely uninformative: V6 regresses vs V0 on 0.01/0.05 because test baseline is saturated while validation suggests missed rare cells.
- small_intestine should be tracked as abstain-necessity negative control, not a primary rescue benchmark.
- Conformal τ evidence is empirical and near-boundary on pancreas_baron; avoid claiming strict FFR guarantee without CI/multi-seed.
- Watch benchmark-overfitting risk around rank_grid={1,2}, especially any rationale derived from prior test FFR for rank=3.
- Future reviews should check cache provenance and cell-ID alignment before trusting ablation rows.

---

## 作者回应 / 行动项映射

| codex 提出 | 处理决定 | 跟进位置 |
|-----------|---------|---------|
| 多 seed 是 #1 blocker | 用户已明确"前期单 seed 迭代，后期统一补 multi-seed"；codex 提早一轮，记入 G01 标 from codex r1 优先级提升 | G01（升级） |
| pancreas_integrated 负回归 | 真问题，确认。Round 11 优先处理：要么换数据集，要么加 val-rare-support 守门 | G60 (new) |
| small_intestine 改称 negative-control | 接受，文档措辞同步修正 | Round 10 决策段 |
| conformal claim 过强 | 接受，conformal section 改成 empirical 描述 + binomial CI | G61 (new) |
| rank_grid={1,2} 文档含 test FFR 信息 | 接受，潜在 R1 违规：要么消除注释里的 test 引用，要么补 rank=3 sensitivity 并以 train+val 论证 | G62 (new) |
| cache provenance 证据链 | 接受，ablation 输出加 manifest hash + cache path | G63 (new) |
