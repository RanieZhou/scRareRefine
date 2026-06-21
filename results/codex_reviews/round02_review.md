# Round 02 — codex 评审原文（Round 11 / G60+G62+G63）

**Date**: 2026-06-19
**threadId**: `019edb7d-fede-7622-9b61-e2b66beb4439` (continued from Round 1)
**Model / config**: `gpt-5.5` + `{"model_reasoning_effort": "xhigh"}`
**Reviewer Difficulty**: hard (Reviewer Memory enabled)
**Score**: 7.2/10 (↑0.4 from Round 1)
**Verdict**: almost

---

## 原文响应（verbatim, 不裁剪）

**1. Score**：7.2/10。Round 11 是实质进步，但还没过二区稳态线。

**2. Verdict**：almost。方法层主要漏洞补了一半以上；投稿前最大 blocker 仍是 single seed + testbed adequacy。

**3. 上轮怀疑裁定**
- Adaptive rank 证据：**SUSTAINED**。V7_rank3_fixed 加强了"固定 rank=3 不可用"的论证，V6 选择更可信。
- Necessity 只是 safety 证据：**PARTIALLY**。`MIN_VAL_MISSED=3` 把它升级成 split-shift guard，但仍主要是 safety/abstain 机制，不是广义性能增益。
- pancreas_integrated 负回归：**SUSTAINED**。V6 已回 baseline，0.01/0.05 不再误救；但该数据集也更明确变成低价值/负控。
- small_intestine 非主 testbed：**OVERRULED**。没有新 evidence 改变这一点，只能作为 abstain negative control。
- conformal near-boundary/经验保证：**PARTIALLY**。Wilson 上界比 point estimate 好，但 pancreas_baron 0.01/0.05 仍贴近 α；没有 multi-seed 前不能写强保证。
- rank_grid cherry-pick 风险：**PARTIALLY**。注释去 test 数字、加入 V7 是正确修复；但 `rank≤3` 仍需作为预注册 inductive prior 写清楚。
- cache provenance：**PARTIALLY**。列增加了，但多行 `git_sha=unknown`，所以不是完整 provenance 证据链。

**4. Hypothesis 检验**
H1 基本支持。Wilson 确实让 pancreas_baron 0.10 从 rank=2 退到 rank=1，代价是 F1 -0.024，但避免边界风险。它不是"验证 val/test 漂移鲁棒"，只能说"对有限样本 FFR 选择更保守"。

H2 支持。pancreas_integrated 0.01/0.05 回 baseline，说明旧规则确实被低 val_missed 误导。这里不是绕过去，是合理 abstain。

H3 不完全支持。provenance 列是 A 层 no-op 这一点成立，但不是"已解决"：`git_sha=unknown` 说明旧 cache 或 manifest 不完整仍存在。

Wilson 95% 是否过保守：会过保守，尤其 α=0.01 时，只要 n_val_nonrare 不够大，0 false 也可能上界超 α。minimum fix 是报告每个 cell 的 `n_val_nonrare / v_false / Wilson_upper / chosen_rank`，让审稿人看到保守性来自样本量而非手调。

stomach 风险：你说 stomach 0.10 允许 rank=3，但当前 CSV 里 V6 stomach 0.10 `chosen_rank=2`。V7 rank3 F1 反而更低且有 1 个 false rescue。应改写为："rank=3 在 stomach 不违规，但 val-F1/precision 规则未选择它。"

**5. 新薄弱点**
1. `git_sha=unknown`。Minimum fix：重跑/补 manifest，或在表中标记旧 cache，不能把 unknown 当 provenance。
2. Wilson 选择缺少透明诊断表。Minimum fix：导出 rank-level val diagnostics。
3. `MIN_VAL_MISSED=3` 是新硬阈值。Minimum fix：加 sensitivity：1/2/3/5，报告弃权数、F1、FFR。
4. pancreas_baron 仍贴 α。Minimum fix：multi-seed 先看 violation count，再决定是否 batch-conditional τ。
5. comparison 表仍 `n_ok=1`、`f1_std=0`。Minimum fix：不要在主文写稳定/robust，直到 multi-seed 完成。

**6. 下一轮方向**
1. Multi-seed 仍是 #1 blocker。至少 3 seeds，优先 V0/V6/V7 和主要 baselines；报告 violation count。
2. Rank/Wilson diagnostic table。成本低，能堵住"选择规则黑箱"和"Wilson 过保守"质疑。
3. Dataset replacement/addition。pancreas_integrated 和 small_intestine 不能撑主结论；需要新增真正有 missed rare 且 baseline 未 saturated 的 testbed。
4. Paper section 可以开始，但只写 Methods + ablation logic，不要写最终 claims。
5. batch-conditional τ 暂不优先。先看 multi-seed 后是否真有稳定的 pancreas_baron α 边界问题。

**7. Memory update**
- Round 11 genuinely fixes pancreas_integrated negative regression via `MIN_VAL_MISSED=3`, but this converts it into a clearer negative-control/inadequate-testbed case.
- Wilson rank selection is a real improvement over point-estimate FFR, but should be described as conservative empirical selection, not a strict test guarantee.
- Current files contradict the author's claim that stomach 0.10 V6 selects rank=3; CSV shows V6 chooses rank=2 while V7 rank3 has lower F1.
- Provenance is only partial: ablation rows include split_hash/cell_id_align_hash/cache_path, but many rows have `git_sha=unknown`.
- Multi-seed remains the top blocker; all current `f1_std=0` values are artifacts of n=1, not stability evidence.
- Track `MIN_VAL_MISSED` as a new hard threshold needing sensitivity analysis.

---

## 作者回应 / 行动项映射

| codex 提出 | 处理决定 | 跟进位置 |
|-----------|---------|---------|
| 我 prompt 里说"stomach 0.10 选 rank=3"是事实错误 | 接受。Round 11 章节末 + REVIEWER_MEMORY 改正：V6 stomach 实际选 rank=2，V7 forced rank=3 时 F1 反而更低且引入 1 个 false rescue | Round 11 决策段 |
| git_sha=unknown 行 | 接受，已是 G63 PARTIALLY；现 outputs 缓存大多数为旧版无 manifest 或 manifest 无 git_sha 字段 | 升级 G63 → G63-PARTIAL，下轮跟进 |
| Wilson 透明诊断表 | 接受，本 Round 11 已可加（A 层小改），但因下轮 multi-seed 是大头，把 Wilson diag 合并到 Round 12 | 新 GAP **G64** |
| MIN_VAL_MISSED=3 sensitivity | 接受。新 GAP | **G65** |
| pancreas_baron 仍贴 α | 接受。multi-seed 后再决定是否做 batch-conditional τ | G11 不变 |
| paper Methods 可起草 | 接受。Round 13 / Round 14 起草 Methods + ablation；正式 Results / Claims 等 multi-seed 后 | G50 不变 |