# Adaptive Separability Gate 初始实验结果

**日期**：2026-07-29  
**计划**：`refine-logs/EXPERIMENT_PLAN.md`  
**实现**：`tools/analysis/adaptive_separability_gate.py`

## 结论

Cross-fitted adaptive gate 在当前 cache-only 评估中通过了预设的收益与安全标准：相对固定 `S_min=1.3`，8数据集96个实验单元的平均 rare-cell F1 从 0.814654 提升到 0.851613，结果为 7 wins / 89 ties / 0 losses；最大 test incremental FPR 保持为 0.009768，未出现 `alpha=0.01` 违规。

该结果支持继续审计和补充稳健性实验，但在完成正式 experiment audit 前暂不替换主线 `conformal_rescue()`。

## M0：实现与Sanity — PASSED

- 新增独立cache-only实现，未修改 `src/rescue.py`。
- 14个相关测试通过，包括：低S安全放行、低support弃权、不安全OOF拒绝、确定性、test-feature不影响gate decision，以及原有conformal测试。
- Adaptive decision函数不接受test ground-truth参数；真实标签仅在runner末端计算最终指标。
- Human完整运行后写入 `policy_manifest.json`；mouse阶段验证脚本SHA和policy完全一致后才允许执行。

## M1：6-human低-S开发审计 — PASSED

8个 `S<1.3` 单元中：

| Variant | Mean F1 | Wins/Ties/Losses vs fixed | Max FFR | Violations |
|---|---:|---:|---:|---:|
| fixed_s1.3 | 0.683069 | 0/8/0 | 0.000000 | 0 |
| no_sep_gate | 0.844973 | 4/2/2 | 0.015263 | 2 |
| adaptive_sep_gate | 0.809345 | 2/6/0 | 0.002442 | 0 |

Adaptive仅放行Baron pancreas seed43的1%和5%配置，两者F1均从0.2449提升到0.7500，test FFR为0.002442。此前no-gate会产生FFR=0.015263的seed44 1%/5%配置被OOF audit拒绝；两个负增益配置同样被拒绝。

## M2：6-human完整开发集 — PASSED

| Variant | N | Mean F1 | Delta vs fixed | W/T/L vs fixed | Max FFR | Violations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_s1.3 | 72 | 0.887511 | 0.000000 | 0/72/0 | 0.009768 | 0 |
| no_sep_gate | 72 | 0.905501 | +0.017989 | 4/66/2 | 0.015263 | 2 |
| adaptive_sep_gate | 72 | 0.901542 | +0.014031 | 2/70/0 | 0.009768 | 0 |

Adaptive恢复了no-gate潜在F1收益的大部分，同时没有继承其两个安全违规和两个退化单元。

## M3：2-mouse冻结确认集 — PASSED

Human开发结束后冻结了脚本SHA和policy；mouse阶段在未修改规则的条件下运行。

| Variant | N | Mean F1 | Delta vs fixed | W/T/L vs fixed | Max FFR | Violations |
|---|---:|---:|---:|---:|---:|---:|
| fixed_s1.3 | 24 | 0.596083 | 0.000000 | 0/24/0 | 0.001520 | 0 |
| no_sep_gate | 24 | 0.702189 | +0.106106 | 6/18/0 | 0.002280 | 0 |
| adaptive_sep_gate | 24 | 0.701827 | +0.105744 | 5/19/0 | 0.002280 | 0 |

Adaptive在mouse上获得5 wins / 19 ties / 0 losses。5个被安全放行的配置均来自mouse lung：

- seed42, rts=0.05：F1 0.000 → 0.548；
- seed43, rts=0.01：F1 0.000 → 0.435；
- seed43, rts=0.05：F1 0.000 → 0.595；
- seed43, rts=0.10：F1 0.000 → 0.744；
- seed44, rts=0.10：F1 0.522 → 0.738。

## 8-dataset汇总

| Region | Variant | N | Mean F1 | Delta vs fixed | W/T/L vs fixed | Max FFR | Violations |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL | fixed_s1.3 | 96 | 0.814654 | 0.000000 | 0/96/0 | 0.009768 | 0 |
| ALL | no_sep_gate | 96 | 0.854673 | +0.040019 | 10/84/2 | 0.015263 | 2 |
| ALL | adaptive_sep_gate | 96 | 0.851613 | +0.036959 | 7/89/0 | 0.009768 | 0 |
| SCARCE | fixed_s1.3 | 72 | 0.784417 | 0.000000 | 0/72/0 | 0.009768 | 0 |
| SCARCE | no_sep_gate | 72 | 0.837837 | +0.053421 | 9/62/1 | 0.015263 | 2 |
| SCARCE | adaptive_sep_gate | 72 | 0.833695 | +0.049279 | 7/65/0 | 0.009768 | 0 |

## Success Criteria

- [x] 相对fixed-1.3提高平均F1。
- [x] 至少恢复一个安全低-S正增益单元。
- [x] 无新增test FFR violation。
- [x] 最大test incremental FPR不超过0.01。
- [x] Human开发中已知不安全和负增益单元被拒绝。
- [x] Frozen mouse confirmation中无负增益、无安全违规。

## 限制与下一步

1. 增益集中于Baron pancreas和mouse lung，共7个实验单元；其余89个单元按设计保持不变。
2. Human数据属于development evidence，因为方法构思已受到既有human low-S结果启发；mouse才是规则冻结后的确认集。
3. 尚未进行fold数/CI规则敏感性与cell-stratified补充实验。
4. 尚未执行独立experiment integrity audit，因此暂不修改主线方法和论文主结果。
5. 若正式审计通过，再决定将其命名为 `Validation-Adaptive Separability Gate` 并加入主方法；否则保留固定1.3，将结果作为方法扩展或附录。

## 产物

- Human：`results/adaptive_separability_gate/v1/human_{run_level,summary,decision_audit}.csv`
- Mouse：`results/adaptive_separability_gate/v1/mouse_{run_level,summary,decision_audit}.csv`
- Frozen policy：`results/adaptive_separability_gate/v1/policy_manifest.json`
- Tests：`tests/test_adaptive_separability_gate.py`

