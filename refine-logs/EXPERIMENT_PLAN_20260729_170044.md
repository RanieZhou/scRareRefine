# Adaptive Separability Gate 实验计划

**问题**：固定 `S_min=1.3` 在部分数据上过于保守，但直接降低阈值会提高 F1 的同时引入 FFR 违规。

**方法主张**：使用只依赖 train/validation 的 cross-fitted safety audit，在低 separability 情况下选择性放宽 gate，从而恢复安全的有效 rescue，同时保持 test incremental FPR 不超过 `alpha=0.01`。

**日期**：2026-07-29

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：Adaptive gate 能恢复固定 1.3 的部分错误弃权，而不破坏 FFR 控制 | 这是修改方法的唯一合理理由 | 相对 fixed-1.3，F1 提高；test FFR max ≤ 0.01；FFR violation 不增加 | B1, B2 |
| C2：决策来自 validation 证据，而非 per-dataset test tuning | 防止 test leakage 和 benchmark-specific 调参质疑 | 所有 gate 规则预先冻结；test 标签仅用于最终指标；在未用于规则开发的 mouse 数据集上确认 | B1, B3 |
| Anti-claim：收益只是取消安全门后的激进 rescue | 区分 adaptive gate 与 `no-sep-gate` | Adaptive 的 FFR 显著优于 no-gate，并拒绝不安全的低-S配置 | B1, B2 |

## Paper Storyline

- Main paper must prove：adaptive gate 在安全预算内提高或保持 rare-cell F1。
- Appendix can support：fold 数、bootstrap 次数以及判定阈值的敏感性。
- Experiments intentionally cut：不在当前阶段训练 gate classifier，不使用 dataset ID，不根据 test F1 搜索每个数据集的最优阈值，也不立即迁移到 TOSICA。

## Frozen Adaptive Rule

当前每次运行的 separability statistic `S` 已由 train 自动计算；自适应对象是是否放行 rescue，而不是重新拟合 `S`。

1. 当 `S >= 1.3`：完全复用当前 `conformal_rescue()`，不改变主线行为。
2. 当 `S < 1.3`：在 validation 上做 5-fold stratified cross-fitting。
3. 每个 fold：
   - 使用其余 folds 校准 conformal `tau`、necessity gate 与 adaptive rank；
   - 暂时关闭 separability gate；
   - 对 held-out fold 产生 out-of-fold prediction。
4. 汇总全部 OOF prediction，并计算：
   - `delta_f1_oof = F1(rescue_oof) - F1(baseline_oof)`；
   - incremental FFR 及其 Wilson 95% upper bound；
   - paired stratified multinomial bootstrap 的 `delta_f1` 单侧 95% lower bound。
5. 仅当以下条件全部满足时放宽 gate：
   - validation 总漏判稀有细胞数 `>= MIN_VAL_MISSED`；
   - 至少 3 个 folds 成功执行非弃权 rescue；
   - `Wilson_UCB(FFR_oof) <= 0.01`；
   - `LCB_95(delta_f1_oof) > 0`。
6. 放行后，在完整 validation 上重新校准 tau/rank（仍只在该步骤关闭 sep gate），再应用于 test；否则保留 backbone prediction。

固定参数：`n_splits=5`、`bootstrap_reps=2000`、`bootstrap_alpha=0.05`、`Wilson z=1.96`。若类别支持不足以完成 5-fold，则自动降低 fold 数；实际 folds 少于 3 时弃权。

## Experiment Blocks

### B0：实现与无泄漏单元测试

- Claim tested：实现严格遵循 train/validation/test 隔离。
- Why this block exists：adaptive gate 比固定 gate 多一层模型选择，最容易发生 validation/test 混用。
- Dataset / split / task：合成小数组，不读取真实 test 标签做决策。
- Compared systems：fixed、no-gate、adaptive。
- Metrics：gate decision、OOF FFR UCB、OOF delta-F1 LCB、输出预测一致性。
- Success criterion：改变 test label 不改变 gate decision；相同输入重复运行结果一致；低支持时安全弃权。
- Table / figure target：仅测试日志。
- Priority：MUST-RUN。

### B1：6-human 开发诊断

- Claim tested：adaptive rule 是否能区分已知的安全低-S rescue 与不安全/无收益 rescue。
- Dataset / split / task：6 human、batch-heldout、3 seeds、4 rare-label budgets，共 72 units；复用现有 embeddings，不重训 backbone。
- Compared systems：`fixed_s1.3`、`no_sep_gate`、`adaptive_sep_gate`。
- Metrics：rare F1/recall/precision、delta F1、test incremental FPR、FFR violations、abstention、wins/ties/losses。
- Setup details：所有规则在运行前按上节冻结；结果用于诊断，不再据此修改规则后宣称 confirmatory。
- Success criterion：adaptive 至少恢复一个 fixed 弃权的安全正增益 unit；相对 fixed 不增加 test FFR violation；不得出现明显 F1 regression。
- Failure interpretation：若没有配置通过，validation证据不足；若通过但test违规，说明cross-fitted validation仍不能抵御batch shift，应保留固定1.3。
- Table / figure target：方法开发表和 decision audit heatmap。
- Priority：MUST-RUN。

### B2：低-S decision audit

- Claim tested：adaptive gate 的每次放行都有可审计的 validation 证据。
- Dataset / split / task：仅 `S<1.3` units。
- Compared systems：fixed/no-gate/adaptive，并列出 OOF evidence。
- Metrics：S、val_missed、active folds、OOF delta F1、delta-F1 LCB、OOF FFR、Wilson UCB、test delta F1、test FFR。
- Success criterion：安全正增益配置被放行；已知违规或负增益配置被拒绝。
- Failure interpretation：如果相同 OOF 证据对应相反 test 结果，说明仅靠validation不能可靠自适应，应停止替换主方法。
- Table / figure target：主文或附录 decision table。
- Priority：MUST-RUN。

### B3：冻结规则后的独立确认

- Claim tested：adaptive rule 不是针对6-human结果调出的。
- Dataset / split / task：2 mouse TMS、batch-heldout、3 seeds、4 budgets，共 24 units；其次可在6-human cell-stratified seed42上做补充。
- Compared systems：同 B1。
- Metrics：同 B1，重点看首次出现的低-S units及安全性。
- Setup details：运行前保存脚本 SHA256 和参数 manifest；看结果后不改规则。
- Success criterion：mouse 上不增加任何FFR violation；若存在低-S配置，至少不劣于fixed。
- Failure interpretation：若mouse违规，adaptive规则不能作为主方法，只能作为负结果/未来工作。
- Table / figure target：confirmatory table。
- Priority：MUST-RUN。

### B4：判定规则敏感性

- Claim tested：结果不依赖单个 fold/CI 选择。
- Dataset / split / task：仅在 B1-B3 通过后进行。
- Compared systems：`n_splits={3,5}`，bootstrap reps `{500,2000}`，LCB criterion `{>0, >=0}`。
- Metrics：decision stability、F1、FFR violations。
- Success criterion：默认规则附近结论稳定。
- Failure interpretation：若决策高度敏感，则方法复杂度和不确定性不值得替换固定gate。
- Table / figure target：appendix。
- Priority：NICE-TO-HAVE。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 实现与测试 | 合成单元测试 | 无test leakage、确定性、低支持弃权 | CPU，<10 min | double-dipping；用OOF测试防护 |
| M1 | 快速低-S审计 | 6-human中 `S<1.3` units | adaptive必须拒绝已知不安全unit | CPU，<10 min | 规则过严导致全弃权 |
| M2 | 6-human完整比较 | 72 units × 3 variants | F1不下降且违规不增加 | CPU，约10–30 min | batch shift使validation失效 |
| M3 | mouse确认 | 24 units × 3 variants | 不增加FFR violation | CPU，约5–15 min | mouse没有低-S unit，无法验证增益 |
| M4 | 图表与敏感性 | 通过后再做 | 结论在邻近规则下稳定 | CPU，<30 min | 过多调参；仅作为附录 |

## Compute and Data Budget

- Total estimated GPU-hours：0；全部复用缓存 embeddings。
- CPU time：预计 30–60 分钟，主要来自 cross-fitting 和 bootstrap。
- Data preparation needs：无新增数据；需要 train/validation/test predictions 与 latent caches。
- Biggest bottleneck：低标注情况下 validation rare support 较少，可能使CI过宽并导致保守弃权。

## Risks and Mitigations

- 风险：使用同一validation同时校准tau/rank和选择gate。
  - 缓解：fold内校准、fold外审计，使用完整OOF预测做最终gate decision。
- 风险：开发方案已受到6-human test结果启发。
  - 缓解：6-human明确标记为development；在查看adaptive mouse结果前冻结实现和SHA。
- 风险：F1 bootstrap受大量nonrare细胞主导。
  - 缓解：rare/nonrare分层的paired multinomial bootstrap。
- 风险：adaptive提高F1但破坏安全主张。
  - 缓解：任何新增FFR violation均判定失败，不替换fixed-1.3主线。
- 风险：方法复杂度超过收益。
  - 缓解：若只恢复极少unit或增益很小，保留fixed gate，将adaptive作为未来工作。

## Final Checklist

- [x] Main paper claims are explicit
- [x] No test labels enter gate selection
- [x] Fixed/no-gate/adaptive are compared
- [x] Development and confirmatory datasets are separated
- [x] Safety failure has a hard stop criterion
- [x] Nice-to-have sensitivity runs are separated from must-run runs
