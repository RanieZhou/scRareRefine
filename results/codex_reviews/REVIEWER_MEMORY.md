# Reviewer Memory（codex 跨轮持久脑）

> 本文件按 [ITERATION_BOUNDARY.md §4.3](../../ITERATION_BOUNDARY.md) 维护：永远追加、绝不删除；每轮把 codex 响应里的 "Memory update" 段原文 copy 进来。
> 调用约定：`mcp__codex__codex` (or `codex-reply`) + `model: gpt-5.5` + `config: {"model_reasoning_effort": "xhigh"}`。

---

## Round 0 — 初始化（2026-06-19）

无历史记录。codex 第 1 轮将作为冷启动评审；其 Memory update 会在 Round 1 后追加到本文件。

**项目状态（提交给 Round 1 codex 的事实集）**
- 任务：scRNA-seq 稀有细胞类型识别的 post-hoc refinement
- 基线：scANVI（半监督）；指标：rare F1 / rare recall + FFR ≤ α=0.01
- 当前主方法：conformal_rescue（separability 安全网 + necessity 守门 + val-自适应候选 rank ∈ {1,2} + conformal τ）
- 当前 benchmark：6 数据集（immune_dc / pancreas_baron / pancreas_integrated / tabula_lung_endo / tabula_sapiens_stomach / tabula_small_intestine），seed=42，4 个 rare_train_size {0.01, 0.05, 0.10, all}
- 第九轮成绩：标注稀缺区 (rts 0.01/0.05/0.10) 15/15 胜过多数方法；零回归
- 目标：生信二区

---

## Round 1 — Score: 6.8/10 — Verdict: almost (2026-06-19)

**Codex thread**: `019edb7d-fede-7622-9b61-e2b66beb4439`
**Submitted**: Round 10 ablation (G03) + dataset adequacy
**Full review**: [round01_review.md](round01_review.md)

### Memory update（codex 原文 verbatim）

- Round 10 strongest evidence is adaptive rank: rank=2 helps pancreas_baron/stomach, rank=1 protects immune/lung_endo.
- Necessity guard evidence is real but mostly negative-control/saturated-regime evidence, not broad performance evidence.
- pancreas_integrated is not merely uninformative: V6 regresses vs V0 on 0.01/0.05 because test baseline is saturated while validation suggests missed rare cells.
- small_intestine should be tracked as abstain-necessity negative control, not a primary rescue benchmark.
- Conformal τ evidence is empirical and near-boundary on pancreas_baron; avoid claiming strict FFR guarantee without CI/multi-seed.
- Watch benchmark-overfitting risk around rank_grid={1,2}, especially any rationale derived from prior test FFR for rank=3.
- Future reviews should check cache provenance and cell-ID alignment before trusting ablation rows.

### Unresolved（Round 1 提出，等后续轮跟踪）

1. **pancreas_integrated 负回归** — Round 11 处理（G60）
2. **conformal claim 经验化重写 + binomial CI** — G61
3. **rank_grid 文档脱钩 test 信息**（潜在 R1） — G62
4. **ablation 行级 cache provenance** — G63
5. **multi-seed 稳定性**（codex 把它列为 #1 blocker，提早 G01 优先级）

### Patterns to track

- 任何方法层声称的 "guarantee" / "上界" 必须 codex 检查是否被 test 经验数据污染过
- 数据集层"degenerate"标签必须区分：abstain-by-design（OK）vs split-shift failure（要 fix）
- 每个新机制必须 ablation 测能否被去除而不损失，避免堆 trick


---

## Round 2 — Score: 7.2/10 (↑0.4) — Verdict: almost (2026-06-19)

**Codex thread**: `019edb7d-...` (continued)
**Submitted**: Round 11 — G62 + G60 + G63 fixes (rank_grid脱钩 + Wilson 上界 + split-shift guard + provenance 列)
**Full review**: [round02_review.md](round02_review.md)

### Memory update（codex 原文 verbatim）

- Round 11 genuinely fixes pancreas_integrated negative regression via `MIN_VAL_MISSED=3`, but this converts it into a clearer negative-control/inadequate-testbed case.
- Wilson rank selection is a real improvement over point-estimate FFR, but should be described as conservative empirical selection, not a strict test guarantee.
- Current files contradict the author's claim that stomach 0.10 V6 selects rank=3; CSV shows V6 chooses rank=2 while V7 rank3 has lower F1.
- Provenance is only partial: ablation rows include split_hash/cell_id_align_hash/cache_path, but many rows have `git_sha=unknown`.
- Multi-seed remains the top blocker; all current `f1_std=0` values are artifacts of n=1, not stability evidence.
- Track `MIN_VAL_MISSED` as a new hard threshold needing sensitivity analysis.

### Suspicion rulings（codex 对 Round 1 怀疑的裁定）

1. Adaptive rank: **SUSTAINED** ✓
2. Necessity 是 safety 证据: **PARTIALLY** （split-shift 拓展是机制层升级，但仍偏 safety）
3. pancreas_integrated 负回归: **SUSTAINED** ✓
4. small_intestine 非主 testbed: **OVERRULED** （无新 evidence 改变这一点）
5. conformal near-boundary: **PARTIALLY** （Wilson 改进，但 multi-seed 前不能写强保证）
6. rank_grid cherry-pick 风险: **PARTIALLY** （注释脱钩 + V7 sensitivity，但需写预注册 prior）
7. cache provenance: **PARTIALLY** （列加了但很多 git_sha=unknown）

### Unresolved（Round 2 提出，等后续轮跟踪）

1. **Multi-seed**（#1 blocker，仍 unresolved；用户允许前期单 seed，但 Round 12-13 必须收尾）
2. **Wilson 选择透明诊断表** — G64
3. **MIN_VAL_MISSED sensitivity (1/2/3/5)** — G65
4. **git_sha=unknown 行** — G63 升级
5. **comparison 表 n_ok=1 / f1_std=0** — 待 multi-seed 后才能写"stable/robust"

### Author errata（codex 抓到的我方失误）

- Round 11 prompt 中写 "stomach 0.10 仍允许 rank=3"，实际 CSV 显示 V6 选 rank=2；V7 forced rank=3 时 F1 反而低。下次提交 codex 前必须本地对照 CSV 再发送。

### Patterns to track

- 任何"选择规则"必须导出 row-level 诊断表，否则 reviewer 会怀疑黑箱
- 新引入的硬阈值必须配 sensitivity sweep（MIN_VAL_MISSED 现是案例）
- 作者对 csv 的口头描述必须与 csv 字段精确一致，codex 会真去读


---

## Round 3 — 无 codex 调用（A 层 consolidation, 2026-06-19）

**Submitted**: 等下轮 multi-seed 完成后一并提交
**项目变化**：
- G64 关闭（Wilson 诊断表落盘，72 行；0 cells 选 rank=3 验证选择规则）
- G65 关闭（k=3 数据驱动最小有效；k=1/2 仍有 pancreas_integrated 回归）
- G63 真闭环（96 行 legacy_pre_git_sha_recording + 96 行 current sha；不再有 unknown）

**未变**：
- 主表 F1 / FFR 数字完全不变（纯 A 层）
- 仍单 seed=42

**下次外审时贴的事实集补充**：
- Wilson 诊断表可作为 "rank 选择规则透明" 的直接证据
- k=3 的 sensitivity 表可作为 "MIN_VAL_MISSED 不是 cherry-pick" 的直接证据
- 还剩 5 个未解决：multi-seed / dataset replacement / paper Methods / pancreas_baron α 边界 / failure modes section

