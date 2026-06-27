# 迭代边界约束（ITERATION_BOUNDARY）

> **本文档是 Claude Code 每次启动新一轮迭代前的必读文件。**
> 目的是把"凭直觉调参 / 加魔法常数 / 越改越偏"的幻觉风险，约束在一个由证据驱动、可回滚、可外审的迭代框架内。

---

## 0. 总目标（不可被迭代过程中改动）

将 scRareRefine 推到**生信二区**（Bioinformatics / Briefings in Bioinformatics / NAR Genomics & Bioinformatics / iScience / Cell Reports Methods 量级）可投稿状态。

衡量标准是「**证据 + 同行可复现**」，不是「指标数字越高越好」。一个诚实呈现局限的 0.78 F1 数据集，比一个 cherry-picked 的 0.95 更接近发表。

---

## 1. 不可逾越的红线（红线一旦触碰必须立即回滚）

以下任何一条被违反 → **本轮迭代作废 + 在 `results/experiment_log.md` 记录"触红线"事件 + 必须回到上一稳定 commit**：

### R1. Test-label 泄漏

- 训练集原型、HVG 选择、marker signature 只能用训练集
- conformal τ、val-自适应 rank、所有阈值只能在 validation 选取
- **test 标签仅用于最终 metrics 计算，不得参与任何调参 / 阈值选择 / 模型选择**
- 跨数据集时不得用 dataset A 的 test 标签调 dataset B 的参数

### R2. 数据集相关魔法常数

- 任何新增的常量必须**二选一**：
  - **(a)** 可在 validation 上自动选取（写明搜索网格 + 选择准则）
  - **(b)** 是**跨数据集固定的先验**（写明 docstring 中的理由，引用诊断证据）
- 禁止「这个数据集 sep 阈值改 1.5、那个改 1.1」式的 per-dataset tuning

### R3. 伪造 / 美化结果

- 不得为达指标删除 seed / 数据集 / rare class
- 不得只 report 单 seed 时却写成「multi-seed」
- 不得用「最大可用 unlabeled set」之类隐性放宽 inductive 约束
- comparison_summary.csv 中 scRareRefine 的行必须与其他 8 方法**同一份 split + 同一份 embeddings 缓存**产出

### R4. 改实验设置不记录

- 随机种子、split_mode、rare_class 定义、label/batch 列、预处理流程、baseline 设置、conformal α、`CONFORMAL_LOW_SEP`、`CONFORMAL_RANK_GRID` 的任何改动 → 必须在 `results/experiment_log.md` 当轮章节写明「为什么改 + 旧值 → 新值 + 对历史结果的影响」

### R5. 论文主张越界

不可写：解决了稀有细胞识别问题、全面优于所有方法、SOTA、临床可用、generalizes to all single-cell data。
可写（须有结果支持）：在评估的 N 个数据集上、在 rare_train_size ∈ {...} 区间内，scRareRefine 相对 baseline 提升 rare F1 X.XX，FFR 受控于 α。

### R6. 不修改原始数据

`data/raw/`、原始 `.h5ad` 文件、Tabula Sapiens 下载产物——只读。任何子集抽取必须在 `tools/extract/` 中以脚本形式落盘 + 输入文件 hash 留存。

---

## 2. 迭代触发条件（无证据不开新轮）

**开启新一轮迭代前**，必须在当轮 experiment_log 章节顶部回答以下三问：

| 必答项 | 通过标准 |
|--------|---------|
| **依据从哪来？** | 引用具体数字（comparison_summary_agg.csv 某行）/ 诊断输出（`tmp/diag_*.py`）/ 外审反馈（codex 第 N 次响应）/ 评审意见。**不接受**「我觉得 / 直觉上 / 应该可以」。 |
| **现有方法在该依据上的具体缺陷是什么？** | 一句话指明：哪个数据集、哪个 rts、哪个指标、和谁比、差多少。**不接受**抽象描述。 |
| **预期改动后达到的最低验收线是什么？** | 写明 falsifiable 的目标：例如「stomach rts=0.10 时 recall 从 0.59 → ≥0.70 且不引起其他 4 数据集任一格回归 >0.005」。**预设验收线必须在跑实验前确定**，避免事后挪门槛。 |

三问任一缺失 → 不开新轮，先补诊断。

---

## 3. 迭代层次（按风险升级）

每轮迭代必须明确声明属于 A / B / C 三层中的哪一层。层次越高，要求的证据越多。

### A 层 · 数值微调
- 例：调网格点、调 top_n_markers、调 val 搜索网格密度
- **要求**：val 上自洽 + 不破红线即可
- **不需要** codex 外审

### B 层 · 机制重构
- 例：新增闸门（如 necessity 守门）、新增 Rescuer 类、改 score 公式
- **要求**：
  1. 诊断脚本（落盘到 `tmp/`）证明现有机制具体在哪个数据集失效、为什么
  2. 新机制的理论动机写进对应函数 docstring
  3. **跑完实验后必须调用 codex 外审**（见 §4）
  4. 至少 1 个数据集明显改进 + 0 数据集回归 > 0.005

### C 层 · 范式切换
- 例：换骨干（不用 scANVI 用 scVI / scGPT / scFoundation）、换框架（监督 → 自监督 / RL）
- **要求**：
  1. 必须先证明 B 层已被穷尽（列出近 3 轮所有 B 层尝试及失败原因）
  2. **改动前**就要 codex 外审过一次（避免大改后才发现方向错）
  3. 改动后必须 codex 外审第二次
  4. 必须保留旧 baseline 可复现（旧代码 commit hash 在 log 中存档）

---

## 4. 外部审阅专家：codex MCP

调用方式：**codex MCP** (`mcp__codex__codex` / `mcp__codex__codex-reply`)。本项目沿用 [auto-review-loop skill](~/.claude/skills/auto-review-loop/SKILL.md) 的 **hard 模式**（Reviewer Memory + 可选 Debate Protocol）。

### 4.0 固定调用配置（不许改）

```yaml
mcp__codex__codex:
  model: gpt-5.5                                  # codex MCP 默认 reviewer 模型
  config: {"model_reasoning_effort": "xhigh"}     # 最深推理档；codex 自报 effort 不可靠，以本字段为准
  sandbox: read-only                              # 评审不允许写文件
  approval-policy: never                          # 非交互
  prompt: |
    [按 §4.2 模板填]
```

第 2 轮起改用 `mcp__codex__codex-reply` + 上一轮保留的 `threadId`（保持记忆 / 对话上下文）。

### 4.1 必须调用 codex 的时机

| 触发点 | 提交给 codex 的内容 |
|--------|--------------------|
| B 层改动后 | 改动 diff + 该轮诊断脚本输出 + 实验结果表 + 你的 hypothesis & 解释。**让 codex 找 hypothesis 与结果是否真的一致、有没有 confirmation bias**。 |
| C 层改动前 | 当前方法 1 段描述 + 待选的 C 层方向（≥2 个）+ 每个方向的预期成本 / 风险。**让 codex 从二区审稿人视角 rank 这些方向**。 |
| C 层改动后 | 改动后完整实验 + 同 B 层 |
| 准备写论文 section | 该 section 的所有数字 + 来源 csv 路径。**让 codex 从审稿人视角找漏洞**：是否有 cherry-picking、是否有未交代的局限、claim 是否被数据真的支持。 |
| 实验结果意外 | 例如某改动在 4 数据集都涨但 1 数据集大跌。**先让 codex 提假设，再针对性诊断**，避免你自己想当然解释。 |

### 4.2 固定 prompt 模板（hard 模式 · 含 Reviewer Memory）

每轮调用前，先把 `results/codex_reviews/REVIEWER_MEMORY.md` 的**全文**贴进 prompt 顶部。这是 codex 跨轮的持久"大脑"，让它能追踪你是否真的解决了上一轮的疑点，而不是绕过去。

```text
[Round N / MAX_ROUNDS — scRareRefine 二区评审循环]

## Your Reviewer Memory（持久跨轮）
[贴 results/codex_reviews/REVIEWER_MEMORY.md 全文]

IMPORTANT: 你有上一轮留下的记忆。请核查你之前的怀疑究竟是被真正解决，
还是被作者绕过去了。作者（Claude）控制你看到什么——对"方便的省略"保持警惕。

## 研究背景（事实，不要自行补全 / 想象）
- 任务：scRNA-seq 稀有细胞类型识别的 post-hoc refinement
- 基线：scANVI（半监督）
- 指标：rare F1 / rare recall + FFR ≤ α=0.01
- 数据集：[N 个具体数据集 + 稀有类]
- Inductive 约束：仅 train + val 选参，绝不接触 test 标签
- 目标投稿区间：生信二区（Bioinformatics / BIB / NAR-GAB / iScience / Cell Reports Methods）

## 本轮改动
[改动层次 A/B/C；改动 diff 摘要；相关 docstring / 理论动机]

## 实验结果（完整表，含不利数据）
[贴 comparison_summary_agg.csv 相关行 + 与上一轮的对照]
[贴 FFR、recall、precision 等次级指标，不只 F1]

## 我的解释（请你审查是否过度解释 / confirmation bias）
[本轮 hypothesis 与因果链]

## 请按以下结构回答（不要恭维；如方向错就直说）

1. **Score**：1-10，针对生信二区接收线
2. **Verdict**：ready / almost / not ready
3. **Hypothesis 检验**：我的 hypothesis 是否真的被结果支持？是否存在 confirmation bias？
4. **遗漏检查**：有没有该报但我没报的实验 / 局限 / 不利结果？
5. **薄弱点（按严重度排序）**：本轮工作最大薄弱点在哪？每条给 minimum fix。
6. **下一轮方向（≥3 个）**：按"距离二区接收的边际贡献 / 实施成本"排序。
7. **Memory update**：列出本轮新增怀疑、未解决疑点、想跨轮追踪的 pattern。
   （我会把它原文追加进 REVIEWER_MEMORY.md，下轮再贴给你。）
```

### 4.3 处理 codex 反馈的纪律

- **原文落盘**：每轮 codex 完整响应原文存到 `results/codex_reviews/round{N}_review.md`（verbatim，不裁剪 / 不改写）。
- **Reviewer Memory 维护**：`results/codex_reviews/REVIEWER_MEMORY.md` 永远追加、绝不删除前轮记录（审计链）。codex 响应里的 "Memory update" 段原文 copy 进去。
- **threadId 留档**：第 1 轮调用返回的 `threadId` 写到当轮 `experiment_log.md` 章节头，后续轮用 `codex-reply` 续。
- **不许选择性引用**：即使 codex 否定了你最想做的方向，也必须记录原文 + 你为什么仍要做（或不做）的理由。
- **未补上的局限**：codex 指出的薄弱点若本轮未处理 → 必须新增进 §5 GAP 清单，标记 "from codex round N"。

### 4.4 可选：Debate Protocol（仅当 codex 明显误判时启用）

若 codex 某条 criticism 基于**误读代码 / 误解结果 / 引用错指标**，可走辩论流程（每轮最多 3 条 rebuttal）：

```text
[贴本轮 codex 原响应中你想反驳的 weakness]

作者 rebuttal：
- Accept / Partially Accept / Reject
- 论据：[为什么这个 criticism 无效 / 已解决 / 基于误解]
- 证据：[指到具体代码行、结果文件、之前轮的 fix]

请对每条 rebuttal 裁定：
- SUSTAINED（作者论据成立，撤销该条 weakness）
- OVERRULED（原 criticism 仍然成立，说明理由）
- PARTIALLY SUSTAINED（修订为更窄范围）

裁定后请更新 score 与 memory。
```

规则：
- rebuttal 必须诚实，不许伪造证据 / 曲解结果
- 每轮最多反驳 3 条（挑最关键的）
- 完整 debate 原文也要进 `round{N}_review.md`

---

## 5. 当前到二区的 GAP 清单（迭代待办池）

这是**当前已识别的 gap**。每轮迭代必须从这里挑（或新增有依据的 gap），不允许凭空冒出方向。

> 命名规则：`G{ID}-{layer}-{topic}`，已完成的 gap 在 experiment_log 当轮章节注明 `closes: G##`。
> codex 评审若指出本清单未覆盖的 gap，必须以 `G##-{layer}-{topic}  *(from codex round N)*` 形式新增条目，不许私下消化。

### 5.1 实验完整性（多为 A / B 层，相对容易补）

- **G01-A-multiseed**：当前正式对比只有 seed=42。二区要求 ≥3 seed + mean ± std。补 seed=43, 44 跑全部 9 方法 × 5 数据集 × 4 rts。
- **G02-A-statest**：缺统计显著性检验（paired Wilcoxon 或 bootstrap CI）。
- **G03-A-ablation**：`results/ablation/` 已被清空。系统化 ablation：去掉 separability 闸门 / necessity 闸门 / 自适应 rank / 改 α / 改 LOW_SEP，记录每个组件贡献。
- **G04-B-baseline-fairness**：检查每个 baseline 是否在该方法的"合理设置"下运行（默认超参 vs 调优过的超参的 fairness 问题）。
- **G05-A-runtime**：补 wall-clock + peak RAM + GPU mem 对比表。

### 5.2 方法稳健性（多为 B 层）

- **G10-B-stomach-recall**：stomach recall 卡在 0.59，mast cell 与多数类几何纠缠在 rank ≥ 3。诊断后判断是 (a) prototype 几何天花板（接受）还是 (b) 用 expression-side 信号能救（如 marker / 自定义距离）。
- **G11-B-pancreas-ffr**：pancreas rank=2 时 test FFR=0.0098 逼近 α=0.01。是 batch shift 问题，需要 split-aware 校准或 batch-conditional τ。
- **G12-B-rts-zero**：rts=0 / rts=1 标注（极端情形）下的退化分析。
- **G13-C-foundation-model-init**：用 scGPT / scFoundation 替换 scANVI 作 backbone 的 latent，看 conformal 是否在 foundation model 之上仍有价值。

### 5.3 理论 / 表述（B 层）

- **G20-B-conformal-coverage**：写一个简短的 coverage proof / 引用现成 conformal 文献，说明在 exchangeability 假设下 FFR ≤ α 的成立条件、以及 batch shift 时假设的破坏程度。
- **G21-B-separability-justify**：`CONFORMAL_LOW_SEP=1.3` 现在是经验值。需补一个跨数据集 sep vs rescue 收益的 scatter，说明 1.3 不是 cherry-picked。
- **G22-B-necessity-formalize**：necessity 守门用 val rare recall==1.0 触发，是硬阈值。考虑 soft 版本（val rare recall ≥ 1 - δ）+ δ 选择准则。

### 5.4 数据集广度（A 层）

- **G30-A-more-datasets**：5 数据集偏少，二区典型 6-10。候选：HLCA、Tabula Muris、Heart Cell Atlas、Liver Cell Atlas 中的稀有类。
- **G31-A-cross-species**：补一个小鼠数据集证明方法不仅限于人。
- **G32-A-disease-state**：补 disease vs healthy 数据集，看 rescue 在病理状态下是否仍 inductive。

### 5.5 可视化与可读性（A 层）

- **G40-A-umap-systematic**：当前只有 immune + pancreas 两张 UMAP 对照。补 stomach / endo / small_intestine 的 UMAP rescue 面板。
- **G41-A-figure-quality**：所有 figure 检查：300 dpi PNG + 矢量 PDF + 一致字号 + 黑边 marker + σ 误差带。

### 5.6 写作（B 层）

- **G50-B-paper-structure**：从 experiment_log.md 出发，先用 `paper-plan` skill 出大纲，再用 `paper-write` 起草各 section。
- **G51-B-failure-modes**：写一个独立的 "Failure modes & limitations" 节，包含 stomach recall ceiling、pancreas FFR pressure、rts=all 时 necessity 弃权的语义。**这是诚实的二区论文必备**。

### 5.7 codex Round 1 新增 GAP（2026-06-19）

- **G60-B-split-shift-guard**  *(from codex round 1)* — pancreas_integrated 在 rts=0.01/0.05 上 V6 vs V0 真回归（1.000→0.939, 1.000→0.979）：val 上 baseline 漏判稀有（recall<1.0）但 test 上 baseline 已 saturated。necessity 守门基于 val，对此 split shift 失效。最小修：加 val rare support / val missed-rare 的最小支持数守门，或让 "k=0 abstain" 进入 val 自适应 rank 选择。
- **G61-A-conformal-empirical-CI**  *(from codex round 1)* — 当前文档把 conformal τ 写成 "FFR ≤ α 保证"，但 pancreas_baron V6 FFR=0.0098 贴边 α=0.01。改写为 "empirical FFR control under val calibration"，所有 FFR 数字补 Wilson / binomial CI；说明 α 是固定先验、非调参。
- **G62-A-rank-grid-leakage**  *(from codex round 1)* — [src/rescue.py:512-515](src/rescue.py:512) 的注释含 "离线验证 rank=3 在 batch_heldout 的 val/test 漂移下 test FFR 会冲破 alpha" → **潜在 R1 违规**（test 信息回流到 design）。改：(a) 移除注释中对 test FFR 的引用，仅以 val 数据论证；或 (b) 把 rank=3 加入 grid，由 val 自适应自动选择；并跑 rank=3 sensitivity 表明它在 val 上就被淘汰。
- ~~**G63-A-cache-provenance**~~ *(closed Round 12)* — legacy 行透明标记为 `legacy_pre_git_sha_recording`，新行有 current sha。
- ~~**G64-A-wilson-diagnostic**~~ *(closed Round 12)* — [wilson_diagnostics.csv](results/ablation/diagnostics_round12/wilson_diagnostics.csv) 落盘。
- ~~**G65-A-min-val-missed-sensitivity**~~ *(closed Round 12)* — [min_val_missed_sensitivity_agg.csv](results/ablation/diagnostics_round12/min_val_missed_sensitivity_agg.csv)：k=3 是最小消除 pancreas_integrated 回归的阈值。

---

## 6. 单轮迭代标准流程

```text
[1] 读本文档 + experiment_log.md 最近一轮章节 + MEMORY.md
[2] 从 §5 GAP 清单选 1 个 gap（或新增有依据的）→ 声明层次 A/B/C
[3] 写本轮章节起点（在 experiment_log.md 新建一轮）：
     - closes: G##
     - 依据 / 缺陷 / 验收线（§2 三问）
     - hypothesis（可证伪）
[4] 写代码 / 改 config / 跑诊断 → 验证 inductive 合规
[5] 跑实验（必要时跑 seed=43,44 验证稳定性）
[6] 写本轮章节中段：实验结果（包括不利数据）
[7] B/C 层必须调 codex 外审 → 落盘 codex_reviews/round{N}_review.md
[8] 写本轮章节末段：
     - 决策（保留 / 回滚 / 部分采纳）
     - 局限（诚实写）
     - 触发的新 gap（加入 §5）
[9] 若改了任何 §1 R4 列出的设置 → 写明影响
[10] 提交（仅在用户确认时）
```

---

## 7. 回滚标准（出现以下任一即回滚）

- val 大涨但 test 不动或回退（典型过拟合 val）
- 1 数据集大涨但 ≥2 数据集回归 > 0.005
- 引入了 §1 R2 禁止的 per-dataset 魔法常数
- codex 外审明确指出方向有误且你无法反驳
- 触发任一 §1 红线
- 改动后无法解释为什么涨了（结果对但机制黑箱）→ 至少不能基于此发论文，先诊断清楚

---

## 8. 元规则（关于本文档本身）

- 本文档可以被改，但**只能在两种情况下改**：
  - (a) 用户明确要求
  - (b) 一个 gap 关闭后，把对应条目从 §5 移到 `results/gap_closed.md` 留档（不算"改约束"，算"维护清单"）
- 红线 §1 不允许在没有用户批准的情况下被弱化
- 任何对本文档的修改都要在 experiment_log 当轮章节简短记录
