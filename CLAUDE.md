# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

always reply in Chinese

## 项目目标

**scRareRefine** 基于 scANVI 的 latent embedding 与预测概率，为稀有细胞类型设计一个 post-hoc refinement 模块。当前正式方法是 **prototype 距离评分 + conformal 阈值校准 + Validation-Adaptive Separability Gate**，目标是在 validation OOF 风险约束下提升 rare cell type 的 F1，并在评估数据上维持 incremental FPR/FFR 不超过 1%。

实验环境（conda）：
- `scanvi311` — 主流水线（scANVI、scvi-tools）
- `sandbox310` — 部分对比方法（CellTypist / TOSICA / scBalance / ProtoCloud / HiCat / scCAD）

## 常用命令

```bash
# 安装依赖
pip install -e .[dev]

# 运行完整管道（rare_train_size 支持 float 比例、int 计数或 "all"）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05

# 强制重新训练（忽略 embeddings 缓存与 manifest 校验）
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --force

# 显式复现原固定 S=1.3 对照；8 个主配置默认使用 adaptive
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --separability_gate_mode fixed

# 冻结 v1 的 20-decision-seed 稳定性复现（CPU/cache-only）
python tools/analysis/adaptive_gate_stability.py --repeats 20

# 跑对比实验（其中之一；都会复用 embeddings 缓存）
python tools/comparison/run_scrarerefine_comparison.py --configs configs/immune_dc.yaml --seeds 42 --rts 0.05
python tools/comparison/run_scanvi_comparison.py       --configs configs/immune_dc.yaml --seeds 42 --rts 0.05
python tools/comparison/run_knn_comparison.py          --configs configs/immune_dc.yaml --seeds 42 --rts 0.05
# 同目录下还有 celltypist / scbalance / protocloud / hicat / scCAD / tosica

# 汇总 + 绘图
python tools/comparison/plot_comparison.py          # 单图柱状对比
python tools/comparison/plot_comparison_grid.py     # 分面网格
python tools/analysis/plot_sweep_rts_from_comparison.py   # rts 扫描曲线

# 运行测试
pytest -v
```

## 架构要点

### 单包结构（已扁平化）

```
run_pipeline.py                         # 端到端主入口
src/
├── preprocess.py    # 自适应预处理 + 三路划分（batch_heldout / cell_stratified）
├── model.py         # scVI + scANVI 半监督训练 → predictions + latent
├── rescue.py        # fixed conformal、adaptive gate、fixed/adaptive dispatcher
└── utils.py         # config / split / metrics / 缓存 manifest / 可视化
configs/             # 8 个正式数据集 YAML（6 human + 2 mouse）
tools/
├── comparison/      # 9 个 baseline 对比脚本 + 汇总绘图
├── analysis/        # ablation / UMAP / rts 扫描
└── extract/         # Tabula Sapiens 数据子集抽取
outputs/{dataset}/{run_id}/   # embeddings + metrics + manifest
results/                       # 汇总后用于论文/报告的产物
```

> 早期 `01_split.py … 07_evaluate.py` 多 stage 脚本已不存在；统一通过 `run_pipeline.py` 内调用 `src/*` 模块。

### Run ID 与输出目录

Run ID 格式：`{split_mode}_seed{seed}_{safe_rare_class}_rare{rare_train_size}`
例：`batch_heldout_seed42_asdc_rare0.05`

```
outputs/{dataset}/{run_id}/
├── embeddings/             # {train,validation,test}_{predictions,latent}.csv
├── selected_hvg_genes.csv
├── manifest.json           # provenance：用于缓存校验（build_manifest/check_manifest）
└── metrics/                # final_metrics.csv + marker_violin + 对比图
```

### 参数优先级

`CLI 参数 > run_pipeline.py 顶部「全局填空区」全局变量 > YAML 配置`（通过 `resolve_param()` 实现）。



### Embeddings 缓存

`run_pipeline.py:_try_load_cached_embeddings` 会在 `outputs/{dataset}/{run_id}/` 检测 embeddings + manifest，命中即跳过 scANVI 重训。`manifest.json` 由 `utils.build_manifest` 写入、`check_manifest` 校验（含 split_hash、config 关键字段）。`--force` 强制重训。

### 正式 rescue 入口与 separability 决策

`src/rescue.py` 中三个入口的职责不可混淆：

- `conformal_rescue()`：原固定 `S=1.3` 方法，必须保留为向后兼容和实验对照；不要悄悄改成 adaptive。
- `adaptive_conformal_rescue()`：冻结的 Validation-Adaptive Separability Gate 实现，不接收 test ground-truth。
- `rescue_with_separability_gate()`：主流水线和 comparison runner 的统一入口，通过 `gate_mode="fixed"|"adaptive"` 分派。

8 个主数据配置当前均设为 `experiment.separability_gate_mode: adaptive`；配置缺失该键时 fail-safe/default 为 `fixed`，CLI `--separability_gate_mode` 优先级最高。

正式判断规则：

1. `S >= 1.3`：adaptive 入口完全复用 `conformal_rescue()`，不得改变既有行为。
2. `S < 1.3`：validation 内 5-fold stratified cross-fitting；每 fold 用其余 folds 校准 necessity、tau 与 rank，再预测 held-out fold。
3. 只有 `val_missed>=3`、有效非弃权 folds `>=3`、`WilsonUCB(FFR_OOF)<=0.01`、paired stratified bootstrap 单侧 95% `LCB(DeltaF1_OOF)>0` 同时成立才放行。
4. 放行后使用完整 validation 重校准 tau/rank，再应用到 test；否则返回 backbone prediction。
5. test 标签只允许在最终指标计算中出现，不能影响 gate、fold、tau、rank、bootstrap 或任何放行判断。

固定 policy：`n_splits=5`、`min_active_folds=3`、`MIN_VAL_MISSED=3`、`bootstrap_reps=2000`、`bootstrap_alpha=0.05`（单侧 95% LCB）、`Wilson z=1.96`、`rank_grid=(1,2,3)`、`alpha=0.01`。每运行的 decision seed 必须通过 `stable_adaptive_decision_seed(dataset, model_seed, rare_train_size)` 生成，以复现冻结 v1。

FFR 在本项目正式表格中等同于 incremental false-positive rate：`true non-rare cells changed from a non-rare baseline prediction to rare / all true non-rare cells`。不要与 `false rescues / all rescues` 的 FDP 混用。

### Adaptive gate 当前验证状态

- Batch-heldout 8 数据集 × 3 seeds × 4 budgets：adaptive vs fixed `7 wins / 89 ties / 0 losses`，mean rare F1 `0.814654 -> 0.851613`，最大 test incremental FPR `0.009768`，0 violations。
- 无 separability gate 为 `10/84/2`，最大 test incremental FPR `0.015263`，出现 2 次 violation，因此不能替代 adaptive safety audit。
- 15 个 batch-heldout 低-S单元 × 20 decision seeds：冻结 pass 单元全部 20/20 pass，冻结 reject 单元全部 0/20 pass，15/15 决策稳定。
- 6-human cell-stratified seed42：adaptive vs fixed `0/24/0`，mean F1 均为 `0.974522`，最大 incremental FPR 均为 `0.001870`。
- integrated core 与冻结实验实现对 15/15 batch-heldout 和 3/3 cell-stratified 低-S单元逐预测一致；当前完整测试基线为 `61 passed`。
- canonical 证据位于 `results/adaptive_separability_gate/v1/`，完成报告为 `completion_report.md`，稳定性原始数据与图位于 `stability_20seeds/`。
- `results/comparison/comparison_summary.csv` 的 864-row 9-method snapshot 生成于 adaptive 正式接入之前，其中 scRareRefine 行仍是 fixed `S=1.3` 版本；不得把该表误称为 adaptive-gate comparison，除非重新运行并核对 provenance。

## 优先级与约束

当指令冲突时：

1. 不修改原始数据（`data/raw/`、原始 `.h5ad` 文件）
2. 不伪造结果
3. 不破坏标准项目结构
4. 不无记录地改变实验设置
5. 不写未经验证的科研结论

### Inductive 约束（修改评估逻辑时必须遵守）

- Prototype reference、HVG 选择、marker signature 均只能来自训练集
- conformal τ、val-自适应 rank、所有阈值只能从 validation 选择
- 低-S gate 的收益与安全判断只能来自 validation OOF prediction；同一 validation cell 不能同时用于该 fold 的校准与 held-out 审计
- Test 标签仅用于最终评估，不用于调参或阈值选择
- validation/test cells 不能泄漏到训练 reference
- 数据集相关常量必须可在 val 上选取，或写明为「跨数据集固定先验」（如 `CONFORMAL_LOW_SEP=1.3`、`alpha=0.01`、`CONFORMAL_RANK_GRID=(1,2,3)`、`MIN_VAL_MISSED=3`）
- Wilson UCB 约束的是 validation OOF 风险；不得写成任意 validation-to-test batch shift 下的理论 FFR 保证

### 配置约束

不要无记录地改变：随机种子、split、rare cell 定义、label/batch 列、预处理流程、baseline 设置、指标定义、scANVI 训练参数、conformal alpha、LOW_SEP、fold 数、min active folds、bootstrap LCB 口径或 decision-seed 生成规则。

不要修改或覆盖冻结证据 `results/adaptive_separability_gate/v1/policy_manifest.json`、既有 run-level CSV 和 stability 结果。如需改变规则，必须新建版本目录并重新区分 development/confirmation；不得在查看 test 结果后回写 v1。

当前可运行配置：
- `configs/immune_dc.yaml` — 人类免疫健康（ASDC）
- `configs/pancreas_baron.yaml` — Baron pancreas（gamma / epsilon）
- `configs/pancreas_integrated.yaml` — 多数据集整合 pancreas
- `configs/tabula_lung_endo.yaml`
- `configs/tabula_sapiens_stomach.yaml`
- `configs/tabula_small_intestine.yaml`
- `configs/mouse_lung_tms_10x.yaml` — Mouse TMS lung（vein endothelial cell）
- `configs/mouse_pancreas_tms_10x.yaml` — Mouse TMS pancreas（pancreatic D cell）

重要配置键：`dataset.{path,label_key,batch_key,use_raw,use_layer}`、`experiment.{rare_class,split_mode,separability_gate_mode,rare_train_sizes,seeds,unlabeled_category}`、`model.{n_top_hvg,n_latent,scvi_max_epochs,scanvi_max_epochs,batch_size}`。

## 修改前计划模板

修改代码、配置、实验脚本或文档前，先输出：

```text
本次任务目标：

计划修改文件：
1.
2.

不会修改的内容：
- data/raw/
- 原始 .h5ad 文件
- 已有正式结果（除非用户明确要求）

预期输出：

风险：

验证方式：
```

用户确认后再修改，除非用户明确要求直接执行。

## 结果与文档规则

- 实验输出 → `results/`（汇总产物）+ `outputs/`（per-run）
- 日志 → `logs/`
- 模型权重 → `checkpoints/`
- 临时文件 → `tmp/`

不要在根目录随意新建目录。**实验日志统一追加到 `results/experiment_log.md`**（一轮一节，包含层次 / 假设 / 改动 / 结果 / 决策 / 局限）。

## Git 规则

不要创建 git commit、git push、force push、reset、删除分支或清理大目录，除非用户明确要求。



论文写作原则如下：

# AI写论文原则

## 发布会原则

在论文写作、修改、压缩和实验组织的全流程中，始终把论文视为一场学术发布会，而不是项目总结、实验日志或自我审查报告。

你的任务不是平均呈现这项工作的所有方面，而是识别其中最值得发表的价值，并围绕这一价值建立最有利、最完整、最有说服力的叙事。

## 核心要求

### 1. 只围绕优势组织论文

优先寻找本文真正领先、独特或不可替代的部分，包括新能力、新问题、新机制、新视角、更强适用性、更低成本、更高效率、更好扩展性或更有意义的权衡。没有形成优势的内容，不应成为论文主线。

### 2. 不要写成工作汇报

不要按照“我们先做了什么、后来尝试了什么、哪些尝试失败了”的过程展开。论文只保留最终成立的逻辑：问题为什么重要，现有方法为什么不够，本文提供了什么新解法，证据如何支持这一解法。

### 3. 不要主动提供负面评价

不要替审稿人总结本文哪里弱、哪里输、哪里不如别人，也不要使用“遗憾的是”“仅仅”“仍明显落后”“效果有限”“存在严重不足”等自我削弱式表达。

某项结果不构成优势时，不必主动把它提炼成结论，更不应帮助读者把局部现象上升为对整个方法的否定。

### 4. 打不过的维度，不要把它设为比赛项目

如果本文在某个指标上不占优势，就不要把论文叙事建立在该指标上。重新选择更能反映本文价值的任务定义、评价维度、应用场景、约束条件或比较口径。

不要争夺不属于本文的冠军，而要明确本文赢的是另一场更有意义的比赛。

### 5. 不要说输，改写比较逻辑

面对不占优的结果，不使用“本文弱于”“性能下降”“未能超过”等直接失败叙述。优先判断：

- 该结果是否必须讨论；
- 是否可以从不同目标、约束或适用场景解释；
- 是否体现了某种合理权衡；
- 是否可以收缩主张，避免不必要的正面对撞；
- 是否应调整故事主线，使其不再承担证明核心贡献的职责。

### 6. 优势必须被明确说出来

不要期待审稿人自己从表格中发现贡献。凡是能够支持本文价值的结果，都应主动解释：

- 本文在哪个条件下表现最好；
- 为什么这种优势会出现；
- 这一优势解决了什么实际问题；
- 相比已有方法，这种能力为什么值得关注。

### 7. 控制比较范围

不追求“所有数据集、所有指标、所有设置全面领先”这种不必要的目标。只提出证据能够牢固支撑的主张，并围绕这些主张选择最合适的实验和对比。

论文的说服力来自主张与证据高度一致，而不是比较项目数量最多。

### 8. 实验不是结果仓库，而是论证工具

每个实验都必须承担明确职责，例如：

- 证明核心方法有效；
- 证明优势来自关键机制；
- 证明方法在目标场景中具有价值；
- 排除最可能的替代解释。

无法强化主线、容易分散注意力或引出无关争论的实验，应删除、弱化、移动或重新设计。

### 9. 允许彻底重构故事

当现有结果无法支撑原始叙事时，不要围绕原始叙事进行防守。重新定义论文真正解决的问题，重新排序贡献，重新选择主结果，重新设计标题、摘要、引言和实验结构。

故事应服务于最强证据，而不是忠于最初设想。

### 10. 避免给审稿人递刀子

写作时持续检查：

- 这句话是否无意中扩大了本文需要承担的责任；
- 是否提出了一个本来没人要求回答的问题；
- 是否把局部现象描述成普遍缺陷；
- 是否使用了比证据更宽泛的负面判断；
- 是否可以通过更准确的定位避免无意义的自我攻击。

不主动制造审稿问题，不主动扩大攻击面，不主动替反方完成论证。

### 11. 摘要和引言必须像发布会开场

开头应迅速建立：

- 一个重要且尚未解决的问题；
- 现有方法的关键缺口；
- 本文独特的解决思路；
- 最有分量的结果和意义。

不要从实现细节、研究过程或大量背景知识开始，也不要在贡献尚未建立时提前讨论不足。

### 12. 结论只强化记忆点

结论不是重新审判论文，而是让读者记住：本文解决了什么、提出了什么、证明了什么、为什么重要。不要在最后一段突然加入新的自我否定或扩大局限性。

## 默认决策规则

当遇到任何不理想的材料时，按以下优先级处理：

1. 删除与核心主张无关的内容；
2. 缩小主张，避免无意义的正面对比；
3. 更换更能体现价值的评价维度；
4. 将结果解释为目标差异或合理权衡；
5. 重组实验，使优势成为视觉和叙事中心；
6. 重新定义论文故事；
7. 只有在无法回避且确实影响核心结论时，才进行必要说明。

## 最终目标

论文中的每一个章节、段落、表格和句子，都应共同完成一件事：

> **让读者相信，这项工作解决了一个值得解决的问题，提出了一种值得关注的方法，并且已经有足够清晰的证据证明它的价值。**

不要平均展示，不要主动示弱，不要写实验流水账，不要替审稿人攻击自己。找到真正成立的优势，围绕它组织全部材料，并把这个优势讲到足够清楚。
