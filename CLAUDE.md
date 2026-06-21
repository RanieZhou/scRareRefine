# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

always reply in Chinese

> **迭代前必读**：[ITERATION_BOUNDARY.md](ITERATION_BOUNDARY.md) 定义了红线、迭代触发条件、A/B/C 层次纪律、codex 外审节点、当前 GAP 清单。新开一轮迭代前必须先读它 + `results/experiment_log.md` 最近一轮章节。

## 项目目标

**scRareRefine** 基于 scANVI 的 latent embedding 与预测概率，为稀有细胞类型设计一个 post-hoc refinement 模块。核心方法是 **prototype 距离评分 + conformal 阈值校准** 的拯救流程，目标是在 FFR（False Rescue Rate）严格受控的前提下提升 rare cell type 的 F1。

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

# 跑对比实验（其中之一；都会复用 embeddings 缓存）
python tools/comparison/run_scrarerefine_comparison.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
python tools/comparison/run_scanvi_comparison.py       --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
python tools/comparison/run_knn_comparison.py          --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05
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
├── rescue.py        # 4 个 Rescuer 类 + conformal_rescue() 单一入口
└── utils.py         # config / split / metrics / 缓存 manifest / 可视化
configs/             # 7 个数据集 YAML（见下）
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

### 核心算法（src/rescue.py）

**PrototypeRescuer**
- 按训练集 labeled 样本计算各类均值原型，每类 `radius = median(dist to prototype)`（<3 样本时取 1.0）。
- `separability_ratio = d(rare_proto, nearest_majority_proto) / mean(intra_rare_radius)`。
- `rare_membership_score`：各向异性 softmax(-d_c/r_c)，对各类紧致度归一化。
- `rare_rank` / `rank_candidate(max_rank)`：候选筛选用各向同性欧氏距离 rank。

**ConformalRescuer + 顶层 `conformal_rescue()`**（主路径，单一来源；run_pipeline 与对比脚本共用）

三道全 inductive 闸门：
1. **separability 安全网** — `sep < CONFORMAL_LOW_SEP=1.3` → 弃权。
2. **necessity 守门** — val baseline rare recall==1.0 → 弃权（避免对已经救满的数据集添乱）。
3. **val-自适应候选 rank ∈ {1, 2, 3}** — 在 val FFR Wilson 95% 上界≤α 约束下选「val rare F1 最高」的 max_rank，平手取小 rank。再以 conformal τ（val 非稀有 score 的有限样本 (1-α) 顺序统计量）控 FFR，应用到 test。

发表级 FFR 上界 `DEFAULT_CONFORMAL_ALPHA = 0.01`（跨数据集固定常量，**不调参**）。

**保留但非默认的策略**：`gate_only` / `gate_marker` / `fusion`（FFR 约束 `max_false_rescue_rate=0.001`），通过 `run_post_hoc_rescue(strategy=...)` 切换。

### Embeddings 缓存

`run_pipeline.py:_try_load_cached_embeddings` 会在 `outputs/{dataset}/{run_id}/` 检测 embeddings + manifest，命中即跳过 scANVI 重训。`manifest.json` 由 `utils.build_manifest` 写入、`check_manifest` 校验（含 split_hash、config 关键字段）。`--force` 强制重训。

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
- Test 标签仅用于最终评估，不用于调参或阈值选择
- validation/test cells 不能泄漏到训练 reference
- 数据集相关常量必须可在 val 上选取，或写明为「跨数据集固定先验」（如 `LOW_SEP=1.1`、`CONFORMAL_LOW_SEP=1.3`、`alpha=0.01`、`CONFORMAL_RANK_GRID=(1,2,3)`、`MIN_VAL_MISSED=3`）

### 配置约束

不要无记录地改变：随机种子、split、rare cell 定义、label/batch 列、预处理流程、baseline 设置、指标定义、scANVI 训练参数、conformal alpha 与 LOW_SEP 常量。

当前可运行配置：
- `configs/immune_dc.yaml` — 人类免疫健康（ASDC）
- `configs/pancreas_baron.yaml` — Baron pancreas（gamma / epsilon）
- `configs/pancreas_integrated.yaml` — 多数据集整合 pancreas
- `configs/tabula_lung_endo.yaml`、`tabula_lung_stroma.yaml`
- `configs/tabula_sapiens_stomach.yaml`
- `configs/tabula_small_intestine.yaml`

重要配置键：`dataset.{path,label_key,batch_key,use_raw,use_layer}`、`experiment.{rare_class,split_mode,rare_train_sizes,seeds,unlabeled_category}`、`model.{n_top_hvg,n_latent,scvi_max_epochs,scanvi_max_epochs,batch_size}`。

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

## 论文主张限制

不可写：解决了稀有细胞识别问题、全面优于所有方法、state-of-the-art、临床可用、适用于所有单细胞数据。

可写（需结果支持）：在评估的数据集上，scRareRefine 相比原始 scANVI 提升了稀有细胞相关指标，在标注稀缺区（rare_train_size ≤ 0.10）相对多数对比方法占优。

## Git 规则

不要创建 git commit、git push、force push、reset、删除分支或清理大目录，除非用户明确要求。
