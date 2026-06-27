# PAPER_PLAN — scRareRefine（2026-06-21）

> 目标：生信二区（Bioinformatics / BIB / NAR-GAB / Cell Reports Methods）。模板用通用 `article`+natbib（投稿时换刊 .cls）。
> 诚实底线：不写 SOTA/全面最优；主张限于"标注稀缺区相对 baseline 提升 rare F1 且 FFR≤α 受控"。

## 一句话贡献

A post-hoc module that recovers rare cell types missed by a semi-supervised classifier (scANVI) under label scarcity, using train-only prototype-distance scoring and validation-only conformal calibration to raise rare-cell F1 **while keeping the false-rescue rate (FFR) under a fixed budget α=0.01**, and abstaining when rescue is not warranted.

## 工作标题（候选）

- A. *Risk-controlled post-hoc rescue of rare cell types in single-cell RNA-seq under label scarcity*
- B. *scRareRefine: conformal prototype rescue for rare cell-type identification under label scarcity*

## Claims–Evidence Matrix（每条 falsifiable + 证据）

| # | Claim | 证据（图/表/CSV）|
|---|---|---|
| **C1** | 标注稀缺区，scRareRefine 的 rare F1 高于 scANVI 及 7 个对比方法 | Fig 2a（稀缺区 F1=0.878 vs 次高 0.725）；significance_test.csv（vs scANVI 0 负、p=1.3e-6，对 8 法 CI 均>0）；Supp grid |
| **C2** | scRareRefine 把 worst-case FFR 控在 α=0.01 内，多数对比方法不受控 | Fig 2b（scCAD 0.045 / scBalance 0.016 / TOSICA 0.014 破 α；SRR 0.0098≤α）|
| **C3** | 各组件各司其职：自适应 rank 拉 F1、τ 控 FFR、两闸门保安全/防回归 | Fig 3（ablation 表1 组件留一法 + 表2 rank 敏感性）|
| **C4** | sep 闸门阈值 1.3 是保守 pre-specified 先验，风险轴数据集相关 | Fig 4（合成 sep 扫描：safe 到 ~0.76；真实 low_sep sensitivity：1.3 是 FFR≤α 最小阈值）；codex Round 3 |
| **C5** | 方法是**条件性**的：不需要时弃权(=no-op)；有已知失败模式（几何纠缠 recall 天花板、极端稀缺 seed 不稳）| Discussion/Failure modes；Fig 5（UMAP，stomach 13 仍漏=天花板）|

## 章节计划（生信结构）

| 章 | 内容要点 | 主图/表 | 目标长度 |
|---|---|---|---|
| Abstract | what/why-hard/how/evidence/strongest（含 1 个定量：稀缺区 F1 0.72→0.88、FFR≤α）| — | 200词 |
| 1 Introduction | 痛点（稀有细胞漏判+硬救会涨 FP）→ gap → 方法概览 → 贡献列表(C1–C5) → Fig 1 | Fig 1 | 1–1.5p |
| 1.x Related work | 监督注释(CellTypist/scBalance/TOSICA)、稀有检测(scCAD)、原型/半监督(scANVI/ProtoCloud)、conformal | — | ≥0.75p |
| 2 Methods | 2.1 scANVI backbone + inductive 三路 split；2.2 prototype scoring（隶属度 s=softmax(-d/r)、sep）；2.3 conformal calibration（sep 闸门 / necessity 守门 / val-自适应 rank Wilson / τ）；2.4 FFR 定义与控制 | Fig 1, Algorithm 1 | 1.5–2p |
| 3 Results | 3.1 setup（6 数据集/9 方法/3 seed/指标）；3.2 main(C1+C2)；3.3 ablation(C3)；3.4 separability(C4)；3.5 qualitative UMAP | Fig 2,3,4,5 + Table 1 | 2.5–3p |
| 4 Discussion | failure modes & limitations(C5，引 failure_modes 草稿)；与 baseline 操作点差异；future work | — | 0.75p |
| 5 Conclusion | 贡献复述 + 1-2 future | — | 0.3p |
| Appendix/Supp | grid(S1)、显著性表(S2)、distinct 计数(S3)、sep_vs_gain(S4)、rts 曲线(S5)、runtime(S6 待补)、更多 UMAP(S7)、provenance | — | — |

## 图映射

- Fig 1 = docs/Fig 1-overview.png（方法总览，暂用 AI 图）
- Fig 2 = results/comparison/main_summary.png
- Fig 3 = results/ablation/ablation_table1_components.png + ablation_table2_rank.png
- Fig 4 = results/sep_sweep/fig4_separability.png
- Fig 5 = results/umap/umap_rescue_immune_dc.png（+ stomach/pancreas_baron 进 Supp）
- Table 1 = comparison_summary_agg.csv 选列（3-seed mean±std）

## 关键引用（待 DBLP 取真 bib）

scANVI(Xu 2021)、scVI(Lopez 2018)、CellTypist(Domínguez Conde 2022)、scBalance、TOSICA(Chen 2023)、scCAD(2024)、HiCat、ProtoCloud、conformal prediction(Vovk; Angelopoulos&Bates 2023)、scANVI/UMAP(McInnes 2018)、Tabula Sapiens(2022)、Baron pancreas(2016)。

## 诚实/写作纪律

- p 值写成 directional/robust paired improvement，effect size + CI 优先（codex）。
- HiCat 标 transductive 上界；TOSICA 降配披露。
- sep 闸门写"conservative prior"，不写"定位崩塌边界"（codex Round 3）。
- rts 塌缩：稀缺区按 15 distinct 报，不写"15/15"独立。
