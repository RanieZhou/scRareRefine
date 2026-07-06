# PAPER_PLAN — scRareRefine（2026-06-21，2026-07-05 重定向）

> 目标：生信二区（Bioinformatics / BIB / NAR-GAB / Cell Reports Methods）。模板用通用 `article`+natbib（投稿时换刊 .cls）。
> 诚实底线：不写 SOTA/全面最优；主张限于"标注稀缺区相对 baseline 提升 rare F1 且 FFR≤α 受控"。
>
> **2026-07-05 重定向（用户决定，记于 experiment_log 第十五轮）**：科学问题重心由"FFR-controlled selective rescue"改为 **"标注稀缺下的召回恢复（recall recovery under label scarcity）"**。
> 后果：① **rts×指标恢复曲线升为主图**（原 Supp S5 → 主 Fig，展示"标注越稀缺、增益越大、rts=all 优雅弃权收敛 baseline"）；② **FFR 由并列头条降为安全约束**（支撑性 claim，不是故事中心）；③ **F1 仍为唯一头条指标，recall 作补充 panel**（用户决定）；④ G10 stomach recall 天花板由边缘局限升为**头条失败模式**。方法定位锁定为**框架 B（完整 pipeline 当一个方法，新颖性钉在 refinement），不给其他 baseline 加模块**，仅补一个弱 backbone 泛化 demo 堵 reviewer。

## 一句话贡献

A post-hoc module that recovers rare cell types missed by a semi-supervised classifier (scANVI) under label scarcity, using train-only prototype-distance scoring and validation-only conformal calibration to raise rare-cell F1 **while keeping the false-rescue rate (FFR) under a fixed budget α=0.01**, and abstaining when rescue is not warranted.

## 工作标题（候选）

- A. *Risk-controlled post-hoc rescue of rare cell types in single-cell RNA-seq under label scarcity*
- B. *scRareRefine: conformal prototype rescue for rare cell-type identification under label scarcity*

## Claims–Evidence Matrix（每条 falsifiable + 证据）

| # | Claim | 证据（图/表/CSV）|
|---|---|---|
| **C1（头条·恢复）** | scANVI 在标注稀缺下系统性欠召回稀有细胞；scRareRefine 恢复这部分召回，**增益随标注变稀缺而扩大**，rts=all 时优雅弃权收敛到 baseline | **Fig 2（rts 恢复曲线，升为主图）**：immune_dc rts=0.01 scANVI F1 0.000→SRR 0.927、rts=0.10 gain 缩到 +0.033；stomach 全 rts gain +0.112；core_agg.csv（3-seed mean±std）|
| **C1b（幅度·对比）** | 稀缺区 scRareRefine 的 rare F1 高于 scANVI 及 7 个对比方法 | Fig 3a（稀缺区 F1=0.878 vs 次高 0.725）；significance_test.csv（vs scANVI 29胜/25平/0负、ΔF1+0.160 CI[+0.085,+0.244] p=1.3e-6，对 8 法 CI 均>0）；scarce_region_distinct.csv（15 distinct，win-most 15/15、best 14/15）|
| **C2（安全约束，非头条）** | scRareRefine 把 worst-case FFR 控在 α=0.01 内（从不制造假拯救泛滥、弃权=返回 baseline=不伤害），多数对比方法不受控 | Fig 3b（scCAD 0.045 / scBalance 0.016 / TOSICA 0.014 破 α；SRR 0.0098≤α）|
| **C-supp（recall panel）** | rare recall 是 scANVI 丢失、被恢复的量；用 recall 补充 panel 展示机制（非头条指标）| Supp：ablation_table2 recall 列（R_adaptive 0.853）+ core_agg rescue recall；stomach recall 天花板 0.59（见 C5/失败模式）|
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

> **2026-07-05 重定向后的图序（recall 恢复主线）**：Fig 1 总览 → **Fig 2 = rts 恢复曲线（新主图，原 sweep_rts_curves 升格）** → Fig 3 = 对比主结果（原 Fig 2，F1 + FFR 双面板）→ Fig 4 = ablation → Fig 5 = separability → Fig 6 = UMAP。recall 曲线进 Supp panel。下面旧编号待正式定稿时统一顺移。

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

## 2026-07-06 状态更新

- 已完成：从 `results/multiseed/core_agg.csv` 生成三种子主图 `paper/figures/fig2_recovery_curves.png` 和召回补图 `paper/figures/figS_recall_recovery_panel.png`。
- 已完成：补充弱 backbone demo（kNN base + unchanged rescue），稀缺区 F1 0.7248→0.8603、recall 0.6506→0.8085、FFR_max 0.009768；有 1 个稀缺区负向 paired cell，不能写 backbone-agnostic no-regression。
- 已完成：补充缓存 provenance 审计，76/76 split hash 与缓存 cell IDs 一致；64 个缓存来自旧 commit，12 个 legacy unknown，投稿前严格复跑需 clean output 或 `--force`。
- 已同步：LaTeX 正文/讨论/附录与 Markdown 草稿已纳入新图、弱 backbone 和 provenance 证据；TOSICA 显著性表修正为 53/1/0、ΔF1 +0.387、CI [+0.321,+0.454]。
- 期刊约束：Bioinformatics Original Paper 官方限制为 7 页左右（约 5000 words excluding figures），当前通用 `article` 编译稿为 15 页，若冲 Bioinformatics 必须主文压缩 + 大量转 Supplement；BMC Bioinformatics 接受 LaTeX 且更适合完整附录/数据材料。
- 仍待：外部/二层数值审稿、runtime/peak-memory benchmark、最终 Bioinformatics/BMC Bioinformatics 模板转换、参考文献 bib 核对，以及 Fig 1 是否需要重绘为最终出版图。
