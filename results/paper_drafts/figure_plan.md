# Figure Plan（论文配图规划，2026-06-21）

> 决策（用户 2026-06-21）：① 风格统一字号/配色；② UMAP 做；③ Fig2 精简汇总 + 全 grid 进 Supp；
> ④ **只出 PNG，不生成 PDF**（投稿矢量图走 AI→矢量化 工作流另行处理，不在分析图内）。
> Fig2 含 FFR 面板（FFR≤α 是核心卖点，必须可见）。

## 主图（4 核心 + 1 定性）

| 图 | 叙事角色 | 来源 | 状态 |
|---|---|---|---|
| **Fig 1 · Method overview** | 方法 pipeline：scANVI→prototype→conformal→FFR受控 rescue | AI 示意图（提示词已给，见对话）| ⬜ 待生成+矢量化（用户出）|
| **Fig 2 · Main result（精简）** | "它 work"：稀缺区 rare-F1 胜 baseline **且 FFR 受控** | [main_summary.png](../comparison/main_summary.png)（a: 稀缺区 9方法 F1=0.88 断层最高；b: worst-case FFR，仅 SRR≤α）| ✅ 已建 |
| **Fig 3 · Ablation** | 每个组件 earn its place | [ablation_table1](../ablation/ablation_table1_components.png) + [table2](../ablation/ablation_table2_rank.png)| ✅ 已建（PNG）|
| **Fig 4 · Separability gate** | 保守闸门有原则（G21）| [fig4_separability.png](../sep_sweep/fig4_separability.png)（合成 sweep + 真实 sensitivity 三联）| ✅ 已建 |
| **Fig 5 · UMAP rescue（定性）** | latent 空间被救回的稀有细胞 before/after | [immune](../umap/umap_rescue_immune_dc.png)/[stomach](../umap/umap_rescue_tabula_sapiens_stomach.png)/[pancreas_baron](../umap/umap_rescue_pancreas_baron.png)（已修一致性=主方法）| ✅ 已建 3 张 |

## 补充材料（Supplementary）

| Supp | 内容 | 来源 |
|---|---|---|
| S1 | 全 grid（6×4×9，F1）| comparison_bars_grid |
| S2 | 配对显著性表 | significance_test.csv |
| S3 | 稀缺区 distinct 计数 + rts 塌缩说明 | scarce_region_distinct.csv |
| S4 | sep vs gain 散点 | sep_vs_gain |
| S5 | rts 稳健性曲线 | sweep_rts_curves |
| S6 | runtime / peak RAM（G05，未做）| 待补 |
| S7 | 更多数据集 UMAP | plot_umap_rescue |

## 构建待办

- [x] Fig 2 plot_main_summary.py（F1 + FFR 双面板，PNG）✅
- [x] Fig 4 plot_fig4_separability.py（合成 sweep + 真实 sensitivity 三联，PNG）✅
- [x] Fig 5 UMAP rescue 生成 3 数据集（immune/stomach/pancreas_baron，已修一致性）✅
- [x] **移除所有 plot 脚本的 .pdf 保存**（PNG-only durable）✅
- [x] 共享风格统一：scRareRefine 绿+加粗、α=0.01 红虚线、DejaVu、3-seed 误差棒 ✅
- [ ] Fig 1 overview AI 图（用户出）+ 矢量化
- [ ] 图注（caption）逐图写，回链 results/ CSV（写作时）
- [ ] Supp S6 runtime/RAM（G05，未做）

## 风格规范（统一）

- 字体 DejaVu Sans；标题 11-12pt，轴标 10pt，刻度 9pt，图例 9pt。
- scRareRefine 固定绿色 #2ca02c + 加粗边框高亮；HiCat 标 † transductive。
- FFR 图统一画 α=0.01 红虚线。
- 误差棒 = 跨 3 seed SD。
- **输出仅 PNG（dpi=300），不出 PDF。**
