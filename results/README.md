# results/ 索引

scRareRefine 实验结果。完整 8 轮迭代记录见 [experiment_log.md](experiment_log.md)。

## 目录结构

```
results/
├── experiment_log.md        # 主日志（8 轮迭代：诊断→方法→消融→对比→扫描→可视化）
├── README.md                # 本索引
├── ablation/                # 第五轮：消融实验（组件贡献拆解）
├── comparison/              # 第六轮：对比实验（vs scANVI/kNN/CellTypist/scBalance）
├── sweep_rts/               # 第七轮：rare_train_size 稳健性扫描 + 论文级曲线
├── umap/                    # 第八轮：UMAP rescue 可视化 + 高/低 sep 对照
└── legacy/                  # 早期结果（已被后续轮次取代，保留备查）
```

## 各目录内容

| 目录 | 轮次 | 文件 | 生成脚本 |
|------|------|------|---------|
| `ablation/` | 第五轮 | `ablation_summary.csv`（逐 run×变体）、`ablation_summary_agg.csv`（3-seed 聚合）、`ablation_log.md` | `tools/ablation.py` |
| `comparison/` | 第六轮 | `comparison_summary.csv`、`comparison_summary_agg.csv`、`comparison_log.md` | `tools/compare_baselines.py` |
| `sweep_rts/` | 第七轮 | `sweep_rts_summary.csv`（180 行）、`sweep_rts_agg.csv`（60 行）、`sweep_rts_log.md`、`sweep_rts_curves.png/.pdf` | `tools/sweep_rare_train_size.py`、`tools/plot_sweep_rts.py` |
| `umap/` | 第八轮 | `umap_rescue_{immune_dc,pancreas_baron}.png/.npz`、`umap_contrast_sep.png/.pdf` | `tools/plot_umap_rescue.py`、`tools/plot_umap_contrast.py` |
| `legacy/` | 第四轮 | `eval_summary.csv`（baseline/gate/conformal 9-run 评估，已被 comparison 取代） | — |

## 主要结论速查

**对比实验（3 数据集 × 3 seed，rare F1 均值）** — 见 `comparison/`

| 数据集 | scANVI | kNN | CellTypist | scBalance | **scRareRefine** |
|-------|--------|-----|-----------|-----------|-----------------|
| immune_dc | 0.025 | 0.673 | 0.560 | 0.546 | **0.939** |
| pancreas_baron | 0.825 | 0.617 | 0.628 | 0.664 | **0.849** |
| tabula_lung_endo | 0.969 | 0.952 | 0.775 | 0.915 | **0.980** |

**稳健性扫描核心结论** — 见 `sweep_rts/sweep_rts_curves.png`

- 稀有标注越少，scRareRefine 优势越大（极少标注时领先次优 +0.06~+0.32）
- 曲线最平坦（对标注量最不敏感）；优势幅度随可分性 sep 递增
- 标注充足（all）时纯监督方法追平或反超 → 定位为「标注稀缺」场景

**机理可视化** — 见 `umap/umap_contrast_sep.png`

rescue 收益 ∝（scANVI 漏判量 × 稀有簇几何独立性）：高 sep（immune）救回 120/130，低 sep（pancreas）仅多救 6 个。

## 复跑方式

所有脚本输出已指向各自子目录，重跑自动归位。依赖关系：
- 绘图脚本依赖对应实验脚本的输出（`plot_sweep_rts` 读 `sweep_rts/sweep_rts_agg.csv`；`plot_umap_contrast` 读 `umap/*.npz`）
- 所有实验复用 `outputs/{dataset}/{run_id}/embeddings/` 的缓存（由 `tools/train_cache.py` 生成），无需重训
- 运行环境：`D:/setup/anaconda/envs/scanvi311/python.exe`
