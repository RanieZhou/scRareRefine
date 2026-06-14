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
| `ablation/` | 第五轮 | `ablation_summary.csv`（逐 run×变体）、`ablation_summary_agg.csv`（3-seed 聚合）、`ablation_log.md`、`ablation_bars.png/.pdf`（双指标对比图） | `tools/ablation.py`、`tools/plot_ablation.py` |
| `comparison/` | 第六轮 | `comparison_summary.csv`（135 行，9 方法×15 run，5 数据集×3 seed）、`comparison_summary_agg.csv`、`comparison_log.md`、`comparison_bars.png/.pdf`（方法对比柱状图） | `tools/compare_baselines.py`（scANVI/kNN/CellTypist/scBalance/scRareRefine）、`tools/run_protocloud_comparison.py`（ProtoCloud，sandbox310）、`tools/run_hicat_comparison.py`（HiCat，sandbox310）、`tools/run_scCAD_comparison.py`（scCAD，scanvi311）、`tools/run_tosica_comparison.py`（TOSICA，sandbox310）、`tools/plot_comparison.py` |
| `sweep_rts/` | 第七轮 | `sweep_rts_summary.csv`（180 行）、`sweep_rts_agg.csv`（60 行）、`sweep_rts_log.md`、`sweep_rts_curves.png/.pdf` | `tools/sweep_rare_train_size.py`、`tools/plot_sweep_rts.py` |
| `umap/` | 第八轮 | `umap_rescue_{immune_dc,pancreas_baron}.png/.npz`、`umap_contrast_sep.png/.pdf` | `tools/plot_umap_rescue.py`、`tools/plot_umap_contrast.py` |
| `legacy/` | 第四轮 | `eval_summary.csv`（baseline/gate/conformal 9-run 评估，已被 comparison 取代） | — |

## 主要结论速查

**对比实验（9 方法，5 数据集 × 3 seed，rare F1 均值）** — 见 `comparison/`

| 数据集 | scANVI | kNN | CellTypist | scBalance | ProtoCloud | HiCat | scCAD | TOSICA | **scRareRefine** |
|-------|--------|-----|-----------|-----------|-----------|-------|-------|--------|-----------------|
| immune_dc | 0.025 | 0.673 | 0.560 | 0.544 | 0.494 | 0.214 | 0.013 | **0.954** | 0.939 |
| pancreas_baron | 0.825 | 0.617 | 0.628 | 0.710 | 0.661 | 0.102 | 0.618 | **0.899** | 0.849 |
| tabula_lung_endo | 0.969 | 0.952 | 0.775 | 0.918 | 0.938 | 0.020 | 0.968 | 0.901 | **0.980** |
| tabula_small_intestine | **0.980** | 0.972 | **0.982** | 0.967 | **0.985** | 0.345 | 0.501 | 0.956 | 0.977 |
| tabula_lung_stroma | **0.390** | 0.354 | 0.205 | 0.289 | 0.286 | 0.000 | 0.352 | 0.349 | **0.390** |

**数据集说明**

- **immune_dc**（Granja et al.）：ASDC（外周血稀有 DC 亚型），high-sep，scANVI baseline 几乎为零，rescue 提升最显著。
- **pancreas_baron**（Baron et al.）：gamma 细胞（胰岛稀有类型），中等 sep，rescue 稳定提升。
- **tabula_lung_endo**（Tabula Sapiens，Endothelial 10X）：淋巴内皮细胞，high-sep，所有方法均高，scRareRefine 最优。
- **tabula_small_intestine**（Tabula Sapiens，Small Intestine 10X）：肠道微绒毛细胞（intestinal tuft cell），high-sep（前三名方法 F1>0.97），所有监督方法均趋近上界；HiCat/scCAD 例外（设计场景不匹配）。
- **tabula_lung_stroma**（Tabula Sapiens，Lung Stromal 10X）：支气管平滑肌细胞，**low-sep 极端困难**场景，全部方法 F1 均在 0.4 以下；scRareRefine 与 scANVI 持平（0.39，0 rescues），prototype 可分性评分触发 abstain（separability < 1.1），符合设计预期。

**新增方法说明（2023-2026）**

- **ProtoCloud**（Cell Genomics 2026）：原型 VAE + 稀有类过采样。在 high-sep 数据集（tabula_small_intestine）达最优（0.985），中等 sep 表现中等，low-sep（tabula_lung_stroma）也未能提升（0.286 < scANVI 0.390）。
- **HiCat**（Briefings in Bioinformatics 2025）：Harmony PCA + UMAP + DBSCAN + CatBoost + 置信度阈值。F1 极低（0.00–0.35），原因是**设计场景不匹配**：HiCat 专为"测试集含完全未见新型别"设计，在该场景下用 DBSCAN 簇标签替代低置信度预测；但我们的 few-shot 场景中稀有类在训练集中已有少量样本，阈值仍将约 20–51% 的测试细胞重置为 DBSCAN 标签（如 "3"、"-1"），导致稀有类 recall 接近零且方差极高（seed 间差异达 0.46）。这是方法适用场景的根本差异，而非实现错误。
- **scCAD**（Nature Communications 2024）：无监督稀有细胞异常检测。immune_dc F1=0.013（高 sep 稀有类难以无监督检出），pancreas_baron F1=0.618，tabula_lung_endo F1=0.968（强离群信号），tabula_small_intestine F1=0.501（高 recall 但 precision 极低，过度检测），tabula_lung_stroma F1=0.352（all methods low）。
- **TOSICA**（Nature Communications 2023）：Transformer + pathway token 可解释注释（max_gs=100，epochs=10，免疫数据集用 human_immune 掩码，其余用 human_gobp）。在 immune_dc（0.954）和 pancreas_baron（0.899）上超过 scRareRefine，在 tabula_lung_endo（0.901 vs 0.980）和 tabula_small_intestine（0.956 vs 0.977）上低于 scRareRefine，在 low-sep tabula_lung_stroma 上（0.349 vs 0.390）也略低。TOSICA 的优势来自 pathway 先验知识，对有免疫/胰腺特异通路的稀有类型尤为有效；三 seed 结果完全一致（std=0.000），训练稳定（内部使用固定 GLOBAL_SEED=1）。

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
- 主体运行环境：`D:/setup/anaconda/envs/scanvi311/python.exe`（NumPy 1.26.4，PyTorch 2.2.2+cu118）
- ProtoCloud / HiCat / TOSICA 运行环境：`D:/setup/anaconda/envs/sandbox310/python.exe`（NumPy 1.x，torch 2.0.1+cu118）
- scCAD 运行环境：`D:/setup/anaconda/envs/scanvi311/python.exe`
- NumPy 版本注意：scanvi311 环境需 NumPy<2.0（已固定为 1.26.4）；PyTorch 2.2.2+cu118 用 NumPy 1.x 编译，NumPy 2.x 导致 DataLoader collate 失败
- immune_dc h5ad 格式问题：sandbox310 anndata 0.11 读取失败，sandbox310 脚本均通过 scanvi311 subprocess 提取 count 矩阵（自动 fallback，无需手动干预）
- TOSICA 磁盘注意：每次训练约产生 10 个 ×57 MB 临时模型文件，脚本在预测完成后自动清理（`tmp/tosica/` 及 `outputs/*/tosica_cache/` 均为可再生临时文件）
- **新增数据集**（Tabula Sapiens，CellxGene）：tabula_small_intestine（肠道微绒毛 tuft cell，40350 cells，5 donors）和 tabula_lung_stroma（支气管平滑肌细胞，2283 cells，4 donors）；两者均用 `tools/extract_tabula_*.py` 提取 10X compartment
