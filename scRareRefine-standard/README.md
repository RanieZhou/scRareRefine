# scRareRefine 项目说明

本项目暂定名为 **scRareRefine**，目标是基于 **scANVI** 的输出结果，包括预测概率和 latent embedding，设计一个面向稀有细胞类型识别的 refinement 模块。

项目当前定位：

```text
scANVI 基础模型
  ├── 预测概率 probability
  ├── latent embedding
  ↓
prototype / uncertainty / rare-aware refinement
  ↓
提升稀有细胞类型识别效果
```

本仓库建议按照 `PROJECT_STRUCTURE.md` 中的目录结构维护。所有 AI agent，包括 ARIS、Codex、Claude Code，都必须优先阅读：

```text
AGENTS.md
CLAUDE.md
PROJECT_STRUCTURE.md
docs/PROJECT_BRIEF.md
docs/EXPERIMENT_PLAN.md
docs/CLAIMS.md
RESULTS_LOG.md
```

## 当前核心目标

1. 以 scANVI 为基础模型。
2. 使用 scANVI 输出的预测概率和 latent embedding。
3. 设计 prototype-based / uncertainty-aware / rare-aware refinement 模块。
4. 在小规模数据集上验证稀有细胞识别效果。
5. 目标产出一篇 Q2 级别小论文，并作为硕士论文三个工作之一。

## 当前数据集

已知数据集：

```text
human_pancreas_norm_complexBatch.h5ad
Tabula Sapiens
human_immune_health_atlas_dc.h5ad
```

注意：原始数据应放在 `data/raw/`，并且该目录默认只读，禁止 AI agent 修改。
