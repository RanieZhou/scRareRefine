# Ablation Report（重构版，3-seed）

**Date**: 2026-06-21  |  **Seeds**: [42, 43, 44]  |  **Datasets**: 6  |  **rts**: ['0.01', '0.05', '0.10', 'all']

真正可拆组件 = 4 个（sep / necessity 弃权闸门 + 自适应rank / τ 拯救机制）。
表 1 答「每个组件该不该留」，表 2 答「自适应 rank 为何优于任何固定值」。
聚合单元 = 每 (dataset, variant) 的 4 rts × 3 seed = 12 cell；f1_std 含 rts 轴差异（非纯 seed 方差）。

## 表 1 · 组件留一法（leave-one-out）

| dataset | variant | F1 mean±std | recall | gain vs baseline | **Δ=Full−变体** | FFR_max | abstain |
|---|---|---|---|---|---|---|---|
| OVERALL | A0_baseline | 0.761±0.301 | 0.710 | +0.000 | +0.127 | 0.00000 | 0 |
| OVERALL | A1_minus_sep | 0.905±0.103 | 0.879 | +0.144 | -0.018 | 0.01526 | 31 |
| OVERALL | A2_minus_necessity | 0.885±0.150 | 0.854 | +0.124 | +0.002 | 0.00977 | 8 |
| OVERALL | A3_minus_adaptive_rank | 0.877±0.164 | 0.838 | +0.116 | +0.010 | 0.00488 | 37 |
| OVERALL | A4_minus_tau | 0.885±0.153 | 0.853 | +0.124 | +0.002 | 0.01648 | 37 |
| OVERALL | A5_full | 0.887±0.151 | 0.853 | +0.127 | +0.000 | 0.00977 | 37 |
| immune_dc | A0_baseline | 0.681±0.395 | 0.631 | +0.000 | +0.258 | 0.00000 | 0 |
| immune_dc | A1_minus_sep | 0.939±0.014 | 0.906 | +0.258 | +0.000 | 0.00100 | 0 |
| immune_dc | A2_minus_necessity | 0.939±0.014 | 0.906 | +0.258 | +0.000 | 0.00100 | 0 |
| immune_dc | A3_minus_adaptive_rank | 0.939±0.014 | 0.906 | +0.258 | +0.000 | 0.00100 | 0 |
| immune_dc | A4_minus_tau | 0.938±0.017 | 0.906 | +0.257 | +0.001 | 0.00100 | 0 |
| immune_dc | A5_full | 0.939±0.014 | 0.906 | +0.258 | +0.000 | 0.00100 | 0 |
| pancreas_baron | A0_baseline | 0.624±0.285 | 0.533 | +0.000 | +0.098 | 0.00000 | 0 |
| pancreas_baron | A1_minus_sep | 0.830±0.057 | 0.811 | +0.205 | -0.108 | 0.01526 | 2 |
| pancreas_baron | A2_minus_necessity | 0.722±0.227 | 0.653 | +0.098 | +0.000 | 0.00977 | 8 |
| pancreas_baron | A3_minus_adaptive_rank | 0.702±0.227 | 0.612 | +0.077 | +0.020 | 0.00488 | 8 |
| pancreas_baron | A4_minus_tau | 0.712±0.225 | 0.658 | +0.087 | +0.010 | 0.01648 | 8 |
| pancreas_baron | A5_full | 0.722±0.227 | 0.653 | +0.098 | +0.000 | 0.00977 | 8 |
| pancreas_integrated | A0_baseline | 0.906±0.274 | 0.917 | +0.000 | +0.083 | 0.00000 | 0 |
| pancreas_integrated | A1_minus_sep | 0.990±0.016 | 1.000 | +0.083 | +0.000 | 0.00000 | 11 |
| pancreas_integrated | A2_minus_necessity | 0.977±0.020 | 1.000 | +0.071 | +0.012 | 0.00176 | 0 |
| pancreas_integrated | A3_minus_adaptive_rank | 0.990±0.016 | 1.000 | +0.083 | +0.000 | 0.00000 | 11 |
| pancreas_integrated | A4_minus_tau | 0.990±0.016 | 1.000 | +0.083 | +0.000 | 0.00000 | 11 |
| pancreas_integrated | A5_full | 0.990±0.016 | 1.000 | +0.083 | +0.000 | 0.00000 | 11 |
| tabula_lung_endo | A0_baseline | 0.780±0.323 | 0.746 | +0.000 | +0.196 | 0.00000 | 0 |
| tabula_lung_endo | A1_minus_sep | 0.977±0.009 | 0.986 | +0.196 | +0.000 | 0.00233 | 6 |
| tabula_lung_endo | A2_minus_necessity | 0.977±0.008 | 0.995 | +0.197 | -0.001 | 0.00233 | 0 |
| tabula_lung_endo | A3_minus_adaptive_rank | 0.977±0.009 | 0.986 | +0.196 | +0.000 | 0.00233 | 6 |
| tabula_lung_endo | A4_minus_tau | 0.976±0.009 | 0.986 | +0.195 | +0.001 | 0.00233 | 6 |
| tabula_lung_endo | A5_full | 0.977±0.009 | 0.986 | +0.196 | +0.000 | 0.00233 | 6 |
| tabula_sapiens_stomach | A0_baseline | 0.595±0.074 | 0.435 | +0.000 | +0.124 | 0.00000 | 0 |
| tabula_sapiens_stomach | A1_minus_sep | 0.719±0.021 | 0.570 | +0.124 | +0.000 | 0.00000 | 0 |
| tabula_sapiens_stomach | A2_minus_necessity | 0.719±0.021 | 0.570 | +0.124 | +0.000 | 0.00000 | 0 |
| tabula_sapiens_stomach | A3_minus_adaptive_rank | 0.679±0.042 | 0.523 | +0.084 | +0.040 | 0.00000 | 0 |
| tabula_sapiens_stomach | A4_minus_tau | 0.719±0.021 | 0.570 | +0.124 | +0.000 | 0.00000 | 0 |
| tabula_sapiens_stomach | A5_full | 0.719±0.021 | 0.570 | +0.124 | +0.000 | 0.00000 | 0 |
| tabula_small_intestine | A0_baseline | 0.979±0.004 | 1.000 | +0.000 | +0.000 | 0.00000 | 0 |
| tabula_small_intestine | A1_minus_sep | 0.979±0.004 | 1.000 | +0.000 | +0.000 | 0.00000 | 12 |
| tabula_small_intestine | A2_minus_necessity | 0.976±0.004 | 1.000 | -0.003 | +0.003 | 0.00032 | 0 |
| tabula_small_intestine | A3_minus_adaptive_rank | 0.979±0.004 | 1.000 | +0.000 | +0.000 | 0.00000 | 12 |
| tabula_small_intestine | A4_minus_tau | 0.979±0.004 | 1.000 | +0.000 | +0.000 | 0.00000 | 12 |
| tabula_small_intestine | A5_full | 0.979±0.004 | 1.000 | +0.000 | +0.000 | 0.00000 | 12 |

> 读法：`Δ=Full−变体`（逐 cell 均值）。**正 = 去掉该组件 F1 掉这么多（对 F1 有正贡献）**；
> **负 = 去掉反而升 → 该组件价值在 FFR/安全而非 F1，须看 FFR_max**（如 −sep 升 F1 但 FFR 破 α；−τ 同理）。
> A5_full 的 Δ 恒为 0（自比）；A0_baseline 的 gain vs baseline 恒为 0。

## 表 2 · rank 敏感性（自适应 vs 固定）

| dataset | variant | F1 mean±std | recall | FFR_max | abstain |
|---|---|---|---|---|---|
| OVERALL | R1_rank1 | 0.877±0.164 | 0.838 | 0.00488 | 37 |
| OVERALL | R2_rank2 | 0.865±0.148 | 0.868 | 0.00999 | 37 |
| OVERALL | R3_rank3 | 0.853±0.150 | 0.875 | 0.04640 | 37 |
| OVERALL | R_adaptive | 0.887±0.151 | 0.853 | 0.00977 | 37 |
| immune_dc | R1_rank1 | 0.939±0.014 | 0.906 | 0.00100 | 0 |
| immune_dc | R2_rank2 | 0.821±0.016 | 0.972 | 0.00999 | 0 |
| immune_dc | R3_rank3 | 0.808±0.015 | 0.973 | 0.01133 | 0 |
| immune_dc | R_adaptive | 0.939±0.014 | 0.906 | 0.00100 | 0 |
| pancreas_baron | R1_rank1 | 0.702±0.227 | 0.612 | 0.00488 | 8 |
| pancreas_baron | R2_rank2 | 0.723±0.228 | 0.663 | 0.00977 | 8 |
| pancreas_baron | R3_rank3 | 0.697±0.227 | 0.689 | 0.04640 | 8 |
| pancreas_baron | R_adaptive | 0.722±0.227 | 0.653 | 0.00977 | 8 |
| pancreas_integrated | R1_rank1 | 0.990±0.016 | 1.000 | 0.00000 | 11 |
| pancreas_integrated | R2_rank2 | 0.983±0.025 | 1.000 | 0.00235 | 11 |
| pancreas_integrated | R3_rank3 | 0.971±0.060 | 1.000 | 0.00764 | 11 |
| pancreas_integrated | R_adaptive | 0.990±0.016 | 1.000 | 0.00000 | 11 |
| tabula_lung_endo | R1_rank1 | 0.977±0.009 | 0.986 | 0.00233 | 6 |
| tabula_lung_endo | R2_rank2 | 0.957±0.025 | 0.989 | 0.00758 | 6 |
| tabula_lung_endo | R3_rank3 | 0.927±0.060 | 0.991 | 0.01690 | 6 |
| tabula_lung_endo | R_adaptive | 0.977±0.009 | 0.986 | 0.00233 | 6 |
| tabula_sapiens_stomach | R1_rank1 | 0.679±0.042 | 0.523 | 0.00000 | 0 |
| tabula_sapiens_stomach | R2_rank2 | 0.729±0.043 | 0.583 | 0.00000 | 0 |
| tabula_sapiens_stomach | R3_rank3 | 0.735±0.033 | 0.596 | 0.00011 | 0 |
| tabula_sapiens_stomach | R_adaptive | 0.719±0.021 | 0.570 | 0.00000 | 0 |
| tabula_small_intestine | R1_rank1 | 0.979±0.004 | 1.000 | 0.00000 | 12 |
| tabula_small_intestine | R2_rank2 | 0.979±0.004 | 1.000 | 0.00000 | 12 |
| tabula_small_intestine | R3_rank3 | 0.979±0.004 | 1.000 | 0.00000 | 12 |
| tabula_small_intestine | R_adaptive | 0.979±0.004 | 1.000 | 0.00000 | 12 |

> R1_rank1 == A3_minus_adaptive_rank（去自适应=退回固定 rank=1）；R_adaptive == A5_full（交叉引用+一致性自检）。
> 看点：固定 rank=3 时 FFR_max 是否冲破 α=0.01；自适应是否在 FFR≤α 下拿到 ≥ 任何固定值的 F1。