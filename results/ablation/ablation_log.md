# Round 10 Ablation Report

**Date**: 2026-06-19  |  **Seed**: 42  |  **Datasets**: 6  |  **rts**: 0.01/0.05/0.10/all

## 变体定义

| 变体 | 改动 |
|------|------|
| V0 baseline_scanvi   | 完全不 rescue（参照线） |
| V1 no_sep_gate       | 关 separability 安全网（LOW_SEP=0） |
| V2 no_necessity      | 关 necessity 守门 |
| V3 rank1_fixed       | 固定 rank=1，无 val-自适应 |
| V4 rank2_fixed       | 固定 rank=2，无 val-自适应 |
| V5 no_conformal_tau  | 候选直接全 relabel（去 τ） |
| V6 full              | 当前 conformal_rescue（reference） |

## 聚合表（按数据集 × 变体）

| dataset | variant | n | F1 mean | recall mean | prec mean | FFR_max | gain mean | n_abstain | rescued_total | false_total |
|---------|---------|---|---------|-------------|-----------|---------|-----------|-----------|---------------|-------------|
| immune_dc | V0_baseline_scanvi | 4 | 0.6648 | 0.6038 | 0.7434 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| immune_dc | V1_no_sep_gate | 4 | 0.9317 | 0.8846 | 0.9855 | 0.00033 | +0.2668 | 0 | 150 | 4 |
| immune_dc | V2_no_necessity | 4 | 0.9317 | 0.8846 | 0.9855 | 0.00033 | +0.2668 | 0 | 150 | 4 |
| immune_dc | V3_rank1_fixed | 4 | 0.9317 | 0.8846 | 0.9855 | 0.00033 | +0.2668 | 0 | 150 | 4 |
| immune_dc | V4_rank2_fixed | 4 | 0.8225 | 0.9730 | 0.7130 | 0.00966 | +0.1576 | 0 | 394 | 202 |
| immune_dc | V5_no_conformal_tau | 4 | 0.9279 | 0.8846 | 0.9766 | 0.00050 | +0.2631 | 0 | 154 | 8 |
| immune_dc | V6_full | 4 | 0.9317 | 0.8846 | 0.9855 | 0.00033 | +0.2668 | 0 | 150 | 4 |
| immune_dc | V7_rank3_fixed | 4 | 0.8133 | 0.9730 | 0.6995 | 0.01033 | +0.1485 | 0 | 408 | 216 |
| pancreas_baron | V0_baseline_scanvi | 4 | 0.5399 | 0.4622 | 0.9704 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| pancreas_baron | V1_no_sep_gate | 4 | 0.8184 | 0.7965 | 0.8496 | 0.00977 | +0.2785 | 0 | 155 | 40 |
| pancreas_baron | V2_no_necessity | 4 | 0.8217 | 0.7907 | 0.8608 | 0.00977 | +0.2818 | 1 | 148 | 35 |
| pancreas_baron | V3_rank1_fixed | 4 | 0.7609 | 0.6686 | 0.9179 | 0.00244 | +0.2210 | 1 | 82 | 11 |
| pancreas_baron | V4_rank2_fixed | 4 | 0.8278 | 0.8169 | 0.8398 | 0.00977 | +0.2880 | 1 | 165 | 43 |
| pancreas_baron | V5_no_conformal_tau | 4 | 0.8040 | 0.7907 | 0.8247 | 0.01465 | +0.2641 | 1 | 164 | 51 |
| pancreas_baron | V6_full | 4 | 0.8217 | 0.7907 | 0.8608 | 0.00977 | +0.2818 | 1 | 148 | 35 |
| pancreas_baron | V7_rank3_fixed | 4 | 0.7503 | 0.8954 | 0.6606 | 0.04640 | +0.2104 | 1 | 326 | 177 |
| pancreas_integrated | V0_baseline_scanvi | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| pancreas_integrated | V1_no_sep_gate | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| pancreas_integrated | V2_no_necessity | 4 | 0.9636 | 1.0000 | 0.9303 | 0.00176 | -0.0206 | 0 | 4 | 4 |
| pancreas_integrated | V3_rank1_fixed | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| pancreas_integrated | V4_rank2_fixed | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| pancreas_integrated | V5_no_conformal_tau | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| pancreas_integrated | V6_full | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| pancreas_integrated | V7_rank3_fixed | 4 | 0.9842 | 1.0000 | 0.9696 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_lung_endo | V0_baseline_scanvi | 4 | 0.6553 | 0.6231 | 0.7241 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| tabula_lung_endo | V1_no_sep_gate | 4 | 0.9756 | 0.9962 | 0.9561 | 0.00233 | +0.3203 | 2 | 103 | 6 |
| tabula_lung_endo | V2_no_necessity | 4 | 0.9720 | 1.0000 | 0.9458 | 0.00233 | +0.3167 | 0 | 107 | 9 |
| tabula_lung_endo | V3_rank1_fixed | 4 | 0.9756 | 0.9962 | 0.9561 | 0.00233 | +0.3203 | 2 | 103 | 6 |
| tabula_lung_endo | V4_rank2_fixed | 4 | 0.9465 | 0.9962 | 0.9038 | 0.00758 | +0.2912 | 2 | 120 | 23 |
| tabula_lung_endo | V5_no_conformal_tau | 4 | 0.9756 | 0.9962 | 0.9561 | 0.00233 | +0.3203 | 2 | 103 | 6 |
| tabula_lung_endo | V6_full | 4 | 0.9756 | 0.9962 | 0.9561 | 0.00233 | +0.3203 | 2 | 103 | 6 |
| tabula_lung_endo | V7_rank3_fixed | 4 | 0.9179 | 0.9962 | 0.8577 | 0.01399 | +0.2626 | 2 | 139 | 42 |
| tabula_sapiens_stomach | V0_baseline_scanvi | 4 | 0.5091 | 0.3438 | 1.0000 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| tabula_sapiens_stomach | V1_no_sep_gate | 4 | 0.7323 | 0.5782 | 1.0000 | 0.00000 | +0.2232 | 0 | 30 | 0 |
| tabula_sapiens_stomach | V2_no_necessity | 4 | 0.7323 | 0.5782 | 1.0000 | 0.00000 | +0.2232 | 0 | 30 | 0 |
| tabula_sapiens_stomach | V3_rank1_fixed | 4 | 0.6309 | 0.4610 | 1.0000 | 0.00000 | +0.1218 | 0 | 15 | 0 |
| tabula_sapiens_stomach | V4_rank2_fixed | 4 | 0.7323 | 0.5782 | 1.0000 | 0.00000 | +0.2232 | 0 | 30 | 0 |
| tabula_sapiens_stomach | V5_no_conformal_tau | 4 | 0.7323 | 0.5782 | 1.0000 | 0.00000 | +0.2232 | 0 | 30 | 0 |
| tabula_sapiens_stomach | V6_full | 4 | 0.7323 | 0.5782 | 1.0000 | 0.00000 | +0.2232 | 0 | 30 | 0 |
| tabula_sapiens_stomach | V7_rank3_fixed | 4 | 0.7246 | 0.5860 | 0.9494 | 0.00011 | +0.2154 | 0 | 35 | 4 |
| tabula_small_intestine | V0_baseline_scanvi | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 0 | 0 | 0 |
| tabula_small_intestine | V1_no_sep_gate | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_small_intestine | V2_no_necessity | 4 | 0.9701 | 1.0000 | 0.9420 | 0.00032 | -0.0073 | 0 | 4 | 4 |
| tabula_small_intestine | V3_rank1_fixed | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_small_intestine | V4_rank2_fixed | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_small_intestine | V5_no_conformal_tau | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_small_intestine | V6_full | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |
| tabula_small_intestine | V7_rank3_fixed | 4 | 0.9774 | 1.0000 | 0.9560 | 0.00000 | +0.0000 | 4 | 0 | 0 |

## 逐配置明细

| dataset | rts | variant | sep | F1 | recall | prec | rank | rescued | false | FFR | abstain |
|---------|-----|---------|-----|-----|--------|------|------|---------|-------|-----|---------|
| immune_dc | 0.01 | V0_baseline_scanvi | 2.113 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.00000 | N |
| immune_dc | 0.01 | V1_no_sep_gate | 2.113 | 0.9030 | 0.8231 | 1.0000 | 1 | 107 | 0 | 0.00000 | N |
| immune_dc | 0.01 | V2_no_necessity | 2.113 | 0.9030 | 0.8231 | 1.0000 | 1 | 107 | 0 | 0.00000 | N |
| immune_dc | 0.01 | V3_rank1_fixed | 2.113 | 0.9030 | 0.8231 | 1.0000 | 1 | 107 | 0 | 0.00000 | N |
| immune_dc | 0.01 | V4_rank2_fixed | 2.113 | 0.8378 | 0.9538 | 0.7470 | 2 | 166 | 42 | 0.00700 | N |
| immune_dc | 0.01 | V5_no_conformal_tau | 2.113 | 0.8917 | 0.8231 | 0.9727 | 1 | 110 | 3 | 0.00050 | N |
| immune_dc | 0.01 | V6_full | 2.113 | 0.9030 | 0.8231 | 1.0000 | 1 | 107 | 0 | 0.00000 | N |
| immune_dc | 0.01 | V7_rank3_fixed | 2.113 | 0.8378 | 0.9538 | 0.7470 | 3 | 166 | 42 | 0.00700 | N |
| immune_dc | 0.05 | V0_baseline_scanvi | 2.138 | 0.8393 | 0.7231 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| immune_dc | 0.05 | V1_no_sep_gate | 2.138 | 0.9435 | 0.9000 | 0.9915 | 1 | 24 | 1 | 0.00017 | N |
| immune_dc | 0.05 | V2_no_necessity | 2.138 | 0.9435 | 0.9000 | 0.9915 | 1 | 24 | 1 | 0.00017 | N |
| immune_dc | 0.05 | V3_rank1_fixed | 2.138 | 0.9435 | 0.9000 | 0.9915 | 1 | 24 | 1 | 0.00017 | N |
| immune_dc | 0.05 | V4_rank2_fixed | 2.138 | 0.8063 | 0.9769 | 0.6865 | 2 | 91 | 58 | 0.00966 | N |
| immune_dc | 0.05 | V5_no_conformal_tau | 2.138 | 0.9398 | 0.9000 | 0.9832 | 1 | 25 | 2 | 0.00033 | N |
| immune_dc | 0.05 | V6_full | 2.138 | 0.9435 | 0.9000 | 0.9915 | 1 | 24 | 1 | 0.00017 | N |
| immune_dc | 0.05 | V7_rank3_fixed | 2.138 | 0.7962 | 0.9769 | 0.6720 | 3 | 95 | 62 | 0.01033 | N |
| immune_dc | 0.10 | V0_baseline_scanvi | 2.104 | 0.8803 | 0.7923 | 0.9904 | 0 | 0 | 0 | 0.00000 | N |
| immune_dc | 0.10 | V1_no_sep_gate | 2.104 | 0.9274 | 0.8846 | 0.9746 | 1 | 14 | 2 | 0.00033 | N |
| immune_dc | 0.10 | V2_no_necessity | 2.104 | 0.9274 | 0.8846 | 0.9746 | 1 | 14 | 2 | 0.00033 | N |
| immune_dc | 0.10 | V3_rank1_fixed | 2.104 | 0.9274 | 0.8846 | 0.9746 | 1 | 14 | 2 | 0.00033 | N |
| immune_dc | 0.10 | V4_rank2_fixed | 2.104 | 0.8383 | 0.9769 | 0.7341 | 2 | 69 | 45 | 0.00749 | N |
| immune_dc | 0.10 | V5_no_conformal_tau | 2.104 | 0.9274 | 0.8846 | 0.9746 | 1 | 14 | 2 | 0.00033 | N |
| immune_dc | 0.10 | V6_full | 2.104 | 0.9274 | 0.8846 | 0.9746 | 1 | 14 | 2 | 0.00033 | N |
| immune_dc | 0.10 | V7_rank3_fixed | 2.104 | 0.8167 | 0.9769 | 0.7017 | 3 | 77 | 53 | 0.00883 | N |
| immune_dc | all | V0_baseline_scanvi | 1.762 | 0.9398 | 0.9000 | 0.9832 | 0 | 0 | 0 | 0.00000 | N |
| immune_dc | all | V1_no_sep_gate | 1.762 | 0.9528 | 0.9308 | 0.9758 | 1 | 5 | 1 | 0.00017 | N |
| immune_dc | all | V2_no_necessity | 1.762 | 0.9528 | 0.9308 | 0.9758 | 1 | 5 | 1 | 0.00017 | N |
| immune_dc | all | V3_rank1_fixed | 1.762 | 0.9528 | 0.9308 | 0.9758 | 1 | 5 | 1 | 0.00017 | N |
| immune_dc | all | V4_rank2_fixed | 1.762 | 0.8076 | 0.9846 | 0.6845 | 2 | 68 | 57 | 0.00949 | N |
| immune_dc | all | V5_no_conformal_tau | 1.762 | 0.9528 | 0.9308 | 0.9758 | 1 | 5 | 1 | 0.00017 | N |
| immune_dc | all | V6_full | 1.762 | 0.9528 | 0.9308 | 0.9758 | 1 | 5 | 1 | 0.00017 | N |
| immune_dc | all | V7_rank3_fixed | 1.762 | 0.8025 | 0.9846 | 0.6772 | 3 | 70 | 59 | 0.00983 | N |
| pancreas_baron | 0.01 | V0_baseline_scanvi | 1.521 | 0.2268 | 0.1279 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_baron | 0.01 | V1_no_sep_gate | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.01 | V2_no_necessity | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.01 | V3_rank1_fixed | 1.521 | 0.6567 | 0.5116 | 0.9167 | 1 | 37 | 4 | 0.00244 | N |
| pancreas_baron | 0.01 | V4_rank2_fixed | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.01 | V5_no_conformal_tau | 1.521 | 0.7429 | 0.7558 | 0.7303 | 2 | 78 | 24 | 0.01465 | N |
| pancreas_baron | 0.01 | V6_full | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.01 | V7_rank3_fixed | 1.521 | 0.6329 | 0.8721 | 0.4967 | 3 | 140 | 76 | 0.04640 | N |
| pancreas_baron | 0.05 | V0_baseline_scanvi | 1.521 | 0.2268 | 0.1279 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_baron | 0.05 | V1_no_sep_gate | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.05 | V2_no_necessity | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.05 | V3_rank1_fixed | 1.521 | 0.6567 | 0.5116 | 0.9167 | 1 | 37 | 4 | 0.00244 | N |
| pancreas_baron | 0.05 | V4_rank2_fixed | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.05 | V5_no_conformal_tau | 1.521 | 0.7429 | 0.7558 | 0.7303 | 2 | 78 | 24 | 0.01465 | N |
| pancreas_baron | 0.05 | V6_full | 1.521 | 0.7784 | 0.7558 | 0.8025 | 2 | 70 | 16 | 0.00977 | N |
| pancreas_baron | 0.05 | V7_rank3_fixed | 1.521 | 0.6329 | 0.8721 | 0.4967 | 3 | 140 | 76 | 0.04640 | N |
| pancreas_baron | 0.10 | V0_baseline_scanvi | 1.442 | 0.7917 | 0.6628 | 0.9828 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_baron | 0.10 | V1_no_sep_gate | 1.442 | 0.8158 | 0.7209 | 0.9394 | 1 | 8 | 3 | 0.00183 | N |
| pancreas_baron | 0.10 | V2_no_necessity | 1.442 | 0.8158 | 0.7209 | 0.9394 | 1 | 8 | 3 | 0.00183 | N |
| pancreas_baron | 0.10 | V3_rank1_fixed | 1.442 | 0.8158 | 0.7209 | 0.9394 | 1 | 8 | 3 | 0.00183 | N |
| pancreas_baron | 0.10 | V4_rank2_fixed | 1.442 | 0.8402 | 0.8256 | 0.8554 | 2 | 25 | 11 | 0.00672 | N |
| pancreas_baron | 0.10 | V5_no_conformal_tau | 1.442 | 0.8158 | 0.7209 | 0.9394 | 1 | 8 | 3 | 0.00183 | N |
| pancreas_baron | 0.10 | V6_full | 1.442 | 0.8158 | 0.7209 | 0.9394 | 1 | 8 | 3 | 0.00183 | N |
| pancreas_baron | 0.10 | V7_rank3_fixed | 1.442 | 0.8211 | 0.9070 | 0.7500 | 3 | 46 | 25 | 0.01526 | N |
| pancreas_baron | all | V0_baseline_scanvi | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_baron | all | V1_no_sep_gate | 1.160 | 0.9011 | 0.9535 | 0.8542 | 1 | 7 | 5 | 0.00305 | N |
| pancreas_baron | all | V2_no_necessity | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_baron | all | V3_rank1_fixed | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_baron | all | V4_rank2_fixed | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_baron | all | V5_no_conformal_tau | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_baron | all | V6_full | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_baron | all | V7_rank3_fixed | 1.160 | 0.9143 | 0.9302 | 0.8989 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V0_baseline_scanvi | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | 0.01 | V1_no_sep_gate | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V2_no_necessity | 1.452 | 0.9388 | 1.0000 | 0.8846 | 2 | 3 | 3 | 0.00176 | N |
| pancreas_integrated | 0.01 | V3_rank1_fixed | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V4_rank2_fixed | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V5_no_conformal_tau | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V6_full | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.01 | V7_rank3_fixed | 1.452 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V0_baseline_scanvi | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | 0.05 | V1_no_sep_gate | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V2_no_necessity | 1.543 | 0.9787 | 1.0000 | 0.9583 | 1 | 1 | 1 | 0.00059 | N |
| pancreas_integrated | 0.05 | V3_rank1_fixed | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V4_rank2_fixed | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V5_no_conformal_tau | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V6_full | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.05 | V7_rank3_fixed | 1.543 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V0_baseline_scanvi | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | 0.10 | V1_no_sep_gate | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V2_no_necessity | 1.341 | 0.9787 | 1.0000 | 0.9583 | 1 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | 0.10 | V3_rank1_fixed | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V4_rank2_fixed | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V5_no_conformal_tau | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V6_full | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | 0.10 | V7_rank3_fixed | 1.341 | 0.9787 | 1.0000 | 0.9583 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V0_baseline_scanvi | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | all | V1_no_sep_gate | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V2_no_necessity | 1.444 | 0.9583 | 1.0000 | 0.9200 | 1 | 0 | 0 | 0.00000 | N |
| pancreas_integrated | all | V3_rank1_fixed | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V4_rank2_fixed | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V5_no_conformal_tau | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V6_full | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| pancreas_integrated | all | V7_rank3_fixed | 1.444 | 0.9583 | 1.0000 | 0.9200 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.01 | V0_baseline_scanvi | 2.347 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 0 | 0.00000 | N |
| tabula_lung_endo | 0.01 | V1_no_sep_gate | 2.347 | 0.9848 | 1.0000 | 0.9701 | 1 | 67 | 2 | 0.00117 | N |
| tabula_lung_endo | 0.01 | V2_no_necessity | 2.347 | 0.9848 | 1.0000 | 0.9701 | 1 | 67 | 2 | 0.00117 | N |
| tabula_lung_endo | 0.01 | V3_rank1_fixed | 2.347 | 0.9848 | 1.0000 | 0.9701 | 1 | 67 | 2 | 0.00117 | N |
| tabula_lung_endo | 0.01 | V4_rank2_fixed | 2.347 | 0.9286 | 1.0000 | 0.8667 | 2 | 75 | 10 | 0.00583 | N |
| tabula_lung_endo | 0.01 | V5_no_conformal_tau | 2.347 | 0.9848 | 1.0000 | 0.9701 | 1 | 67 | 2 | 0.00117 | N |
| tabula_lung_endo | 0.01 | V6_full | 2.347 | 0.9848 | 1.0000 | 0.9701 | 1 | 67 | 2 | 0.00117 | N |
| tabula_lung_endo | 0.01 | V7_rank3_fixed | 2.347 | 0.8784 | 1.0000 | 0.7831 | 3 | 83 | 18 | 0.01049 | N |
| tabula_lung_endo | 0.05 | V0_baseline_scanvi | 1.879 | 0.6667 | 0.5077 | 0.9706 | 0 | 0 | 0 | 0.00000 | N |
| tabula_lung_endo | 0.05 | V1_no_sep_gate | 1.879 | 0.9630 | 1.0000 | 0.9286 | 1 | 36 | 4 | 0.00233 | N |
| tabula_lung_endo | 0.05 | V2_no_necessity | 1.879 | 0.9630 | 1.0000 | 0.9286 | 1 | 36 | 4 | 0.00233 | N |
| tabula_lung_endo | 0.05 | V3_rank1_fixed | 1.879 | 0.9630 | 1.0000 | 0.9286 | 1 | 36 | 4 | 0.00233 | N |
| tabula_lung_endo | 0.05 | V4_rank2_fixed | 1.879 | 0.9028 | 1.0000 | 0.8228 | 2 | 45 | 13 | 0.00758 | N |
| tabula_lung_endo | 0.05 | V5_no_conformal_tau | 1.879 | 0.9630 | 1.0000 | 0.9286 | 1 | 36 | 4 | 0.00233 | N |
| tabula_lung_endo | 0.05 | V6_full | 1.879 | 0.9630 | 1.0000 | 0.9286 | 1 | 36 | 4 | 0.00233 | N |
| tabula_lung_endo | 0.05 | V7_rank3_fixed | 1.879 | 0.8387 | 1.0000 | 0.7222 | 3 | 56 | 24 | 0.01399 | N |
| tabula_lung_endo | 0.10 | V0_baseline_scanvi | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | N |
| tabula_lung_endo | 0.10 | V1_no_sep_gate | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.10 | V2_no_necessity | 1.659 | 0.9774 | 1.0000 | 0.9559 | 1 | 2 | 1 | 0.00058 | N |
| tabula_lung_endo | 0.10 | V3_rank1_fixed | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.10 | V4_rank2_fixed | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.10 | V5_no_conformal_tau | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.10 | V6_full | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | 0.10 | V7_rank3_fixed | 1.659 | 0.9771 | 0.9846 | 0.9697 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V0_baseline_scanvi | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | N |
| tabula_lung_endo | all | V1_no_sep_gate | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V2_no_necessity | 1.671 | 0.9630 | 1.0000 | 0.9286 | 1 | 2 | 2 | 0.00117 | N |
| tabula_lung_endo | all | V3_rank1_fixed | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V4_rank2_fixed | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V5_no_conformal_tau | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V6_full | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_lung_endo | all | V7_rank3_fixed | 1.671 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_sapiens_stomach | 0.01 | V0_baseline_scanvi | 1.779 | 0.5455 | 0.3750 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V1_no_sep_gate | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V2_no_necessity | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V3_rank1_fixed | 1.779 | 0.6383 | 0.4688 | 1.0000 | 1 | 3 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V4_rank2_fixed | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V5_no_conformal_tau | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V6_full | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.01 | V7_rank3_fixed | 1.779 | 0.7308 | 0.5938 | 0.9500 | 3 | 8 | 1 | 0.00011 | N |
| tabula_sapiens_stomach | 0.05 | V0_baseline_scanvi | 1.779 | 0.5455 | 0.3750 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V1_no_sep_gate | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V2_no_necessity | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V3_rank1_fixed | 1.779 | 0.6383 | 0.4688 | 1.0000 | 1 | 3 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V4_rank2_fixed | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V5_no_conformal_tau | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V6_full | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.05 | V7_rank3_fixed | 1.779 | 0.7308 | 0.5938 | 0.9500 | 3 | 8 | 1 | 0.00011 | N |
| tabula_sapiens_stomach | 0.10 | V0_baseline_scanvi | 1.779 | 0.5455 | 0.3750 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V1_no_sep_gate | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V2_no_necessity | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V3_rank1_fixed | 1.779 | 0.6383 | 0.4688 | 1.0000 | 1 | 3 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V4_rank2_fixed | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V5_no_conformal_tau | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V6_full | 1.779 | 0.7451 | 0.5938 | 1.0000 | 2 | 7 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | 0.10 | V7_rank3_fixed | 1.779 | 0.7308 | 0.5938 | 0.9500 | 3 | 8 | 1 | 0.00011 | N |
| tabula_sapiens_stomach | all | V0_baseline_scanvi | 1.818 | 0.4000 | 0.2500 | 1.0000 | 0 | 0 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V1_no_sep_gate | 1.818 | 0.6939 | 0.5312 | 1.0000 | 2 | 9 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V2_no_necessity | 1.818 | 0.6939 | 0.5312 | 1.0000 | 2 | 9 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V3_rank1_fixed | 1.818 | 0.6087 | 0.4375 | 1.0000 | 1 | 6 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V4_rank2_fixed | 1.818 | 0.6939 | 0.5312 | 1.0000 | 2 | 9 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V5_no_conformal_tau | 1.818 | 0.6939 | 0.5312 | 1.0000 | 2 | 9 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V6_full | 1.818 | 0.6939 | 0.5312 | 1.0000 | 2 | 9 | 0 | 0.00000 | N |
| tabula_sapiens_stomach | all | V7_rank3_fixed | 1.818 | 0.7059 | 0.5625 | 0.9474 | 3 | 11 | 1 | 0.00011 | N |
| tabula_small_intestine | 0.01 | V0_baseline_scanvi | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | N |
| tabula_small_intestine | 0.01 | V1_no_sep_gate | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.01 | V2_no_necessity | 3.165 | 0.9701 | 1.0000 | 0.9420 | 1 | 1 | 1 | 0.00016 | N |
| tabula_small_intestine | 0.01 | V3_rank1_fixed | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.01 | V4_rank2_fixed | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.01 | V5_no_conformal_tau | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.01 | V6_full | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.01 | V7_rank3_fixed | 3.165 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V0_baseline_scanvi | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | N |
| tabula_small_intestine | 0.05 | V1_no_sep_gate | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V2_no_necessity | 2.624 | 0.9701 | 1.0000 | 0.9420 | 1 | 1 | 1 | 0.00016 | N |
| tabula_small_intestine | 0.05 | V3_rank1_fixed | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V4_rank2_fixed | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V5_no_conformal_tau | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V6_full | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.05 | V7_rank3_fixed | 2.624 | 0.9774 | 1.0000 | 0.9559 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V0_baseline_scanvi | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | N |
| tabula_small_intestine | 0.10 | V1_no_sep_gate | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V2_no_necessity | 2.301 | 0.9701 | 1.0000 | 0.9420 | 1 | 2 | 2 | 0.00032 | N |
| tabula_small_intestine | 0.10 | V3_rank1_fixed | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V4_rank2_fixed | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V5_no_conformal_tau | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V6_full | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | 0.10 | V7_rank3_fixed | 2.301 | 0.9848 | 1.0000 | 0.9701 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V0_baseline_scanvi | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | N |
| tabula_small_intestine | all | V1_no_sep_gate | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V2_no_necessity | 2.448 | 0.9701 | 1.0000 | 0.9420 | 1 | 0 | 0 | 0.00000 | N |
| tabula_small_intestine | all | V3_rank1_fixed | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V4_rank2_fixed | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V5_no_conformal_tau | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V6_full | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |
| tabula_small_intestine | all | V7_rank3_fixed | 2.448 | 0.9701 | 1.0000 | 0.9420 | 0 | 0 | 0 | 0.00000 | Y |