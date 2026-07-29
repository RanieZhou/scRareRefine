# Adaptive Gate Decision-Seed Stability

Frozen v1 was rerun under 20 deterministic fold/bootstrap seeds for every batch-heldout unit with `S < 1.3`.
Test labels were not loaded and no frozen rule or threshold was modified.

- Low-S units: 15
- Consistent with frozen decision: 15/15
- Unstable/inconsistent units: 0
- Stable-pass definition: pass rate >= 0.80
- Stable-reject definition: pass rate <= 0.20

## Unit-level decisions

| dataset | seed | rts | S | frozen | pass rate | band | consistent | active folds | Wilson UCB q05–q95 | ΔF1 LCB q05–q95 |
|---|---:|---:|---:|---|---:|---|---|---|---|---|
| mouse_lung_tms_10x | 42 | 0.05 | 1.188 | pass | 1.00 | stable_pass | yes | 5/5.0/5 | 0.0037–0.0037 | 0.4363–0.4584 |
| mouse_lung_tms_10x | 42 | all | 1.208 | reject | 0.00 | stable_reject | yes | 0/1.0/2 | 0.0025–0.0086 | -0.0553–0.0000 |
| mouse_lung_tms_10x | 43 | 0.01 | 1.098 | pass | 1.00 | stable_pass | yes | 5/5.0/5 | 0.0058–0.0067 | 0.3025–0.3289 |
| mouse_lung_tms_10x | 43 | 0.05 | 1.105 | pass | 1.00 | stable_pass | yes | 5/5.0/5 | 0.0048–0.0048 | 0.4499–0.4737 |
| mouse_lung_tms_10x | 43 | 0.10 | 1.244 | pass | 1.00 | stable_pass | yes | 5/5.0/5 | 0.0067–0.0067 | 0.3528–0.3732 |
| mouse_lung_tms_10x | 43 | all | 1.241 | reject | 0.00 | stable_reject | yes | 5/5.0/5 | 0.0058–0.0067 | -0.0225–-0.0119 |
| mouse_lung_tms_10x | 44 | 0.10 | 1.149 | pass | 1.00 | stable_pass | yes | 5/5.0/5 | 0.0037–0.0037 | 0.1451–0.1544 |
| pancreas_baron | 42 | all | 1.160 | reject | 0.00 | stable_reject | yes | 0/1.0/2 | 0.0057–0.0094 | -0.0527–-0.0252 |
| pancreas_baron | 43 | 0.01 | 1.124 | pass | 1.00 | stable_pass | yes | 3/4.0/5 | 0.0071–0.0083 | 0.2611–0.4385 |
| pancreas_baron | 43 | 0.05 | 1.124 | pass | 1.00 | stable_pass | yes | 3/4.0/5 | 0.0071–0.0083 | 0.2600–0.4462 |
| pancreas_baron | 43 | all | 1.122 | reject | 0.00 | stable_reject | yes | 0/0.0/0 | nan–nan | nan–nan |
| pancreas_baron | 44 | 0.01 | 1.222 | reject | 0.00 | stable_reject | yes | 0/1.0/2 | 0.0031–0.0094 | -0.0360–0.0103 |
| pancreas_baron | 44 | 0.05 | 1.222 | reject | 0.00 | stable_reject | yes | 0/1.0/1 | 0.0031–0.0071 | -0.0286–0.0000 |
| pancreas_baron | 44 | 0.10 | 1.124 | reject | 0.00 | stable_reject | yes | 1/3.0/4 | 0.0046–0.0083 | -0.0529–-0.0216 |
| pancreas_baron | 44 | all | 1.093 | reject | 0.00 | stable_reject | yes | 0/0.0/0 | nan–nan | nan–nan |
