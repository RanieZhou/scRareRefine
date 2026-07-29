# Adaptive Separability Gate 实验跟踪

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ASG-R001 | M0 | deterministic unit tests | adaptive gate core | synthetic | decision, OOF stats | MUST | DONE | 14 related tests passed |
| ASG-R002 | M0 | insufficient support | adaptive gate core | synthetic | abstain reason | MUST | DONE | val_missed<3 safely abstains |
| ASG-R003 | M1 | low-S development audit | fixed/no-gate/adaptive | 6-human batch-heldout | F1, FFR, decision evidence | MUST | DONE | 2/6/0; max FFR 0.002442 |
| ASG-R004 | M2 | full human comparison | fixed/no-gate/adaptive | 6-human batch-heldout | mean F1, max FFR, W/T/L | MUST | DONE | +0.014031 F1; 0 violations |
| ASG-R005 | M3 | frozen confirmatory evaluation | fixed/no-gate/adaptive | 2-mouse batch-heldout | mean F1, max FFR, W/T/L | MUST | DONE | 5/19/0; +0.105744 F1; 0 violations |
| ASG-R006 | M3 | split sensitivity confirmation | fixed/no-gate/adaptive | 6-human cell-stratified seed42 | F1, FFR | NICE | TODO | only after audit decision |
| ASG-R007 | M4 | decision sensitivity | adaptive variants | batch-heldout | decision stability | NICE | TODO | only after experiment audit |

