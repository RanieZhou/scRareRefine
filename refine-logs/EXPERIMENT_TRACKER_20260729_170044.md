# Adaptive Separability Gate 实验跟踪

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| ASG-R001 | M0 | deterministic unit tests | adaptive gate core | synthetic | decision, OOF stats | MUST | TODO | test-label invariance |
| ASG-R002 | M0 | insufficient support | adaptive gate core | synthetic | abstain reason | MUST | TODO | rare support < 2 folds |
| ASG-R003 | M1 | low-S development audit | fixed/no-gate/adaptive | 6-human batch-heldout | F1, FFR, decision evidence | MUST | TODO | only S<1.3 |
| ASG-R004 | M2 | full human comparison | fixed/no-gate/adaptive | 6-human batch-heldout | mean F1, max FFR, W/T/L | MUST | TODO | 72 units |
| ASG-R005 | M3 | frozen confirmatory evaluation | fixed/no-gate/adaptive | 2-mouse batch-heldout | mean F1, max FFR, W/T/L | MUST | TODO | 24 units; no rule changes |
| ASG-R006 | M3 | split sensitivity confirmation | fixed/no-gate/adaptive | 6-human cell-stratified seed42 | F1, FFR | NICE | TODO | only after mouse |
| ASG-R007 | M4 | decision sensitivity | adaptive variants | batch-heldout | decision stability | NICE | TODO | only if R004/R005 pass |

