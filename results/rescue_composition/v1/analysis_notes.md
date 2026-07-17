# Rescue composition analysis notes

## Scope

- Cache-only reconstruction using the formal train-only prototype and validation-calibrated conformal rescue implementation.
- Test labels are used only to characterize true rescues, false rescues, and remaining misses.
- Raw candidates are undefined for abstention runs because chosen_rank=0 is a sentinel, not an active rank.

## Completeness

- Expected configurations: 96.
- Status counts: `{"success": 96}`.
- Non-abstaining runs: 40; runs with at least one rescue: 51.
- Historical-count-only rows: 11; rank, tau, raw candidates, and abstention metadata are unavailable for these rows.

## Scarce-label descriptive composition

- Across all successful scarce-label runs, true rescues=947, false rescues=95.
- Pooled rescue precision across those rescue events=0.9088.
- Maximum run-level incremental FPR=0.009768.

## Interpretation limits

- Pooled counts are descriptive and do not replace dataset-level inference.
- Rescue precision/FDP is undefined when no cells are rescued; recovery rate is undefined when the baseline has no missed rare cells.
- Empirical test error rates under batch shift are safety outcomes, not unconditional conformal guarantees.
