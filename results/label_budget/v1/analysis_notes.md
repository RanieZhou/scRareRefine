# Rare-label budget accounting notes

## Scope

- The ledger covers the frozen 8 datasets x 3 seeds x 4 nominal budgets grid.
- Training labels and validation calibration labels are reported separately.
- Test labels contribute only split-support counts; they are not used for model selection, calibration, thresholds, or collapse.

## Completeness and identity

- Expected ledger rows: 96.
- Status counts: `{"success": 96}`.
- Verified labeled-rare identity hashes: 96.
- Within-seed identity-collapse groups: 6.
- Within-seed count-only collision groups: 0.
- Rows after identity collapse: 87; seed-count units: 87.

## Interpretation

- Nominal budgets are requests; observed labeled-rare counts and fractions are the auditable supervision quantities.
- Equal counts do not imply equal labeled-cell identity. Count-only collisions retain every identity hash and are averaged only at the frozen seed-count aggregation step.
- Cross-dataset pooled cell counts are not inferential summaries because dataset sizes and rare-cell prevalence differ.
