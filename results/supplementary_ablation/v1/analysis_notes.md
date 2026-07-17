# Supplementary ablation notes

## Scope

- Current-code cache replay over the frozen eight datasets, four nominal budgets, and three seeds.
- Prototypes use labeled training cells; all gates, rank choices, and tau use validation only; test labels are used only for final empirical metrics.
- This P1 replay does not reconstruct unavailable historical mouse-pancreas cell identities; it evaluates every variant consistently under the current frozen implementation.

## Completeness

- Expected run-variant rows: 960; observed: 960.
- Identity-collapsed run-variant rows: 870.
- Dataset-effective-budget-variant summary rows: 290.
- Empirical alpha violations by variant: `{"baseline": 0, "minus_separability_gate": 2, "minus_necessity_gate": 0, "fixed_rank_1": 0, "minus_conformal_tau": 2, "full_method": 0, "rank_1": 0, "rank_2": 6, "rank_3": 15, "adaptive_rank": 0}`.
- Abstentions by variant: `{"baseline": 0, "minus_separability_gate": 44, "minus_necessity_gate": 27, "fixed_rank_1": 44, "minus_conformal_tau": 57, "full_method": 56, "rank_1": 44, "rank_2": 44, "rank_3": 44, "adaptive_rank": 56}`.

## Interpretation limits

- Rescue precision is NA when no cells are rescued and is never replaced with zero.
- Empirical test incremental-FPR exceedance under split shift is reported completely but does not alter the frozen defaults.
- Dataset-equal summaries prevent large datasets or duplicated floor budgets from dominating the headline aggregation.
