# Residual-signal selection-pathway analysis notes

## Scope and interpretation

- This is a cache-only audit of the formal rescue selection pathway, not independent biological validation.
- Rare score, rank, distance, and prototype margin share the geometry used by rescue and therefore partly reflect selection by construction.
- Cached scANVI rare probability is a frozen non-selection model readout; disagreement is scientifically expected in the target failure mode.
- Test labels are used only for final group characterization and historical agreement diagnostics, never for current rescue selection or run eligibility.

## Completeness

- Expected runs: 96; status counts: `{"success": 96}`.
- Cell-level current-code replay rows: 355,116 cells across 96 runs.
- Historical cell identity unavailable: 11 runs; current replay is reported separately and is not presented as historical reconstruction.
- Cached scANVI rare probability available in 96 runs.
- Current-code replay group totals: baseline-correct rare=4086, true rescued rare=864, unrescued rare=1326, non-target=348840, false rescues=83.

## Prespecified ordering summary

- H1_baseline_correct_vs_true_rescued: median dataset direction rates by budget: rare_membership_score=1.000, rare_rank=0.325, rare_standardized_distance=1.000, standardized_prototype_margin=1.000.
- H2_true_rescued_vs_unrescued: median dataset direction rates by budget: rare_membership_score=1.000, rare_rank=1.000, rare_standardized_distance=1.000, standardized_prototype_margin=1.000.
- H3a_true_rescued_vs_non_target: median dataset direction rates by budget: rare_membership_score=1.000, rare_rank=1.000, rare_standardized_distance=1.000, standardized_prototype_margin=1.000.
- H3b_true_rescued_vs_closest_competitor: median dataset direction rates by budget: rare_membership_score=1.000, rare_rank=1.000, rare_standardized_distance=1.000, standardized_prototype_margin=1.000.

## Frozen scANVI probability readout

- H1_baseline_correct_vs_true_rescued: n_runs=33, median raw Cliff's delta=1.0000, positive-direction rate=1.000.
- H2_true_rescued_vs_unrescued: n_runs=35, median raw Cliff's delta=0.7778, positive-direction rate=1.000.
- H3a_true_rescued_vs_non_target: n_runs=40, median raw Cliff's delta=0.9985, positive-direction rate=1.000.
- H3b_true_rescued_vs_closest_competitor: n_runs=40, median raw Cliff's delta=0.9972, positive-direction rate=1.000.

## Limitations

- Direction rates are descriptive; datasets, not pooled cells, are the main evidence units.
- Empty groups yield missing effects and are not counted as failures or successes.
- Current-code replay can differ from historical count-only outputs; both provenance fields are retained.
- Biological marker validation remains a separate P2 analysis.
