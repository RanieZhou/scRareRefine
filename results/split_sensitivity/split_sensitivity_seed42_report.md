# Split Sensitivity Report (seed=42)

Scope: scANVI + scRareRefine only. Primary claims remain based on `batch_heldout`; `cell_stratified` is a supplementary easier-setting sensitivity analysis.

- Conformal alpha: 0.01
- Low-separability gate: 1.3
- Completed rows: 48/48
- Missing/non-ok rows: 0

## Scarce-region aggregate

| split_mode | n | scANVI_f1 | scRareRefine_f1 | delta_f1 | scANVI_recall | scRareRefine_recall | delta_recall | rescue_ffr_max | n_abstain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batch_heldout | 18 | 0.6757 | 0.8639 | 0.1881 | 0.614 | 0.8199 | 0.2059 | 0.002442 | 1 |
| cell_stratified | 18 | 0.9016 | 0.9741 | 0.0725 | 0.8657 | 0.9637 | 0.09804 | 0.00187 | 10 |

## Cell-stratified paired delta vs scANVI

- scarce-region wins/ties/losses: 6/10/2
- scarce-region mean delta F1: 0.0725
- scarce-region max rescue FFR: 0.001870

## Output files

- `results\split_sensitivity\cell_stratified_seed42_summary.csv`
- `results\split_sensitivity\batch_vs_cell_stratified_seed42.csv`
