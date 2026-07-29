# Validation-Adaptive Separability Gate — Completion Report

Date: 2026-07-29

## Decision-seed stability (raw unit table)

Every batch-heldout unit with train-derived `S < 1.3` was rerun under 20 deterministic decision seeds. Stability was predeclared as pass rate `>=0.80` for a frozen pass or `<=0.20` for a frozen rejection. Test labels were not loaded.

| Dataset | Model seed | RTS | S | Frozen decision | Passes / 20 | Active folds (min/median/max) | Stability |
|---|---:|---:|---:|---|---:|---:|---|
| pancreas_baron | 42 | all | 1.160 | reject | 0/20 | 0/1/2 | stable reject |
| pancreas_baron | 43 | 0.01 | 1.124 | pass | 20/20 | 3/4/5 | stable pass |
| pancreas_baron | 43 | 0.05 | 1.124 | pass | 20/20 | 3/4/5 | stable pass |
| pancreas_baron | 43 | all | 1.122 | reject | 0/20 | 0/0/0 | stable reject |
| pancreas_baron | 44 | 0.01 | 1.222 | reject | 0/20 | 0/1/2 | stable reject |
| pancreas_baron | 44 | 0.05 | 1.222 | reject | 0/20 | 0/1/1 | stable reject |
| pancreas_baron | 44 | 0.10 | 1.124 | reject | 0/20 | 1/3/4 | stable reject |
| pancreas_baron | 44 | all | 1.093 | reject | 0/20 | 0/0/0 | stable reject |
| mouse_lung_tms_10x | 42 | 0.05 | 1.188 | pass | 20/20 | 5/5/5 | stable pass |
| mouse_lung_tms_10x | 42 | all | 1.208 | reject | 0/20 | 0/1/2 | stable reject |
| mouse_lung_tms_10x | 43 | 0.01 | 1.098 | pass | 20/20 | 5/5/5 | stable pass |
| mouse_lung_tms_10x | 43 | 0.05 | 1.105 | pass | 20/20 | 5/5/5 | stable pass |
| mouse_lung_tms_10x | 43 | 0.10 | 1.244 | pass | 20/20 | 5/5/5 | stable pass |
| mouse_lung_tms_10x | 43 | all | 1.241 | reject | 0/20 | 5/5/5 | stable reject |
| mouse_lung_tms_10x | 44 | 0.10 | 1.149 | pass | 20/20 | 5/5/5 | stable pass |

Result: **15/15 units were consistent with the frozen decision**. All seven frozen passes remained passes in `20/20` repeats; all eight frozen rejections remained rejections in `20/20` repeats.

Detailed Wilson-UCB and OOF-ΔF1-LCB distributions are in `stability_20seeds/stability_summary.csv`; all 300 repeat-level rows are in `stability_20seeds/stability_repeats.csv`.

## Cell-stratified robustness (6 human datasets, seed 42)

| Region | Variant | N | Mean rare F1 | Delta vs fixed | W/T/L vs fixed | Max incremental FPR | Violations |
|---|---|---:|---:|---:|---:|---:|---:|
| ALL | fixed S=1.3 | 24 | 0.974522 | 0.000000 | 0/24/0 | 0.001870 | 0 |
| ALL | no separability gate | 24 | 0.973777 | -0.000745 | 1/22/1 | 0.003208 | 0 |
| ALL | adaptive gate | 24 | 0.974522 | 0.000000 | 0/24/0 | 0.001870 | 0 |
| SCARCE | fixed S=1.3 | 18 | 0.974102 | 0.000000 | 0/18/0 | 0.001870 | 0 |
| SCARCE | no separability gate | 18 | 0.973109 | -0.000993 | 1/16/1 | 0.003208 | 0 |
| SCARCE | adaptive gate | 18 | 0.974102 | 0.000000 | 0/18/0 | 0.001870 | 0 |

Three cell-stratified units had `S < 1.3`, all from pancreas_baron. Adaptive audit rejected all three: RTS 0.01 and 0.10 failed the positive-gain criterion; RTS all had only one validation missed rare cell. Thus adaptive matched the fixed gate exactly and avoided the negative no-gate result at RTS 0.01.

## Mainline integration

- `src/rescue.py` retains the original `conformal_rescue()` fixed-S behavior.
- `adaptive_conformal_rescue()` implements the frozen validation-only rule.
- `rescue_with_separability_gate(gate_mode="fixed"|"adaptive")` is the public dispatcher.
- Eight main dataset configs now select `experiment.separability_gate_mode: adaptive`; CLI `--separability_gate_mode fixed` reproduces the original control.
- The new core implementation matched the frozen experiment implementation on predictions and key audit fields for **15/15 batch-heldout low-S units** and **3/3 cell-stratified low-S units**.
- Related targeted tests: **20 passed**; full regression suite: **61 passed**.

## Decision

The preregistered stability and split-robustness conditions are met. The adaptive gate is accepted as the configurable mainline extension, while fixed `S=1.3` remains an explicit reproducible control. Evidence supports empirical validation-constrained safety on the evaluated datasets; it does not constitute a formal guarantee of test FFR under arbitrary distribution shift.
