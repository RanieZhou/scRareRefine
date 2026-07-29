# Findings

## 2026-07-29 — Validation-Adaptive Separability Gate

### Claim verdict

- Proposed claim: For runs with train-derived separability `S < 1.3`, a validation-only 5-fold cross-fitted audit can safely decide whether to override the fixed separability gate, while keeping `S >= 1.3` behavior unchanged.
- Verdict: **partial** `[pending Codex review]`.
- Confidence: **medium**.
- Integrity status: **unavailable / provisional**. No independent `EXPERIMENT_AUDIT.md` or `EXPERIMENT_AUDIT.json` is available.

### Evidence

- Related unit tests: 14 passed, including low-support abstention, unsafe-OOF rejection, determinism, and exclusion of test labels from the gate decision.
- Human development set (6 datasets, 72 units): adaptive gate mean rare-cell F1 `0.901542` versus fixed gate `0.887511` (`+0.014031`), with `2/70/0` wins/ties/losses and no observed test FFR violation.
- Frozen mouse confirmation set (2 datasets, 24 units): adaptive gate mean rare-cell F1 `0.701827` versus fixed gate `0.596083` (`+0.105744`), with `5/19/0` wins/ties/losses and no observed test FFR violation.
- Combined 8-dataset evaluation (96 units): adaptive gate mean rare-cell F1 `0.851613` versus fixed gate `0.814654` (`+0.036959`), with `7/89/0` wins/ties/losses; maximum observed test incremental false-positive rate was `0.009768`.
- Removing the separability gate unconditionally gave higher mean F1 (`0.854673`) but introduced two test safety violations, with maximum test incremental false-positive rate `0.015263`.

### Interpretation and permitted wording

- Supported: the rule is implementable, uses no test labels for any decision, and in the current evaluation recovered low-S gains without an observed regression or new test safety violation.
- Not supported: a universal or formal guarantee that test FFR is at most `0.01`. The Wilson check constrains validation OOF errors and relies on approximate cell-level exchangeability; batch shift can break transfer from validation to test.
- Human results are development evidence because they influenced the rule. The frozen mouse results are the cleaner confirmatory evidence.
- Gains are concentrated in Baron pancreas and mouse lung (7 changed units); 89/96 units remain unchanged by design.

### Rule definitions to freeze

- `FFR_OOF` is currently the incremental false-positive rate: true non-rare validation cells changed from a non-rare baseline prediction to the rare label, divided by all true non-rare validation cells.
- A fold is “active/valid” when its calibration subset yields a non-abstaining tau/rank rule. It need not actually change a held-out cell. Avoid the ambiguous phrase “a fold successfully rescued cells.”
- The gain criterion is a **one-sided 95% paired stratified-bootstrap lower confidence bound**, implemented as the fifth percentile of bootstrap `Delta F1`.
- The Wilson upper bound uses `z=1.96`; with zero false rescues it still needs roughly 381 OOF non-rare cells to fall below `0.01`.
- Full-validation recalibration occurs only after the OOF audit passes. Test labels are used only for final retrospective reporting.

### Remaining gap and next experiment

1. Run decision-stability diagnostics across repeated fold partitions without changing the frozen v1 policy or selecting settings using test labels.
2. Run the predeclared 6-human cell-stratified seed-42 robustness experiment.
3. Audit cell-independence assumptions; use donor/batch-grouped folds or grouped bootstrap when validation contains usable donor/batch groups.
4. Obtain an independent experiment-integrity review before replacing the main `conformal_rescue()` implementation or making a strong paper claim.

### Completion update (2026-07-29)

- Fold-decision stability passed: all 7 frozen-pass low-S units passed in 20/20 repeats, and all 8 frozen-reject units passed in 0/20 repeats; 15/15 were consistent.
- Six-human cell-stratified seed-42 robustness passed: adaptive versus fixed was 0/24/0 wins/ties/losses with identical maximum incremental FPR 0.001870; no-gate was 1/22/1.
- The adaptive rule was integrated behind an explicit `fixed|adaptive` dispatcher. The original fixed `conformal_rescue()` behavior remains available and unchanged.
- The integrated core matched the frozen experimental implementation for predictions and key audit fields on 15/15 batch-heldout and 3/3 cell-stratified low-S units; the full test suite passed 61/61 tests.
- Claim remains empirical rather than a universal FFR guarantee; independent experiment audit remains unavailable.
