# Methodology: approved supplementary analysis program

## Scope and fixed boundaries

This program completes four approved analyses without changing the formal
scRareRefine algorithm, cached predictions, thresholds, splits, or historical
benchmark files. All analysis outputs are versioned under `results/`.

The formal evidence hierarchy is dataset-first. Seeds are aggregated within a
dataset and effective label budget before cross-dataset summaries. Test labels
are used only for final characterization and empirical metrics.

## P0: rare-label budget accounting

- Read the 96 formal train/validation/test prediction caches using explicit,
  unique `cell_id` alignment.
- Record training rare-pool size, actual labeled rare training cells,
  validation rare support, test rare support, all-split rare support, total
  labeled training support, and total rare supervision
  (`training_labeled_rare + validation_rare`).
- The canonical IDs are the `cell_id` strings in
  `train_predictions.csv` where `true_label == rare_class` and
  `is_labeled_for_scanvi == True`. Reject missing/duplicate IDs, sort unique
  strings lexicographically, serialize as compact UTF-8 JSON with
  `ensure_ascii=False` and separators `(',', ':')`, and hash the exact bytes
  with lowercase hexadecimal SHA-256. Verify that the ID-set size equals the
  observed labeled rare count and that every ID belongs to the training split.
- Distinguish count collapse from identity collapse. The effective-budget key
  is `(dataset, rare_class, split_hash, actual_training_labeled_rare_count,
  labeled_rare_id_sha256)`. Nominal budgets are folded within dataset/seed only
  when this complete key is identical. Missing identity provenance is labeled
  `identity_unverifiable` and is never merged by count alone.
- All supports are cardinalities of unique canonical `cell_id` sets:
  `train_rare_pool = {train cell_id | true_label == rare_class}`,
  `all_split_rare = train_rare_ids union validation_rare_ids union
  test_rare_ids`, and `all_training_cells = {train cell_id}`. The training rare
  pool includes both labeled and unlabeled true rare cells. Split-specific IDs
  must be unique and the three split sets must be pairwise disjoint; overlap is
  a failed run, never silently deduplicated. Validation/test rare support is
  used only for retrospective cost accounting and never for collapse,
  selection, calibration, or thresholds.
- After within-seed identity collapse, the frozen cross-seed unit is
  `(dataset, rare_class, actual_training_labeled_rare_count)`. Each seed
  contributes at most once to a unit. If one seed has multiple identity-distinct
  runs with the same count, those identity runs are first averaged equally
  within `(dataset, rare_class, count, seed)`; no nominal budget is selected as
  representative. Runs with equal counts but different split/identity hashes
  therefore enter the same cross-seed descriptive unit while all hashes and
  folded nominal budgets remain explicit provenance. Runs labeled
  `identity_unverifiable` remain separate and do not enter identity/count
  folding. Dataset summaries are equal-weighted; raw run rows are descriptive
  only.
- Construct the complete expected grid before reading inputs: eight frozen
  configs by three seeds by four nominal budgets. Every expected key produces
  exactly one ledger row. Missing/unreadable files, missing required columns,
  duplicate/missing IDs, split overlap, manifest/config mismatch, or cached
  split-hash mismatch produce `status=failed` with an explicit error reason;
  rows are never skipped or inferred from another cache. Any failed row blocks
  formal collapse, summary, and acceptance even though the closed ledger is
  still written for diagnosis.
- Report explicit ratios: `actual_training_labeled_rare / train_rare_pool`,
  `actual_training_labeled_rare / all_split_rare`,
  `(actual_training_labeled_rare + validation_rare) / all_split_rare`, and
  `actual_training_labeled_rare / all_training_cells`. No test label is part of
  training or calibration; test support is reported only to make the full
  split accounting transparent.

## P1: gate/rank/conformal ablation

- Replay the current code on all eight datasets, four budgets, and three seeds.
- Component variants: baseline, minus separability gate, minus necessity gate,
  fixed rank 1, minus conformal tau, and full method.
- Rank sensitivity: fixed rank 1/2/3 and validation-adaptive rank.
- Primary outcomes: rare F1, recall, rescue precision, incremental FPR,
  abstention, and empirical alpha violation.
- The primary aggregation first collapses nominal budgets sharing the complete
  effective-budget key within dataset/seed. For cross-seed summaries, the
  frozen unit key is `(dataset, rare_class,
  actual_training_labeled_rare_count)`; after within-seed identity collapse,
  each seed contributes at most once to this unit. Split/hash provenance and
  folded nominal budgets remain reported. Dataset summaries
  are equal-weighted; run-level rates are descriptive only. Identity-unverifiable
  runs remain separate and are not count-folded.

## P2: prespecified marker-expression validation

Markers are frozen from external literature before expression analysis:

| Dataset | Markers |
|---|---|
| immune_dc | AXL, SIGLEC6, CD2, CD5, CD22 |
| pancreas_baron | PPY, PNLIPRP1, CARTPT |
| pancreas_integrated | PECAM1, VWF, KDR, ESAM, CDH5 |
| tabula_lung_endo | PROX1, FLT4, PDPN, CCL21, LYVE1 |
| tabula_sapiens_stomach | TPSAB1, TPSB2, KIT, CPA3, MS4A2 |
| tabula_small_intestine | POU2F3, TRPM5, GNG13, SH2D6, AVIL |
| mouse_lung_tms_10x | Nr2f2, Ephb4, Ackr1, Vcam1, Selp |
| mouse_pancreas_tms_10x | Sst, Hhex, Ghsr, Rbp4 |

Expression is transformed per cell as log1p counts per 10,000. For every run,
gene means and standard deviations are fitted on training cells only; test
cells are transformed with those frozen values. The signature score is the
unweighted mean z-score over available prespecified markers. No marker is
selected or weighted using validation/test outcomes.

Registry version `v1` is frozen on 2026-07-17. Every marker has expected
direction `higher`. Exact gene-symbol matches are used after stripping
whitespace; duplicate symbols are rejected. No cross-species ortholog mapping
is performed because each panel is species-specific. Missing genes are retained
in the availability table but omitted from the unweighted score. A run with no
available marker is `not_evaluable`, not zero, and no replacement marker is
chosen. The registry SHA-256 is recorded in the analysis manifest before any
expression group effect is computed.

Primary groups follow the current formal replay: baseline-correct rare, true
rescued rare, unrescued rare, and non-target. The closest competitor is fixed
from labeled training prototypes using Euclidean distance between class mean
prototypes; ties are resolved by lexicographic class label. Run-level median
differences and Cliff's delta use higher marker score as the favorable direction
for: H1 baseline-correct rare versus true-rescued rare; H2 true-rescued rare
versus unrescued rare; H3a true-rescued rare versus all non-target cells; and
H3b true-rescued rare versus the closest-competitor subset.
This is expression-program concordance, not proof of biological identity.

## P3a: parameter sensitivity

Frozen exploratory grids:

- `alpha`: 0.005, 0.01, 0.02
- `low_sep`: 0.0, 1.0, 1.3, 1.6
- fixed/adaptive rank: 1, 2, 3, adaptive (1,2,3)
- `min_val_missed`: 0, 1, 3, 5

One axis is changed at a time from the current default. Every alpha recalibrates
its own validation-derived conformal threshold. The scan cannot change the
default settings and is reported completely, including adverse results.

Primary outcomes are rare F1, recall, rescue precision, incremental FPR,
abstention rate, and the number of empirical test incremental-FPR values above
the corresponding alpha. Undefined rescue precision remains NA. Summaries use
the P1 effective-budget aggregation and dataset-equal weighting with
dataset-cluster bootstrap 95% intervals. The full grid is exploratory: no
per-combination significance claims or default-parameter selection are made.

## P3b: separability association

Separability is calculated only from labeled training prototypes. Frozen test
outcomes are rare-F1 gain, recovery rate, rescue precision, and incremental
FPR. Association is descriptive: seed medians are formed within
dataset/effective-budget, then dataset-cluster bootstrap intervals and
leave-one-dataset-out ranges are reported. No association is used to select a
new threshold or support a causal claim.

Zero denominators remain NA and are never replaced with zero. Every estimate
reports the number of contributing datasets, effective-budget units, and seeds.

## P3c: minimal second representation backbone

- Use the same raw expression source and cached train/validation/test cell IDs.
- Fit variance filtering, standardization, and TruncatedSVD on training cells
  only; validation/test are transformed only.
- Deterministically split validation into mutually exclusive `val_base` and
  `val_rescue` using `train_test_split(test_size=0.5, random_state=seed+1000)`
  stratified by true class when every class has at least two cells; otherwise
  use the same deterministic unstratified split and disclose the fallback.
  Report rare support in both subsets. If `val_rescue` has no rare cells or
  fewer than the formal `MIN_VAL_MISSED` target misses, that run is retained as
  `not_evaluable_for_rescue` rather than bypassing the guard.
- Conformal calibration additionally requires at least
  `ceil(1 / alpha) - 1` non-target `val_rescue` cells so the finite-sample
  order statistic is defined. Runs below this support, or with infinite tau,
  are retained as `not_evaluable_for_calibration`.
- Freeze SVD dimensions `(10, 20, 50)` capped by matrix rank and kNN
  `k=(3,5,10,15)`. `val_base` selects the pair maximizing rare F1, with ties
  resolved by smaller dimension then smaller k. `val_rescue` alone selects the
  formal rank and calibrates tau. The subsets are never reused across roles.
- SVD+kNN and scANVI use the same cached training cell universe and the same
  `is_labeled_for_scanvi` label permissions for each run. SVD does not access
  scANVI latent vectors or probabilities. Training prototypes use labeled
  training cells only. The primary paired
  comparison is SVD+kNN versus SVD+kNN+unchanged rescue. scANVI comparisons are
  secondary descriptions.
- This analysis tests portability to one prespecified linear expression
  representation; it does not claim backbone universality.

## Acceptance criteria

- P0: 96/96 ledger rows, explicit label-ID hashes, and transparent collapse
  classification.
- P1: complete eight-dataset replay, all variants, no silent failures, and all
  empirical alpha violations reported.
- P2: marker registry hash precedes computation; train-only standardization;
  at least one real-data PNG/PDF figure; unavailable genes yield explicit NA.
- P3: complete frozen grids, independent tau per alpha, clustered association
  summaries, and a leakage-free paired second-backbone evaluation.

## P4: independent CellTypist classifier backbone

P4 is a minimal portability screen requested after P0--P3 were specified. It
does not replace P3c. A custom CellTypist 1.7.1 logistic classifier is trained
only on cells carrying the frozen `is_labeled_for_scanvi` permission. Cached
HVG identities are reused because they were selected by train-only variance;
no scANVI latent vector, probability, or predicted label enters P4.

The native multiclass CellTypist decision-function vector, ordered by the
classifier's class registry, is the representation supplied to the unchanged
`PrototypeRescuer`. Training prototypes use labeled training cells only.
CellTypist validation predictions select the candidate rank and validation
non-target scores calibrate conformal tau. Test labels are used only after both
CellTypist and refined test predictions are frozen.

The prespecified screen is three heterogeneous human datasets
(`immune_dc`, `pancreas_baron`, `tabula_sapiens_stomach`), seed 42, and rare
training-label fractions 0.01, 0.05, and 0.10. Formal rescue constants remain
unchanged (`alpha=0.01`, `low_sep=1.3`, ranks 1--3, `min_val_missed=3`). Every
attempt, abstention, failure, and empirical alpha exceedance is retained.

P4 acceptance requires a closed 9-run ledger, at least two datasets with
evaluable CellTypist predictions, no test-label leakage, and complete paired
metrics. Positive gain is not required for technical completion. Any scientific
claim is limited to a single-seed portability screen; null or mixed outcomes
must be reported as such and may not support backbone universality.
