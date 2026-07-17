# Methodology: formal rescue selection-pathway characterization

## Research Question and Hypotheses

- Question: What cells does the formal rescue pathway select in the frozen prototype geometry, and does the non-selection scANVI rare probability provide concordant or discordant information?
- H1: Within the selection geometry, baseline-correct rare cells show stronger target signal than true rescued rare cells.
- H2: Within the selection geometry, true rescued rare cells show stronger target signal than unrescued rare cells.
- H3: Within the selection geometry, true rescued rare cells remain separated from non-target cells and the train-defined closest competing class.
- Orthogonal readout: cached scANVI rare probability is reported without a prespecified favorable direction because the method is designed for cases where the backbone probability can fail.
- Success criteria: complete fail-closed accounting; one-hot primary groups; separate current-code and historical-identity provenance; at least one real-data figure; and reported direction consistency for all frozen contrasts without suppressing contradictory results.
- Interpretation boundary: the prototype metrics characterize the same decision pathway that selected rescued cells. They do not constitute independent mechanism or biological validation.

## Data Sources

- Existing `outputs/<dataset>/<run>/embeddings/` train, validation, and test prediction/latent caches for 8 datasets, 4 rare-label budgets, and 3 seeds.
- Existing `results/rescue_composition/v1/run_level.csv` for authoritative reconstruction-basis metadata.
- No raw data, model retraining, or historical prediction overwrite.

## Analysis Pipeline

### Step 1: Fail-closed cache reconstruction

- Align prediction and latent tables by unique `cell_id`.
- Verify train/validation/test IDs are disjoint, latent dimensions match, values are finite, and the labeled rare count matches the formal budget rule.
- Fit class prototypes and radii using only labeled training cells.
- Replay `conformal_rescue()` using validation labels for formal rank/tau selection and test latent values for application.
- Assign current-code replay identities for every valid cache using the current formal implementation, independent of test-label-derived historical metrics.
- Separately record whether an authoritative historical cell-level identity exists. For the known count-only historical rows it remains unavailable; current replay identities must not be represented as reconstructed historical identities.

### Step 2: Prespecified groups and metrics

- Mutually exclusive and exhaustive primary groups: `baseline_correct_rare`, `true_rescued_rare`, `unrescued_rare`, and `non_target`. Enforce exactly one group per test cell.
- `false_rescue` is a flag and subset of `non_target`, not a fifth primary group.
- A train-defined closest-competitor flag identifies non-target cells whose true label equals the non-rare prototype nearest to the rare prototype.
- Primary selection-pathway metrics: anisotropic rare-membership score, isotropic rare-prototype rank, rare-radius-standardized distance, and standardized prototype margin.
- Secondary metrics: raw rare-prototype distance, nearest-nonrare distance, and raw prototype margin (`nearest_nonrare_distance - rare_distance`).
- Non-selection model readout: cached `prob_<rare_class>` from the frozen scANVI prediction table when present. Missing probabilities remain missing and are never reconstructed.

### Step 3: Contrasts and aggregation

- Compute per-run group counts, medians, and interquartile ranges.
- Freeze contrasts before execution: H1=`baseline_correct_rare` versus `true_rescued_rare`; H2=`true_rescued_rare` versus `unrescued_rare`; H3a=`true_rescued_rare` versus `non_target`; H3b=`true_rescued_rare` versus `closest_competitor`.
- For every frozen contrast and metric, compute median differences and Cliff's delta per run. Orient selection-pathway effects so positive values support stronger target geometry: higher score/margin and nearest-nonrare distance, but lower rare rank and rare distance.
- First summarize seeds within each dataset x budget, then summarize dataset-level effects within each budget. Run-level direction rates remain descriptive diagnostics only.
- The scANVI probability readout is reported as raw direction/effect without declaring disagreement a failure.

### Step 4: Visualization

- Plot group distributions for the four primary selection metrics and the scANVI rare probability using run-centered values to reduce cross-dataset scale differences.
- Plot dataset x budget seed-median prototype-margin effects for the prespecified contrasts with zero-reference lines and explicit budget encoding.
- Export 300 dpi PNG and vector PDF; include sample sizes and traceability limitations in captions/notes.

## Controls and Validation

- Positive ordering control: baseline-correct rare versus non-target.
- Safety control: false rescues are reported separately and are never merged with true rescued rare cells.
- Historical limitation: unsupported rows remain marked `historical_cell_identity_available=false`; current-code replay identities are available but are not inferred historical identities.
- Test labels define final characterization groups and historical agreement diagnostics only. They do not select thresholds, metrics, directions, competitors, exclusions, or current-replay eligibility.

## Statistical Plan

- Primary effect: per-run Cliff's delta and median difference for each frozen contrast/metric.
- Evidence hierarchy: seed summary within dataset x budget, followed by dataset-level median and direction count within budget.
- Zero-sized groups yield NA, not zero effects.
- No pooled-cell p values, because cells within a run are not independent dataset-level replicates.

## Compute Requirements

- Platform: local CPU in `scanvi311`.
- Expected duration: minutes.
- Estimated cost: zero cloud/API cost.

## Limitations and Assumptions

- Distances are representation-dependent and not directly comparable across runs; figures use run-centered values and inference uses within-run effects.
- Cell-level residual-signal evidence is unavailable for historical runs without authoritative rescue identities.
- Prototype-derived results are selection-pathway audits and are partly expected by construction.
- Cached scANVI probability is a non-selection model readout but is not independent biological validation.
- This analysis characterizes the frozen model and does not establish biological identity independently of the benchmark labels.
