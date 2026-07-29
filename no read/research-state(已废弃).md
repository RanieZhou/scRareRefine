# Research State: approved supplementary analysis program

## Current Stage
COMPUTE

## Research Question
Do transparent label accounting, component ablation, prespecified expression markers, robustness analyses, and a minimal independent representation support the safety and scope of scRareRefine?

## Key Decisions
- SCOPE: Complete the four user-approved supplementary tasks; manuscript updating remains excluded.
- LITERATURE: Freeze eight target-specific marker panels from verified external literature before expression analysis.
- REASON: Prefer cache-only, dataset-first analyses; use a train-fitted linear expression representation as the simplest genuine second backbone.
- METHODOLOGY: Follow `results/supplementary_program/v1/methodology.md`; preserve all adverse outcomes and prohibit test-driven parameter changes.

## Experiment Log
| Attempt | Method | Result | Status |
|---------|--------|--------|--------|
| 1 | P0 label budget accounting | 96/96 verified; 87 identity-collapsed units; 29 dataset-count summaries | completed |
| 2 | P1 eight-dataset component/rank ablation | pending | planned |
| 3 | P2 prespecified marker validation | pending | planned |
| 4 | P3 sensitivity, association, and SVD backbone | pending | planned |

## Critique History
- Pre-COMPUTE: P0, P2, P3a, and P3c initially BLOCKING. Fixed explicit label-ID hashes and budget definitions, externally frozen markers with train-only standardization, frozen sensitivity grids with alpha-specific tau, and mutually exclusive validation subsets for the SVD backbone. P1 and P3b passed with dataset-first aggregation requirements.
- P0 pre-COMPUTE: PASS after specifying unique-ID denominators, closed 96-key failure rows, cross-seed units, and deterministic handling of same-seed count collisions.
- P0 post-COMPUTE: PASS; all numerical outputs trace to the cache-only script and no blocking integrity or logic issue remains.

## What Worked
- P0 verified all 96 formal runs, including exact labeled-rare ID hashes and manifest split hashes, with no failed or identity-unverifiable row.
- P0 found six true identity-collapse groups from the five-cell floor and no count-only identity collision; 96 nominal runs reduce to 87 identity units.
- Rescue composition established a validated fail-closed cache alignment and formal replay path for 85 traceable runs.
- Current-code replay succeeded for all 96 runs and produced one-hot groups for 355,116 test cells.
- H2 and H3 orderings were direction-consistent across all informative dataset-budget summaries for all four primary prototype metrics.
- Frozen scANVI rare probability independently ordered baseline-correct > true-rescued > unrescued/non-target in all informative runs.

## What Didn't Work
- Eleven historical mouse-pancreas runs lack authoritative cell-level rescue identities and cannot support residual-signal grouping.
- Rare rank did not support H1 consistently; baseline-correct and true-rescued rare cells often shared similar prototype ranks.

## Open Questions
- Whether all prespecified markers are present in every source expression matrix.
- Whether the simple SVD representation has enough validation rare support after splitting for meaningful rescue calibration.

## Artifacts
- literature-review.md: marker sources are frozen in `results/supplementary_program/v1/marker_registry.csv` and prior project literature remains applicable.
- reasoning.md: decisions and claim boundaries are recorded in the methodology and experiment log.
- methodology.md: `results/supplementary_program/v1/methodology.md`.
- figures/: P0 PNG/PDF completed under `results/label_budget/v1/figures/`; P1-P3 figures pending.
- experiments.tsv: not used; formal defaults will not be optimized or committed as experiments.
- P0 outputs: `results/label_budget/v1/` and `logs/label_budget/label_budget_v1.log`.
