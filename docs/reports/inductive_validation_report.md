# scRareRefine Inductive Validation Report

## Current Scope

The project now uses only inductive train/validation/test validation.

- Train cells are the only cells used for scVI/scANVI training.
- Validation cells are used only for fusion or marker-threshold selection.
- Test cells are used only for final metrics.
- Rare-label downsampling is sampled only from rare cells inside the train split.
- HVG genes are selected from train cells and then applied to validation/test cells.
- Latent prototypes and marker signatures are built only from train cells with `is_labeled_for_scanvi=True`.

## Main Entry Points

Run inductive scANVI plus validation-tuned probability fusion:

```powershell
conda run -n scanvi311 python scripts/run_scanvi_inductive.py --config configs/immune_dc.yaml --rare-class ASDC,cDC1 --split-mode cell_stratified,batch_heldout
```

Run prototype rank1 gate and validation-tuned marker verification on existing inductive runs:

```powershell
conda run -n scanvi311 python scripts/evaluate_inductive_prototype_marker.py --config configs/immune_dc.yaml --rare-class ASDC,cDC1 --split-mode batch_heldout
```

Pancreas batch-heldout validation:

```powershell
conda run -n scanvi311 python scripts/run_scanvi_inductive.py --config configs/pancreas_gamma.yaml --rare-class gamma --split-mode batch_heldout
conda run -n scanvi311 python scripts/run_scanvi_inductive.py --config configs/pancreas_epsilon.yaml --rare-class epsilon --split-mode batch_heldout
```

## Output Layout

Current outputs are organized by dataset, split mode, and rare class:

- `outputs/immune_dc/inductive_cell/asdc/`
- `outputs/immune_dc/inductive_cell/cdc1/`
- `outputs/immune_dc/inductive_batch/asdc/`
- `outputs/immune_dc/inductive_batch/cdc1/`
- `outputs/pancreas/inductive_batch/gamma/`
- `outputs/pancreas/inductive_batch/epsilon/`

Each run stores core intermediate assets under `runs/<run>/artifacts/`.
Stage-level summaries are under `stages/fusion/` and `stages/prototype_marker_validation/`.

## Method Summary

The current method has three inductive stages:

1. `scANVI baseline`: train on train cells and predict held-out validation/test query cells.
2. `prototype rank1 gate`: build class prototypes from labeled train latent embeddings and identify rare-cell rescue candidates on validation/test cells.
3. `validation-tuned marker`: select the marker-margin threshold on validation cells under a false-rescue constraint, then apply the selected threshold once to test cells.

Only the inductive workflow is documented and supported in the active code path.
