# scRareRefine

scRareRefine is an inductive post-hoc refinement method for recovering rare cell types missed by a frozen scANVI classifier. It combines train-derived prototypes, validation-calibrated candidate selection, and a safety-aware adaptive separability gate.

The current evaluation covers six human scRNA-seq datasets and two mouse Tabula Muris Senis datasets under rare-label budgets of 1%, 5%, 10%, and all available rare labels.

## Highlights

- **Post-hoc rescue**: improves rare-cell recall without retraining or modifying the scANVI backbone.
- **Validation-only calibration**: prototypes come from labeled training cells; tau, candidate rank, and gate decisions use validation only.
- **Adaptive safety gate**: low-separability runs are rescued only when cross-fitted validation evidence supports both positive F1 gain and controlled false-rescue risk.
- **Reproducible fallback**: unsafe or weakly supported runs keep the original backbone prediction; the fixed `S=1.3` method remains available as a control.

## Method

```text
scANVI prediction -> train prototypes -> rare candidates
                  -> validation calibration -> rescue or abstain
```

The train-derived separability statistic is

```text
S = distance(rare prototype, nearest non-rare prototype)
    / mean rare within-class distance
```

- `S >= 1.3`: use the original conformal rescue pipeline unchanged.
- `S < 1.3`: run 5-fold cross-fitting on validation. Relax the gate only when validation misses at least three rare cells, at least three folds are valid, `WilsonUCB(FFR_OOF) <= 0.01`, and the one-sided 95% lower bound of `Delta F1_OOF` is positive.
- After approval, recalibrate tau and rank on full validation and apply them to test. Test labels never enter a decision.

The project reports FFR as the incremental false-positive rate: false non-rare-to-rare relabelings divided by all true non-rare cells.

## Installation

The main pipeline uses Python 3.11 and the `scanvi311` environment.

```bash
conda create -n scanvi311 python=3.11
conda activate scanvi311
pip install -r requirements.txt
```

Some comparison methods with older dependencies use a separate `sandbox310` environment.

## Quick Start

The eight main dataset configurations enable the adaptive gate by default.

```bash
# Main batch-heldout experiment
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05

# Reproduce the original fixed S=1.3 control
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --separability_gate_mode fixed

# Supplementary cell-stratified setting
python run_pipeline.py --config configs/immune_dc.yaml --seed 42 --rare_train_size 0.05 --split_mode cell_stratified
```

Existing embeddings are reused when their provenance manifest matches the requested dataset, split, seed, and label budget. Add `--force` to retrain the backbone.

Outputs are written to `outputs/<dataset>/<run_id>/`:

- `embeddings/`: train, validation, and test predictions/latent representations;
- `metrics/final_metrics.csv`: baseline and refined metrics plus adaptive-gate audit fields;
- `manifest.json`: split and cache provenance.

## Results

Batch-heldout comparison across 8 datasets, 3 seeds, and 4 label budgets (`n=96` paired units):

| Separability policy | Mean rare F1 | Delta vs fixed | W/T/L vs fixed | Max incremental FPR |
|---|---:|---:|---:|---:|
| Fixed `S=1.3` | 0.814654 | 0.000000 | 0/96/0 | 0.009768 |
| No separability gate | 0.854673 | +0.040019 | 10/84/2 | 0.015263 |
| **Adaptive gate** | **0.851613** | **+0.036959** | **7/89/0** | **0.009768** |

- Decision stability: all 7 frozen-pass units passed in 20/20 repeats, and all 8 frozen-reject units passed in 0/20 repeats.
- Cell-stratified sensitivity: adaptive vs fixed was 0 wins / 24 ties / 0 losses, with identical maximum incremental FPR of 0.001870.
- Full regression suite: 61 tests passed; the integrated core matched the frozen implementation on every evaluated low-separability unit.

These are empirical results on the evaluated datasets, not a formal guarantee under arbitrary validation-to-test distribution shift.

## Reproducibility

- [Adaptive-gate completion report](results/adaptive_separability_gate/v1/completion_report.md)
- [Decision-stability report](results/adaptive_separability_gate/v1/stability_20seeds/stability_report.md)
- [Nine-method benchmark snapshot](results/comparison/comparison_summary.csv) — generated with the earlier fixed `S=1.3` scRareRefine policy
- [Full experiment log](results/experiment_log.md)
- [Dataset configurations](configs/)

The main implementation is in [`src/rescue.py`](src/rescue.py); `run_pipeline.py` exposes `--separability_gate_mode {fixed,adaptive}`.

## Citation

The manuscript is in preparation. Citation information will be added upon publication.
