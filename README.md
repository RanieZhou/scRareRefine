# scRareRefine

Post-hoc refinement of rare cell-type annotations under limited supervision.

scRareRefine uses a frozen scANVI annotation model to refine predictions for one
predefined rare cell type. It combines train-derived latent prototypes with
validation-based calibration and selective correction. The method does not
retrain the base annotation model or discover unknown cell populations.

## Installation

The main workflow was developed for Python 3.11.

```bash
conda create -n scanvi311 python=3.11
conda activate scanvi311
pip install -r requirements.txt
```

Some comparison methods may require separate environments because of their
dependency versions.

## Quick start

Dataset paths, label columns and experiment settings are specified in
`configs/*.yaml`. After preparing the required input dataset, run:

```bash
python run_pipeline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_train_size 0.05
```

To use the fixed separability rule or the cell-stratified split, pass the
corresponding command-line options:

```bash
python run_pipeline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_train_size 0.05 \
  --separability_gate_mode fixed

python run_pipeline.py \
  --config configs/immune_dc.yaml \
  --seed 42 \
  --rare_train_size 0.05 \
  --split_mode cell_stratified
```

Generated outputs are written to the locations defined by the selected
configuration. Existing embeddings can be reused when their provenance matches
the requested dataset, split, seed and label budget.

## Repository layout

```text
src/                    Core scRareRefine implementation
configs/                Dataset and experiment configurations
baseline/               Baseline integration wrappers
tools/comparison/       Benchmark comparison scripts
tools/analysis/         Analysis scripts
tools/figures/          Figure-generation scripts
tests/                  Unit and regression tests
```

Raw datasets, checkpoints, generated results and manuscript files are not
included in this repository. Obtain the required datasets from their original
sources and update the corresponding configuration paths before running the
experiments.

## Citation

If you use scRareRefine, please cite this repository. Citation metadata are
provided in [`CITATION.cff`](CITATION.cff). The associated manuscript is in
preparation.

## License

This project is released under the MIT License; see [`LICENSE`](LICENSE).
