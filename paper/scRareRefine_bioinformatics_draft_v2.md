# scRareRefine: risk-controlled rescue of low-frequency cell types under rare-label scarcity

[Internal author note: this is an AI-assisted working draft for author rewriting. Bioinformatics follows the ISCB policy on large language models; before submission, the manuscript must be substantially revised by the authors, all references and numerical claims must be checked against the final artifacts, and AI assistance must be disclosed where required.]

Manuscript category: Gene expression  
Article type: Original Paper  
Running title: Risk-controlled rare-cell rescue

Authors: [AUTHOR NAMES]  
Affiliations: [AFFILIATIONS]  
Corresponding author: [CONTACT EMAIL]

## Structured Abstract

### Motivation

Semi-supervised single-cell annotation models can miss low-frequency cell types when only a small number of rare-class labels are available. A useful refinement method should recover these missed cells without creating a broad false rare-cell call set.

### Results

We introduce scRareRefine, a post-hoc refinement module for scANVI that combines train-derived latent prototypes with validation-only conformal calibration. scRareRefine targets a pre-specified low-frequency class and abstains when validation evidence does not support rescue. Across eight human and mouse scRNA-seq datasets, nine methods, four rare-label budgets and three seeds, the scarce-label settings contained only 0.0206%-0.3526% labeled rare cells in training. In these settings, scRareRefine achieved mean rare-cell F1 of 0.801, compared with 0.646 for scANVI and 0.665 for kNN, and was best or tied-best in 23 of 24 dataset-budget cells. Its maximum false rare-call rate was 0.009878 under a pre-specified 0.01 budget.

### Availability and Implementation

Source code, configuration files, and result-generation scripts will be released at [GITHUB URL] and archived at [ZENODO DOI]. Implemented in Python using scvi-tools, Scanpy and AnnData.

### Contact

[CONTACT EMAIL]

### Supplementary Information

Supplementary data are available at [JOURNAL/SUPPLEMENT URL].

## 1 Introduction

Single-cell RNA sequencing has made it routine to profile complex tissues at cellular resolution, but the cell types that are most biologically interesting are often not the most abundant. Rare immune subsets, endocrine subtypes, tissue-resident stromal cells and lineage intermediates may account for only a few percent of a dataset. They are also the classes for which labels are most fragile: small reference panels contain few positive examples, batch-heldout evaluation can place the target class in different technical contexts, and semi-supervised classifiers may collapse an under-represented class into a nearby majority class.

This creates a practical failure mode for cell-type annotation. A model can achieve acceptable global accuracy while systematically losing the cell type under study. The problem is not simply that the class is rare in the full dataset. For a supervised or semi-supervised annotation model, the more direct quantity is the fraction of training cells that are labeled examples of the target rare class. In the experiments below, every formal target rare class has dataset-level prevalence below 5%, but the scarce-label regimes are more severe: after downsampling rare labels in the training split, labeled rare cells comprise only 0.0206%-0.3526% of all training cells in the main scarce-label settings. This distinction matters because a dataset can contain a real low-frequency class while providing too few labeled examples for a classifier to maintain a stable decision region.

Existing annotation methods approach the problem from different directions. scANVI learns a semi-supervised latent representation and classifier from labeled and unlabeled cells, making it a natural backbone when only part of the dataset is annotated. CellTypist, TOSICA and related supervised systems provide scalable annotation using marker-informed or transformer-based models. Rebalancing and rare-cell detection methods, including scBalance and scCAD, are designed to counter class imbalance or discover unusual populations. Prototype-based approaches such as ProtoCloud use geometric summaries of reference labels. These methods are valuable, but they do not directly address a specific deployment question: when a known low-frequency target class is missed by a semi-supervised classifier, can we selectively recover likely false negatives while controlling false rare-cell calls?

scRareRefine is designed for this setting. It is not a de novo rare-cell discovery method and it does not change the scANVI training objective. Instead, it is an inductive post-hoc module applied after scANVI. It builds latent-space prototypes from labeled training cells, scores candidate cells by their proximity to the target rare prototype relative to other class prototypes, and uses validation cells to decide whether rescue is needed and how permissive the rescue rule should be. The final relabeling threshold is calibrated from validation non-rare cells under a fixed false rare-call budget. If the target class is not geometrically separable, if the baseline already recovers the validation rare cells, or if no candidate rank satisfies the validation safety criterion, the method abstains and returns the baseline predictions.

The contribution is therefore a constrained recovery procedure rather than a new end-to-end classifier. This positioning is important for evaluation. A useful method must improve rare-class F1 and recall in the label-scarce region, but it must not do so by broadly assigning the rare label. We evaluate scRareRefine against eight baselines or alternative methods across eight datasets covering human and mouse samples, multiple tissues, three random seeds, and four rare-label budgets. The benchmark includes six human datasets and two mouse Tabula Muris Senis 10x add-on datasets, enabling cross-tissue and cross-species breadth while preserving within-dataset train-validation-test evaluation. The main claim is that scRareRefine recovers rare-cell annotations under rare-label scarcity with a pre-specified false rare-call budget, and that it abstains when rescue is not warranted.

## 2 Materials and Methods

### 2.1 Problem setup

Let a single-cell dataset contain gene-expression matrix `X`, cell-type labels `y` for a subset of cells, and a pre-specified target rare class `r`. The target class is supplied by the dataset configuration rather than inferred automatically from a frequency threshold. We use the full labeled dataset only to define the evaluation target and then construct inductive train, validation and test splits. Model fitting, prototype estimation, rank selection and threshold calibration use only training and validation information. Test labels are reserved for final evaluation.

We distinguish two notions of rarity. The first is the dataset-level target prevalence,

`pi_r = N_r / N`,

where `N_r` is the number of cells of the configured target rare class and `N` is the dataset size. This is the biological and benchmarking definition used to select target classes. In the formal benchmark, after excluding the exploratory `tabula_lung_stroma` dataset, all configured target rare classes have `pi_r < 5%`.

The second is rare-label availability in the training split,

`lambda_r = L_r_train / N_train`,

where `L_r_train` is the number of labeled rare-class training cells retained after rare-label downsampling and `N_train` is the total number of training cells. This is the operational scarcity seen by the classifier. The label budgets `rare_train_size = 0.01, 0.05, 0.10` retain approximately 1%, 5% or 10% of training rare labels, with a minimum of five labeled rare cells when available; `rare_train_size = all` retains all training rare labels. Across the formal eight-dataset benchmark and three seeds, the scarce-label settings (`0.01, 0.05, 0.10`) correspond to `lambda_r = 0.0206%-0.3526%` of training cells, with mean `lambda_r` of 0.0559%, 0.0999% and 0.2003% at the three budgets, respectively.

### 2.2 Data splitting and label masking

The primary experiments use batch-heldout splits. Cells are partitioned into training, validation and test sets by batch when dataset metadata supports this design, with fallback logic handled by the preprocessing pipeline. Because batches have unequal sizes, the resulting split proportions need not exactly match nominal ratios, but the split is fixed before model fitting and is recorded in the run manifest. This design tests whether a rare-cell recovery rule transfers across held-out experimental batches rather than only across randomly sampled cells.

Within the training split, rare-class labels are downsampled according to the selected label budget. Non-rare labeled cells remain available, while rare cells whose labels are masked are treated as unlabeled for semi-supervised training. Validation and test labels are not downsampled for evaluation; validation labels are used only for method selection and calibration, and test labels are used only for final metrics.

### 2.3 scANVI backbone

The backbone model is scANVI, trained through scvi-tools on highly variable genes selected within the pipeline. The model first learns a latent representation with scVI and then fits the semi-supervised scANVI classifier using labeled and unlabeled training cells. For each split, the pipeline exports predicted labels, class probabilities and latent embeddings. These exported embeddings are shared by scRareRefine and by comparison scripts where applicable, with manifest checks to prevent accidental reuse across incompatible configurations, splits or datasets.

scRareRefine is applied only after the backbone predictions are generated. It does not alter scVI or scANVI optimization, does not retrain the classifier, and does not use test labels during threshold selection. This makes the method a post-hoc refinement module that can be inspected separately from the backbone.

### 2.4 Prototype scoring

scRareRefine builds class prototypes in the scANVI latent space using labeled training cells. For each class `c`, the prototype `mu_c` is the mean latent vector of labeled training cells of class `c`. A class-specific radius `rho_c` is estimated from within-class distances to the prototype, using the median distance with a conservative fallback for extremely small classes. Distances from a cell embedding `z_i` to each prototype are normalized by these radii.

The rare-membership score is computed from normalized prototype distances so that cells closer to the target rare prototype, relative to competing prototypes, receive higher scores. We also compute the rank of the rare prototype among all class prototypes for each cell. A cell is considered a rescue candidate only if the scANVI prediction is not the target rare class and the rare prototype rank is at most `k`, where `k` is selected on validation data from the grid `{1, 2, 3}`. This candidate rule limits rescue to cells that are already geometrically close to the rare prototype.

The same training prototypes are used for validation and test scoring. Validation and test cells never contribute to prototype coordinates, class radii, highly variable gene selection or the learned scANVI representation.

### 2.5 Validation-only gates and conformal calibration

scRareRefine applies three validation-only safeguards before relabeling test cells.

First, a separability gate compares the distance between the target rare prototype and the nearest non-rare prototype against the within-rare radius. If the target rare class is insufficiently separated (`separability < 1.3`), the method abstains. This constant is treated as a pre-specified conservative prior rather than tuned per dataset.

Second, a necessity gate checks whether rescue is supported by validation errors. If the validation split contains too few missed rare cells by the baseline, or if scANVI already recovers the target rare class on validation, scRareRefine abstains. This prevents unnecessary relabeling when the baseline is already saturated or when validation evidence is too sparse to select a reliable rescue rule.

Third, the method selects the maximum candidate rank `k` from `{1, 2, 3}` using validation performance under a safety constraint. Candidate ranks are evaluated by validation rare F1 subject to a Wilson upper-bound constraint on false rare-cell calls. Ties are broken toward the smaller rank, yielding a more conservative rule.

After a rank is selected, the final rare-score threshold is calibrated from validation non-rare cells. Let `s_i` be the rare-membership score for validation cells whose true label is not `r`. The threshold is a finite-sample high quantile corresponding to a pre-specified false rare-call budget `alpha = 0.01`. At test time, a non-rare scANVI prediction is changed to the target rare class only if the cell passes the selected rank rule and its rare-membership score exceeds the calibrated threshold. Otherwise, the original scANVI prediction is retained.

### 2.6 Evaluation metrics

The primary metric is rare-class F1 on the independent test split. We also report rare-class precision and recall, because the method is intended to recover rare false negatives rather than only improve aggregate annotation accuracy. The safety metric is the false rare-call rate among true non-rare cells, reported in the result files as `rare_fp_rate` or `fp_rate_max` depending on aggregation level. For scRareRefine, this rate is evaluated after the full rescue procedure has been applied.

We define the scarce-label region as `rare_train_size` in `{0.01, 0.05, 0.10}`. The `all` setting is retained to verify that scRareRefine does not create unnecessary changes when the backbone has substantially more rare labels.

### 2.7 Datasets and comparison methods

The formal benchmark contains eight datasets. Six are human datasets spanning immune, pancreas, lung, stomach and small intestine contexts. Two mouse Tabula Muris Senis 10x datasets were added to broaden the evaluation across species and tissues. The configured target rare classes are fixed in dataset configuration files and are not selected post hoc from model performance.

| Dataset configuration | Organism/context | Target rare class |
|---|---|---|
| `immune_dc` | human immune atlas | ASDC |
| `pancreas_baron` | human pancreas | gamma / epsilon |
| `pancreas_integrated` | integrated human pancreas | endothelial |
| `tabula_lung_endo` | human lung | endothelial subtype |
| `tabula_sapiens_stomach` | human stomach | mast cell |
| `tabula_small_intestine` | human small intestine | intestinal tuft cell |
| `mouse_lung_tms_10x` | mouse lung, Tabula Muris Senis 10x | vein endothelial cell |
| `mouse_pancreas_tms_10x` | mouse pancreas, Tabula Muris Senis 10x | pancreatic D cell |

We compare scRareRefine with scANVI, kNN on the shared latent representation, CellTypist, scBalance, ProtoCloud, HiCat, scCAD and TOSICA. All methods are evaluated on the same dataset splits, seeds and rare-label budgets where applicable. The complete comparison grid contains eight datasets, four label budgets, three seeds and nine methods, for 864 runs. The final result table contains 864 successful runs.

## 3 Results

### 3.1 Rare-label scarcity is substantially more severe than target-class prevalence

All formal target classes are low-frequency classes in their respective datasets, with full-dataset prevalence below 5%. However, the label-scarcity experiment is governed by the number of labeled rare cells available to the training process. In the most stringent settings, the classifier sees only a handful of rare labels against thousands of non-rare training cells. For example, in `immune_dc` at `rare_train_size = 0.01`, the training split contains 277 rare cells and 11,014 non-rare cells, but only five rare cells remain labeled, giving `lambda_r = 5 / (277 + 11014) = 0.0443%`.

This framing clarifies the evaluation target. The task is not to discover an arbitrary cluster below a fixed frequency threshold. The task is to preserve and recover a known low-frequency class when labeled positive examples are scarce enough that a semi-supervised classifier may erase the class from its predictions.

### 3.2 scRareRefine improves rare-cell F1 in scarce-label settings

In the scarce-label region (`rare_train_size` in `{0.01, 0.05, 0.10}`), scRareRefine achieved the highest mean rare-cell F1 across methods. Averaged over datasets, seeds and scarce-label budgets, its rare-cell F1 was 0.8008, compared with 0.6645 for kNN and 0.6463 for scANVI. The improvement is driven by recall recovery: scRareRefine mean rare recall was 0.7543, compared with 0.5883 for kNN and 0.5855 for scANVI.

| Method | Rare F1, scarce settings | Rare recall, scarce settings | Mean false rare-call rate | Max false rare-call rate |
|---|---:|---:|---:|---:|
| scRareRefine | 0.8008 | 0.7543 | 0.0012 | 0.0098 |
| kNN | 0.6645 | 0.5883 | 0.0002 | 0.0035 |
| scANVI | 0.6463 | 0.5855 | 0.0003 | 0.0023 |
| scBalance | 0.5827 | 0.4967 | 0.0001 | 0.0021 |
| ProtoCloud | 0.5592 | 0.4776 | 0.0002 | 0.0018 |
| CellTypist | 0.5514 | 0.4688 | 0.0002 | 0.0012 |
| scCAD | 0.4726 | 0.6698 | 0.0246 | 0.0752 |
| TOSICA | 0.4345 | 0.3918 | 0.0030 | 0.0209 |
| HiCat | 0.1445 | 0.1193 | 0.0000 | 0.0002 |

The method was also robust at the dataset-budget level. Collapsing over seeds gives 24 scarce-label dataset-budget cells. scRareRefine was exactly best or tied-best in 23 of 24 cells. In the only non-best cell, `tabula_small_intestine` at `rare_train_size = 0.10`, ProtoCloud achieved rare F1 of 0.9848 and scRareRefine achieved 0.9823, a difference of 0.0025.

### 3.3 Paired comparisons show consistent gains over the backbone and baselines

We next compared scRareRefine against each method in matched dataset-budget-seed units in the scarce-label region (`n = 72` paired units per baseline). Against scANVI, scRareRefine had 41 wins, 30 ties and one loss, with mean paired `Delta F1 = +0.1545` and bootstrap confidence interval `[+0.0956, +0.2199]` (one-sided paired Wilcoxon `P = 8.82e-09`). The large number of ties is expected: if validation evidence indicates that rescue is unnecessary or unsafe, scRareRefine returns the baseline prediction. Because paired units are correlated across seeds and some rare-label budgets collapse to the same effective label count, these `P` values should be read as directional evidence alongside effect sizes and confidence intervals.

| Comparator | Wins | Ties | Losses | Mean paired Delta F1 | Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| scANVI | 41 | 30 | 1 | +0.1545 | [+0.0956, +0.2199] |
| kNN | 59 | 8 | 5 | +0.1363 | [+0.0967, +0.1773] |
| CellTypist | 65 | 4 | 3 | +0.2494 | [+0.1950, +0.3049] |
| scBalance | 61 | 5 | 6 | +0.2181 | [+0.1642, +0.2737] |
| ProtoCloud | 62 | 6 | 4 | +0.2416 | [+0.1884, +0.2972] |
| HiCat | 66 | 5 | 1 | +0.6563 | [+0.5795, +0.7291] |
| scCAD | 66 | 5 | 1 | +0.3282 | [+0.2815, +0.3771] |
| TOSICA | 67 | 2 | 3 | +0.3663 | [+0.3064, +0.4261] |

These paired results support the interpretation that scRareRefine is not merely benefiting from a favorable average across datasets. The gain appears in most matched scarce-label runs, while abstention limits changes in runs where the backbone is already sufficient.

### 3.4 False rare-cell calls remain within the pre-specified budget for scRareRefine

Recovering more rare cells is only useful if the rare label remains specific. Across the complete eight-dataset comparison, including the `all` label-budget setting, the maximum false rare-call rate observed for scRareRefine was 0.009878, below the pre-specified `alpha = 0.01` budget. This maximum occurred in `mouse_lung_tms_10x` at `rare_train_size = all` for one seed. In the scarce-label region, the largest values remained close to, but below, the same budget.

The comparison methods show different safety profiles. Some methods, such as kNN and scANVI, have low false rare-call rates but miss many rare cells. Others, such as scCAD in this benchmark, recover more rare cells but can exceed the false rare-call budget. scRareRefine occupies the intended middle ground: it increases rare recall relative to scANVI while using validation calibration to constrain broad rare-label expansion.

### 3.5 The all-label setting tests graceful abstention

When all training rare labels are available, the backbone has more information about the target class. In this setting, scRareRefine should not be expected to produce large gains. The desired behavior is graceful convergence toward the baseline, with rescue applied only when validation evidence still indicates missed rare cells. Across all four label budgets, scRareRefine mean rare F1 was 0.8267 and mean rare recall was 0.7871, remaining the strongest aggregate method while preserving the same maximum false rare-call constraint.

This behavior is consistent with the method design. The necessity gate and validation-selected rank prevent rescue from becoming a fixed post-processing step that always changes predictions. Instead, scRareRefine is conditional: it acts when the backbone misses validation rare cells and when the latent geometry supports a controlled rescue rule.

### 3.6 Cross-tissue and mouse add-on datasets support broader applicability

The original human benchmark already covers multiple tissue contexts, including immune, pancreas, lung, stomach and small intestine datasets. The two mouse Tabula Muris Senis 10x add-on datasets extend this breadth to a second species and additional tissue contexts while keeping the same evaluation contract. These mouse datasets are not treated as a separate domain-transfer experiment in which a model is trained on human cells and tested on mouse cells. Rather, they test whether the same label-scarcity and rescue procedure remains useful outside the original human-only panel.

The inclusion of mouse lung and mouse pancreas also reduces the risk that the method is tuned to a single organism or to a single tissue geometry. The main aggregate results include these two datasets, and the complete 864-run comparison succeeded without method-specific changes to the scRareRefine algorithm. This supports the claim that the procedure is a general rare-label refinement strategy for scRNA-seq annotation, while leaving formal cross-species transfer as future work.

### 3.7 Failure modes are informative and bounded

scRareRefine is intentionally conservative, and its failures are tied to interpretable conditions. If the target rare prototype is not separated from nearby majority classes, the separability gate abstains. If validation contains too few missed rare cells, the necessity gate abstains rather than selecting a rule from weak evidence. If missed rare cells are far from the rare prototype and rank below the candidate grid, the method cannot rescue them without expanding the candidate set and risking false rare calls.

These conditions appear in the benchmark. The stomach dataset shows a recall ceiling in which a subset of mast cells remains geometrically entangled with majority classes. In such cases, increasing the conformal threshold alone cannot recover cells that fail the candidate rank rule. The pancreas Baron dataset is sensitive in the most extreme rare-label settings, where only the minimum number of rare labels is available and validation-test distribution shift can place the method close to the false rare-call budget. These limitations argue against presenting scRareRefine as a universal rare-cell detector. Its strength is a specific, auditable rescue rule for known rare classes under scarce labels.

## 4 Discussion

The central result is that a constrained post-hoc rescue module can recover rare-cell annotations lost by a semi-supervised backbone under severe label scarcity. scRareRefine improves rare-cell F1 and recall across an eight-dataset benchmark while keeping the observed false rare-call rate below a fixed 1% budget. The method is deliberately narrow: it targets one configured low-frequency class, builds prototypes only from labeled training cells, calibrates thresholds only on validation cells, and abstains when validation evidence is insufficient.

This narrowness is a practical advantage. Many single-cell studies do not need a fully new annotation model; they need to know whether a small, known cell type has been erased by the combination of class imbalance and batch shift. Because scRareRefine operates after scANVI, it can be attached to an established semi-supervised workflow and audited through its gates, candidate ranks and calibrated threshold. When it changes predictions, the change is traceable to latent prototype proximity and validation-calibrated rare-score evidence. When it does not change predictions, the abstention is also informative.

The evaluation also highlights why rare-cell benchmarking should report both target prevalence and labeled rare-cell availability. A target class below 5% is biologically low-frequency, but the model's difficulty depends on how many labeled rare examples remain in the training set. In our benchmark, the scarce-label settings expose the classifier to labeled rare cells comprising at most 0.3526% of the training split. Reporting `lambda_r` makes clear that the method is being tested under a true rare-label regime, rather than simply on datasets containing low-frequency classes.

There are several limitations. First, scRareRefine assumes the target rare class is known in advance. It is therefore complementary to de novo rare-cell discovery methods rather than a replacement for them. Second, the current implementation refines a single target rare class per run. Multi-rare-class settings may require class-wise calibration or a joint error-budget allocation. Third, the method inherits the quality of the scANVI latent representation. If rare and majority cells are not separable in the latent space, the conservative gates will abstain or recall will remain bounded. Fourth, while the benchmark now includes mouse add-on datasets, it does not test direct human-to-mouse model transfer. Finally, the current manuscript reports computational results from public scRNA-seq datasets; experimental validation of newly recovered rare cells would require independent biological assays or marker-level confirmation.

The comparison methods also motivate careful interpretation. A method with very low false rare-call rate can still be undesirable if it misses most rare cells. Conversely, a method with high recall can be unsafe if it broadly assigns the rare label. For the use case studied here, rare-cell F1, recall and false rare-call rate must be considered together. scRareRefine is designed around this three-way trade-off rather than around global accuracy.

## 5 Conclusions

scRareRefine provides a validation-calibrated post-hoc rescue procedure for known low-frequency cell types in scRNA-seq annotation. By combining train-only latent prototypes, validation-selected candidate ranks, conformal score thresholds and abstention gates, it improves rare-cell recovery under severe rare-label scarcity while maintaining a fixed false rare-call budget. The current eight-dataset benchmark supports scRareRefine as a practical refinement module for studies where preserving a known rare cell type is more important than maximizing aggregate annotation accuracy.

## Data availability

All datasets used in this study are derived from public single-cell resources, including human atlas datasets and CELLxGENE/Tabula Muris Senis mouse 10x subsets. Processed dataset identifiers, download instructions and preprocessing scripts will be provided in the repository and Supplementary Data. No new sequencing data were generated for this study.

## Code availability

The scRareRefine implementation, configuration files, comparison scripts, plotting scripts and result manifests will be made freely available at [GITHUB URL] under [LICENSE]. The submitted version will be archived at [ZENODO DOI]. Reproducibility instructions will include the two conda environments used for the comparison grid: `scanvi311` for the scANVI/scRareRefine pipeline and most baselines, and `sandbox310` for baselines requiring older dependency stacks.

## Funding

This work was supported by [FUNDING INFORMATION].

## Conflict of Interest

The authors declare no competing interests.

## Author contributions

[AUTHOR CONTRIBUTIONS]

## Acknowledgements

The authors thank the maintainers of scvi-tools, Scanpy, AnnData, CELLxGENE, Tabula Sapiens and Tabula Muris Senis for public data and software resources.

## References

[To be verified and converted to Oxford SCIMED style before submission.]

- Lopez et al. Deep generative modeling for single-cell transcriptomics. Nature Methods, 2018.
- Xu et al. Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models. Molecular Systems Biology, 2021.
- Angelopoulos and Bates. Conformal prediction: a gentle introduction. Foundations and Trends in Machine Learning, 2023.
- Vovk et al. Algorithmic Learning in a Random World. Springer, 2005.
- The Tabula Sapiens Consortium. The Tabula Sapiens: a multiple-organ, single-cell transcriptomic atlas of humans. Science, 2022.
- Tabula Muris Consortium. Single-cell transcriptomics of 20 mouse organs creates a Tabula Muris. Nature, 2018.
- Tabula Muris Senis Consortium. A single-cell transcriptomic atlas characterizes ageing tissues in the mouse. Nature, 2020.
- Baron et al. A single-cell transcriptomic map of the human and mouse pancreas reveals inter- and intra-cell population structure. Cell Systems, 2016.
- Dominguez Conde et al. Cross-tissue immune cell analysis reveals tissue-specific features in humans. Science, 2022.
- Chen et al. TOSICA: a transformer-based supervised cell-type annotation method for single-cell RNA-seq data. [VERIFY].
- scBalance reference. [VERIFY].
- ProtoCloud reference. [VERIFY].
- HiCat reference. [VERIFY].
- scCAD reference. [VERIFY].

## Figure plan for current draft

Figure 1. Workflow schematic. scANVI produces latent embeddings and baseline predictions; scRareRefine builds train-only prototypes, selects candidate rank and threshold on validation cells, and rescues only calibrated test candidates.  
Figure 2. Rare-label scarcity curves across `rare_train_size`, showing rare F1 and recall for scANVI and scRareRefine.  
Figure 3. Eight-dataset comparison grid for the nine methods, with rare F1 and false rare-call rate.  
Figure 4. Component ablation for separability gate, necessity gate, adaptive rank and conformal threshold.  
Figure 5. Representative UMAP or latent-space panels for successful rescue and a bounded failure case.  
Supplementary Figure S1. Dataset-wise split composition and labeled rare-cell availability.  
Supplementary Figure S2. Random proportional split sensitivity analysis.  
Supplementary Figure S3. Runtime and memory comparison.  

## Internal checks still needed before journal submission

1. Use the recomputed 8-dataset `results/comparison/significance_test.csv`; report p-values only as directional evidence because paired units are correlated across seeds and collapsed rare-label budgets.
2. Verify every reference and replace placeholder entries with exact journal metadata and DOIs.
3. Decide whether ablation results should be rerun on all eight datasets or clearly labeled as six-human ablation.
4. Add final GitHub and Zenodo URLs before submission.
5. Convert this working draft into the final author-written Bioinformatics or BMC Bioinformatics template.
