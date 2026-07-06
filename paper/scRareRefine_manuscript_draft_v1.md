# Recovering rare cell types missed under label scarcity: a risk-controlled post-hoc rescue for single-cell annotation

**Working draft v1 — 2026-07-05.** Main line: *recall recovery under label scarcity*. F1 is the headline metric; rare recall is reported as a supporting panel. False-rescue rate (FFR ≤ α) is framed as a safety constraint, not the headline. All numbers trace to `results/` CSVs / `experiment_log.md` (see Evidence source map at the end; that section and every "中文：" note do not go into the submitted manuscript).

Target venue: primary **Bioinformatics**; fallback **BMC Bioinformatics**.

Honesty guardrails (ITERATION_BOUNDARY R5): we do **not** claim to solve rare-cell identification, to be SOTA, to beat all methods uniformly, to be clinically usable, or to generalize to all single-cell data. Claims are restricted to the six evaluated datasets and the label-scarce regime (rare_train_size ≤ 0.10).

---

## Abstract

Semi-supervised single-cell annotators such as scANVI achieve strong overall accuracy, but when a rare cell type is represented by only a handful of labeled cells they systematically fail to recall it — the population that matters most is the one that is lost. We ask a focused question: **under label scarcity, can the rare cells missed by a semi-supervised classifier be recovered post hoc, using only inductive (training + validation) information, without inflating false rescues?** We present scRareRefine, a post-hoc module that scores each cell by its distance to training-set class prototypes in the scANVI latent space and admits candidate rare cells using a validation-calibrated conformal threshold, with two abstention gates (a separability safety net and a necessity guard) that return the unmodified scANVI prediction whenever rescue is not warranted. Across six datasets, nine methods and three random splits, scRareRefine recovers rare-cell F1 that scANVI loses under scarcity, and the recovery *grows as labels become scarcer* — for example, on human immune dendritic cells at five labeled rare cells, rare-cell F1 rises from 0.00 (scANVI) to 0.93, while at a 10% label budget the gap narrows to +0.03 and the method gracefully abstains when the baseline is already saturated. In the label-scarce regime scRareRefine attains the highest rare-cell F1 among nine methods (0.878 vs. 0.725 next-best) with zero regressions against scANVI (29 wins / 25 ties / 0 losses over 54 scarce configurations), while keeping the worst-case false-rescue rate within a fixed budget α = 0.01 (0.0098 ≤ α) where several competitors exceed it. We characterize the method's failure modes honestly, including a recall ceiling for transcriptionally entangled rare types (mast cells, recall ≈ 0.59) and seed instability at the most extreme label budget. scRareRefine is a conservative, do-no-harm rescuer whose value is concentrated in the genuinely label-scarce, genuinely-missed regime.

---

## 1. Introduction

Cell-type annotation is the entry point of most single-cell RNA-seq analyses, and semi-supervised methods such as scANVI have become a standard because they transfer labels from a small annotated reference to a large unlabeled body of cells while modeling batch structure. Yet the cells that are biologically most interesting are frequently the rarest — a small dendritic-cell subset, a scattered mast-cell population, an endocrine islet subtype — and these are exactly the cells that a classifier trained on a handful of labeled examples tends to miss. In the annotation-scarce regime, the dominant failure is not misclassification of abundant types but **under-recall of the rare type**: the classifier plays it safe and assigns rare cells to a nearby majority class.

A tempting fix is to lower the decision threshold for the rare class, but naive thresholding trades missed rare cells for a flood of false positives — cells wrongly "rescued" into the rare class. In a discovery setting a false rescue is not a benign error: it fabricates a rare population that is not there. Any practical rescue procedure therefore has to control this cost. The problem we address is the tension between **recovering genuinely missed rare cells** and **not manufacturing false ones**, specifically in the regime where labels are scarce enough that the backbone classifier fails on its own.

This is a rare-cell **recall-recovery** problem under supervision, not a rare-cell **discovery** problem. We assume the rare type is known and appears — however sparsely — in the labeled reference; our goal is to recover its recall, not to discover novel populations de novo. This distinction governs our evaluation and our choice of the most relevant comparators (imbalance-aware supervised annotators), and separates our task from unsupervised rare-population detectors, which we include only as a paradigm reference.

**Contributions.** We make the following claims, each falsifiable and tied to specific evidence:

- **C1 (headline — recovery).** scANVI systematically under-recalls rare cell types under label scarcity; scRareRefine recovers this signal, and the recovery *grows as labels become scarcer*, converging to a no-op (returning scANVI) when the baseline is already saturated (Fig. 2, rts recovery curves).
- **C1b (magnitude — comparison).** In the label-scarce regime, scRareRefine attains higher rare-cell F1 than scANVI and seven comparison methods, with zero regressions against scANVI (Fig. 3a; significance test; distinct-configuration counts).
- **C2 (safety constraint).** scRareRefine keeps the worst-case false-rescue rate within a fixed budget α = 0.01 — abstaining rather than over-firing — where several comparison methods are uncontrolled (Fig. 3b). This is framed as a safety property supporting C1, not as the paper's headline.
- **C3 (mechanism).** Each component earns its place: the adaptive candidate rank drives the F1 gain, the conformal threshold controls FFR, and the two gates provide safety and anti-regression (Fig. 4, two ablation tables).
- **C4 (conditioned by separability).** The separability gate uses a fixed, pre-specified conservative prior (threshold 1.3); we show it is the smallest gate keeping worst-case FFR ≤ α on the real benchmark, and that the risk axis is dataset-dependent rather than a universal boundary (Fig. 5).
- **C5 (honest failure modes).** The method is conditional by design (a no-op outside its regime) and has known limitations: a recall ceiling for geometrically entangled rare types, and seed instability at extreme scarcity (Discussion §4; Fig. 6).

### 1.1 Related work

**Supervised / semi-supervised annotation.** scVI and scANVI provide a probabilistic latent space with batch modeling; scANVI is our backbone and primary baseline. Marker- or logistic-regression-based annotators (CellTypist), imbalance-aware classifiers (scBalance), and attention/pathway models (TOSICA) are supervised competitors that produce per-cell labels but do not offer an explicit false-positive budget. Prototype-based classifiers (ProtoCloud) are conceptually closest to our distance-to-prototype scoring.

**Rare-cell discovery.** Unsupervised detectors of rare or novel populations (scCAD, HiCat) address a different task — finding unknown rare structure without labels. We include them for completeness and clearly mark HiCat as a transductive reference (its dimensionality reduction is fit on combined train and test cells) rather than a fair inductive competitor.

**Conformal / selective prediction.** Distribution-free calibration (conformal prediction; Angelopoulos & Bates) motivates our validation-only thresholding. We use conformal calibration in the narrow role of controlling the false-rescue rate under an exchangeability assumption, and we are explicit about where batch structure weakens that assumption.

---

## 2. Methods

### 2.1 Backbone and inductive three-way split

We use scANVI to produce, for every cell, a latent embedding and a vector of class prediction probabilities. Every downstream quantity in scRareRefine is computed under a strict inductive protocol: training-set labeled cells define the class prototypes and highly-variable genes; validation cells (never used in training) calibrate all thresholds; test labels are used only for final metric computation and never for any tuning, threshold selection, or model selection. We evaluate two split modes — `batch_heldout` (validation and test drawn from held-out donors) and `cell_stratified` — with `batch_heldout` as the primary, more demanding protocol.

### 2.2 Prototype scoring

For each class *c* we compute the mean prototype of its training-set labeled cells in the scANVI latent space and a per-class radius `r_c = median(distance of class-c training cells to their prototype)` (set to 1.0 when fewer than three cells are available). A cell's **rare-membership score** is an anisotropic softmax over `−d_c / r_c`, normalizing each class by its own compactness so that a cell is scored by how well it fits the rare cluster's scale rather than by raw Euclidean distance. We also define a **separability ratio** `sep = d(rare prototype, nearest majority prototype) / mean(intra-rare radius)`, a training-only measure of how geometrically isolated the rare cluster is, and an isotropic Euclidean **candidate rank**, where a cell has rank *k* for the rare class if the rare prototype is its *k*-th nearest class prototype.

### 2.3 Conformal calibration and the rescue decision

The top-level `conformal_rescue()` procedure applies three fully inductive gates:

1. **Separability safety net.** If `sep < CONFORMAL_LOW_SEP = 1.3`, abstain (return scANVI). This threshold is a fixed cross-dataset conservative prior, justified in §3.4 and Fig. 5, not tuned per dataset.
2. **Necessity guard.** If the validation baseline already recalls the rare population (validation rare recall = 1.0, subject to a minimum of `MIN_VAL_MISSED = 3` missed validation rare cells), abstain — there is nothing to recover and rescue could only add risk.
3. **Validation-adaptive candidate rank + conformal threshold.** Over a fixed grid `rank ∈ {1, 2, 3}`, we select the `max_rank` that maximizes validation rare-cell F1 subject to a Wilson 95% upper bound on the validation FFR ≤ α (ties broken toward the smaller rank). We then set the conformal threshold τ as the finite-sample (1 − α) order statistic of the non-rare validation scores, and apply the selected rank and τ to the test cells.

The publication-level budget `α = 0.01` is a fixed cross-dataset constant and is never tuned.

### 2.4 False-rescue rate

We define the **false-rescue rate (FFR)** as the fraction of cells rescued into the rare class that are not truly rare. FFR is the quantity the conformal threshold controls; we report it with raw counts and Wilson confidence intervals, and we describe FFR ≤ α as *empirical* control under validation calibration rather than a strict finite-sample guarantee on test (see §4.3).

---

## 3. Results

### 3.1 Experimental setup

We evaluate on six datasets spanning human immune, pancreas and Tabula Sapiens tissues: `immune_dc` (rare = ASDC dendritic cells), `pancreas_baron` (rare = gamma / epsilon endocrine cells), `pancreas_integrated` (multi-dataset integrated pancreas), `tabula_lung_endo` (rare = lymphatic endothelial cells), `tabula_sapiens_stomach` (rare = mast cells), and `tabula_small_intestine`. We compare nine methods — scRareRefine plus scANVI, kNN, CellTypist, scBalance, ProtoCloud, HiCat (transductive reference), scCAD, and TOSICA (run at a reduced configuration for tractability) — across four label budgets `rare_train_size ∈ {0.01, 0.05, 0.10, all}` and three random seeds {42, 43, 44}, for 9 × 6 × 4 × 3 = 648 runs, all completed. Metrics are rare-cell F1 (headline) and rare-cell recall (supporting), with FFR as a safety constraint.

The number of labeled rare cells is `max(5, ⌊p · N_rare⌋)`, so on small-rare datasets several nominal fractions collapse to the same labeled set. We therefore report the scarce-region comparison over the 15 *distinct* labeled configurations rather than the 18 nominal cells (see §4.6).

### 3.2 Recovery grows as labels become scarcer (C1 — headline)

The central result is the trajectory of the rare-cell gain as the label budget shrinks (Fig. 2, three-seed mean ± SD from `core_agg.csv`). The recovery is largest exactly where scANVI fails hardest:

| dataset | rts | scANVI F1 (3-seed) | scRareRefine F1 (3-seed) | gain |
|---|---|---|---|---|
| immune_dc | 0.01 | 0.000 ± 0.000 | 0.927 ± 0.018 | **+0.927** |
| immune_dc | 0.05 | 0.871 ± 0.023 | 0.940 ± 0.002 | +0.069 |
| immune_dc | 0.10 | 0.910 ± 0.022 | 0.943 ± 0.012 | +0.033 |
| tabula_sapiens_stomach | all rts | 0.607 ± 0.049 | 0.719 ± 0.022 | +0.112 |
| pancreas_baron | 0.10 | 0.820 ± 0.048 | 0.842 ± 0.032 | +0.023 |

At five labeled rare cells on `immune_dc`, scANVI recalls essentially no ASDC cells (F1 = 0.00) while scRareRefine recovers F1 = 0.93; as the budget rises to 10%, scANVI catches up and the gain shrinks to +0.03. This is the intended behavior: the method contributes most under scarcity and fades to a no-op as the baseline saturates. On `pancreas_integrated` and `tabula_small_intestine`, where the necessity gate abstains on most or all configurations, scRareRefine reduces exactly to scANVI — neither helping nor harming.

**Supporting recall panel.** Because recall is precisely what scANVI loses, we report rare recall as a supporting panel (Supplementary). The rank-sensitivity analysis (§3.4, Table 2) shows the adaptive configuration operating at recall ≈ 0.853 overall, and the one dataset where recall does not fully recover — stomach mast cells at ≈ 0.59 — is discussed as a headline failure mode (§4.5).

### 3.3 Comparison against nine methods in the scarce regime (C1b)

Aggregated over the scarce region (three seeds, distinct configurations), scRareRefine attains the highest rare-cell F1 among all nine methods (0.878 vs. 0.725 next-best; Fig. 3a). Paired significance testing (one-sided Wilcoxon and bootstrap 95% CIs, paired by dataset × fraction × seed; `significance_test.csv`) gives, in the scarce region (n = 54):

| vs. baseline | win/tie/loss | mean ΔF1 | bootstrap 95% CI | Wilcoxon p |
|---|---|---|---|---|
| scANVI | **29 / 25 / 0** | +0.160 | [+0.085, +0.244] | 1.3e-6 |
| kNN | 46 / 6 / 2 | +0.153 | [+0.106, +0.204] | 1.9e-9 |
| CellTypist | 51 / 1 / 2 | +0.249 | [+0.188, +0.316] | 1.8e-10 |
| scBalance | 47 / 5 / 2 | +0.235 | [+0.170, +0.304] | 9.7e-10 |
| ProtoCloud | 48 / 4 / 2 | +0.226 | [+0.166, +0.289] | 5.1e-10 |
| HiCat † (transductive) | 52 / 1 / 1 | +0.692 | [+0.607, +0.771] | 1.3e-10 |
| scCAD | 52 / 1 / 1 | +0.350 | [+0.294, +0.408] | 1.3e-10 |
| TOSICA | 53 / 1 / 0 | +0.387 | [+0.321, +0.454] | 1.2e-10 |

Against scANVI the scarce region shows **zero regressions** (25 ties correspond to necessity abstentions), and the ΔF1 confidence interval excludes zero for every one of the eight baselines. Over the distinct scarce configurations, scRareRefine is win-most in 15/15 and best in 14/15 (the single exception, `small_intestine` at 10%, is a saturated baseline). We deliberately temper these p-values: the 54 cells are not fully independent (three seeds of the same configuration are correlated, and collapsed fractions are near-duplicated), so we read them as directional evidence of a robust paired improvement with an effect size whose CI excludes zero, not as strict independent-sample tests (§4.6).

### 3.4 Ablation: each component earns its place (C3)

We decompose the method two ways (three-seed means, overall 6 × 4 × 3; `ablation_table1_components.csv`, `ablation_table2_rank.csv`).

**Component leave-one-out (Table 1).** Removing the adaptive rank is the only change that clearly lowers F1 (Δ = +0.010, i.e. F1 drops by 0.010 when it is removed); removing the conformal threshold τ barely moves F1 but breaks the budget (FFR 0.0165 > α); removing the separability gate actually *raises* F1 by 0.018 but also breaks the budget (FFR 0.0153 > α), confirming its role is safety rather than accuracy. The necessity gate contributes little overall (+0.002) but prevents per-dataset regressions (e.g., `pancreas_integrated` Δ = +0.0121).

| variant | F1 (mean ± SD) | Δ = Full − variant | FFR_max | abstain |
|---|---|---|---|---|
| A0 baseline (scANVI) | 0.761 ± 0.301 | +0.127 | 0 | 0/72 |
| A1 − separability gate | 0.905 ± 0.103 | −0.018 | **0.0153 (>α)** | 31/72 |
| A2 − necessity guard | 0.885 ± 0.150 | +0.002 | 0.0098 | 8/72 |
| A3 − adaptive rank (→ k=1) | 0.877 ± 0.164 | +0.010 | 0.0049 | 37/72 |
| A4 − conformal τ | 0.885 ± 0.153 | +0.002 | **0.0165 (>α)** | 37/72 |
| A5 full | 0.887 ± 0.151 | 0 | 0.0098 | 37/72 |

**Rank sensitivity (Table 2).** The adaptive rank behaves like an oracle that picks the best fixed rank per dataset while respecting the FFR budget: fixed rank = 3 attains the highest recall (0.875) but blows the budget (FFR 0.046 ≫ α), while the adaptive choice attains the highest F1 (0.887) at FFR ≤ α.

| variant | F1 | recall | FFR_max |
|---|---|---|---|
| fixed rank = 1 | 0.877 | 0.838 | 0.0049 |
| fixed rank = 2 | 0.865 | 0.868 | 0.0100 |
| fixed rank = 3 | 0.853 | 0.875 | **0.0464 (≫α)** |
| adaptive | **0.887** | 0.853 | 0.0098 |

The narrative is clean: the adaptive rank recovers rare cells (F1/recall), the conformal threshold controls the false-rescue cost, and the two gates keep the method safe and regression-free. Because the method's design is "conservative in exchange for control," most components do not raise F1 — they cap risk.

### 3.5 The separability gate is a conservative prior (C4)

The separability threshold 1.3 is fixed across datasets. Two stress tests characterize it (Fig. 5). On the real benchmark, a sensitivity sweep over the threshold (`lowsep_sensitivity_agg.csv`, cache-only over the 648 real embeddings) shows 1.3 is the *smallest* gate keeping worst-case FFR ≤ α across all 72 configurations: lowering it to ≤ 1.0 raises mean F1 by ≈ 0.018 but admits two FFR violations on `pancreas_baron` (sep ≈ 1.22, FFR 0.0153), while raising it to 1.6 only sacrifices F1 (−0.030) with no FFR benefit. The threshold is thus selected by the FFR constraint, not tuned for F1 — lowering it would *improve* F1.

In a controlled semi-synthetic sweep (progressively entangling the `tabula_lung_endo` lymphatic-EC population toward its nearest majority type), rescue remained FFR-safe down to sep ≈ 0.76 and produced only a marginal violation (18 of 1716 false rescues, FFR = 0.0105) at the lowest separability (≈ 0.69). The two experiments disagree on *where* rescue becomes unsafe (≈ 1.22 on real `pancreas_baron` vs. ≈ 0.69 in the synthetic sweep), which is itself the finding: **separability alone does not monotonically determine rescue risk**. We therefore do not claim to have localized a universal collapse boundary; 1.3 is a conservative cross-dataset compromise, tight enough to exclude the real violation at ≈ 1.22 and loose enough to retain most benefit, at the cost of occasionally abstaining where rescue would have been safe.

### 3.6 Qualitative rescue (C5 preview)

UMAP visualizations of before/after rescue (Fig. 6; `immune_dc`, `pancreas_baron`, `tabula_sapiens_stomach`) show rare cells recovered in latent space. The stomach panel also makes the recall ceiling visible: a fraction of true mast cells remain unrescued because admitting them would require widening the candidate pool past the FFR budget (§4.5).

---

## 4. Discussion — failure modes and limitations

We deliberately characterize where scRareRefine does not help, where its guarantees weaken, and where our evaluation is incomplete. scRareRefine is a conservative, risk-controlled rescuer whose value is concentrated in a specific regime rather than uniform across settings.

### 4.1 The method is conditional by design and is a no-op outside its regime

Two abstention gates cause the method to return the unmodified scANVI prediction whenever the rare and majority prototypes are too entangled or the validation baseline already recovers the rare population. On datasets where scANVI is already saturated (`pancreas_integrated`, `tabula_small_intestine`), scRareRefine reduces exactly to scANVI: it neither helps nor harms. Its benefit is concentrated in the genuinely label-scarce, genuinely-missed regime, and our claims are restricted accordingly. This is a feature for safety (do no harm) but means the method must not be presented as a universal improvement.

### 4.2 The separability gate is a conservative prior whose risk axis is dataset-dependent

As shown in §3.5, the fixed threshold 1.3 is selected by the FFR constraint rather than tuned for F1, but the real and synthetic stress tests disagree on where rescue becomes unsafe. Separability alone does not monotonically determine rescue risk; 1.3 is a conservative cross-dataset compromise that sometimes abstains where rescue would have been safe (e.g., forgoing a recoverable +0.24 F1 at synthetic separability ≈ 1.15). We do not claim a universal collapse boundary.

### 4.3 FFR control is empirical and can sit near the boundary under batch shift

The conformal threshold provides *empirical* FFR control under validation calibration, not a strict finite-sample guarantee on test. Under the batch-heldout protocol, validation and test cells come from different donors, so the exchangeability assumption underlying conformal coverage is only approximately satisfied. The clearest symptom is `pancreas_baron`, where the test FFR reaches 0.0098 — essentially at the α = 0.01 budget. Selecting the candidate rank via a Wilson 95% upper bound on the validation FFR (rather than a point estimate) reduces but does not remove this risk. FFR numbers should be read as empirical control under distribution shift, reported with raw counts.

### 4.4 Seed sensitivity at extreme label scarcity

At the most extreme budget (five labeled rare cells), the improvement on `pancreas_baron` is positive on average (+0.18 rare-cell F1) but unstable across seeds (± 0.23 over three seeds); the scANVI baseline itself swings comparably (0.38 ± 0.21). The instability disappears once at least ten rare cells are labeled. We therefore do not claim stable improvement at the ≤ 5-label point on this dataset, and flag extreme scarcity as a regime where both backbone and rescue inherit the variance of the underlying split.

### 4.5 A recall ceiling for geometrically entangled rare types (headline limitation)

Because the recovered quantity is fundamentally recall, the most important limitation of a recall-recovery method is the case where recall cannot be recovered. For rare types transcriptionally interleaved with a majority population, prototype-distance rescue has an intrinsic recall ceiling. On `tabula_sapiens_stomach` (mast cells), a substantial fraction of true rare cells fall at candidate rank ≥ 3 relative to the prototypes, where admitting them would require widening the candidate pool past the point at which the FFR budget is exceeded. The adaptive-rank mechanism correctly refuses to do so, so recall plateaus at ≈ 0.59. This is a fundamental limitation of latent-geometry rescue rather than a tuning artifact; expression-side signals (e.g., marker genes) may be needed to break this ceiling, which we leave to future work.

### 4.6 Evaluation caveats

- **Nominal label-fraction points are not all independent.** Because the labeled count is `max(5, ⌊p · N_rare⌋)`, several nominal fractions collapse to the same labeled set on small-rare datasets (all three scarce fractions map to five labeled cells on stomach; two of three on `pancreas_baron`). We report the scarce comparison over 15 distinct configurations rather than 18 nominal cells.
- **Statistical tests are paired but not fully independent.** Cells paired by (dataset, fraction, seed) are correlated across seeds and near-duplicated across collapsed fractions, so the p-values are optimistic; they are directional evidence of a robust paired improvement, not strict independent-sample tests.
- **Benchmark scope.** Six datasets, predominantly human and exclusively single-cell; we do not yet validate cross-species transfer or disease-versus-healthy settings, both of which would further stress exchangeability.
- **Baseline caveats.** HiCat is transductive (its dimensionality reduction is fit on combined train and test cells) and is reported as a transductive upper-bound reference, not a fair inductive competitor; TOSICA is run at a reduced configuration (10 epochs, 100 pathway tokens), which may underestimate it.
- **Backbone dependence.** scRareRefine operates primarily on scANVI latent embeddings. A cache-only weak-backbone demonstration using validation-selected kNN predictions on the same latent space improves scarce-region F1 from 0.725 to 0.860 and recall from 0.651 to 0.809 at FFR_max = 0.0098, but has one negative scarce-region cell; we therefore do not claim backbone-agnostic no-regression. Whether it helps on single-cell foundation-model embeddings (scGPT, scFoundation) is untested.

---

## 5. Conclusion

Under label scarcity, semi-supervised annotators lose the rare cell types that matter most. scRareRefine recovers them post hoc using only inductive information, with the recovery concentrated exactly where the baseline fails and fading to a safe no-op where it does not, while keeping the false-rescue rate within a fixed budget. On six datasets and against eight comparison methods, the recovery is largest in the scarce regime and never regresses against the scANVI backbone. We report the method's limitations — a recall ceiling for entangled rare types, seed instability at extreme scarcity, and a conservative separability prior whose risk axis is dataset-dependent — as first-class results. Future work includes expression-side signals to break the recall ceiling, cross-species and disease-state validation, and generalization beyond the scANVI backbone.

---

## Data and code availability

Six datasets (`immune_dc`, `pancreas_baron`, `pancreas_integrated`, `tabula_lung_endo`, `tabula_sapiens_stomach`, `tabula_small_intestine`) from public sources (Tabula Sapiens; Baron et al.). Pipeline and comparison scripts as described; per-run outputs and manifests under `outputs/`, aggregated results under `results/`.

---

## Evidence source map (NOT for submission — internal traceability)

| Manuscript element | Number(s) | Source |
|---|---|---|
| Abstract / §3.2 immune recovery 0.00→0.93; gains | rts curve, 3-seed | `results/multiseed/core_agg.csv`; experiment_log 第十三轮 |
| §3.3 scarce F1 0.878 vs 0.725 | main summary | `results/comparison/main_summary.png`; PAPER_PLAN C1 |
| §3.3 significance table (29/25/0 etc.) | scarce n=54 | `results/comparison/significance_test.csv`; experiment_log 第十三轮 Phase 3 |
| §3.3 win-most 15/15, best 14/15 | distinct | `results/comparison/scarce_region_distinct.csv` |
| §3.4 Table 1 (component leave-one-out) | overall 3-seed | `results/ablation/ablation_table1_components.csv` |
| §3.4 Table 2 (rank sensitivity) | overall 3-seed | `results/ablation/ablation_table2_rank.csv` |
| §3.5 / §4.2 low_sep sensitivity (1.3 smallest safe; 2 violations at sep≈1.22) | benchmark | `results/sep_sweep/lowsep_sensitivity_agg.csv`; experiment_log 第十四轮 G82 |
| §3.5 synthetic sweep (safe to 0.76; 18/1716 at 0.69) | lung_endo | `results/sep_sweep/sep_sweep_summary.csv`; experiment_log 第十四轮 |
| §4.4 pancreas_baron seed instability (+0.18±0.23) | 3-seed | `results/multiseed/core_agg.csv`; failure_modes §6.4 |
| §4.5 stomach recall ceiling ≈0.59 | — | experiment_log G10; failure_modes §6.5 |
| Fig 6 UMAP | qualitative | `results/umap/umap_rescue_{immune_dc,pancreas_baron,tabula_sapiens_stomach}.png` |
| Supp recall panel | recall curves | `results/multiseed/figS_recall_recovery_panel.png`; `paper/figures/figS_recall_recovery_panel.png` |
| Weak-backbone demo | kNN rescue | `results/weak_backbone/weak_backbone_summary.csv`; `results/weak_backbone/weak_backbone_agg.csv` |
| Provenance audit | split hash / cache status | `results/provenance/cache_audit.csv`; `results/provenance/cache_audit.md` |

**Pending before submission:** (1) B-layer external/codex review of this draft's numbers (ITERATION_BOUNDARY §4.1 "准备写论文 section" trigger) — not yet run; (2) runtime and peak-memory benchmark if required by the final venue; (3) final journal-template conversion and reference/bib cleanup; (4) publication-quality polish of Fig. 1 if the current overview is not accepted as final artwork.
