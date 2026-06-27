# Failure Modes & Limitations（论文草稿，2026-06-21）

> 草稿状态：英文 paper-ready；每小节前的「中文：」是导航/依据，不进论文正文。
> 全部 claim 有 experiment_log / results CSV 支撑，已过 codex Round 3 外审的"窄而实"标准。

---

## 6. Failure modes and limitations

We deliberately characterize where scRareRefine does *not* help, where its guarantees weaken, and where our evaluation is incomplete. We view this transparency as essential: scRareRefine is a conservative, risk-controlled rescuer, and its value is concentrated in a specific regime rather than uniform across all settings.

### 6.1 The method is conditional by design, and is a no-op outside its regime

中文：必要性弃门 + 收益集中在"真稀缺、真漏判"区，pancreas_integrated/small_intestine 全弃权 = scANVI。

scRareRefine is explicitly conditional. Two abstention gates — a separability safety net and a necessity guard — cause the method to return the unmodified scANVI prediction whenever (i) the rare and majority prototypes are too entangled, or (ii) the validation baseline already recovers the rare population. As a consequence, on datasets where scANVI is already saturated (e.g., `pancreas_integrated` and `tabula_small_intestine`, where the gate abstains on most or all configurations), scRareRefine reduces exactly to scANVI: it neither helps nor harms. Its benefit is therefore concentrated in the genuinely label-scarce, genuinely-missed regime, and our claims are restricted accordingly. This is a feature for safety ("do no harm") but means the method should not be presented as a universal improvement.

### 6.2 The separability gate is a conservative prior whose risk axis is dataset-dependent

中文：G21。1.3 是保守先验；合成扫描 sep≈0.7 才破、真实 pancreas_baron sep≈1.22 就破 → sep-风险数据集相关，非普适边界。1.3 在真实 benchmark 上是 FFR≤α 的最小阈值（非为 F1 调）。

The separability gate uses a fixed, dataset-independent threshold (`CONFORMAL_LOW_SEP = 1.3`). We stress-tested this choice in two ways. (i) On the real benchmark, a sensitivity sweep over the threshold shows that 1.3 is the *smallest* gate that keeps the worst-case false-rescue rate (FFR) ≤ α across all 72 configurations: lowering it to ≤1.0 increases mean rare-cell F1 by ≈0.02 but admits two FFR violations on `pancreas_baron` (separability ≈1.22), while raising it to 1.6 only sacrifices F1 with no FFR benefit. The threshold is thus selected by the FFR constraint, not tuned for F1 (lowering it would *improve* F1). (ii) In a controlled semi-synthetic stress test (progressively entangling a rare population toward its nearest majority type on `tabula_lung_endo`), rescue remained FFR-safe down to separability ≈0.76 and only produced a marginal violation (18 false rescues out of 1716, FFR = 0.0105) at the lowest separability (≈0.69). These two experiments disagree on *where* rescue becomes unsafe (≈1.22 on real `pancreas_baron` vs ≈0.69 on the synthetic sweep), which is itself the key finding: **separability alone does not monotonically determine rescue risk**. We therefore do not claim to have localized a universal "collapse boundary." Instead, 1.3 is a conservative cross-dataset compromise — tight enough to exclude the real FFR violation at ≈1.22, loose enough to retain most benefit — and it occasionally abstains where rescue would have been safe (e.g., forgoing a recoverable +0.24 F1 at synthetic separability ≈1.15).

### 6.3 FFR control is empirical and can sit near the boundary under batch shift

中文：G61/G11。FFR≤α 是 val 校准下的经验控制非严格保证；pancreas_baron test FFR=0.0098 贴边；batch_heldout 破坏 exchangeability，Wilson 上界缓解但不消除。

The conformal threshold provides *empirical* FFR control under validation calibration, not a strict finite-sample guarantee on test. Under our batch-heldout protocol, validation and test cells come from different donors, so the exchangeability assumption underlying conformal coverage is only approximately satisfied. The clearest symptom is `pancreas_baron`, where the test FFR reaches 0.0098 — essentially at the α = 0.01 budget. We mitigate this by selecting the candidate rank via a Wilson 95% upper bound on the validation FFR rather than a point estimate, which prevents over-firing in several cases, but this reduces rather than removes the risk. FFR numbers should accordingly be read as empirical control under distribution shift, and we report them with their raw counts.

### 6.4 Seed sensitivity at extreme label scarcity

中文：G71。pancreas_baron ≤5 标注点 gain +0.18 但 ±0.23（seed 不稳）；scANVI 自身 0.38±0.21；≥10 标注即稳。

At the most extreme label budget (five labeled rare cells), the improvement on `pancreas_baron` is positive on average (+0.18 rare-cell F1) but unstable across random seeds (± 0.23 over three seeds); the scANVI baseline itself swings comparably (0.38 ± 0.21). The instability disappears once at least ten rare cells are labeled. We therefore do not claim stable improvement at the ≤5-label point on this dataset, and flag extreme scarcity as a regime where both the backbone and the rescue inherit the variance of the underlying split.

### 6.5 A recall ceiling for geometrically entangled rare types

中文：G10。stomach mast cell 与多数类在 rank≥3 纠缠，recall 卡 ~0.59；自适应 rank 无法在不破 FFR 前提下进一步召回。

For rare types that are transcriptionally interleaved with a majority population, prototype-distance rescue has an intrinsic recall ceiling. On `tabula_sapiens_stomach` (mast cells), a substantial fraction of true rare cells fall at candidate rank ≥ 3 relative to the prototypes, where admitting them would require widening the candidate pool past the point at which the FFR budget is exceeded. The adaptive-rank mechanism correctly refuses to do so, so recall plateaus (≈0.59). This is a fundamental limitation of latent-geometry rescue rather than a tuning artifact; expression-side signals (e.g., marker genes) may be needed to break this ceiling, which we leave to future work.

### 6.6 Evaluation caveats

中文：rts 塌缩去重、统计非独立、benchmark 广度、HiCat transductive、TOSICA 降配、foundation-model 未测。

- **Nominal label-fraction points are not all independent.** Because the number of labeled rare cells is `max(5, ⌊p · N_rare⌋)`, several nominal `rare_train_size` values collapse to the same labeled set on small-rare datasets (e.g., all three scarce fractions map to five labeled cells on `tabula_sapiens_stomach`; two of three on `pancreas_baron`). We therefore report the scarce-region comparison over the 15 *distinct* labeled configurations rather than the 18 nominal cells.
- **Statistical tests are paired but not fully independent.** Our significance results (one-sided Wilcoxon, bootstrap CIs) pair predictions by (dataset, fraction, seed); these cells are correlated across seeds of the same configuration and near-duplicated across collapsed fractions, so the p-values are optimistic. They should be read as directional evidence supporting a robust paired improvement and an effect size whose confidence interval excludes zero, not as strict independent-sample tests.
- **Benchmark scope.** We evaluate six datasets, predominantly human and exclusively single-cell. We do not yet validate cross-species transfer or disease-versus-healthy settings, both of which would further stress the exchangeability assumption.
- **Baseline caveats.** One of the eight baselines (HiCat) is transductive — its dimensionality reduction is fit on combined train and test cells — so it is reported as a transductive upper-bound reference rather than a fair inductive competitor. TOSICA is run at a reduced configuration (10 epochs, 100 pathway tokens) for tractability, which may underestimate its performance.
- **Backbone dependence.** scRareRefine operates on scANVI latent embeddings; whether the same conformal rescue adds value on top of single-cell foundation-model embeddings (e.g., scGPT, scFoundation) is untested.

---

## 待办 / 自检（不进论文）

- 6.2 的 G82 真实 benchmark 数字、6.5 的 stomach recall、6.4 的 seed std 均可在 results/ 找到对应 CSV，写作时逐一回链。
- codex Round 3 未完全闭合项：G80（第二 stress 数据集）、G81（sep sweep raw counts 正式入表）—— 若 reviewer 追问 6.2 的合成结论稳健性，需补 G80。
- 语言：最终随论文（英文二区）；此草稿可直接进 paper-write 的 Limitations section。
