***1. 这个方向到底在研究什么？*** **                                                                                                                                                                                                                                                           **

---

**  **“单细胞稀有细胞识别”其实不是一个单一问题，至少分成三类：**                                                                                                                                                                                                                               **

---

**  ****A. Rare cell discovery：不知道稀有细胞是什么，要从数据里发现它**

---

**  **典型问题：**                                                                                                                                                                                                                                                                             **

---

**  **▎** ***这个数据里是否存在一个很小、但真实的细胞群？*

---

**  **代表方法：**                                                                                                                                                                                                                                                                             **

---

**  **- **RaceID**

**  **- **GiniClust / GiniClust2 / GiniClust3**

**  **- **CellSIUS**

**  **- **FiRE**

**  **- **GapClust**

**  **- **scSID** **                                                                                                                                                                                                                                                                               **

**  **- **scCAD** **                                                                                                                                                                                                                                                                               **

---

**  **这些方法通常基于：**                                                                                                                                                                                                                                                                     **

---

**  **- 异常检测**                                                                                                                                                                                                                                                                             **

**  **- 稀有 marker gene**                                                                                                                                                                                                                                                                   **

**  **- 聚类分解**                                                                                                                                                                                                                                                                             **

**  **- outlier score **                                                                                                                                                                                                                                                                       **

**  **- 局部密度或相似性结构**                                                                                                                                                                                                                                                                 **

---

**  **它们适合  **de novo discovery** **，也就是在没有明确训练标签时发现 rare cluster。 **

---

**  **但它们通常不解决：**                                                                                                                                                                                                                                                                     **

---

**  **- 已知 rare class 只有极少训练样本时，如何提升 supervised classifier；**                                                                                                                                                                                                                 **

**  **- 新 batch/query 数据上的 inductive annotation；**                                                                                                                                                                                                                                     **

**  **- 在 scANVI / atlas mapping 之后如何 post-hoc 修正 rare-cell 错误。 **                                                                                                                                                                                                                   **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****B. Cell type annotation / label transfer：已有参考图谱或训练集，要给 query cells 标注** **                                                                                                                                                                                                 **

---

**  **典型方法： **                   **

---

**  **- **SingleR** **                                                                                                                                                                                                                                                                             **

**  **- **Seurat label transfer / MapQuery**

**  **- **scmap** **                                                                                                                                                                                                                                                                               **

**  **- **CellTypist**

**  **- **scANVI**

**  **- **scArches**

**  **- **Azimuth** **                                                                                                                                                                                                                                                                             **

**  **- **scPred**

**  **- **popV** 这类 consensus annotation framework**                                                                                                                                                                                                                                             **

---

**  **这类方法的目标是：**                                                                                                                                                                                                                                                                     **

---

**  **▎** ***给新数据中的 cell 分配已知 cell type 标签。* **                                                                                                                                                                                                                                         **

---

**  **这和我们的任务更接近。**                                                                                                                                                                                                                                                                 **

---

**  **其中：**                                                                                                                                                                                                                                                                                 **

---

**  **-  **CellTypist** **：典型 supervised classifier，尤其在免疫细胞 annotation 中常用。**

**  **-  **Seurat label transfer** **：基于 reference-query anchoring / transfer。 **

**  **-  **scANVI / scArches** **：深度生成模型 + transfer learning，可处理 batch effect 和 reference mapping。 **

**  **-  **popV** **：2024 年左右出现的 consensus framework，把多个 annotation 方法组合起来提高鲁棒性。 **

---

**  **但这些方法的主要风险是：**                                                                                                                                                                                                                                                               **

---

**  **- 如果 rare class 样本少，classifier 容易被 majority class 主导； **                                                                                                                                                                                                                     **

**  **- batch-heldout 场景下，softmax confidence 可能不可靠； **                                                                                                                                                                                                                             **

**  **- reference 数据中 rare class 代表性不足时，label transfer 容易 miss rare cells； **                                                                                                                                                                                                     **

**  **- 很多方法报告的是整体 accuracy / macro-F1，不一定专门优化 extreme rare class。 **                                                                                                                                                                                                       **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****C. Imbalance-aware rare-cell annotation：明确处理类别不平衡** **                                                                                                                                                                                                                           **

---

**  **这一类更直接针对我们的痛点。**                                                                                                                                                                                                                                                         **

---

**  **代表方法包括：**                                                                                                                                                                                                                                                                         **

---

**  ****1. sc-SynO / synthetic oversampling** **                                                                                                                                                                                                                                                   **

---

**  **2021 年 BMC Bioinformatics 的方法，用 synthetic oversampling 处理 rare-cell annotation。它把 rare class 通过类似 SMOTE / LoRAS 的方式扩增，从而缓解训练集不平衡。 **                                                                                                                     **

---

**  **优点：**                                                                                                                                                                                                                                                                                 **

---

**  **- 直接处理 class imbalance；**                                                                                                                                                                                                                                                           **

**  **- 思路清晰。 **                                                               **

---

**  **限制： **                                                                     **

---

**  **- synthetic cells 是否保留真实生物结构不总是容易保证；**                                                                                                                                                                                                                                 **

**  **- 通常需要重新训练 classifier； **                                                                                                                                                                                                                                                     **

**  **- 对 batch-heldout / inductive query 的稳定性需要具体验证。 **                                                                                                                                                                                                                           **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****2. scBalance**

---

**  **2023 年 Communications Biology 的方法，提出 sparse neural network + adaptive weighted sampling + dropout，目标就是处理 single-cell annotation 中的 rare-cell / imbalance 问题。

---

**  **文献摘要明确说：**                                                                                                                                                                                                                                                                       **

---

**  **▎** ***现有方法经常没有充分考虑 scRNA-seq 数据集不平衡，会忽视小群体，导致生物分析错误。scBalance 通过 adaptive weight sampling 和 dropout 处理不平衡，并在 20 个数据集上评估。*

---

**  **这说明我们的问题是文献承认的真实问题，不是人为制造的。**                                                                                                                                                                                                                                 **

---

**  **但 scBalance 是一个  **重新训练的 annotation model** **，而我们现在的方法更像： **

---

**  **▎** ***在已有 scANVI embedding / prediction 之上做 post-hoc rare-cell rescue。* **                                                                                                                                                                                                             **

---

**  **这两者处在不同层级。**                                                                                                                                                                                                                                                                   **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****D. Foundation model / large single-cell model**

---

**  **近两年出现很多单细胞 foundation model：**                                     **

---

**  **- **scGPT** **                                                                                                                                                                                                                                                                               **

**  **- **Geneformer**

**  **- **scFoundation**

**  **- **UCE** **                                                                                                                                                                                                                                                                                 **

**  **- **CellPLM** **                                                                                                                                                                                                                                                                             **

**  **- **scCello** 等**                                                                                                                                                                                                                                                                           **

---

**  **它们的定位是：**                                                                                                                                                                                                                                                                         **

---

**  **▎** ***通过大规模预训练学习 gene/cell representation，再迁移到 annotation、integration、perturbation、gene network 等任务。*

---

**  **例如 scGPT 论文称其在超过 3300 万 cells 上预训练，支持 cell type annotation、multi-batch integration、multi-omic integration 等。 **                                                                                                                                                     **

---

**  **但是 2025 年的 zero-shot evaluation 论文明确指出：**                                                                                                                                                                                                                                     **

---

**  **▎** ***scGPT / Geneformer 等 foundation models 在 zero-shot 场景下还没有被充分严格验证，并且在 cell-type identification、batch robustness 等任务上存在限制。* **                                                                                                                               **

---

**  **所以目前不能简单说“大模型已经解决 rare cell annotation”。 **                                                                                                                                                                                                                             **

---

**  **更准确的判断是：**                                                                                                                                                                                                                                                                       **

---

**  **▎** ***Foundation model 可能提供更好的 embedding 或 prior，但 rare-cell、batch-heldout、极小样本 annotation 的可靠性仍然需要专门方法处理。* **                                                                                                                                                 **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****2. 目前领域的核心挑战是什么？***

---

**  ****挑战 1：class imbalance**

---

**  **稀有细胞天然数量少。普通 classifier 优化整体 loss 时，更关注 majority class。 **                                                                                                                                                                                                         **

---

**  **结果是：**                                                                                                                                                                                                                                                                               **

---

**  **- rare class recall 低；**                                                                                                                                                                                                                                                               **

**  **- rare cells 被预测成相近 majority class；**                                                                                                                                                                                                                                           **

**  **- softmax confidence 不一定能反映真实不确定性。 **                                                                                                                                                                                                                                       **

---

**  **2024 年 Nature Biotechnology 关于 dataset imbalance 的研究明确指出，single-cell integration 中的 sample/cell-type imbalance 会显著影响：**                                                                                                                                               **

---

**  **- unsupervised clustering **                                                                                                                                                                                                                                                             **

**  **- cell type classification **                                                 **

**  **- differential expression **                                                                                                                                                                                                                                                             **

**  **- marker gene annotation**                                                                                                                                                                                                                                                               **

**  **- query-to-reference mapping**                                                                                                                                                                                                                                                           **

**  **- trajectory inference**                                                                                                                                                                                                                                                                 **

---

**  **这说明 imbalance 会系统性影响后续分析，不只是一个分类指标问题。 **                                                                                                                                                                                                                       **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****挑战 2：batch effect / batch-heldout generalization**

---

**  **很多 annotation benchmark 是 random split 或 cell-stratified split。**                                                                                                                                                                                                                 **

---

**  **但真实场景常常是：**                                                                                                                                                                                                                                                                     **

---

**  **▎** ***reference 和 query 来自不同 batch、donor、平台或疾病状态。*

---

**  **这时：**                                                                                                                                                                                                                                                                                 **

---

**  **- scANVI / scArches 这类方法能缓解 batch effect； **                                                                                                                                                                                                                                     **

**  **- 但 rare class 在新 batch 中仍可能被 majority class 吞掉；**                 **

**  **- softmax probability 对 rare class 尤其不稳定。**                                                                                                                                                                                                                                       **

---

**  **我们现在的 immune_dc batch-heldout 结果正好说明这一点： **                                                                                                                                                                                                                               **

---

**  **- cDC1 rts=5 时 baseline F1 ≈ 0； **                                                                                                                                                                                                                                                     **

**  **- 说明 scANVI 的 rare-class softmax 在极端小样本 + heldout batch 下失效。**   **

---

**  **--- **                                                                                                                                                                                                                                                                                 **

**  ****挑战 3：rare cell marker stochasticity**

---

**  **稀有细胞数量少，所以 marker gene 估计不稳定： **                                                                                                                                                                                                                                       **

---

**  **- dropout 高；**                                                                                                                                                                                                                                                                         **

**  **- marker expression sparse；**                                                                                                                                                                                                                                                           **

**  **- 不同 batch marker 表达强度不同；**                                                                                                                                                                                                                                                     **

**  **- top marker 容易受噪声影响。 **                                                                                                                                                                                                                                                         **

---

**  **这也是为什么单纯 marker-based annotation 可能不稳。 **                                                                                                                                                                                                                                   **

---

**  **我们的方法里 marker verification 不是单独决定标签，而是放在 prototype candidate 之后作为验证层，这一点比较合理。**                                                                                                                                                                       **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****挑战 4：known rare type vs unknown rare type**

---

**  **这两个问题容易混淆。 **         **

---

**  **-  **unknown rare discovery** **：不知道 rare cell 是什么，需要发现新群体。 **

**  **-  **known rare rescue** **：知道 rare class 存在，但训练样本很少，需要在 query/test 中把它救回来。 **

---

**  **我们的方法主要解决第二类，不是第一类。**                                                                                                                                                                                                                                                 **

---

**  **也就是说：**                                                                                                                                                                                                                                                                             **

---

**  **▎** ***scRareRefine 不是一个 de novo rare-cell discovery 方法，而是一个 known rare-cell annotation refinement / rescue 方法。*

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****3. 当前方法发展阶段怎么划分？***

---

**  **可以把这个方向的发展分成几个阶段：

---

**  ****阶段 1：聚类 + marker 人工注释**

---

**  **典型流程：**                                                                                                                                                                                                                                                                             **

---

**  **1. clustering **                                                                                                                                                                                                                                                                         **

**  **2. 找 marker gene **                                                                                                                                                                                                                                                                   **

**  **3. 人工判断 cell type **                                                                                                                                                                                                                                                                 **

---

**  **优点：**                                                                                                                                                                                                                                                                                 **

---

**  **- 可解释；**                                                                                                                                                                                                                                                                             **

**  **- 生物学直观。 **               **

---

**  **问题：**                                                                                                                                                                                                                                                                                 **

---

**  **- 对 rare cells 敏感；**                                                                                                                                                                                                                                                                 **

**  **- cluster resolution 难选； **                                                                                                                                                                                                                                                         **

**  **- rare cells 可能被并入大 cluster； **                                                                                                                                                                                                                                                   **

**  **- 不适合大规模自动化 query annotation。 **                                                                                                                                                                                                                                               **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****阶段 2：rare-cell discovery 专门方法**

---

**  **代表：**                                                                                                                                                                                                                                                                               **

---

**  **- GiniClust **                                                                                                                                                                                                                                                                           **

**  **- RaceID**                                                                                                                                                                                                                                                                               **

**  **- CellSIUS**                                                                                                                                                                                                                                                                             **

**  **- FiRE**                                                                                                                                                                                                                                                                                 **

**  **- GapClust**                                                                                                                                                                                                                                                                             **

**  **- scSID **                                                                                                                                                                                                                                                                               **

**  **- scCAD **                                                                                                                                                                                                                                                                               **

---

**  **目标：**                                                                                                                                                                                                                                                                                 **

---

**  **▎** ***更敏感地发现小 cluster / rare cluster。* **                                                                                                                                                                                                                                             **

---

**  **这是 rare-cell 方法的第一条主线。 **                                                                                                                                                                                                                                                     **

---

**  **但它们更偏向  **unsupervised discovery** **，不是 supervised inductive annotation。 **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****阶段 3：reference-based / supervised annotation**

---

**  **代表： **                       **

---

**  **- SingleR **                                                                                                                                                                                                                                                                             **

**  **- Seurat label transfer **                                                                                                                                                                                                                                                               **

**  **- CellTypist**                                                                                                                                                                                                                                                                           **

**  **- scANVI**                                                                                                                                                                                                                                                                               **

**  **- scArches**                                                                                                                                                                                                                                                                             **

**  **- Azimuth **                                                                                                                                                                                                                                                                             **

---

**  **目标：**                                                                                                                                                                                                                                                                                 **

---

**  **▎** ***用已有参考图谱给新数据标注。*

---

**  **这是当前实际单细胞分析里非常主流的一条线。**                                                                                                                                                                                                                                             **

---

**  **问题是：**                                                                                                                                                                                                                                                                               **

---

**  **- reference 中 rare class 不足时，label transfer 容易失败； **                                                                                                                                                                                                                           **

**  **- class imbalance 导致 rare type recall 差； **                               **

**  **- batch-heldout 场景下 rare-cell annotation 不稳。**                                                                                                                                                                                                                                     **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****阶段 4：imbalance-aware annotation**

---

**  **代表： **                                                                     **

---

**  **- sc-SynO **                                                                                                                                                                                                                                                                             **

**  **- scBalance **                                                                                                                                                                                                                                                                           **

**  **- 其他 weighted sampling / contrastive learning / imbalance-aware classifier**                                                                                                                                                                                                           **

---

**  **目标：**                                                                                                                                                                                                                                                                                 **

---

**  **▎** ***在模型训练阶段主动处理 rare class 不平衡。*

---

**  **这是和我们最接近的方向之一。**                                                                                                                                                                                                                                                           **

---

**  **区别是：**                                                                                                                                                                                                                                                                               **

---

**  **- scBalance / sc-SynO 多数是在训练 classifier 时解决 imbalance；**                                                                                                                                                                                                                       **

**  **- 我们是在已经训练好的 scANVI 之后，用 prototype + gate + marker + fusion 做  **post-hoc rescue** **。**

---

**  **---**                                                                         **

**  ****阶段 5：foundation model + reliability layer**

---

**  **代表：**                                                                                                                                                                                                                                                                               **

---

**  **- scGPT **                                                                                                                                                                                                                                                                             **

**  **- Geneformer**                                                                                                                                                                                                                                                                           **

**  **- UCE **                                                                                                                                                                                                                                                                                 **

**  **- scFoundation**                                                                                                                                                                                                                                                                         **

**  **- CellPLM **                                                                                                                                                                                                                                                                             **

**  **- scCello **                                                                                                                                                                                                                                                                             **

---

**  **趋势是：**                                                                                                                                                                                                                                                                               **

---

**  **▎** ***用大模型提供更通用的 embedding，再在具体任务上 fine-tune / transfer。* **                                                                                                                                                                                                               **

---

**  **但目前文献也指出 zero-shot foundation model 仍有限制。**                                                                                                                                                                                                                                 **

---

**  **所以未来很可能不是“foundation model 直接解决一切”，而是： **                                                                                                                                                                                                                             **

---

**  **▎** ***foundation embedding + task-specific calibration / uncertainty / rare-cell rescue / biological validation。* **                                                                                                                                                                         **

---

**  **我们的 scRareRefine 更接近这个阶段里的  **reliability / refinement layer** **。 **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****4. 我们的方法处在整体发展中的哪个位置？***

---

**  **我建议这样定位： **                                                           **

---

**  **▎** ***scRareRefine 是一个面向 known rare cell type 的 * ***post-hoc inductive rescue framework*** *。它不是替代 scANVI、CellTypist 或 scArches 的基础 annotation model，而是在已有 semi-supervised / reference-based annotation 的基础上，专门修正 rare-class 在小样本和 batch-heldout 场景下的失败。*

---

**  **更具体地说，我们的方法位于：**                                                                                                                                                                                                                                                           **

---

**  **reference / semi-supervised annotation**                                                                                                                                                                                                                                                 **

**          **↓ **                                                                                                                                                                                                                                                                             **

**  **rare-cell failure detection **                                                                                                                                                                                                                                                           **

**          **↓ **                                                                                                                                                                                                                                                                             **

**  **prototype-based rescue**                                                                                                                                                                                                                                                                 **

**          **↓ **                                                                                                                                                                                                                                                                             **

**  **marker-gene verification**                                                                                                                                                                                                                                                               **

**          **↓ **                                                                                                                                                                                                                                                                             **

**  **validation-tuned fusion / abstention**                                                                                                                                                                                                                                                   **

---

**  **它不是：**                                                                                                                                                                                                                                                                               **

---

**  **- 不是 de novo rare-cell discovery 方法； **                                                                                                                                                                                                                                             **

**  **- 不是新的 foundation model； **                                                                                                                                                                                                                                                       **

**  **- 不是通用 annotation model； **                                                                                                                                                                                                                                                         **

**  **- 不是单纯 marker-based classifier；**                                                                                                                                                                                                                                                   **

**  **- 不是 oversampling 方法。**                                                                                                                                                                                                                                                             **

---

**  **它是：**                                                                                                                                                                                                                                                                                 **

---

**  **▎** ***一个轻量、可解释、inductive-safe 的 rare-cell rescue/refinement 层。*

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****5. 我们可以解决哪些问题？*** **                                                                                                                                                                                                                                                           **

---

**  ****可以解决 1：极小 rare labeled samples 下，scANVI baseline 失效的问题** **       **

---

**  **我们最强证据：**                                                                                                                                                                                                                                                                         **

---

**  **cDC1, rare_train_size=5, n=3 seeds**                                                                                                                                                                                                                                                     **

---

**  **Baseline F1 = 0.003 ± 0.005 **                                                                                                                                                                                                                                                           **

**  **kNN F1**      **= 0.000 ± 0.000 **                                                                                                                                                                                                                                                           **

**  **Ours F1 **    **= 0.986 ± 0.004 **                                                                                                                                                                                                                                                           **

---

**  **这个结果说明：**                                                                                                                                                                                                                                                                         **

---

**  **- 在只有 5 个 rare training cells 时，scANVI softmax 几乎完全不识别 cDC1；**                                                                                                                                                                                                             **

**  **- kNN 也完全失败； **                                                         **

**  **- 但 latent geometry + prototype + marker verification 能把 rare class 救回来。 **                                                                                                                                                                                                       **

---

**  **这是我们的核心卖点。**                                                                                                                                                                                                                                                                   **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****可以解决 2：kNN 在极端不平衡下被 majority vote 压倒的问题**

---

**  **kNN 在 rts=5 对 ASDC 和 cDC1 都是 0。 **                                                                                                                                                                                                                                               **

---

**  **这很重要，因为很多人会自然问：**                                                                                                                                                                                                                                                         **

---

**  **▎** ***既然 scANVI latent 好，为什么不直接 kNN？* **                                                                                                                                                                                                                                           **

---

**  **我们的数据回答是：**                                                                                                                                                                                                                                                                     **

---

**  **▎** ***因为 rare class 太少时，kNN 的 majority voting 会被大类压倒。* **                                                                                                                                                                                                                       **

---

**  **我们的方法不是简单看最近邻多数，而是：**                                                                                                                                                                                                                                                 **

---

**  **- 建 rare prototype； **                                                                                                                                                                                                                                                                 **

**  **- 看 rare-vs-majority 的几何 separability；**                                 **

**  **- 加 gate 控制 false rescue； **                                                                                                                                                                                                                                                         **

**  **- 加 marker verification 做生物学确认。 **                                                                                                                                                                                                                                               **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****可以解决 3：在 separable rare class 上做有效 rescue** **                                                                                                                                                                                                                                   **

---

**  **我们发现 separability ratio 很关键：**                                                                                                                                                                                                                                                 **

---

**  **sep > 1.3**  **→ rescue works **                                                                                                                                                                                                                                                             **

**  **sep < 1.1**  **→ abstain correctly**                                                                                                                                                                                                                                                         **

---

**  **支持数据：**                                                                                                                                                                                                                                                                             **

---

**  **- ASDC sep=1.526，F1 gain +0.277**                                                                                                                                                                                                                                                       **

**  **- cDC1 sep=1.408，F1 gain +0.777**                                                                                                                                                                                                                                                     **

**  **- tabula_liver NCM sep≈1.98，F1 gain +0.27 左右 **                                                                                                                                                                                                                                       **

**  **- epsilon/gamma/beta sep 低，方法基本 abstain 或不强行提升**                                                                                                                                                                                                                             **

---

**  **这使我们的方法不是“盲目 rescue”，而是： **                                                                                                                                                                                                                                               **

---

**  **▎** ***有几何条件才 rescue；没有条件则 abstain。* **                                                                                                                                                                                                                                           **

---

**  **这点在论文里很重要，因为 rare-cell 方法最容易被质疑 false positive。**                                                                                                                                                                                                                   **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****可以解决 4：给 rare-cell rescue 提供可解释证据** **                             **

---

**  **我们不仅输出标签，还能解释： **                                               **

---

**  **- prototype distance 支持； **                                                                                                                                                                                                                                                           **

**  **- gate rule 是否通过；**                                                                                                                                                                                                                                                                 **

**  **- marker gene 是否验证；**                                                                                                                                                                                                                                                               **

**  **- fusion 权重如何选择； **                                                                                                                                                                                                                                                               **

**  **- separability ratio 是否支持 rescue。**                                                                                                                                                                                                                                                 **

---

**  **这比黑箱 classifier 更容易给生物学审稿人解释。**                                                                                                                                                                                                                                         **

---

**  **尤其 marker gene：**                                                                                                                                                                                                                                                                     **

---

**  **- ASDC: AXL, TCF4, LILRA4 **                                                                                                                                                                                                                                                             **

**  **- cDC1: CLEC9A, BATF3, ID2**                                                                                                                                                                                                                                                           **

---

**  **这些是生物学上合理的 marker，能增强可信度。 **                                                                                                                                                                                                                                           **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****可以解决 5：inductive evaluation 下的合法 rare rescue** **                                                                                                                                                                                                                               **

---

**  **我们的设置强调：**                                                                                                                                                                                                                                                                     **

---

**  **- test labels 不用于调参；**                                                                                                                                                                                                                                                             **

**  **- marker signature 只来自训练集 labeled cells； **                                                                                                                                                                                                                                       **

**  **- prototype reference 只来自训练集；**                                                                                                                                                                                                                                                   **

**  **- fusion 参数只从 validation 选择。 **                                                                                                                                                                                                                                                   **

---

**  **这让结果比很多 transductive clustering / 全数据调参更严格。 **                                                                                                                                                                                                                           **

---

**  **可以在组会上这样说：**                                                                                                                                                                                                                                                                   **

---

**  **▎** ***我们不是在 test set 上看到了 rare cluster 再回头调规则，而是在训练/验证阶段固定规则，然后应用到 held-out query cells。*

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****6. 我们暂时不能解决哪些问题？***

---

**  **这部分必须诚实讲。 **           **

---

**  ****不能解决 1：完全未知 rare cell type discovery** **                                                                                                                                                                                                                                         **

---

**  **如果训练集中没有这个 rare class 的任何 labeled cells，我们现在的方法不能自动命名它。**                                                                                                                                                                                                   **

---

**  **它可能发现异常候选，但不能可靠 annotation。 **                                                                                                                                                                                                                                           **

---

**  **这类问题更接近：**                                                                                                                                                                                                                                                                       **

---

**  **- scCAD **                                                                                                                                                                                                                                                                               **

**  **- CellSIUS **                                                                 **

**  **- GiniClust **                                                                                                                                                                                                                                                                           **

**  **- FiRE**                                                                                                                                                                                                                                                                                 **

**  **- GapClust**                                                                                                                                                                                                                                                                             **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****不能解决 2：latent space 不可分的 rare class**

---

**  **如果 rare class 与 majority class 在 latent 中高度重叠，我们的方法会 abstain 或无提升。 **                                                                                                                                                                                             **

---

**  **例如：**                                                                                                                                                                                                                                                                                 **

---

**  **- pancreas gamma：baseline 已经很好，sep 低，没必要 rescue；**                                                                                                                                                                                                                           **

**  **- tabula_pancreas beta：sep 低，方法基本 abstain；**                                                                                                                                                                                                                                   **

**  **- epsilon：中间状态，效果不稳定。 **                                                                                                                                                                                                                                                     **

---

**  **这不是失败，而是方法设计的一部分：**                                                                                                                                                                                                                                                     **

---

**  **▎** ***不可分就不强行 rescue，避免 false positive。*

---

**  **但论文里不能说“适用于所有 rare cell”。**                                                                                                                                                                                                                                                 **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****不能解决 3：marker gene 本身不稳定或不可区分的 rare class**

---

**  **如果 rare class 没有稳定 marker，或者 marker 与相近 majority class 高度共享，marker verification 会弱。

---

**  **这种情况下我们只能依赖 prototype geometry；如果 geometry 也弱，就应该 abstain。 **                                                                                                                                                                                                       **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****不能解决 4：替代大规模 reference atlas / foundation model** **                                                                                                                                                                                                                           **

---

**  **我们的方法不是 atlas builder，也不是 foundation model。 **                                                                                                                                                                                                                             **

---

**  **更合理的未来定位是：**                                                                                                                                                                                                                                                                   **

---

**  **▎** ***可以接在 scANVI / scArches / CellTypist / foundation model embedding 后面，作为 rare-cell reliability layer。* **                                                                                                                                                                       **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****不能解决 5：目前还没有全面打败所有 rare-cell 方法** **                                                                                                                                                                                                                                   **

---

**  **我们目前主要比较了： **                                                       **

---

**  **- baseline scANVI **                                                                                                                                                                                                                                                                     **

**  **- kNN **                                                                                                                                                                                                                                                                                 **

**  **- Gate+Marker **                                                                                                                                                                                                                                                                         **

**  **- Fusion-gated**                                                                                                                                                                                                                                                                         **

---

**  **还没有系统比较：**                                                                                                                                                                                                                                                                       **

---

**  **- scBalance **                                                                                                                                                                                                                                                                           **

**  **- sc-SynO **                                                                                                                                                                                                                                                                           **

**  **- scCAD **                                                                                                                                                                                                                                                                               **

**  **- CellSIUS**                                                                                                                                                                                                                                                                             **

**  **- FiRE**                                                                                                                                                                                                                                                                                 **

**  **- GiniClust **                                                                                                                                                                                                                                                                           **

**  **- CellTypist**                                                                                                                                                                                                                                                                           **

**  **- Seurat label transfer **                                                                                                                                                                                                                                                               **

**  **- foundation model embedding**                                                                                                                                                                                                                                                           **

---

**  **所以不能说：**                                                                                                                                                                                                                                                                           **

---

**  **▎** ***state-of-the-art*

---

**  **只能说：**                                                                                                                                                                                                                                                                               **

---

**  **▎** ***在我们评估的数据集和 inductive rare-train-size 设置下，scRareRefine 显著提升了 separable rare classes 的 rare-class F1，并在 low-separability classes 上倾向于 abstain。*

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****7. 我们的方法的新意在哪里？***

---

**  **我建议把 novelty 讲成 4 点，不要夸大。

---

**  ****Novelty 1：从“整体 annotation accuracy”转向“rare-cell rescue reliability”** **                                                                                                                                                                                                             **

---

**  **很多 annotation 方法追求整体准确率。**                                                                                                                                                                                                                                                   **

---

**  **我们聚焦的是：**                                                                                                                                                                                                                                                                         **

---

**  **▎** ***当 rare class 只有 5–20 个 labeled cells 时，如何避免它被 majority class 吞掉。* **                                                                                                                                                                                                     **

---

**  **这是一个更细、更实际的问题。**                                                                                                                                                                                                                                                           **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****Novelty 2：prototype geometry + marker biology 的组合**

---

**  **我们不是只做 embedding distance，也不是只做 marker gene。**                   **

---

**  **而是：**                                                                                                                                                                                                                                                                                 **

---

**  **latent geometry 提供候选**                                                                                                                                                                                                                                                               **

**  **gate 控制 false positive**                                                                                                                                                                                                                                                               **

**  **marker gene 提供生物验证**                                                                                                                                                                                                                                                               **

**  **fusion 做 validation-tuned decision **                                                                                                                                                                                                                                                   **

---

**  **这个组合适合 rare-cell 场景。 **                                                                                                                                                                                                                                                         **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****Novelty 3：separability ratio 作为“是否应该 rescue”的诊断量** **                                                                                                                                                                                                                         **

---

**  **这是目前我们最有论文潜力的点之一。**                                                                                                                                                                                                                                                   **

---

**  **不是所有 rare class 都应该 rescue。 **                                                                                                                                                                                                                                                   **

---

**  **我们的 separability ratio 给出一个简单规则：**                                                                                                                                                                                                                                           **

---

**  **sep > 1.3**  **→ rescue likely useful **                                                                                                                                                                                                                                                     **

**  **sep < 1.1**  **→ abstain**                                                                                                                                                                                                                                                                   **

---

**  **如果后续更多数据集验证，这个可以成为方法核心贡献。**                                                                                                                                                                                                                                     **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  ****Novelty 4：严格 inductive setting 下的极小样本结果**

---

**  **cDC1 rts=5 的结果非常强： **                                                                                                                                                                                                                                                           **

---

**  **baseline≈0, kNN=0, ours≈0.986 **                                                                                                                                                                                                                                                         **

---

**  **而且是 batch-heldout，不是 random split。 **                                                                                                                                                                                                                                             **

---

**  **这比普通 random split 更能说明方法在真实 query 场景下有意义。 **                                                                                                                                                                                                                         **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****8. 组会上可以怎么讲我们的定位？*** **                                                                                                                                                                                                                                                     **

---

**  **可以用下面这段作为汇报话术： **                                               **

---

**  **▎** ***当前单细胞稀有细胞识别主要有三条路线：第一类是无监督 rare-cell discovery，例如 GiniClust、CellSIUS、FiRE、scCAD，目标是在没有标签时发现小群体；第二类是 reference-based 或 supervised annotation，例如 CellTypist、Seurat label transfer、scANVI/scArches，目标是把 query cells *

**  **▎** ***映射到已知 cell types；第三类是 imbalance-aware annotation，例如 sc-SynO 和 scBalance，直接处理 rare class 不平衡。 ** *** **                                                                                                                                                               **

**  **▎** **

**  **▎** ***我们的方法不属于 de novo discovery，也不是新的 foundation model，而是一个接在 scANVI 之后的 post-hoc rare-cell rescue layer。它针对的是 known rare cell type 在极小训练样本和 batch-heldout query 中被 baseline classifier 漏掉的问题。我们的核心思想是：先判断 rare class 在 latent** ***

**  **▎** ***space 中是否有足够 separability；如果有，再用 prototype distance 生成候选，用 gate 控制 false positives，用 marker gene 做生物学验证，最后通过 validation-tuned fusion 输出结果；如果 separability 不足，则 abstain，不强行 rescue。 ** ***

**  **▎** **

**  **▎** ***因此，我们解决的不是“所有 rare-cell identification”问题，而是一个更具体的问题：在已有少量 rare labels、已有 scANVI embedding、且 rare class 与 majority class 有可分离几何结构时，如何可靠地把被 baseline 漏掉的 rare cells 救回来。*

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****9. 最推荐的论文写作定位*** **                                                                                                                                                                                                                                                             **

---

**  **我建议论文题目/摘要中避免说：**                                               **

---

**  **- “general rare cell discovery” **                                                                                                                                                                                                                                                       **

**  **- “state-of-the-art cell type annotation” **                                                                                                                                                                                                                                             **

**  **- “solves rare cell identification” **                                                                                                                                                                                                                                                   **

**  **- “foundation model alternative”**                                                                                                                                                                                                                                                       **

---

**  **更稳妥的表述是：**                                                                                                                                                                                                                                                                       **

---

**  **▎** *****A post-hoc, inductive rare-cell rescue framework for semi-supervised single-cell annotation under extreme label scarcity.*** **                                                                                                                                                           **

---

**  **或者：**                                                                                                                                                                                                                                                                                 **

---

**  **▎** *****scRareRefine improves rare-cell recall in separable rare classes by combining prototype geometry, validation-controlled gating, and marker-gene verification, while abstaining when rare-majority separability is insufficient.*** **                                                     **

---

**  **--- **                                                                                                                                                                                                                                                                                   **

**  *****10. 一句话总结*** **                                                             **

---

**  **当前领域已经有很多方法做  **rare-cell discovery** **、****reference annotation** 和  **imbalance-aware classification** **，但在“已知 rare class + 极少 labeled cells + batch-heldout query + inductive-safe post-hoc rescue”这个具体场景上，仍然存在空缺。**

---

**  **我们的 scRareRefine 正好处在这个空缺上：**                                                                                                                                                                                                                                               **

---

**  **▎** ***它不是替代主流 annotation model，而是给 scANVI / reference-based annotation 增加一个可解释、可控 false-positive、能自动 abstain 的 rare-cell rescue 层。*

---

**  **---**                                                             **
