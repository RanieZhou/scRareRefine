# scRareRefine 对比方法汇总

> 整理时间：2026-05
> 说明：✅ 有第三方库可直接安装；⚠️ 仅源码需手动配置；🔵 R 包

---

## 第一层（必选 baseline）

### 1. scANVI（无 refinement）

- **论文**：Lopez et al., 2021, *Molecular Systems Biology*
- **用途**：scRareRefine 的直接前置模型，最核心的 ablation baseline
- **安装**：✅ PyPI
  ```bash
  pip install scvi-tools
  ```
- **使用**：
  ```python
  import scvi
  scvi.model.SCANVI.setup_anndata(adata, batch_key="batch", labels_key="label")
  model = scvi.model.SCANVI(adata, unlabeled_category="Unknown")
  model.train()
  predictions = model.predict(adata)
  ```
- **文档**：https://docs.scvi-tools.org

---

### 2. kNN（k=15，latent embedding）

- **用途**：最弱基准，用 scANVI latent 空间的 15-NN 投票
- **安装**：✅ PyPI
  ```bash
  pip install scikit-learn
  ```
- **使用**：
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  clf = KNeighborsClassifier(n_neighbors=15)
  clf.fit(train_latent, train_labels)
  pred = clf.predict(test_latent)
  ```
- **备注**：已在 `src/03b_knn_baseline.py` 中实现

---

### 3. CellTypist

- **论文**：Dominguez Conde et al., 2022, *Science*
- **用途**：完全监督方法上界参考（One-vs-Rest 逻辑回归）
- **安装**：✅ PyPI
  ```bash
  pip install celltypist
  ```
- **使用**：
  ```python
  import celltypist
  from celltypist import models
  # 使用自定义训练集
  new_model = celltypist.train(
      adata[train_mask],
      labels=adata[train_mask].obs["label"],
      n_jobs=4,
      max_iter=200,
  )
  predictions = celltypist.annotate(adata[test_mask], model=new_model)
  ```
- **文档**：https://celltypist.readthedocs.io
- **备注**：已在 `src/03c_celltypist_baseline.py` 中实现（sklearn 复现版）

---

### 4. GapClust

- **论文**：Fang et al., 2021, *Nature Communications*
- **用途**：无监督稀有细胞检测，计算 PCA 空间 kNN 距离的二阶导数（ΔΔD）
- **GitHub**：https://github.com/fabotao/GapClust
- **安装**：🔵 R 包（devtools）
  ```r
  install.packages("devtools")
  devtools::install_github("fabotao/GapClust")
  # 依赖：Seurat >= 3.1.0, rflann >= 1.8.4, irlba >= 2.3.3
  ```
- **使用**：
  ```r
  library(GapClust)
  result <- GapClust(seurat_obj, reduction = "pca", dims = 1:30)
  rare_cells <- result$rare_cells
  ```
- **备注**：无监督方法，不需要标注；输出为稀有细胞的 barcode 列表

---

### 5. scCAD（选）

- **论文**：Xu et al., 2024, *Nature Communications*
- **用途**：当前无监督稀有细胞检测 SOTA，在 25 个数据集上 F1=0.417
- **GitHub**：https://github.com/xuyp-csu/scCAD
- **安装**：⚠️ 源码（Python 3.7）
  ```bash
  git clone https://github.com/xuyp-csu/scCAD.git
  cd scCAD
  conda create -n scCAD_env python=3.7
  conda activate scCAD_env
  pip install -r requirements.txt
  ```
- **使用**：
  ```python
  import scCAD
  import numpy as np, h5py

  data_mat = h5py.File('./data.h5')
  data = np.array(data_mat['X'])

  result, score, sub_clusters, degs_list = scCAD.scCAD(
      data=data,
      dataName='sample_name',
      save_path='./'
  )
  ```
- **备注**：输入为原始表达矩阵（非标准化），运行约 40 秒/样本

---

## 第二层（选定方法）

### 6. SRNC

- **论文**：2026, *BMC Bioinformatics*
- **用途**：半监督新细胞类型识别，自监督特征学习 + 半监督分类，类不平衡场景
- **GitHub**：https://github.com/chisquare09/SRNC
- **安装**：⚠️ 仅源码（Python 3.8，底层使用 LightGBM）
  ```bash
  git clone https://github.com/chisquare09/SRNC
  cd SRNC
  pip install -r requirements.txt
  ```
- **使用**：
  ```python
  from model.srnc import SequentialRadiusNeighborsClassifier

  Y_predict = SequentialRadiusNeighborsClassifier(
      X_embedded,        # 降维后的特征矩阵
      y_all_labels,      # 全量标签（未标注用特殊值）
      X_train, X_test,
      Y_train,
      predictive_alg,    # 底层分类器（默认 LightGBM）
      control_neighbor,
      shrink_parameter,
      filter_proportion,
      threshold_rejection
  )
  ```
- **数据**：需从 Google Drive 单独下载示例数据集
- **备注**：在 6 个基准数据集上优于 scANVI、scNym 等半监督方法

---

### 7. HiCat（选）

- **论文**：Chang et al., 2025, *Briefings in Bioinformatics*
- **用途**：混合监督+无监督六步流程，可识别 ≥20 个细胞的稀有亚群
- **GitHub**：https://github.com/changbiHub/HiCat
- **安装**：⚠️ 源码（Python 3.9 + R 混合）

  ```bash
  git clone https://github.com/changbiHub/HiCat.git
  cd HiCat
  conda create --name hicat-env python=3.9
  conda activate hicat-env
  pip install -r requirements.txt
  ```

  R 依赖（需在 R 中安装）：
  ```r
  install.packages(c("Seurat", "harmony", "dplyr", "ggplot2", "cowplot"))
  ```
- **使用**：

  ```bash
  # 将 train.rds / test.rds 放入 data/ 目录
  python script/run.py
  # 结果在 outputs/ 目录
  # 可用 walkThrough.ipynb 查看分析过程
  ```
- **流程**：Harmony 批次校正 → UMAP → DBSCAN 聚类 → 多分辨率融合 → CatBoost 分类 → 冲突协调
- **注意**：PyPI 上的 `hicat` 包是另一个不相关工具，不要混淆

---

### 8. scBalance

- **论文**：Xu et al., 2023, *Briefings in Bioinformatics*
- **用途**：针对稀有细胞的训练集感知重采样 + 深度学习分类器，专门解决 scRNA-seq 类别不平衡问题
- **GitHub**：https://github.com/rpmccordlab/scBalance
- **安装**：✅ PyPI
  ```bash
  pip install scBalance
  ```
- **使用**：
  ```python
  import scBalance

  # 输入：表达矩阵（cells × genes）+ 标签向量
  # X_train: numpy array 或 sparse matrix（已归一化）
  # y_train: 字符串标签向量，含 rare class

  result = scBalance.scBalance(
      X_train=X_train,
      y_train=y_train,
      X_test=X_test,
      rare_class=rare_class,   # 指定稀有类名称
      random_state=42,
  )
  # result.pred: 测试集预测标签
  # result.prob: 各类别概率矩阵
  ```
- **依赖**：torch, scikit-learn, imbalanced-learn（pip 安装时自动拉取）
- **备注**：
  - 内部先用 SMOTE/ADASYN 对 rare class 过采样，再训练 MLP 分类器
  - 输入特征建议用 scANVI latent embedding 或 log1p 归一化表达矩阵
  - 与 AnnData 兼容，可直接接入当前管道

---

## 第二层其他候选方法（供参考，未选入）

| 方法              | 安装方式                              | 备注                                                  |
| ----------------- | ------------------------------------- | ----------------------------------------------------- |
| **smartid** | 🔵`BiocManager::install("smartid")` | 纯 R 包，Bioconductor 3.19+；TF-IDF 式 marker 评分    |
| **scGAD**   | ⚠️ 源码                             | 同用 prototype，方法论最近；GitHub: aimeeyaoyao/scGAD |

---

## 安装方式速查

| 方法       | 层级   | 语言     | pip | conda |     源码     |
| ---------- | ------ | -------- | :-: | :---: | :----------: |
| scANVI     | 第一层 | Python   | ✅ |  ✅  |      —      |
| kNN        | 第一层 | Python   | ✅ |  ✅  |      —      |
| CellTypist | 第一层 | Python   | ✅ |  ✅  |      —      |
| GapClust   | 第一层 | R        | — |  —  | ✅ devtools |
| scCAD      | 第一层 | Python   | — |  —  | ✅ git clone |
| SRNC       | 第二层 | Python   | — |  —  | ✅ git clone |
| HiCat      | 第二层 | Python+R | — |  —  | ✅ git clone |
| scBalance  | 第二层 | Python   | ✅ |  ✅  |      —      |

---

## 说明

- 所有方法均在实验中遵循 **inductive** 评估约束：prototype、marker、阈值只来自 train/val，不泄露 test 标签
- GapClust 和 scCAD 为无监督方法，不依赖标注；评估时需后处理将其输出映射到 rare class 标签
- 版本信息以 2026-05 为准，使用前建议检查各 GitHub 仓库是否有更新
