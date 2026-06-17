"""从 tabula_sapiens_lung 提取 Stromal compartment (10X) 子数据集，分离为独立 .h5ad。

稀有类: bronchial smooth muscle cell (~8%)。保留 raw counts 到 X，仅保留必要 obs 列。
输出: data/raw/tabula_lung_stroma/tabula_lung_stroma.h5ad
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path

SRC = "data/raw/tabula/tabula_sapiens_lung.h5ad"
OUT = Path("data/raw/tabula_lung_stroma/tabula_lung_stroma.h5ad")

a = ad.read_h5ad(SRC, backed="r")

# 先打印 Stromal 的 method 分布，再决定是否过滤
stroma_mask = a.obs["compartment"].astype(str) == "Stromal"
stroma_obs = a.obs[stroma_mask]
print(f"Stromal 全量: {int(stroma_mask.sum())} cells")
print("method 分布:", stroma_obs["method"].value_counts().to_dict())
print("\n稀有类 per donor × method:")
bsmc = stroma_obs[stroma_obs["cell_type"].astype(str) == "bronchial smooth muscle cell"]
print(pd.crosstab(bsmc["donor_id"], bsmc["method"]))

# 过滤 10X only（与 tabula_lung_endo 保持一致）
mask = (a.obs["method"].astype(str) == "10X") & stroma_mask
print(f"\n筛选 Stromal+10X: {int(mask.sum())} / {a.n_obs} cells")

sub = a[mask.to_numpy()].to_memory()

# 取原始 counts（raw.X，全基因）
if sub.raw is not None:
    X = sub.raw.X
    var = sub.raw.var.copy()
    print("使用 raw.X 作为 counts, genes:", var.shape[0])
else:
    X = sub.X
    var = sub.var.copy()
    print("raw 不存在，使用 X")

X = X.tocsr() if sp.issparse(X) else np.asarray(X)

# 校验是否为整数 counts
vals = X.data[:1000] if sp.issparse(X) else X.ravel()[:1000]
is_int = np.allclose(vals % 1, 0, atol=1e-4)
print(f"X 抽样 max={float(vals.max()):.2f} 纯整数={is_int}")

# 仅保留必要 obs 列
keep = ["cell_type", "donor_id", "compartment", "free_annotation", "method"]
obs = sub.obs[[c for c in keep if c in sub.obs.columns]].copy()
obs["cell_type"] = obs["cell_type"].astype(str)
obs["donor_id"] = obs["donor_id"].astype(str)

new = ad.AnnData(X=X, obs=obs, var=var)
new.obs_names = sub.obs_names
new.var_names_make_unique()

OUT.parent.mkdir(parents=True, exist_ok=True)
new.write_h5ad(OUT)
print(f"\n[done] 写出: {OUT}  shape={new.shape}")
print("\ncell_type 分布:")
print(new.obs["cell_type"].value_counts().to_string())
print("\ndonor 分布:")
print(new.obs["donor_id"].value_counts().to_string())
print("\nbronchial smooth muscle cell per donor:")
bsmc2 = new.obs[new.obs["cell_type"] == "bronchial smooth muscle cell"]
print(bsmc2["donor_id"].value_counts().to_string())
