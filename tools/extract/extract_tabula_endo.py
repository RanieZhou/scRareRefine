"""从 tabula_sapiens_lung 提取 Endothelium compartment (10X) 子数据集，分离为独立 .h5ad。

稀有类: endothelial cell of lymphatic vessel (~3%)。保留 raw counts 到 X，仅保留必要 obs 列。
输出: data/raw/tabula_lung_endo/tabula_lung_endo.h5ad
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp
from pathlib import Path

SRC = "data/raw/tabula/tabula_sapiens_lung.h5ad"
OUT = Path("data/raw/tabula_lung_endo/tabula_lung_endo.h5ad")

a = ad.read_h5ad(SRC, backed="r")
mask = (a.obs["method"].astype(str) == "10X") & (a.obs["compartment"].astype(str) == "Endothelium")
print(f"筛选 Endothelium+10X: {int(mask.sum())} / {a.n_obs} cells")

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
print("cell_type 分布:", new.obs['cell_type'].value_counts().to_dict())
print("donor 分布:", new.obs['donor_id'].value_counts().to_dict())
import pandas as pd
ct = pd.crosstab(new.obs['donor_id'], new.obs['cell_type'])
print("\nlymphatic per donor:\n", ct.get('endothelial cell of lymphatic vessel'))
