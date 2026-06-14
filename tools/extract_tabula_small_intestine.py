"""从 tabula_sapiens_small_intestine 提取 10X 子数据集。

稀有类: intestinal tuft cell (~1.1%)。保留 raw counts 到 X，仅保留必要 obs 列。
输出: data/raw/tabula_sapiens_small_intestine/tabula_si_10x.h5ad
"""
import anndata as ad
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path

SRC = "data/raw/tabula_sapiens_small_intestine/tabula_sapiens_small_intestine.h5ad"
OUT = Path("data/raw/tabula_sapiens_small_intestine/tabula_si_10x.h5ad")

a = ad.read_h5ad(SRC, backed="r")
print(f"原始数据: {a.shape}")

mask = a.obs["method"].astype(str) == "10X"
print(f"筛选 10X: {int(mask.sum())} / {a.n_obs} cells")

sub = a[mask.to_numpy()].to_memory()

# 取 raw.X 作为 counts
if sub.raw is not None:
    X = sub.raw.X
    var = sub.raw.var.copy()
    print("使用 raw.X, genes:", var.shape[0])
else:
    X = sub.X
    var = sub.var.copy()
    print("raw 不存在，使用 X")

X = X.tocsr() if sp.issparse(X) else np.asarray(X)

vals = X.data[:1000] if sp.issparse(X) else X.ravel()[:1000]
is_int = np.allclose(vals % 1, 0, atol=1e-4)
print(f"X max={float(X.data.max()):.1f}  纯整数={is_int}")

keep = ["cell_type", "donor_id", "method", "tissue_in_publication", "anatomical_position"]
obs = sub.obs[[c for c in keep if c in sub.obs.columns]].copy()
obs["cell_type"] = obs["cell_type"].astype(str)
obs["donor_id"]  = obs["donor_id"].astype(str)

new = ad.AnnData(X=X, obs=obs, var=var)
new.obs_names = sub.obs_names
new.var_names_make_unique()

OUT.parent.mkdir(parents=True, exist_ok=True)
new.write_h5ad(OUT)
print(f"\n[done] {OUT}  shape={new.shape}")
print("\ncell_type top 10:")
print(new.obs["cell_type"].value_counts().head(10).to_string())
print("\ndonor 分布:")
print(new.obs["donor_id"].value_counts().to_string())
print("\ntuft cell per donor:")
tuft = new.obs[new.obs["cell_type"] == "intestinal tuft cell"]
print(tuft["donor_id"].value_counts().to_string())
