"""从 scIB human_pancreas_norm_complexBatch.h5ad 提取「counts 层为真整数」的平台子集。

整合文件的 layers['counts'] 跨平台不一致：inDrop1-4 与 smartseq2 是整数 counts，
而 celseq/celseq2/smarter/fluidigmc1 为非整数（已被处理），不适合 scVI 的 counts 似然。
本脚本只保留整数-counts 平台，并把 counts 提到 X，产出干净的 raw-counts h5ad。

输出: data/raw/human_pancreas/pancreas_integrated_clean.h5ad
"""
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy import sparse

SRC = Path("data/raw/human_pancreas/human_pancreas_norm_complexBatch.h5ad")
OUT = Path("data/raw/human_pancreas/pancreas_integrated_clean.h5ad")
KEEP_TECH = ["inDrop1", "inDrop2", "inDrop3", "inDrop4", "smartseq2"]  # counts 为整数的平台

a = sc.read_h5ad(SRC)
a = a[a.obs["tech"].isin(KEEP_TECH)].copy()

# 用 counts 层作为 X（原始 counts），丢弃 lognorm 的旧 X
C = a.layers["counts"]
import anndata as ad
clean = ad.AnnData(
    X=C.copy(),
    obs=a.obs[["tech", "celltype"]].copy(),
    var=a.var.copy(),
)
# 校验整数性
d = clean.X.data if sparse.issparse(clean.X) else np.asarray(clean.X).ravel()
d = d[d != 0]
print(f"输出 {clean.shape}  counts nonzero: min={d.min():.2f} max={d.max():.1f} "
      f"frac_integer={np.mean(np.isclose(d % 1, 0, atol=1e-3)):.4f}")
print("tech:", clean.obs['tech'].value_counts().to_dict())
print("endothelial:", int((clean.obs['celltype'] == 'endothelial').sum()),
      "per tech:", clean.obs[clean.obs.celltype=='endothelial']['tech'].value_counts().to_dict())
OUT.parent.mkdir(parents=True, exist_ok=True)
clean.write_h5ad(OUT)
print(f"[saved] {OUT}")
