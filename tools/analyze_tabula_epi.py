"""分析 tabula_sapiens_lung.h5ad Epithelium compartment 的 donor 分布。"""
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("data/raw/tabula/tabula_sapiens_lung.h5ad")
epi = adata[adata.obs["compartment"] == "Epithelium"].copy()

print(f"Epithelium: {epi.shape[0]} cells × {epi.shape[1]} genes")
print(f"\ncell_type counts:")
print(epi.obs["cell_type"].value_counts().to_string())

if "donor_id" in epi.obs.columns:
    print(f"\ndonor_id unique: {epi.obs['donor_id'].nunique()}")
    print(epi.obs["donor_id"].value_counts().to_string())

if "method" in epi.obs.columns:
    print(f"\nmethod distribution:")
    print(epi.obs["method"].value_counts().to_string())

# ionocyte per donor × method
if "donor_id" in epi.obs.columns and "method" in epi.obs.columns:
    iono = epi[epi.obs["cell_type"] == "pulmonary ionocyte"]
    print(f"\nionocyte donor × method:")
    print(pd.crosstab(iono.obs["donor_id"], iono.obs["method"]))

# 10X only epi: donor distribution
if "method" in epi.obs.columns:
    epi_10x = epi[epi.obs["method"] == "10X"]
    print(f"\n10X Epithelium: {epi_10x.shape[0]} cells")
    if "donor_id" in epi_10x.obs.columns:
        print(f"donor_id unique (10X): {epi_10x.obs['donor_id'].nunique()}")
        print(epi_10x.obs["donor_id"].value_counts().to_string())
        # ionocyte per donor in 10X
        iono_10x = epi_10x[epi_10x.obs["cell_type"] == "pulmonary ionocyte"]
        print(f"\n10X ionocyte per donor:")
        print(iono_10x.obs["donor_id"].value_counts().to_string())
