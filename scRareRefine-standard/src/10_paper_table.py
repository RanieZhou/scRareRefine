"""Generate paper-ready summary tables from aggregate metrics.

Usage:
    python src/10_paper_table.py --out_dir figures/paper

Writes:
    figures/paper/
        table_main_results.csv         mean ± std of rare_f1 by (dataset, rare_class, method)
        table_separability.csv         separability metrics per dataset/rare_class/seed
        table_trainsize.csv            rare_f1 by (dataset, rare_class, rare_train_size, method)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["baseline", "knn_k15", "celltypist", "prototype", "prototype_gate",
                "prototype_gate_best", "prototype_gate_marker", "fusion", "fusion_gated"]
METRIC_COLS = ["rare_f1", "rare_precision", "rare_recall", "overall_accuracy",
               "major_to_rare_false_rescue_rate"]


def collect_metrics(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(outputs_dir.glob("*/*/metrics/final_metrics.csv")):
        try:
            df = pd.read_csv(p)
            df["dataset"] = p.parts[-4]
            rows.append(df)
        except Exception:
            pass

    # Also collect CellTypist results from celltypist/ subdirectory
    # (avoids needing to re-run Stage 7 for runs completed before CellTypist support)
    for p in sorted(outputs_dir.glob("*/*/celltypist/test_metrics.csv")):
        try:
            df = pd.read_csv(p)
            if "dataset" not in df.columns:
                df["dataset"] = p.parts[-4]
            rows.append(df)
        except Exception:
            pass

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def collect_separability(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(outputs_dir.glob("*/*/prototype/separability.csv")):
        try:
            df = pd.read_csv(p)
            df["dataset"] = p.parts[-4]
            df["run_id"] = p.parts[-3]
            seed_str = p.parts[-3]  # e.g. batch_heldout_seed42_asdc_rare20
            import re
            m = re.search(r"seed(\d+)", seed_str)
            df["seed"] = int(m.group(1)) if m else -1
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _fmt(mean, std, fmt=".3f"):
    return f"{mean:{fmt}} ± {std:{fmt}}"


def table_main_results(df: pd.DataFrame, out_path: Path) -> None:
    """Mean ± std of key metrics at rare_train_size=20 across seeds."""
    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    if sub.empty:
        return
    methods = [m for m in METHOD_ORDER if m in sub["method"].values]
    groups = sub.groupby(["dataset", "rare_class", "method"])

    rows = []
    for (ds, rc, m) in [(d, r, me) for d in sub["dataset"].unique()
                        for r in sub.loc[sub["dataset"]==d, "rare_class"].unique()
                        for me in methods]:
        g = sub[(sub["dataset"]==ds) & (sub["rare_class"]==rc) & (sub["method"]==m)]
        if g.empty:
            continue
        row = {"dataset": ds, "rare_class": rc, "method": m}
        for col in METRIC_COLS:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce").dropna()
                row[f"{col}_mean"] = round(vals.mean(), 4) if len(vals) else np.nan
                row[f"{col}_std"] = round(vals.std(), 4) if len(vals) > 1 else 0.0
                row[f"{col}_n"] = len(vals)
        rows.append(row)

    result = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Also print a compact view
    print("\n=== Main Results (rare_train_size=20) ===")
    pivot_f1 = result.pivot_table(values="rare_f1_mean", index=["dataset","rare_class"], columns="method")
    pivot_f1 = pivot_f1[[m for m in METHOD_ORDER if m in pivot_f1.columns]]
    print(pivot_f1.round(3).to_string())


def table_trainsize_ablation(df: pd.DataFrame, out_path: Path) -> None:
    """F1 by (dataset, rare_class, rare_train_size, method)."""
    focus_methods = ["baseline", "prototype_gate_marker", "fusion_gated"]
    sub = df[df["method"].isin(focus_methods)].copy()

    def to_num(s):
        return 9999 if str(s).lower() == "all" else float(s)

    sub["rts_num"] = sub["rare_train_size"].apply(to_num)
    groups = sub.groupby(["dataset", "rare_class", "method", "rare_train_size", "rts_num"])["rare_f1"]

    result = groups.agg(["mean", "std", "count"]).reset_index()
    result = result.sort_values(["dataset", "rare_class", "method", "rts_num"])
    result.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")


def table_separability_results(sep_df: pd.DataFrame, metrics_df: pd.DataFrame, out_path: Path) -> None:
    """Separability with corresponding rescue performance. Merges on (dataset, seed, rare_class)."""
    if sep_df.empty:
        return

    # sep_df has rare_class column from separability.csv
    sep_sub = sep_df[["dataset", "seed", "rare_class", "separability_ratio", "nearest_majority_class",
                       "intra_rare_radius", "dist_to_nearest_majority", "n_rare_train"]].copy()

    if not metrics_df.empty:
        m20 = metrics_df[metrics_df["rare_train_size"].astype(str) == "20"].copy()
        for method_col, method in [("baseline_f1", "baseline"), ("gate_marker_f1", "prototype_gate_marker")]:
            sub = m20[m20["method"] == method][["dataset", "seed", "rare_class", "rare_f1"]]
            sub = sub.rename(columns={"rare_f1": method_col})
            sep_sub = sep_sub.merge(sub, on=["dataset", "seed", "rare_class"], how="left")
        sep_sub["f1_gain"] = sep_sub["gate_marker_f1"] - sep_sub["baseline_f1"]

    sep_sub.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    print("\n=== Separability Summary ===")
    agg = sep_sub.groupby(["dataset", "rare_class"])[["separability_ratio", "baseline_f1", "gate_marker_f1", "f1_gain"]].mean().round(3)
    print(agg.to_string())


def latex_main_table(df: pd.DataFrame, out_path: Path) -> None:
    """LaTeX-formatted main results table for rts=20, key methods only."""
    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    focus_methods = ["baseline", "knn_k15", "prototype_gate_marker", "fusion_gated"]
    method_labels_latex = {
        "baseline": "scANVI Baseline",
        "knn_k15": r"kNN ($k$=15)",
        "prototype_gate_marker": r"\textbf{Gate+Marker}",
        "fusion_gated": r"\textbf{Fusion-gated}",
    }
    dataset_labels_latex = {
        "immune_dc": "Immune DC",
        "pancreas": "Pancreas",
        "tabula_liver": "Tabula Liver",
        "tabula_pancreas": "Tabula Pancreas",
        "tabula_spleen": "Tabula Spleen",
        "tabula_kidney": "Tabula Kidney",
        "pbmc_pdc": "PBMC",
    }

    rare_class_labels_latex = {
        "non-classical monocyte": "NCM",
        "type B pancreatic cell": r"$\beta$-cell",
        "innate lymphoid cell": "ILC",
        "endothelial cell": "EC",
        "pDC": "pDC",
    }

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Rare-class F1 across datasets and methods (rare\_train\_size=20, mean $\pm$ SD over 3 seeds)}")
    ncols = 2 + len(focus_methods)
    lines.append(r"\begin{tabular}{ll" + "r" * len(focus_methods) + "}")
    lines.append(r"\hline")
    header = "Dataset & Rare class & " + " & ".join(method_labels_latex[m] for m in focus_methods) + r" \\"
    lines.append(header)
    lines.append(r"\hline")

    for ds in ["immune_dc", "pancreas", "tabula_liver", "tabula_pancreas",
                "tabula_spleen", "tabula_kidney", "pbmc_pdc"]:
        ds_sub = sub[sub["dataset"] == ds]
        rare_classes = sorted(ds_sub["rare_class"].unique())
        for i, rc in enumerate(rare_classes):
            rc_sub = ds_sub[ds_sub["rare_class"] == rc]
            ds_label = dataset_labels_latex.get(ds, ds) if i == 0 else ""
            rc_label = rare_class_labels_latex.get(rc, rc.replace("_", r"\_"))
            row_parts = [ds_label, rc_label]
            means_by_method = {}
            for m in focus_methods:
                m_sub = rc_sub[rc_sub["method"] == m]["rare_f1"].dropna()
                if len(m_sub) > 0:
                    means_by_method[m] = m_sub.mean()
            best_mean = max(means_by_method.values()) if means_by_method else None
            for m in focus_methods:
                m_sub = rc_sub[rc_sub["method"] == m]["rare_f1"].dropna()
                if len(m_sub) > 0:
                    mean = m_sub.mean()
                    std = m_sub.std() if len(m_sub) > 1 else 0.0
                    if std > 0.001:
                        val = f"${mean:.3f} \\pm {std:.3f}$"
                    else:
                        val = f"${mean:.3f}$"
                    if best_mean is not None and abs(mean - best_mean) < 1e-4:
                        val = r"\textbf{" + val + "}"
                    row_parts.append(val)
                else:
                    row_parts.append("--")
            lines.append(" & ".join(row_parts) + r" \\")
        if ds != "pbmc_pdc":
            lines.append(r"\hline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\end{table}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out_dir", default="figures/paper")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting metrics ...")
    df = collect_metrics(outputs_dir)
    sep_df = collect_separability(outputs_dir)
    print(f"  {len(df)} metric rows, {len(sep_df)} separability rows")

    table_main_results(df, out_dir / "table_main_results.csv")
    table_trainsize_ablation(df, out_dir / "table_trainsize.csv")
    table_separability_results(sep_df, df, out_dir / "table_separability.csv")
    latex_main_table(df, out_dir / "table_main_results.tex")
    print(f"\nDone. Tables saved to {out_dir}")


if __name__ == "__main__":
    main()
