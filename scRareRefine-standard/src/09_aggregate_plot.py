"""Aggregate paper-quality plots across all completed experiment runs.

Usage:
    python src/09_aggregate_plot.py --out_dir figures/paper

Writes:
    figures/paper/
        fig_dataset_comparison.png    multi-dataset F1 comparison heatmap
        fig_trainsize_ablation.png    rare_train_size scaling curves
        fig_separability.png          separability ratio vs rescue F1 gain
        fig_all_methods_summary.png   boxplot across seeds per method × dataset
        aggregate_metrics.csv         raw aggregated metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


METHOD_ORDER = ["baseline", "knn_k15", "celltypist", "prototype", "prototype_gate",
                "prototype_gate_best", "prototype_gate_marker", "fusion", "fusion_gated"]
METHOD_LABELS = {
    "baseline":              "scANVI\nBaseline",
    "knn_k15":               "kNN\n(k=15)",
    "celltypist":            "Logistic\nRegr.",
    "prototype":             "Prototype\nRescue",
    "prototype_gate":        "Proto\nGate",
    "prototype_gate_best":   "Gate\n(best)",
    "prototype_gate_marker": "Gate+\nMarker",
    "fusion":                "Fusion\n(global)",
    "fusion_gated":          "Fusion\n(gated)",
}
METHOD_COLORS = {
    "baseline":              "#8da0cb",
    "knn_k15":               "#b3b3b3",
    "celltypist":            "#d4a5a5",
    "prototype":             "#66c2a5",
    "prototype_gate":        "#fc8d62",
    "prototype_gate_best":   "#ff6b35",
    "prototype_gate_marker": "#e78ac3",
    "fusion":                "#a6d854",
    "fusion_gated":          "#ffd92f",
}

DATASET_LABELS = {
    "immune_dc":        "Immune DC\n(ASDC/cDC1)",
    "pancreas":         "Pancreas\n(ε/γ)",
    "tabula_liver":     "Tabula Liver\n(NCM)",
    "tabula_pancreas":  "Tabula Pancreas\n(β-cell)",
    "tabula_spleen":    "Tabula Spleen\n(ILC)",
    "tabula_kidney":    "Tabula Kidney\n(endothelial)",
}


def collect_metrics(outputs_dir: Path) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(outputs_dir.glob("*/*/metrics/final_metrics.csv")):
        try:
            df = pd.read_csv(metrics_path)
            # Infer dataset from directory structure
            dataset = metrics_path.parts[-4]
            df["dataset"] = dataset
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    all_metrics = pd.concat(rows, ignore_index=True)

    # Attach separability ratio if available
    sep_rows = []
    for sep_path in sorted(outputs_dir.glob("*/*/prototype/separability.csv")):
        try:
            s = pd.read_csv(sep_path)
            run_dir = sep_path.parents[1]
            dataset = sep_path.parts[-4]
            seed_col = run_dir.name  # e.g. batch_heldout_seed42_asdc_rare20
            s["dataset"] = dataset
            s["run_id"] = run_dir.name
            sep_rows.append(s)
        except Exception:
            pass
    if sep_rows:
        sep_df = pd.concat(sep_rows, ignore_index=True)
        # Extract seed from run_id; keep rare_class for correct join
        sep_df["seed"] = sep_df["run_id"].str.extract(r"seed(\d+)").astype(float).astype("Int64")
        sep_key = ["dataset", "seed", "rare_class"]
        merge_cols = sep_key + ["separability_ratio", "nearest_majority_class"]
        all_metrics = all_metrics.merge(
            sep_df[merge_cols].drop_duplicates(subset=sep_key),
            on=sep_key, how="left",
        )
    return all_metrics


def fig_dataset_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Mean rare_f1 per (dataset × rare_class × method), across seeds at rare_train_size=20."""
    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    if sub.empty:
        print("  No rare_train_size=20 data, skipping dataset_comparison.")
        return

    methods = [m for m in METHOD_ORDER if m in sub["method"].values]
    dataset_rare = sorted(sub[["dataset", "rare_class"]].drop_duplicates().apply(tuple, axis=1).tolist())

    data = np.full((len(dataset_rare), len(methods)), np.nan)
    for i, (ds, rc) in enumerate(dataset_rare):
        for j, m in enumerate(methods):
            vals = sub[(sub["dataset"] == ds) & (sub["rare_class"] == rc) & (sub["method"] == m)]["rare_f1"].dropna()
            if len(vals):
                data[i, j] = vals.mean()

    yticklabels = [f"{DATASET_LABELS.get(ds, ds).replace(chr(10), ' ')}\n({rc})" for ds, rc in dataset_rare]
    xticklabels = [METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods]

    fig, ax = plt.subplots(figsize=(len(methods) * 1.5 + 1.5, len(dataset_rare) * 0.75 + 2))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(xticklabels, fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(dataset_rare)))
    ax.set_yticklabels(yticklabels, fontsize=8)

    for i in range(len(dataset_rare)):
        for j in range(len(methods)):
            val = data[i, j]
            if np.isfinite(val):
                tc = "white" if val < 0.35 or val > 0.88 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=tc, fontweight="bold")

    fig.colorbar(im, ax=ax, shrink=0.7, label="Mean Rare-class F1")
    ax.set_title("Rare-class F1 across datasets and methods (rare_train_size=20, mean over seeds)",
                 fontsize=10, pad=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_trainsize_ablation(df: pd.DataFrame, out_path: Path) -> None:
    """Rare_train_size scaling curves for selected dataset+method combos."""
    size_map = {"all": 9999}

    def to_num(s):
        s_str = str(s).strip().lower()
        if s_str == "nan":
            return np.nan
        if s_str in size_map:
            return size_map[s_str]
        try:
            return float(s_str)
        except ValueError:
            return np.nan

    sub = df.copy()
    sub["rare_train_size_num"] = sub["rare_train_size"].apply(to_num)
    sub = sub[sub["rare_train_size_num"].notna()].copy()

    focus_methods = ["baseline", "prototype_gate_marker", "fusion_gated"]
    sub = sub[sub["method"].isin(focus_methods)]

    dataset_rare = sorted(sub[["dataset", "rare_class"]].drop_duplicates().apply(tuple, axis=1).tolist())
    if not dataset_rare:
        return

    ncols = min(3, len(dataset_rare))
    nrows = (len(dataset_rare) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for idx, (ds, rc) in enumerate(dataset_rare):
        ax = axes[idx]
        panel_df = sub[(sub["dataset"] == ds) & (sub["rare_class"] == rc)]
        for m in focus_methods:
            m_df = panel_df[panel_df["method"] == m]
            if m_df.empty:
                continue
            agg = m_df.groupby("rare_train_size_num")["rare_f1"].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values("rare_train_size_num")
            xs = agg["rare_train_size_num"].values
            xs[xs == 9999] = xs[xs < 9999].max() * 1.5 if (xs < 9999).any() else 999
            ax.plot(xs, agg["mean"].values, "o-", color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m].replace("\n", " "), linewidth=1.8, markersize=5)
            if len(agg) > 1:
                ax.fill_between(xs,
                                agg["mean"] - agg["std"].fillna(0),
                                agg["mean"] + agg["std"].fillna(0),
                                alpha=0.18, color=METHOD_COLORS[m])
        ax.set_title(f"{DATASET_LABELS.get(ds, ds).replace(chr(10), ' ')}  ({rc})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Rare train size", fontsize=8)
        ax.set_ylabel("Rare-class F1", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], label=METHOD_LABELS[m].replace("\n", " "))
               for m in focus_methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(focus_methods),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))

    for ax in axes[len(dataset_rare):]:
        ax.set_visible(False)

    fig.suptitle("Rare-class F1 vs training size", fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_separability(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter: separability_ratio vs rescue F1 gain with confidence zone shading."""
    if "separability_ratio" not in df.columns:
        print("  No separability data, skipping.")
        return

    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    baseline = sub[sub["method"] == "baseline"][["dataset", "rare_class", "seed", "rare_f1", "separability_ratio"]].rename(
        columns={"rare_f1": "baseline_f1"})
    best = sub[sub["method"] == "prototype_gate_marker"][["dataset", "rare_class", "seed", "rare_f1"]].rename(
        columns={"rare_f1": "rescue_f1"})
    merged = baseline.merge(best, on=["dataset", "rare_class", "seed"], how="inner")
    merged["f1_gain"] = merged["rescue_f1"] - merged["baseline_f1"]

    if merged.empty:
        return

    # Marker style by rare class
    rc_markers = {rc: m for rc, m in zip(sorted(merged["rare_class"].unique()),
                                         ["o", "s", "^", "D", "v", "p"])}

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Background confidence zones
    xlim = (0.5, 2.5)
    ax.axvspan(xlim[0], 1.0, alpha=0.08, color="red", label="Low sep (< 1.0)")
    ax.axvspan(1.0, 1.5, alpha=0.08, color="gold", label="Medium sep (1.0–1.5)")
    ax.axvspan(1.5, xlim[1], alpha=0.08, color="green", label="High sep (> 1.5)")

    colors = plt.cm.tab10(np.linspace(0, 0.9, merged["dataset"].nunique()))
    ds_color = {ds: c for ds, c in zip(sorted(merged["dataset"].unique()), colors)}
    ds_rc_mean = merged.groupby(["dataset", "rare_class"])[["separability_ratio", "f1_gain"]].mean()

    plotted = set()
    for _, row in merged.iterrows():
        ds, rc = row["dataset"], row["rare_class"]
        label = f"{DATASET_LABELS.get(ds, ds).replace(chr(10), ' ')} ({rc})"
        ax.scatter(row["separability_ratio"], row["f1_gain"],
                   color=ds_color[ds], marker=rc_markers.get(rc, "o"),
                   s=70, alpha=0.75, edgecolors="white", linewidth=0.5,
                   label=label if (ds, rc) not in plotted else "_nolegend_")
        plotted.add((ds, rc))

    # Annotate dataset+class means
    for (ds, rc), agg_row in ds_rc_mean.iterrows():
        label_short = f"{rc}"
        ax.annotate(label_short, (agg_row["separability_ratio"], agg_row["f1_gain"]),
                    fontsize=7.5, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points", color=ds_color[ds])

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(1.0, color="#c8441b", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.axvline(1.5, color="#388E3C", linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(1.01, ax.get_ylim()[0] if ax.get_ylim()[0] > -0.2 else -0.12, "sep=1.0",
            color="#c8441b", fontsize=7.5, va="bottom")
    ax.text(1.51, ax.get_ylim()[0] if ax.get_ylim()[0] > -0.2 else -0.12, "sep=1.5",
            color="#388E3C", fontsize=7.5, va="bottom")

    ax.set_xlim(*xlim)
    ax.set_xlabel("Separability ratio  (d_nearest_majority / intra_rare_radius)", fontsize=9)
    ax.set_ylabel("Rare-class F1 gain  (Gate+Marker − Baseline)", fontsize=9)
    ax.set_title("Separability ratio predicts rescue success", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = ax.get_legend_handles_labels()
    # Filter zone labels to end of list
    zone_labels = ["Low sep (< 1.0)", "Medium sep (1.0–1.5)", "High sep (> 1.5)"]
    data_handles = [(h, l) for h, l in zip(handles, labels) if l not in zone_labels]
    zone_handles = [(h, l) for h, l in zip(handles, labels) if l in zone_labels]
    all_h = [h for h, _ in data_handles + zone_handles]
    all_l = [l for _, l in data_handles + zone_handles]
    ax.legend(all_h, all_l, fontsize=7.5, frameon=True, framealpha=0.85, loc="upper left",
              ncol=1, borderpad=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_all_methods_summary(df: pd.DataFrame, out_path: Path) -> None:
    """Multi-panel boxplot across seeds, one panel per dataset+rare_class."""
    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    dataset_rare = sorted(sub[["dataset", "rare_class"]].drop_duplicates().apply(tuple, axis=1).tolist())
    if not dataset_rare:
        return

    methods = [m for m in METHOD_ORDER if m in sub["method"].values]
    ncols = min(3, len(dataset_rare))
    nrows = (len(dataset_rare) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5))
    axes = np.array(axes).flatten()

    for idx, (ds, rc) in enumerate(dataset_rare):
        ax = axes[idx]
        panel_df = sub[(sub["dataset"] == ds) & (sub["rare_class"] == rc)]
        xs = range(len(methods))
        for j, m in enumerate(methods):
            vals = panel_df[panel_df["method"] == m]["rare_f1"].dropna().values
            if len(vals) == 0:
                continue
            bp = ax.boxplot(vals, positions=[j], widths=0.5,
                            patch_artist=True, showfliers=True,
                            boxprops=dict(facecolor=METHOD_COLORS[m], alpha=0.7),
                            medianprops=dict(color="black", linewidth=1.5),
                            whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                            flierprops=dict(marker="o", markersize=4, alpha=0.6))
        ax.set_xticks(list(xs))
        ax.set_xticklabels([METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods],
                           fontsize=7, rotation=30, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Rare-class F1", fontsize=8)
        ax.set_title(f"{DATASET_LABELS.get(ds, ds).replace(chr(10), ' ')}  ({rc})",
                     fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[len(dataset_rare):]:
        ax.set_visible(False)

    fig.suptitle("Rare-class F1 across methods and seeds (rare_train_size=20)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_main_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: baseline vs kNN vs gate+marker vs fusion_gated at rare_train_size=20."""
    sub = df[df["rare_train_size"].astype(str) == "20"].copy()
    compare_methods = ["baseline", "knn_k15", "prototype_gate_marker", "fusion_gated"]
    compare_labels = {"baseline": "scANVI\nBaseline", "knn_k15": "kNN\n(k=15)",
                      "prototype_gate_marker": "Gate+\nMarker", "fusion_gated": "Fusion\n(gated)"}
    compare_colors = {"baseline": "#8da0cb", "knn_k15": "#aec7e8",
                      "prototype_gate_marker": "#e78ac3", "fusion_gated": "#ffd92f"}
    methods_present = [m for m in compare_methods if m in sub["method"].values]
    if len(methods_present) < 2:
        return

    dataset_rare = sorted(sub[["dataset", "rare_class"]].drop_duplicates().apply(tuple, axis=1))
    n_groups = len(dataset_rare)
    n_methods = len(methods_present)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(n_groups * (n_methods * 0.6 + 0.8) + 1.5, 5.5))

    for j, m in enumerate(methods_present):
        means, stds, xs = [], [], []
        for i, (ds, rc) in enumerate(dataset_rare):
            vals = sub[(sub["dataset"] == ds) & (sub["rare_class"] == rc) & (sub["method"] == m)]["rare_f1"].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) > 1 else 0.0)
            xs.append(i + (j - n_methods / 2 + 0.5) * width)
        ax.bar(xs, means, width=width * 0.9, color=compare_colors[m], alpha=0.82,
               label=compare_labels[m], edgecolor="white", linewidth=0.5)
        for x, mean, std in zip(xs, means, stds):
            if np.isfinite(mean) and std > 0:
                ax.errorbar(x, mean, yerr=std, fmt="none", color="black", capsize=3, linewidth=1.2)

    xlabels = [f"{DATASET_LABELS.get(ds, ds).replace(chr(10), ' ')} ({rc})" for ds, rc in dataset_rare]
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Rare-class F1 (mean ± std over seeds)", fontsize=9)
    ax.set_title("scRareRefine: rare cell identification across datasets  (rare_train_size=20)",
                 fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(facecolor=compare_colors[m], label=compare_labels[m].replace("\n", " "))
               for m in methods_present]
    ax.legend(handles=handles, fontsize=9, frameon=False, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_data_efficiency(df: pd.DataFrame, out_path: Path) -> None:
    """Two-panel ablation: left=ASDC, right=cDC1. Shows dramatic data efficiency."""
    size_map = {"all": 9999}

    def to_num(s):
        s_str = str(s).strip().lower()
        if s_str == "nan":
            return np.nan
        if s_str in size_map:
            return size_map[s_str]
        try:
            return float(s_str)
        except ValueError:
            return np.nan

    focus_methods = ["baseline", "knn_k15", "prototype_gate_marker"]
    method_labels = {"baseline": "scANVI Baseline", "knn_k15": "kNN (k=15)",
                     "prototype_gate_marker": "scRareRefine\n(Gate+Marker)"}
    method_colors = {"baseline": "#8da0cb", "knn_k15": "#aec7e8", "prototype_gate_marker": "#e78ac3"}
    method_markers = {"baseline": "s", "knn_k15": "^", "prototype_gate_marker": "o"}

    sub = df[df["dataset"] == "immune_dc"].copy()
    sub["rts_num"] = sub["rare_train_size"].apply(to_num)
    sub = sub[sub["rts_num"].notna() & sub["method"].isin(focus_methods)]

    rare_classes = ["ASDC", "cDC1"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for ax, rc in zip(axes, rare_classes):
        panel = sub[sub["rare_class"] == rc]
        for m in focus_methods:
            m_df = panel[panel["method"] == m]
            if m_df.empty:
                continue
            agg = m_df.groupby("rts_num")["rare_f1"].agg(["mean", "std", "count"]).reset_index()
            agg = agg.sort_values("rts_num")
            xs = agg["rts_num"].values.copy()
            # Map "all" (9999) to 1.5x the largest finite value for display
            finite = xs[xs < 9999]
            if len(finite):
                xs[xs == 9999] = finite.max() * 1.5

            ax.plot(xs, agg["mean"].values, marker=method_markers[m],
                    color=method_colors[m], label=method_labels[m].replace("\n", " "),
                    linewidth=2, markersize=7, zorder=5)
            # Error band (std / sqrt(n))
            se = agg["std"].fillna(0) / np.sqrt(agg["count"].clip(lower=1))
            ax.fill_between(xs,
                            (agg["mean"] - se).clip(0, 1),
                            (agg["mean"] + se).clip(0, 1),
                            alpha=0.18, color=method_colors[m])
        ax.set_title(f"Immune DC — {rc}", fontsize=11, fontweight="bold")
        ax.set_xlabel("# Rare training cells", fontsize=9)
        ax.set_ylabel("Rare-class F1 (mean ± SE)", fontsize=9)
        ax.set_ylim(-0.02, 1.08)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate rts=5 point for cDC1 with callout box
        if rc == "cDC1":
            panel5 = panel[panel["rts_num"] == 5.0]
            if not panel5.empty:
                gm5 = panel5[panel5["method"] == "prototype_gate_marker"]["rare_f1"]
                bl5 = panel5[panel5["method"] == "baseline"]["rare_f1"]
                if len(gm5) and len(bl5):
                    gm_mean = gm5.mean()
                    bl_mean = bl5.mean()
                    ax.annotate(
                        f"n=5: Baseline={bl_mean:.3f}\n       kNN=0.000\n       Ours={gm_mean:.3f}",
                        xy=(5, gm_mean), xytext=(12, 0.55),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                        fontsize=7.5, color="black",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                                  edgecolor="gray", alpha=0.9),
                    )

    handles = [mpatches.Patch(facecolor=method_colors[m], label=method_labels[m].replace("\n", " "))
               for m in focus_methods]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Data efficiency: rare-class F1 vs training size  (immune DC)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def fig_headline_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart contrasting baseline/kNN vs scRareRefine at rts=5 for ASDC and cDC1."""
    sub = df[(df["dataset"] == "immune_dc") &
             (df["rare_train_size"].astype(str) == "5") &
             (df["method"].isin(["baseline", "knn_k15", "prototype_gate_marker"]))].copy()
    if sub.empty:
        return

    method_labels = {"baseline": "scANVI\nBaseline", "knn_k15": "kNN\n(k=15)",
                     "prototype_gate_marker": "scRareRefine\n(ours)"}
    method_colors = {"baseline": "#8da0cb", "knn_k15": "#aec7e8", "prototype_gate_marker": "#e06c75"}
    methods = ["baseline", "knn_k15", "prototype_gate_marker"]
    rare_classes = ["ASDC", "cDC1"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)
    fig.suptitle("With only 5 labeled rare cells: scRareRefine vs baselines",
                 fontsize=11, fontweight="bold")

    for ax, rc in zip(axes, rare_classes):
        panel = sub[sub["rare_class"] == rc]
        xs = np.arange(len(methods))
        for j, m in enumerate(methods):
            vals = panel[panel["method"] == m]["rare_f1"].dropna().values
            mean = vals.mean() if len(vals) else 0.0
            std = vals.std() if len(vals) > 1 else 0.0
            bar = ax.bar(j, mean, color=method_colors[m], alpha=0.88,
                         edgecolor="white", linewidth=0.5, width=0.55)
            if std > 0.001:
                ax.errorbar(j, mean, yerr=std, fmt="none", color="black", capsize=5, linewidth=1.5)
            # Annotate value inside/above bar
            ypos = max(mean + std + 0.02, 0.03)
            ax.text(j, ypos, f"{mean:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="black")

        ax.set_xticks(xs)
        ax.set_xticklabels([method_labels[m] for m in methods], fontsize=9)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Rare-class F1" if rc == "ASDC" else "", fontsize=9)
        ax.set_title(f"Immune DC — {rc}", fontsize=11, fontweight="bold")
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--out_dir", default="figures/paper")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Collecting all metrics ...")
    df = collect_metrics(outputs_dir)
    if df.empty:
        print("No metrics found.")
        return

    df.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    print(f"  Collected {len(df)} rows across {df['dataset'].nunique()} datasets, "
          f"{df['method'].nunique()} methods")

    print("Generating figures ...")
    fig_main_comparison(df, out_dir / "fig_main_comparison.png")
    fig_dataset_comparison(df, out_dir / "fig_dataset_comparison.png")
    fig_trainsize_ablation(df, out_dir / "fig_trainsize_ablation.png")
    fig_data_efficiency(df, out_dir / "fig_data_efficiency.png")
    fig_separability(df, out_dir / "fig_separability.png")
    fig_all_methods_summary(df, out_dir / "fig_all_methods_summary.png")
    fig_headline_bar(df, out_dir / "fig_headline_bar.png")
    print(f"Done. Figures saved to {out_dir}")


if __name__ == "__main__":
    main()
