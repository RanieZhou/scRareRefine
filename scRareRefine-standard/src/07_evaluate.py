"""Stage 7: Multi-seed evaluation and comparison plots.

Reads outputs from all method stages across multiple seeds and produces:
    all_seeds_metrics.csv   — one row per (method, seed)
    comparison_bar.png      — bar chart: mean across seeds, with error bars (±std)
    comparison_box.png      — box plot: distribution across seeds with individual points

Methods included (when available):
    baseline      scANVI softmax (Stage 2)
    knn_k15       k-NN on latent space (Stage 3b)
    lr            Logistic regression on HVG expression (Stage 3c)
    scRareRefine  Prototype gate + marker verification (Stage 5 / main.py)

Usage:
    python src/07_evaluate.py \\
        --config configs/immune_dc.yaml \\
        --rare_class ASDC --rare_train_size 20
    python src/07_evaluate.py \\
        --config configs/immune_dc.yaml \\
        --rare_class ASDC --rare_train_size 20 --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _rts_label, classification_tables, load_config, make_run_dir, parse_rare_train_size, read_table, write_table


# ── Method display config ─────────────────────────────────────────────────────

METHOD_ORDER = ["baseline", "knn_k15", "lr", "scRareRefine"]
METHOD_LABELS = {
    "baseline":     "Baseline\n(scANVI)",
    "knn_k15":      "kNN\n(k=15)",
    "lr":           "LR\n(CellTypist)",
    "scRareRefine": "scRareRefine",
}
METHOD_COLORS = {
    "baseline":     "#8da0cb",
    "knn_k15":      "#66c2a5",
    "lr":           "#fc8d62",
    "scRareRefine": "#e78ac3",
}

METRICS = [
    ("rare_f1",        "Rare-class F1"),
    ("rare_recall",    "Rare Recall"),
    ("rare_precision", "Rare Precision"),
]


# ── IO helpers ────────────────────────────────────────────────────────────────

def _safe_read(path: Path) -> pd.DataFrame:
    try:
        return read_table(path)
    except FileNotFoundError:
        return pd.DataFrame()


def _metrics(y_true, y_pred, *, rare_class: str) -> dict:
    m, _ = classification_tables(y_true, y_pred, rare_class=rare_class)
    return m


# ── Per-method row builders ───────────────────────────────────────────────────

def _baseline_metrics(test_pred: pd.DataFrame, *, rare_class: str) -> dict:
    return _metrics(test_pred["true_label"], test_pred["predicted_label"], rare_class=rare_class)


def _knn_metrics(run_dir: Path) -> dict | None:
    df = _safe_read(run_dir / "knn" / "test_metrics.csv")
    if df.empty:
        return None
    row = df[df["method"] != "baseline"]
    if row.empty:
        return None
    r = row.iloc[0]
    return {c: float(r[c]) for c in ["overall_accuracy", "macro_f1", "rare_precision", "rare_recall", "rare_f1"] if c in r.index}


def _lr_metrics(run_dir: Path) -> dict | None:
    df = _safe_read(run_dir / "celltypist" / "test_metrics.csv")
    if df.empty:
        return None
    row = df[df["method"] == "lr"]
    if row.empty:
        row = df[df["method"] != "baseline"]
    if row.empty:
        return None
    r = row.iloc[0]
    return {c: float(r[c]) for c in ["overall_accuracy", "macro_f1", "rare_precision", "rare_recall", "rare_f1"] if c in r.index}


def _scrarerefine_metrics(test_pred: pd.DataFrame, run_dir: Path, *, rare_class: str) -> dict:
    scored = _safe_read(run_dir / "gate_marker" / "test_scored.csv")
    threshold_df = _safe_read(run_dir / "gate_marker" / "selected_thresholds.csv")
    threshold = float(threshold_df["selected_marker_threshold"].iloc[0]) if not threshold_df.empty else float("inf")

    y_true = test_pred["true_label"].astype(str)
    rescue_pred = test_pred["predicted_label"].astype(str).copy()
    if not scored.empty and "marker_margin" in scored.columns and "cell_id" in scored.columns:
        margins = pd.to_numeric(scored["marker_margin"], errors="coerce")
        verified = set(scored.loc[margins.ge(threshold).fillna(False), "cell_id"].astype(str))
        if "cell_id" in test_pred.columns:
            rescue_pred.loc[test_pred["cell_id"].astype(str).isin(verified)] = rare_class

    return _metrics(y_true, rescue_pred, rare_class=rare_class)


# ── Collect all seeds ─────────────────────────────────────────────────────────

def collect_all_seeds(
    config: dict,
    seeds: list[int],
    *,
    rare_class: str,
    rare_train_size: int | str,
    split_mode: str,
) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        run_dir = make_run_dir(config, split_mode, seed, rare_class, rare_train_size)
        emb_path = run_dir / "embeddings" / "test_predictions.csv"
        if not emb_path.exists():
            print(f"  [skip] seed={seed}: embeddings not found at {run_dir}")
            continue

        test_pred = read_table(emb_path)
        common = {"seed": seed, "rare_class": rare_class,
                  "rare_train_size": str(rare_train_size), "split_mode": split_mode}

        rows.append({"method": "baseline", **common, **_baseline_metrics(test_pred, rare_class=rare_class)})

        knn = _knn_metrics(run_dir)
        if knn:
            rows.append({"method": "knn_k15", **common, **knn})

        lr = _lr_metrics(run_dir)
        if lr:
            rows.append({"method": "lr", **common, **lr})

        rows.append({"method": "scRareRefine", **common,
                     **_scrarerefine_metrics(test_pred, run_dir, rare_class=rare_class)})

    return pd.DataFrame(rows)


# ── Plots ─────────────────────────────────────────────────────────────────────

def _present_methods(df: pd.DataFrame) -> list[str]:
    return [m for m in METHOD_ORDER if m in df["method"].values]


def plot_bar(df: pd.DataFrame, out_path: Path, *, rare_class: str, rts: str) -> None:
    """Bar chart: mean ± std across seeds, one panel per metric."""
    methods = _present_methods(df)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5))
    seeds = sorted(df["seed"].unique())
    fig.suptitle(
        f"{rare_class}  |  rts={rts}  |  seeds={seeds}  —  mean ± std",
        fontsize=10, fontweight="bold",
    )

    x = np.arange(len(methods))
    width = 0.6

    for ax, (col, title) in zip(axes, METRICS):
        means, stds = [], []
        for m in methods:
            vals = df.loc[df["method"] == m, col].dropna().astype(float).tolist()
            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals, ddof=0) if len(vals) > 1 else 0.0)

        bars = ax.bar(x, means, width,
                      color=[METHOD_COLORS.get(m, "#aaa") for m in methods],
                      yerr=stds, capsize=4, error_kw={"linewidth": 1.2, "ecolor": "#333"},
                      edgecolor="white", linewidth=0.8)
        for bar, mean, std in zip(bars, means, stds):
            label = f"{mean:.3f}" if std == 0 else f"{mean:.3f}\n±{std:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.015,
                    label, ha="center", va="bottom", fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8.5)
        ax.set_ylim(0, 1.18)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path.name}")


def plot_box(df: pd.DataFrame, out_path: Path, *, rare_class: str, rts: str) -> None:
    """Box plot with individual seed points overlaid."""
    methods = _present_methods(df)
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4.5))
    seeds = sorted(df["seed"].unique())
    fig.suptitle(
        f"{rare_class}  |  rts={rts}  |  seeds={seeds}  —  distribution",
        fontsize=10, fontweight="bold",
    )

    x = np.arange(len(methods))
    rng = np.random.default_rng(0)

    for ax, (col, title) in zip(axes, METRICS):
        data = [df.loc[df["method"] == m, col].dropna().astype(float).tolist() for m in methods]

        bp = ax.boxplot(data, positions=x, widths=0.45, patch_artist=True,
                        medianprops={"color": "#333", "linewidth": 2},
                        whiskerprops={"linewidth": 1},
                        capprops={"linewidth": 1},
                        flierprops={"marker": ""})
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(m, "#aaa"))
            patch.set_alpha(0.7)

        # Individual seed points
        for xi, (m, vals) in enumerate(zip(methods, data)):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals,
                       color=METHOD_COLORS.get(m, "#aaa"), edgecolors="#333",
                       linewidths=0.8, s=40, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8.5)
        ax.set_ylim(-0.05, 1.12)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {out_path.name}")


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, *, rare_class: str) -> None:
    methods = _present_methods(df)
    cols = ["rare_f1", "rare_recall", "rare_precision"]
    print(f"\n{'─'*65}")
    print(f"  {rare_class}  —  mean (std) across {df['seed'].nunique()} seeds")
    print(f"{'─'*65}")
    header = f"  {'method':<16}" + "".join(f"  {c:<22}" for c in cols)
    print(header)
    for m in methods:
        sub = df[df["method"] == m]
        row_str = f"  {METHOD_LABELS.get(m, m).replace(chr(10), ' '):<16}"
        for col in cols:
            vals = sub[col].dropna().astype(float)
            if vals.empty:
                row_str += f"  {'—':<22}"
            elif len(vals) == 1:
                row_str += f"  {vals.iloc[0]:.3f}{'':17}"
            else:
                row_str += f"  {vals.mean():.3f} ± {vals.std(ddof=0):.3f}         "
        print(row_str)
    print(f"{'─'*65}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: multi-seed evaluation and comparison plots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rare_class", default=None)
    parser.add_argument("--rare_train_size", required=True)
    parser.add_argument("--split_mode", default="batch_heldout", help="batch_heldout | cell_stratified | lobo_<batch>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    config = load_config(args.config)
    rare_class = args.rare_class or config["experiment"]["rare_class"]
    rare_train_size = parse_rare_train_size(args.rare_train_size)
    dataset_name = config["dataset"]["name"]

    print(f"\n07_evaluate  |  {dataset_name}  |  {rare_class}  |  rts={rare_train_size}  |  seeds={args.seeds}")

    df = collect_all_seeds(
        config, args.seeds,
        rare_class=rare_class, rare_train_size=rare_train_size, split_mode=args.split_mode,
    )
    if df.empty:
        print("No results found. Run prerequisite stages first.")
        return

    _print_summary(df, rare_class=rare_class)

    rts_str = str(rare_train_size)

    # All outputs → each seed's run_dir
    for seed in args.seeds:
        seed_df = df[df["seed"] == seed]
        if seed_df.empty:
            continue
        run_dir = make_run_dir(config, args.split_mode, seed, rare_class, rare_train_size)
        run_dir.mkdir(parents=True, exist_ok=True)
        # Aggregate metrics table (all seeds) copied to each run folder
        csv_path = write_table(df, run_dir / "all_seeds_metrics.csv")
        print(f"  Saved: {csv_path}")
        # Per-seed charts
        fig_dir = run_dir / "figure"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_bar(seed_df, fig_dir / "comparison_bar.png", rare_class=rare_class, rts=rts_str)
        plot_box(seed_df, fig_dir / "comparison_box.png", rare_class=rare_class, rts=rts_str)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
