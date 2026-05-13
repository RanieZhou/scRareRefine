"""E33: Logit Adjustment + Mahal-pooled combined (sequential).

E32 showed Logit Adj wins vs scANVI in 66.7% of cases but loses to
Mahal-pooled on average. Key insight: they are COMPLEMENTARY:
- Logit Adj is best for: endothelial (high-sep + high-baseline), ILC rts=20
- Mahal-pooled is best for: epsilon, gamma rts=5, ILC rts=5

Combination strategy:
  1. Compute Logit Adj prediction
  2. Compute Mahal-pooled prediction
  3. If they AGREE → use that prediction
  4. If they DISAGREE → use the one with higher confidence
     (Logit Adj: max adjusted score; Mahal: margin = d_2nd - d_1st)

This is an ENSEMBLE of two different paradigms (probabilistic + geometric).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent, _class_prototypes, _pooled_covariance_shrunk, _mahalanobis, _euclidean, _predict_nearest
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e33_logit_adj_plus_mahal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_GRID = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

DATASET_CONFIGS = [
    ("outputs/immune_dc",       "cDC1",                    "batch_heldout",   "cdc1"),
    ("outputs/immune_dc",       "ASDC",                    "batch_heldout",   "asdc"),
    ("outputs/pancreas",        "epsilon",                 "batch_heldout",   "epsilon"),
    ("outputs/pancreas",        "gamma",                   "batch_heldout",   "gamma"),
    ("outputs/tabula_liver",    "non-classical monocyte",  "cell_stratified", "non-classical_monocyte"),
    ("outputs/tabula_kidney",   "endothelial cell",        "cell_stratified", "endothelial_cell"),
    ("outputs/tabula_spleen",   "innate lymphoid cell",    "batch_heldout",   "innate_lymphoid_cell"),
]

SEEDS = [42, 43, 44]
RTS_VALUES = [5, 20, 50]


def _prob_cols(df):
    return [c for c in df.columns if c.startswith("prob_")]


def _logit_adj_scores(probs_df, log_pi, tau):
    """Returns (predictions, max_adjusted_score) for each cell."""
    classes = [c[len("prob_"):] for c in probs_df.columns]
    log_probs = np.log(probs_df.to_numpy(dtype=float) + 1e-12)
    adj = np.array([tau * log_pi.get(c, 0.0) for c in classes])
    adjusted = log_probs - adj[None, :]
    pred_idx = adjusted.argmax(axis=1)
    max_score = adjusted.max(axis=1)
    return np.array(classes)[pred_idx], max_score, adjusted, classes


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError:
        return None

    prob_cols = _prob_cols(train_pred)
    if not prob_cols:
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool)
    labeled = train_pred[is_labeled]
    class_counts = labeled["true_label"].value_counts()
    total = class_counts.sum()
    log_pi = {c: float(np.log(n / total)) for c, n in class_counts.items()}

    val_labels = val_pred["true_label"].astype(str).to_numpy()
    y_test = test_pred["true_label"].astype(str)

    if rare_class not in y_test.values:
        return None

    # Tune τ on validation
    best_tau = 1.0
    best_val_f1 = -1.0
    for tau in TAU_GRID:
        val_adj, _, _, _ = _logit_adj_scores(val_pred[prob_cols], log_pi, tau)
        m, _ = classification_tables(pd.Series(val_labels), pd.Series(val_adj), rare_class=rare_class)
        if m["rare_f1"] > best_val_f1:
            best_val_f1 = m["rare_f1"]
            best_tau = tau

    # Logit Adj on test
    la_pred, la_conf, la_adj_matrix, la_classes = _logit_adj_scores(test_pred[prob_cols], log_pi, best_tau)
    la_m, _ = classification_tables(y_test, pd.Series(la_pred), rare_class=rare_class)

    # Mahal-pooled on test
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    classes_geo, protos, _ = _class_prototypes(train_z, train_pred["true_label"], is_labeled.to_numpy())
    if rare_class not in classes_geo:
        return None

    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled.to_numpy(), classes_geo)
    pooled_covs = [pooled] * len(classes_geo)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes_geo)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # Mahal confidence: margin = d_2nd_nearest - d_nearest (larger = more confident)
    sorted_dists = np.sort(mahal_dists, axis=1)
    mahal_margin = sorted_dists[:, 1] - sorted_dists[:, 0]  # positive = confident

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes_geo)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Combined: agree → use that; disagree → use higher confidence
    # Normalize confidences to [0,1] for comparison
    la_conf_norm = (la_conf - la_conf.min()) / (la_conf.max() - la_conf.min() + 1e-10)
    mahal_margin_norm = (mahal_margin - mahal_margin.min()) / (mahal_margin.max() - mahal_margin.min() + 1e-10)

    combined_pred = np.where(
        la_pred == mahal_pred,
        la_pred,  # agree
        np.where(la_conf_norm >= mahal_margin_norm, la_pred, mahal_pred)  # disagree: higher conf wins
    )
    combined_m, _ = classification_tables(y_test, pd.Series(combined_pred), rare_class=rare_class)

    # Also try: LA for high-sep (S>1.2), Mahal for low-sep (S<1.2)
    # Compute S
    labeled_z = train_z[is_labeled.to_numpy()]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled.to_numpy()]
    rare_idx_geo = classes_geo.index(rare_class)
    rare_cells = labeled_z[labeled_labels == rare_class]
    if len(rare_cells) >= 2:
        diffs = rare_cells[:, None, :] - rare_cells[None, :, :]
        pairwise = np.sqrt((diffs * diffs).sum(axis=2))
        n = len(rare_cells)
        idx = np.triu_indices(n, k=1)
        d_intra = float(pairwise[idx].mean()) if len(idx[0]) > 0 else 1e-6
    else:
        d_intra = 1e-6
    rare_proto = protos[rare_idx_geo]
    majority_protos = np.delete(protos, rare_idx_geo, axis=0)
    diffs_inter = majority_protos - rare_proto[None, :]
    d_inter = float(np.sqrt((diffs_inter * diffs_inter).sum(axis=1)).min())
    S = d_inter / max(d_intra, 1e-10)

    # S-adaptive: use LA if S >= 1.2, Mahal if S < 1.2
    if S >= 1.2:
        s_adaptive_pred = la_pred
        s_adaptive_method = "logit_adj"
    else:
        s_adaptive_pred = mahal_pred
        s_adaptive_method = "mahal_pooled"
    s_adaptive_m, _ = classification_tables(y_test, pd.Series(s_adaptive_pred), rare_class=rare_class)

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            try: rts = int(part[4:])
            except: pass

    seed = None
    for part in run_dir.name.split("_"):
        if part.startswith("seed"):
            try: seed = int(part[4:])
            except: pass

    print(f"  {run_dir.name}: S={S:.2f}  scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Eucl={euc_m['rare_f1']:.3f}  Mahal={mahal_m['rare_f1']:.3f}  "
          f"LA={la_m['rare_f1']:.3f}  Combined={combined_m['rare_f1']:.3f}  "
          f"S-adapt({s_adaptive_method})={s_adaptive_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "seed": seed,
        "rts": rts,
        "separability_ratio": round(S, 3),
        "n_rare_train": int(class_counts.get(rare_class, 0)),
        "best_tau": best_tau,
        "scanvi_f1": scanvi_m["rare_f1"],
        "euclidean_f1": euc_m["rare_f1"],
        "mahal_f1": mahal_m["rare_f1"],
        "logit_adj_f1": la_m["rare_f1"],
        "combined_f1": combined_m["rare_f1"],
        "s_adaptive_f1": s_adaptive_m["rare_f1"],
        "s_adaptive_method": s_adaptive_method,
        "delta_combined_vs_mahal": combined_m["rare_f1"] - mahal_m["rare_f1"],
        "delta_combined_vs_la": combined_m["rare_f1"] - la_m["rare_f1"],
        "delta_s_adaptive_vs_mahal": s_adaptive_m["rare_f1"] - mahal_m["rare_f1"],
    }


def main() -> pd.DataFrame:
    all_rows = []
    for dataset_dir, rare_class, split_prefix, rare_slug in DATASET_CONFIGS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            continue
        dataset_name = dataset_path.name
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}  rare_class: {rare_class}")
        for seed in SEEDS:
            for rts in RTS_VALUES:
                run_name = f"{split_prefix}_seed{seed}_{rare_slug}_rare{rts}"
                run_dir  = dataset_path / run_name
                if not run_dir.exists():
                    continue
                try:
                    result = run_one(run_dir, rare_class)
                    if result:
                        result["dataset"] = dataset_name
                        all_rows.append(result)
                except Exception as exc:
                    print(f"  ERROR {run_dir.name}: {exc}")

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "per_run_results.csv")

    # Aggregate
    agg_rows = []
    for (dataset, rare_class, rts), grp in df.groupby(["dataset", "rare_class", "rts"]):
        for method, col in [
            ("scANVI",       "scanvi_f1"),
            ("Euclidean",    "euclidean_f1"),
            ("Mahal-pooled", "mahal_f1"),
            ("Logit Adj",    "logit_adj_f1"),
            ("Combined",     "combined_f1"),
            ("S-Adaptive",   "s_adaptive_f1"),
        ]:
            vals = grp[col].dropna().values
            agg_rows.append({
                "dataset": dataset, "rare_class": rare_class, "rts": rts,
                "method": method,
                "mean_f1": float(np.mean(vals)),
                "std_f1":  float(np.std(vals)),
            })
    agg = pd.DataFrame(agg_rows)
    write_table(agg, OUT_DIR / "aggregated_results.csv")

    pivot = agg.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method", values="mean_f1", aggfunc="first"
    ).reset_index()
    pivot.columns.name = None
    method_cols = [c for c in ["scANVI", "Euclidean", "Mahal-pooled", "Logit Adj", "Combined", "S-Adaptive"]
                   if c in pivot.columns]
    pivot["best_method"] = pivot[method_cols].idxmax(axis=1)
    write_table(pivot, OUT_DIR / "best_method_summary.csv")

    print("\n\n=== E33: Logit Adj + Mahal Combined ===")
    print(pivot[["dataset","rare_class","rts"] + method_cols + ["best_method"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Best method distribution ===")
    print(pivot["best_method"].value_counts().to_string())

    total = len(pivot)
    for m1, m2 in [("Combined", "Mahal-pooled"), ("S-Adaptive", "Mahal-pooled"),
                   ("Combined", "Logit Adj"), ("S-Adaptive", "Logit Adj")]:
        if m1 in pivot.columns and m2 in pivot.columns:
            wins = (pivot[m1] > pivot[m2]).sum()
            delta = (pivot[m1] - pivot[m2]).mean()
            print(f"{m1} wins vs {m2}: {wins}/{total} ({100*wins/total:.1f}%), mean Δ={delta:.3f}")

    # Regime analysis
    high_sep = ["cDC1", "ASDC", "gamma", "innate lymphoid cell"]
    low_sep  = ["epsilon", "non-classical monocyte", "endothelial cell"]
    for regime, classes in [("High-sep", high_sep), ("Low-sep", low_sep)]:
        sub = pivot[pivot["rare_class"].isin(classes)]
        print(f"\n{regime} ({len(sub)} configs):")
        for m in method_cols:
            if m in sub.columns:
                print(f"  {m:20s}: {sub[m].mean():.3f} ± {sub[m].std():.3f}")

    return df, agg, pivot


if __name__ == "__main__":
    main()
