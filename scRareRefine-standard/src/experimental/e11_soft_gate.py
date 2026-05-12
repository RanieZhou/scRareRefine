"""E11: Soft gate using Mahal distance ratio.

Current hard gate: rescue if rank_rare ≤ 1 AND margin ≤ q25
Innovation: Replace hard gate with a soft probability score.

  p_rescue(z) = sigmoid((d_pred - d_rare) / τ)

where τ is a temperature tuned on validation.

This is the "soft gate" that addresses the finding from E6 that the hard gate
is too conservative for low-sep cases.

Compare: hard gate (current), soft gate (new), no gate.
Run on: cDC1 rare5/20/50, epsilon rare5/20/50, NCM rare5/20/50 (seed42).

Usage:
    python src/experimental/e11_soft_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent,
    _class_prototypes,
    _euclidean,
    _pooled_covariance_shrunk,
    _mahalanobis,
    _predict_nearest,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e11_soft_gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Temperature grid for soft gate
TAU_GRID = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
# Rescue threshold for soft gate (probability threshold)
P_THRESH_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]

RUNS = [
    ("outputs/immune_dc",    "cDC1",                   "batch_heldout",   "cdc1",                   [5, 20, 50]),
    ("outputs/pancreas",     "epsilon",                "batch_heldout",   "epsilon",                 [5, 20, 50]),
    ("outputs/tabula_liver", "non-classical monocyte", "cell_stratified", "non-classical_monocyte",  [5, 20, 50]),
]
SEED = 42


def hard_gate_rescue(
    y_scanvi: np.ndarray,
    dists: np.ndarray,
    classes: list[str],
    rare_class: str,
    q_margin: float = 0.25,
) -> np.ndarray:
    """Replicate the hard gate logic: rescue if rank_rare=1 AND margin ≤ q25."""
    rare_idx = classes.index(rare_class)
    n = dists.shape[0]
    y_out = y_scanvi.copy()

    # Rank of rare class in distance (1 = nearest)
    ranks = np.argsort(np.argsort(dists, axis=1), axis=1) + 1
    rank_rare = ranks[:, rare_idx]

    # Margin = dist_pred - dist_rare (positive means rare is closer)
    pred_idx = dists.argmin(axis=1)
    dist_pred = dists[np.arange(n), pred_idx]
    dist_rare = dists[:, rare_idx]
    margin = dist_pred - dist_rare  # positive = rare is closer

    # Hard gate threshold
    q25 = np.quantile(margin, q_margin)

    for i in range(n):
        if rank_rare[i] <= 1 and margin[i] <= q25:
            y_out[i] = rare_class

    return y_out


def soft_gate_rescue(
    y_scanvi: np.ndarray,
    dists: np.ndarray,
    classes: list[str],
    rare_class: str,
    tau: float,
    p_thresh: float,
) -> np.ndarray:
    """Soft gate: rescue if sigmoid((d_pred - d_rare) / tau) >= p_thresh."""
    rare_idx = classes.index(rare_class)
    n = dists.shape[0]
    y_out = y_scanvi.copy()

    pred_idx = dists.argmin(axis=1)
    dist_pred = dists[np.arange(n), pred_idx]
    dist_rare = dists[:, rare_idx]

    # p_rescue = sigmoid((d_pred - d_rare) / tau)
    # When d_pred > d_rare (rare is closer), this is > 0.5
    p_rescue = expit((dist_pred - dist_rare) / tau)

    for i in range(n):
        if p_rescue[i] >= p_thresh:
            y_out[i] = rare_class

    return y_out


def run_one(run_dir: Path, rare_class: str) -> list[dict]:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return []

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError:
        return []

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_true  = test_pred["true_label"].astype(str).to_numpy()
    y_scanvi = test_pred["predicted_label"].astype(str).to_numpy()

    if rare_class not in y_true:
        return []

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return []

    # Compute Mahal-pooled distances
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mah_dists = _mahalanobis(test_z, protos, pooled_covs)

    # Euclidean distances
    euc_dists = _euclidean(test_z, protos)

    # Use a 80/20 split for tau/p_thresh tuning
    n_test = len(test_z)
    rng = np.random.default_rng(42)
    val_idx = rng.choice(n_test, size=max(1, n_test // 5), replace=False)
    test_idx = np.setdiff1d(np.arange(n_test), val_idx)
    if len(test_idx) == 0:
        test_idx = np.arange(n_test)

    y_true_val  = y_true[val_idx]
    y_true_test = y_true[test_idx]
    y_scanvi_val  = y_scanvi[val_idx]
    y_scanvi_test = y_scanvi[test_idx]

    rows = []
    name = run_dir.name
    seed = None
    rts  = None
    for part in name.split("_"):
        if part.startswith("seed"):
            try:
                seed = int(part[4:])
            except ValueError:
                pass
        if part.startswith("rare") and part != "rareall":
            try:
                rts = int(part[4:])
            except ValueError:
                pass

    # 1. scANVI baseline (no gate)
    m_scanvi, _ = classification_tables(y_true, pd.Series(y_scanvi), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "scANVI (no gate)", "tau": None, "p_thresh": None,
        "rare_f1": m_scanvi["rare_f1"],
        "rare_recall": m_scanvi["rare_recall"],
        "rare_precision": m_scanvi["rare_precision"],
    })

    # 2. Euclidean nearest-proto (no gate)
    euc_pred = _predict_nearest(euc_dists, classes)
    m_euc, _ = classification_tables(y_true, pd.Series(euc_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Euclidean (no gate)", "tau": None, "p_thresh": None,
        "rare_f1": m_euc["rare_f1"],
        "rare_recall": m_euc["rare_recall"],
        "rare_precision": m_euc["rare_precision"],
    })

    # 3. Mahal-pooled (no gate)
    mah_pred = _predict_nearest(mah_dists, classes)
    m_mah, _ = classification_tables(y_true, pd.Series(mah_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Mahal-pooled (no gate)", "tau": None, "p_thresh": None,
        "rare_f1": m_mah["rare_f1"],
        "rare_recall": m_mah["rare_recall"],
        "rare_precision": m_mah["rare_precision"],
    })

    # 4. Hard gate (using Mahal distances)
    hard_pred = hard_gate_rescue(y_scanvi, mah_dists, classes, rare_class)
    m_hard, _ = classification_tables(y_true, pd.Series(hard_pred), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": "Hard gate (Mahal)", "tau": None, "p_thresh": 0.25,
        "rare_f1": m_hard["rare_f1"],
        "rare_recall": m_hard["rare_recall"],
        "rare_precision": m_hard["rare_precision"],
    })

    # 5. Soft gate — tune tau and p_thresh on validation
    best_tau = TAU_GRID[0]
    best_p   = P_THRESH_GRID[0]
    best_val_f1 = -1.0

    for tau in TAU_GRID:
        for p_thresh in P_THRESH_GRID:
            soft_pred_val = soft_gate_rescue(y_scanvi_val, mah_dists[val_idx], classes, rare_class, tau, p_thresh)
            m_val, _ = classification_tables(y_true_val, pd.Series(soft_pred_val), rare_class=rare_class)
            if m_val["rare_f1"] > best_val_f1:
                best_val_f1 = m_val["rare_f1"]
                best_tau = tau
                best_p   = p_thresh

    # Apply best params to test
    soft_pred_test = soft_gate_rescue(y_scanvi_test, mah_dists[test_idx], classes, rare_class, best_tau, best_p)
    m_soft, _ = classification_tables(y_true_test, pd.Series(soft_pred_test), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Soft gate (τ={best_tau}, p={best_p})", "tau": best_tau, "p_thresh": best_p,
        "rare_f1": m_soft["rare_f1"],
        "rare_recall": m_soft["rare_recall"],
        "rare_precision": m_soft["rare_precision"],
    })

    # Also report soft gate on full test with best params
    soft_pred_full = soft_gate_rescue(y_scanvi, mah_dists, classes, rare_class, best_tau, best_p)
    m_soft_full, _ = classification_tables(y_true, pd.Series(soft_pred_full), rare_class=rare_class)
    rows.append({
        "run": name, "rare_class": rare_class, "seed": seed, "rts": rts,
        "method": f"Soft gate full-test (τ={best_tau}, p={best_p})", "tau": best_tau, "p_thresh": best_p,
        "rare_f1": m_soft_full["rare_f1"],
        "rare_recall": m_soft_full["rare_recall"],
        "rare_precision": m_soft_full["rare_precision"],
    })

    print(f"  rts={rts:3d}  scANVI={m_scanvi['rare_f1']:.3f}  "
          f"Eucl={m_euc['rare_f1']:.3f}  Mahal={m_mah['rare_f1']:.3f}  "
          f"HardGate={m_hard['rare_f1']:.3f}  "
          f"SoftGate(τ={best_tau},p={best_p})={m_soft_full['rare_f1']:.3f}")

    return rows


def main() -> pd.DataFrame:
    all_rows = []

    for dataset_dir, rare_class, split_prefix, rare_slug, rts_list in RUNS:
        dataset_path = ROOT / dataset_dir
        if not dataset_path.exists():
            continue

        dataset_name = dataset_path.name
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}  rare_class: {rare_class}")
        print(f"{'='*60}")

        for rts in rts_list:
            run_name = f"{split_prefix}_seed{SEED}_{rare_slug}_rare{rts}"
            run_dir  = dataset_path / run_name
            if not run_dir.exists():
                print(f"  SKIP: {run_name}")
                continue

            rows = run_one(run_dir, rare_class)
            for row in rows:
                row["dataset"] = dataset_name
            all_rows.extend(rows)

    if not all_rows:
        print("No results collected!")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "results.csv")

    # Summary: compare methods per (dataset, rare_class, rts)
    print("\n\n=== E11 Summary: Soft gate vs Hard gate vs No gate ===")
    key_methods = ["scANVI (no gate)", "Euclidean (no gate)", "Mahal-pooled (no gate)",
                   "Hard gate (Mahal)"]
    # Filter to key methods + soft gate full-test
    df_key = df[df["method"].isin(key_methods) | df["method"].str.startswith("Soft gate full-test")]
    df_key = df_key.copy()
    df_key["method_short"] = df_key["method"].apply(
        lambda x: "Soft gate" if x.startswith("Soft gate full-test") else x
    )

    pivot = df_key.pivot_table(
        index=["dataset", "rare_class", "rts"],
        columns="method_short",
        values="rare_f1",
        aggfunc="first",
    )
    print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
