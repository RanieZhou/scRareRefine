"""E27: Optimal Transport for rare cell prototype alignment.

Literature basis:
- "An Adaptative Optimal Transport Approach for Long-tailed Classification",
  NeurIPS 2023
- "a (Prior-aware) Matching Perspective to (Unbalanced) Classification",
  NeurIPS 2023

Core idea: The rare class prototype is estimated from very few cells (n=5),
making it noisy. Optimal Transport can "transport" the majority class
distribution to estimate a better rare class prototype by finding the
optimal coupling between majority and rare cells.

Specifically:
1. Find the nearest majority class to the rare class (by Euclidean prototype distance)
2. Compute the OT plan from majority cells to rare cells
3. The OT-transported prototype = weighted mean of majority cells using OT plan
4. Use this OT-refined prototype for nearest-prototype classification

This is fundamentally different from distance methods — it uses the
STRUCTURE of the majority class distribution to improve the rare class
prototype estimate.

Paradigm: Optimal Transport / Distribution alignment
Unique advantage: Leverages the rich majority class distribution to
"fill in" the sparse rare class prototype, especially useful when n_rare is tiny.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent, _class_prototypes, _euclidean, _predict_nearest
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e27_optimal_transport"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",    "innate lymphoid cell"),
]


def _sinkhorn_ot(
    source: np.ndarray,
    target: np.ndarray,
    reg: float = 0.1,
    max_iter: int = 200,
) -> np.ndarray:
    """Sinkhorn algorithm for regularized OT.

    Returns transport plan T of shape (n_source, n_target).
    T[i,j] = amount of mass transported from source[i] to target[j].
    """
    n_s, n_t = len(source), len(target)
    # Cost matrix
    C = cdist(source, target, metric="euclidean")
    C = C / (C.max() + 1e-10)  # normalize

    # Uniform marginals
    a = np.ones(n_s) / n_s
    b = np.ones(n_t) / n_t

    # Sinkhorn iterations
    K = np.exp(-C / reg)
    u = np.ones(n_s)
    for _ in range(max_iter):
        v = b / (K.T @ u + 1e-12)
        u = a / (K @ v + 1e-12)

    T = np.diag(u) @ K @ np.diag(v)
    return T


def _ot_refined_prototype(
    rare_cells: np.ndarray,
    majority_cells: np.ndarray,
    reg: float = 0.1,
    max_majority: int = 50,
) -> np.ndarray:
    """Compute OT-refined prototype for rare class.

    Transport majority cells to rare cells using OT.
    The refined prototype is the weighted mean of majority cells
    using the OT plan as weights.
    """
    # Subsample majority cells for efficiency
    if len(majority_cells) > max_majority:
        idx = np.random.choice(len(majority_cells), max_majority, replace=False)
        majority_sub = majority_cells[idx]
    else:
        majority_sub = majority_cells

    # OT plan: majority → rare
    T = _sinkhorn_ot(majority_sub, rare_cells, reg=reg)

    # Weights for majority cells = row sums of T (how much each majority cell contributes)
    weights = T.sum(axis=1)
    weights = weights / (weights.sum() + 1e-12)

    # OT-transported prototype = weighted mean of majority cells
    ot_proto = (majority_sub * weights[:, None]).sum(axis=0)

    # Blend with original rare prototype (50/50)
    rare_proto = rare_cells.mean(axis=0)
    return 0.5 * rare_proto + 0.5 * ot_proto


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)
    if rare_class not in classes:
        return None

    rare_idx = classes.index(rare_class)
    rare_cells = labeled_z[labeled_labels == rare_class]
    n_rare = len(rare_cells)

    # Find nearest majority class
    rare_proto = protos[rare_idx]
    majority_dists = {
        c: float(np.linalg.norm(rare_proto - protos[i]))
        for i, c in enumerate(classes) if c != rare_class
    }
    nearest_majority = min(majority_dists, key=majority_dists.get)
    majority_cells = labeled_z[labeled_labels == nearest_majority]

    # Euclidean baseline
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # OT-refined prototype
    print(f"  Computing OT (n_rare={n_rare}, n_majority={len(majority_cells)}) ...")
    try:
        np.random.seed(42)
        ot_proto = _ot_refined_prototype(rare_cells, majority_cells, reg=0.1)

        # Replace rare prototype with OT-refined version
        protos_ot = protos.copy()
        protos_ot[rare_idx] = ot_proto

        ot_dists = _euclidean(test_z, protos_ot)
        ot_pred  = _predict_nearest(ot_dists, classes)
        ot_m, _  = classification_tables(y_test, pd.Series(ot_pred), rare_class=rare_class)
        ot_f1 = ot_m["rare_f1"]
        ot_recall = ot_m["rare_recall"]
        ot_precision = ot_m["rare_precision"]
    except Exception as ex:
        print(f"  OT failed: {ex}")
        ot_f1 = ot_recall = ot_precision = float("nan")

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"Euclidean={euc_m['rare_f1']:.3f}  OT={ot_f1:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": n_rare,
        "nearest_majority": nearest_majority,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "ot_rare_f1": ot_f1,
        "ot_rare_recall": ot_recall,
        "ot_rare_precision": ot_precision,
        "delta_ot_vs_euclidean": ot_f1 - euc_m["rare_f1"],
        "delta_ot_vs_scanvi": ot_f1 - scanvi_m["rare_f1"],
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        try:
            result = run_one(run_dir, rare_class)
            if result:
                rows.append(result)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E27: Optimal Transport Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train",
            "scanvi_rare_f1", "euclidean_rare_f1", "ot_rare_f1",
            "delta_ot_vs_euclidean"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins_vs_euc = (df["delta_ot_vs_euclidean"] > 0.01).sum()
    print(f"\nOT wins vs Euclidean: {wins_vs_euc}/{len(df)} runs")
    print(f"Mean delta OT vs Euclidean: {df['delta_ot_vs_euclidean'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
