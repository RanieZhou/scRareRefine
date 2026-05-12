"""E12: Rare-class prototype uncertainty quantification via bootstrap.

Algorithm:
1. Bootstrap rare-class prototype B=100 times (resample with replacement)
2. For each test cell, compute distance to each bootstrap prototype → distribution
3. Use 95th percentile of bootstrap distance distribution as rescue threshold
4. A cell is rescued if its distance to mean prototype ≤ 5th percentile of
   bootstrap distances to majority prototypes

Compare vs: euclidean (fixed threshold), mahal-pooled (fixed threshold),
            bootstrap uncertainty (ours).

Run on: cDC1 rare5, ASDC rare5, epsilon rare20 (seed42, 43, 44).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e12_prototype_uncertainty"
OUT_DIR.mkdir(parents=True, exist_ok=True)

B = 100  # bootstrap replicates

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed43_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed44_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed43_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed44_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed43_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed44_epsilon_rare20", "epsilon"),
]


def _bootstrap_predict(
    query_z: np.ndarray,
    labeled_z: np.ndarray,
    labeled_labels: np.ndarray,
    rare_class: str,
    classes: list[str],
    protos: np.ndarray,
    b: int = 100,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap uncertainty-based prediction.

    For each test cell:
    - Compute distance to mean rare prototype
    - Bootstrap B rare prototypes, get 95th percentile distance → threshold_rare
    - Bootstrap B majority prototypes for each majority class, get 5th percentile
      of min-majority distances → threshold_majority
    - Rescue if d_to_mean_rare <= threshold_rare AND d_to_mean_rare <= threshold_majority
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rare_idx = classes.index(rare_class)
    rare_cells = labeled_z[labeled_labels == rare_class]
    n_rare = len(rare_cells)

    # Bootstrap rare prototypes
    boot_rare_protos = np.zeros((b, labeled_z.shape[1]))
    for i in range(b):
        idx = rng.integers(0, n_rare, size=n_rare)
        boot_rare_protos[i] = rare_cells[idx].mean(axis=0)

    # Distance from each test cell to each bootstrap rare prototype
    # shape: (n_test, B)
    diff = query_z[:, None, :] - boot_rare_protos[None, :, :]
    boot_rare_dists = np.sqrt((diff * diff).sum(axis=2))

    # 95th percentile of bootstrap distances → rescue threshold per test cell
    threshold_rare = np.percentile(boot_rare_dists, 95, axis=1)

    # Distance to mean rare prototype
    d_to_mean_rare = np.sqrt(((query_z - protos[rare_idx]) ** 2).sum(axis=1))

    # Bootstrap majority prototypes: for each majority class, bootstrap B prototypes
    # Then for each test cell, compute min distance to any majority bootstrap proto
    majority_indices = [i for i in range(len(classes)) if i != rare_idx]
    boot_majority_min_dists = np.full((len(query_z), b), np.inf)

    for maj_idx in majority_indices:
        maj_class = classes[maj_idx]
        maj_cells = labeled_z[labeled_labels == maj_class]
        n_maj = len(maj_cells)
        if n_maj < 1:
            continue
        for bi in range(b):
            idx = rng.integers(0, n_maj, size=n_maj)
            maj_proto = maj_cells[idx].mean(axis=0)
            d = np.sqrt(((query_z - maj_proto[None, :]) ** 2).sum(axis=1))
            boot_majority_min_dists[:, bi] = np.minimum(boot_majority_min_dists[:, bi], d)

    # 5th percentile of bootstrap majority distances → conservative majority threshold
    threshold_majority = np.percentile(boot_majority_min_dists, 5, axis=1)

    # Rescue condition: rare distance ≤ threshold_rare AND ≤ threshold_majority
    is_rescued = (d_to_mean_rare <= threshold_rare) & (d_to_mean_rare <= threshold_majority)

    # Start from nearest-prototype Euclidean, override with rare where rescued
    euc_dists = _euclidean(query_z, protos)
    base_pred = np.array(classes)[euc_dists.argmin(axis=1)]
    result = base_pred.copy()
    result[is_rescued] = rare_class

    return result, float(threshold_rare.mean()), float(threshold_majority.mean())


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"  WARNING: {emb_dir} not found, skipping.")
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  WARNING: missing file {e}, skipping {run_dir}")
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    classes, protos, counts_map = _class_prototypes(train_z, train_pred["true_label"], is_labeled)

    if rare_class not in classes:
        print(f"  WARNING: rare_class '{rare_class}' not in classes, skipping.")
        return None

    labeled_z = train_z[is_labeled]
    labeled_labels = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    # Euclidean
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # Bootstrap uncertainty
    print(f"  Running bootstrap (B={B}) for {run_dir.name}...")
    rng = np.random.default_rng(42)
    boot_pred, mean_rare_thresh, mean_maj_thresh = _bootstrap_predict(
        test_z, labeled_z, labeled_labels, rare_class, classes, protos, b=B, rng=rng
    )
    boot_m, _ = classification_tables(y_test, pd.Series(boot_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  bootstrap={boot_m['rare_f1']:.3f}  "
          f"(rare_thresh={mean_rare_thresh:.3f}, maj_thresh={mean_maj_thresh:.3f})")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "bootstrap_rare_f1": boot_m["rare_f1"],
        "bootstrap_recall": boot_m["rare_recall"],
        "bootstrap_precision": boot_m["rare_precision"],
        "mean_rare_threshold": mean_rare_thresh,
        "mean_majority_threshold": mean_maj_thresh,
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
            print(f"  ERROR in {run_dir.name}: {exc}")

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E12 Results: Bootstrap Uncertainty ===")
    cols = ["run", "rare_class", "n_rare_train", "scanvi_rare_f1",
            "euclidean_rare_f1", "mahal_pooled_rare_f1", "bootstrap_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
