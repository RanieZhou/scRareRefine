"""E13: Transductive label propagation on latent graph.

NOTE: This is TRANSDUCTIVE — uses test cell positions but NOT test labels.
Valid for evaluation but NOT for deployment. Flagged clearly in results.

Algorithm:
1. Build k-NN graph on ALL cells (train + test) in scANVI latent space (k=15)
2. Initialize: labeled training cells get true labels, unlabeled cells get
   uniform distribution
3. Run LabelSpreading (sklearn) for max_iter=1000
4. Extract predicted labels for test cells

Compare vs: scANVI baseline, euclidean nearest-proto, label propagation.

Run on: cDC1 rare5, ASDC rare5, epsilon rare20 (seed42).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading

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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e13_label_propagation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

K_GRAPH = 15

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
]


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

    # Euclidean baseline
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # Mahal-pooled baseline
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Label propagation (transductive)
    print(f"  Running label propagation for {run_dir.name}...")

    # Combine all cells: train (labeled + unlabeled) + test
    all_z = np.vstack([train_z, test_z])
    n_train = len(train_z)
    n_test  = len(test_z)

    # Labels: -1 for unlabeled (sklearn convention)
    train_labels_str = train_pred["true_label"].astype(str).to_numpy()
    # Map string labels to integers
    label_to_int = {c: i for i, c in enumerate(classes)}
    int_to_label = {i: c for c, i in label_to_int.items()}

    all_labels_int = np.full(len(all_z), -1, dtype=int)
    for i in range(n_train):
        if is_labeled[i]:
            lbl = train_labels_str[i]
            if lbl in label_to_int:
                all_labels_int[i] = label_to_int[lbl]

    # LabelSpreading with RBF kernel on k-NN graph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lp = LabelSpreading(
            kernel="knn",
            n_neighbors=K_GRAPH,
            alpha=0.2,
            max_iter=1000,
            tol=1e-4,
            n_jobs=-1,
        )
        try:
            lp.fit(all_z, all_labels_int)
            lp_pred_int = lp.transduction_
        except Exception as exc:
            print(f"  LabelSpreading failed: {exc}, falling back to euclidean")
            return {
                "run": run_dir.name,
                "rare_class": rare_class,
                "n_rare_train": counts_map.get(rare_class, 0),
                "scanvi_rare_f1": scanvi_m["rare_f1"],
                "euclidean_rare_f1": euc_m["rare_f1"],
                "mahal_pooled_rare_f1": mahal_m["rare_f1"],
                "label_prop_rare_f1": float("nan"),
                "label_prop_recall": float("nan"),
                "label_prop_precision": float("nan"),
                "note": f"LabelSpreading failed: {exc}",
            }

    # Extract test predictions
    test_pred_int = lp_pred_int[n_train:]
    lp_pred_str = np.array([int_to_label.get(i, classes[0]) for i in test_pred_int])
    lp_m, _ = classification_tables(y_test, pd.Series(lp_pred_str), rare_class=rare_class)

    print(f"  {run_dir.name}: scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  label_prop={lp_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_rare_f1": mahal_m["rare_f1"],
        "label_prop_rare_f1": lp_m["rare_f1"],
        "label_prop_recall": lp_m["rare_recall"],
        "label_prop_precision": lp_m["rare_precision"],
        "note": "TRANSDUCTIVE — uses test cell positions, not for deployment",
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

    print("\n=== E13 Results: Label Propagation (TRANSDUCTIVE) ===")
    cols = ["run", "rare_class", "n_rare_train", "scanvi_rare_f1",
            "euclidean_rare_f1", "mahal_pooled_rare_f1", "label_prop_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    main()
