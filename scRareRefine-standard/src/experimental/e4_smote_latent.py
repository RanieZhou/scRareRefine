"""E4: SMOTE oversampling in latent space (sc-SynO equivalent).

Steps:
1. Take labeled training cells
2. Apply SMOTE to oversample rare class to match majority (imbalanced-learn)
3. Train a logistic regression on the oversampled latent embeddings
4. Apply to test set

Compare vs:
  - scANVI baseline
  - standard LR (no oversampling)
  - SMOTE-LR (our variant)

Run on: cDC1 rare5, ASDC rare5, epsilon rare20 (seed42).

Usage:
    python src/experimental/e4_smote_latent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import _latent

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e4_smote_latent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",  "ASDC"),
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

    # Labeled training data only
    X_train = train_z[is_labeled]
    y_train = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    # Encode labels
    le = LabelEncoder()
    le.fit(y_train)
    y_enc = le.transform(y_train)

    # Standard LR (no oversampling)
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_enc)
    lr_pred = le.inverse_transform(lr.predict(test_z))
    lr_m, _ = classification_tables(y_test, pd.Series(lr_pred), rare_class=rare_class)
    print(f"    Standard LR:  rare_f1={lr_m['rare_f1']:.3f}")

    # SMOTE-LR
    try:
        from imblearn.over_sampling import SMOTE
        rare_count = int((y_train == rare_class).sum())
        majority_count = int((y_train != rare_class).sum())

        # Need at least 2 rare samples for SMOTE (k_neighbors=1 if rare_count < 6)
        # SMOTE k_neighbors must be < min class count - 1 (SMOTE uses k+1 internally)
        from collections import Counter
        min_class_count = min(Counter(y_enc).values())
        k_neighbors = min(5, min_class_count - 2)
        if k_neighbors < 1:
            print(f"    SMOTE: min class too small ({min_class_count}), using RandomOverSampler")
            from imblearn.over_sampling import RandomOverSampler
            sampler = RandomOverSampler(random_state=42)
        else:
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)

        X_res, y_res = sampler.fit_resample(X_train, y_enc)
        lr_smote = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_smote.fit(X_res, y_res)
        smote_pred = le.inverse_transform(lr_smote.predict(test_z))
        smote_m, _ = classification_tables(y_test, pd.Series(smote_pred), rare_class=rare_class)
        print(f"    SMOTE-LR:     rare_f1={smote_m['rare_f1']:.3f}")
        smote_rare_f1 = smote_m["rare_f1"]
        smote_overall = smote_m["overall_accuracy"]
    except Exception as ex:
        print(f"    SMOTE failed: {ex}")
        smote_rare_f1 = float("nan")
        smote_overall = float("nan")

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)
    print(f"    scANVI:       rare_f1={scanvi_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": int((y_train == rare_class).sum()),
        "test_rare_f1_scanvi": scanvi_m["rare_f1"],
        "test_rare_f1_lr": lr_m["rare_f1"],
        "test_rare_f1_smote_lr": smote_rare_f1,
        "test_overall_acc_scanvi": scanvi_m["overall_accuracy"],
        "test_overall_acc_lr": lr_m["overall_accuracy"],
        "test_overall_acc_smote_lr": smote_overall,
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        result = run_one(run_dir, rare_class)
        if result:
            rows.append(result)

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E4 Results (SMOTE in latent space, seed42) ===")
    cols = ["run", "rare_class", "n_rare_train",
            "test_rare_f1_scanvi", "test_rare_f1_lr", "test_rare_f1_smote_lr"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
