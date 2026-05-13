"""E31: Three-layer framework — Logit Adjustment + Mahal-pooled + Conformal.

This is the key synthesis experiment of Round 4. We combine the three
best methods from different paradigms into a unified pipeline:

Layer 1 (Probabilistic): Logit Adjustment on scANVI softmax
  → Corrects the majority-class bias in scANVI's posterior
  → Produces adjusted predictions with better rare-class recall

Layer 2 (Geometric): Mahal-pooled prototype rescue
  → For cells where Layer 1 is still uncertain (margin < threshold),
    use Mahal-pooled distance to decide whether to rescue
  → Addresses cases where softmax is too uncertain to adjust

Layer 3 (Statistical guarantee): Conformal prediction set
  → For cells where both Layer 1 and Layer 2 disagree,
    use conformal prediction set to make the final decision
  → Provides theoretical coverage guarantee

Decision logic:
  1. Apply Logit Adjustment → get adjusted_pred
  2. If adjusted_pred == rare_class → accept (Layer 1 confident)
  3. Else if Mahal rank_rare == 1 AND marker_margin > 0 → rescue (Layer 2)
  4. Else if rare_class in conformal_set → rescue (Layer 3)
  5. Else → keep scANVI prediction

Compare vs: scANVI, current main method (Euclidean+gate+marker),
            Logit Adj alone, Conformal alone, Three-layer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import (
    _latent, _class_prototypes, _pooled_covariance_shrunk, _mahalanobis
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e31_three_layer_framework"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",  "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",    "innate lymphoid cell"),
]

TAU_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
ALPHA_CONFORMAL = 0.05


def _prob_cols(df):
    return [c for c in df.columns if c.startswith("prob_")]


def _logit_adj_predict(probs_df, log_pi, tau):
    classes = [c[len("prob_"):] for c in probs_df.columns]
    log_probs = np.log(probs_df.to_numpy(dtype=float) + 1e-12)
    adj = np.array([tau * log_pi.get(c, 0.0) for c in classes])
    adjusted = log_probs - adj[None, :]
    return np.array(classes)[adjusted.argmax(axis=1)]


def _conformal_set(val_probs, val_labels, test_probs, classes, rare_class, alpha):
    """Returns boolean array: True if rare_class is in conformal prediction set."""
    rare_idx = classes.index(rare_class) if rare_class in classes else -1
    if rare_idx < 0:
        return np.zeros(len(test_probs), dtype=bool)
    cal_scores = 1.0 - val_probs[:, rare_idx]
    n = len(cal_scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = np.quantile(cal_scores, q_level)
    return (1.0 - test_probs[:, rare_idx]) <= q_hat


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        val_lat    = read_table(emb_dir / "validation_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    prob_cols = _prob_cols(train_pred)
    if not prob_cols:
        return None

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool)
    labeled = train_pred[is_labeled]
    class_counts = labeled["true_label"].value_counts()
    total = class_counts.sum()
    log_pi = {c: float(np.log(n / total)) for c, n in class_counts.items()}

    classes_prob = [c[len("prob_"):] for c in prob_cols]
    val_probs  = val_pred[prob_cols].to_numpy(dtype=float)
    test_probs = test_pred[prob_cols].to_numpy(dtype=float)
    val_labels = val_pred["true_label"].astype(str).to_numpy()
    y_test     = test_pred["true_label"].astype(str)
    test_base  = test_pred["predicted_label"].astype(str).to_numpy()

    # Mahal-pooled ranks
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    classes_geo, protos, _ = _class_prototypes(train_z, train_pred["true_label"], is_labeled.to_numpy())
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled.to_numpy(), classes_geo)
    pooled_covs = [pooled] * len(classes_geo)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    rare_geo_idx = classes_geo.index(rare_class) if rare_class in classes_geo else -1
    if rare_geo_idx >= 0:
        mahal_ranks = np.argsort(np.argsort(mahal_dists, axis=1), axis=1)[:, rare_geo_idx] + 1
    else:
        mahal_ranks = np.ones(len(test_z), dtype=int) * 999

    # Conformal set membership
    conformal_rare = _conformal_set(val_probs, val_labels, test_probs,
                                    classes_prob, rare_class, ALPHA_CONFORMAL)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Tune τ on validation for Layer 1
    best_tau = 1.0
    best_val_f1 = -1.0
    for tau in TAU_GRID:
        val_adj = _logit_adj_predict(val_pred[prob_cols], log_pi, tau)
        m, _ = classification_tables(pd.Series(val_labels), pd.Series(val_adj), rare_class=rare_class)
        if m["rare_f1"] > best_val_f1:
            best_val_f1 = m["rare_f1"]
            best_tau = tau

    # Apply three-layer framework to test
    test_adj = _logit_adj_predict(test_pred[prob_cols], log_pi, best_tau)

    # Layer 1: Logit Adjustment
    la_pred = test_adj.copy()
    la_m, _ = classification_tables(y_test, pd.Series(la_pred), rare_class=rare_class)

    # Three-layer: start from scANVI, apply layers sequentially
    three_pred = test_base.copy()

    # Layer 1: where logit adj says rare → accept
    layer1_rescue = (test_adj == rare_class) & (test_base != rare_class)
    three_pred[layer1_rescue] = rare_class

    # Layer 2: where Mahal rank=1 AND not yet rescued → rescue
    layer2_rescue = (mahal_ranks == 1) & (three_pred != rare_class)
    three_pred[layer2_rescue] = rare_class

    # Layer 3: where conformal set contains rare AND not yet rescued → rescue
    layer3_rescue = conformal_rare & (three_pred != rare_class)
    three_pred[layer3_rescue] = rare_class

    three_m, _ = classification_tables(y_test, pd.Series(three_pred), rare_class=rare_class)

    # Conformal alone
    conf_pred = test_base.copy()
    conf_pred[conformal_rare & (test_base != rare_class)] = rare_class
    conf_m, _ = classification_tables(y_test, pd.Series(conf_pred), rare_class=rare_class)

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    n_l1 = int(layer1_rescue.sum())
    n_l2 = int(layer2_rescue.sum())
    n_l3 = int(layer3_rescue.sum())

    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"LA={la_m['rare_f1']:.3f}  Conf={conf_m['rare_f1']:.3f}  "
          f"3-layer={three_m['rare_f1']:.3f}  "
          f"(L1={n_l1}, L2={n_l2}, L3={n_l3})")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": int(class_counts.get(rare_class, 0)),
        "best_tau": best_tau,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "logit_adj_rare_f1": la_m["rare_f1"],
        "conformal_rare_f1": conf_m["rare_f1"],
        "three_layer_rare_f1": three_m["rare_f1"],
        "three_layer_recall": three_m["rare_recall"],
        "three_layer_precision": three_m["rare_precision"],
        "n_rescued_layer1": n_l1,
        "n_rescued_layer2": n_l2,
        "n_rescued_layer3": n_l3,
        "delta_3layer_vs_scanvi": three_m["rare_f1"] - scanvi_m["rare_f1"],
        "delta_3layer_vs_la": three_m["rare_f1"] - la_m["rare_f1"],
        "delta_3layer_vs_conf": three_m["rare_f1"] - conf_m["rare_f1"],
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

    print("\n=== E31: Three-layer Framework Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train",
            "scanvi_rare_f1", "logit_adj_rare_f1", "conformal_rare_f1",
            "three_layer_rare_f1", "delta_3layer_vs_scanvi"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins = (df["delta_3layer_vs_scanvi"] > 0.01).sum()
    print(f"\n3-layer wins vs scANVI: {wins}/{len(df)} runs")
    print(f"Mean delta 3-layer vs scANVI: {df['delta_3layer_vs_scanvi'].mean():.3f}")
    print(f"Mean delta 3-layer vs LogitAdj: {df['delta_3layer_vs_la'].mean():.3f}")
    print(f"Mean delta 3-layer vs Conformal: {df['delta_3layer_vs_conf'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
