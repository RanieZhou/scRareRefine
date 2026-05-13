"""E24: Logit Adjustment for scANVI softmax (post-hoc calibration).

Literature basis: Menon et al., "Long-tail learning via logit adjustment", ICLR 2021.
https://arxiv.org/abs/2007.07314

Core idea: scANVI's softmax is biased toward majority classes because training
data is imbalanced. Logit adjustment corrects this by subtracting τ * log(π_c)
from each class log-probability:

    adjusted_score_c = log p_scANVI(c | x) - τ * log(π_c)
    prediction = argmax_c adjusted_score_c

where π_c = n_c / N is the class frequency in labeled training data.

This is a PURELY PROBABILISTIC method — it operates on scANVI's output
probabilities, not on latent geometry. No distance computation involved.

Paradigm: Probabilistic / Statistical calibration
Unique advantage: Directly corrects the softmax miscalibration that causes
rare class under-prediction, without any retraining or geometric assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e24_logit_adjustment"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare5",  "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare20",  "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_kidney/cell_stratified_seed42_endothelial_cell_rare20",      "endothelial cell"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",    "innate lymphoid cell"),
]


def _prob_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_")]


def _class_from_prob_col(col: str) -> str:
    return col[len("prob_"):]


def _logit_adjusted_predict(
    probs_df: pd.DataFrame,
    log_pi: dict[str, float],
    tau: float,
) -> pd.Series:
    """Apply logit adjustment and return predicted labels."""
    classes = [_class_from_prob_col(c) for c in probs_df.columns]
    log_probs = np.log(probs_df.to_numpy(dtype=float) + 1e-12)
    # Subtract τ * log(π_c) for each class
    adjustment = np.array([tau * log_pi.get(c, 0.0) for c in classes])
    adjusted = log_probs - adjustment[None, :]
    pred_idx = adjusted.argmax(axis=1)
    return pd.Series([classes[i] for i in pred_idx], index=probs_df.index)


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        print(f"  SKIP: {emb_dir} not found")
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    prob_cols = _prob_cols(train_pred)
    if not prob_cols:
        print(f"  SKIP: no prob_ columns in {run_dir.name}")
        return None

    # Compute class frequencies from labeled training cells
    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool)
    labeled = train_pred[is_labeled]
    class_counts = labeled["true_label"].value_counts()
    total = class_counts.sum()
    log_pi = {c: float(np.log(n / total)) for c, n in class_counts.items()}

    # Ensure prob columns match class names
    classes_in_probs = [_class_from_prob_col(c) for c in prob_cols]
    val_probs  = val_pred[prob_cols].rename(columns=lambda c: c)
    test_probs = test_pred[prob_cols].rename(columns=lambda c: c)

    y_val  = val_pred["true_label"].astype(str)
    y_test = test_pred["true_label"].astype(str)

    # scANVI baseline (τ=0)
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Grid search τ on validation
    val_results = []
    for tau in TAU_GRID:
        val_pred_adj = _logit_adjusted_predict(val_probs, log_pi, tau)
        m, _ = classification_tables(y_val, val_pred_adj, rare_class=rare_class)
        val_results.append({"tau": tau, "val_rare_f1": m["rare_f1"],
                             "val_rare_recall": m["rare_recall"],
                             "val_rare_precision": m["rare_precision"]})

    val_df = pd.DataFrame(val_results)
    best_tau = float(val_df.loc[val_df["val_rare_f1"].idxmax(), "tau"])
    best_val_f1 = float(val_df["val_rare_f1"].max())

    # Apply best τ to test
    test_pred_adj = _logit_adjusted_predict(test_probs, log_pi, best_tau)
    adj_m, _ = classification_tables(y_test, test_pred_adj, rare_class=rare_class)

    # Parse rts
    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    print(f"  {run_dir.name}: best_τ={best_tau}  "
          f"scANVI={scanvi_m['rare_f1']:.3f}  "
          f"LogitAdj={adj_m['rare_f1']:.3f}  "
          f"(recall: {scanvi_m['rare_recall']:.3f}→{adj_m['rare_recall']:.3f})")

    # Save tau curve
    val_df["run"] = run_dir.name
    val_df["rare_class"] = rare_class
    write_table(val_df, OUT_DIR / f"{run_dir.name}_tau_curve.csv")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": int(class_counts.get(rare_class, 0)),
        "best_tau": best_tau,
        "val_rare_f1_at_best_tau": best_val_f1,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "scanvi_rare_recall": scanvi_m["rare_recall"],
        "scanvi_rare_precision": scanvi_m["rare_precision"],
        "logit_adj_rare_f1": adj_m["rare_f1"],
        "logit_adj_rare_recall": adj_m["rare_recall"],
        "logit_adj_rare_precision": adj_m["rare_precision"],
        "logit_adj_overall_acc": adj_m["overall_accuracy"],
        "delta_f1": adj_m["rare_f1"] - scanvi_m["rare_f1"],
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

    print("\n=== E24: Logit Adjustment Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train", "best_tau",
            "scanvi_rare_f1", "logit_adj_rare_f1", "delta_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins = (df["delta_f1"] > 0.01).sum()
    print(f"\nLogit Adjustment wins vs scANVI: {wins}/{len(df)} runs")
    print(f"Mean delta F1: {df['delta_f1'].mean():.3f}")
    print(f"Best case: {df.loc[df['delta_f1'].idxmax(), 'run']} "
          f"delta={df['delta_f1'].max():.3f}")

    return df


if __name__ == "__main__":
    main()
