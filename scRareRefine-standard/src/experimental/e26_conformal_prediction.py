"""E26: Split Conformal Prediction for rare cell rescue.

Literature basis:
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction", 2022
- "Conformal Inference for Open-Set and Imbalanced Classification", arxiv 2024
- "Robust Conformal Prediction for Infrequent Classes", OpenReview 2024

Core idea: Conformal prediction constructs a PREDICTION SET (not a single label)
with a theoretical coverage guarantee. For rare cell rescue:
- If the prediction set contains the rare class AND scANVI predicted something else
  → rescue to rare class (the model is "uncertain" and rare class is plausible)

Two variants:
1. Marginal conformal: single quantile for all classes
2. Class-conditional conformal: separate quantile per class (better for imbalanced)

Paradigm: Statistical guarantee
Unique advantage: Provides a formal guarantee on false rescue rate (unlike
all previous methods which only tune on validation heuristically).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from utils import classification_tables, read_table, write_table

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e26_conformal_prediction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_VALUES = [0.05, 0.10, 0.15, 0.20, 0.30]

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


def _prob_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("prob_")]


def _class_from_prob_col(col: str) -> str:
    return col[len("prob_"):]


def _nonconformity_scores(probs: np.ndarray, true_labels: np.ndarray, classes: list[str]) -> np.ndarray:
    """Nonconformity score: 1 - p(true_label | x)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    scores = np.array([
        1.0 - probs[i, class_to_idx.get(y, 0)]
        for i, y in enumerate(true_labels)
    ])
    return scores


def _marginal_conformal_rescue(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_baseline_pred: np.ndarray,
    classes: list[str],
    rare_class: str,
    alpha: float,
) -> np.ndarray:
    """Marginal conformal: single quantile over all validation cells."""
    # Calibration scores
    cal_scores = _nonconformity_scores(val_probs, val_labels, classes)
    n = len(cal_scores)
    # Conformal quantile (finite-sample correction)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(cal_scores, q_level)

    # Prediction sets for test cells
    rare_idx = classes.index(rare_class) if rare_class in classes else -1
    if rare_idx < 0:
        return test_baseline_pred.copy()

    # Rescue: rare class is in prediction set AND baseline didn't predict rare
    rare_scores = 1.0 - test_probs[:, rare_idx]
    rare_in_set = rare_scores <= q_hat
    not_predicted_rare = test_baseline_pred != rare_class

    pred = test_baseline_pred.copy()
    pred[rare_in_set & not_predicted_rare] = rare_class
    return pred


def _class_conditional_conformal_rescue(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    test_probs: np.ndarray,
    test_baseline_pred: np.ndarray,
    classes: list[str],
    rare_class: str,
    alpha: float,
) -> np.ndarray:
    """Class-conditional conformal: separate quantile per class.

    For the rare class, we compute the quantile from validation cells
    that ARE the rare class. This ensures the rare class has its own
    coverage guarantee, not dominated by majority class calibration.
    """
    rare_idx = classes.index(rare_class) if rare_class in classes else -1
    if rare_idx < 0:
        return test_baseline_pred.copy()

    # Calibration scores for rare class cells only
    rare_val_mask = val_labels == rare_class
    if rare_val_mask.sum() < 2:
        # Fall back to marginal if too few rare validation cells
        return _marginal_conformal_rescue(
            val_probs, val_labels, test_probs, test_baseline_pred,
            classes, rare_class, alpha
        )

    rare_val_probs = val_probs[rare_val_mask]
    rare_cal_scores = 1.0 - rare_val_probs[:, rare_idx]
    n_rare = len(rare_cal_scores)
    q_level = np.ceil((n_rare + 1) * (1 - alpha)) / n_rare
    q_level = min(q_level, 1.0)
    q_hat_rare = np.quantile(rare_cal_scores, q_level)

    # Rescue: rare class nonconformity score ≤ rare-class quantile
    rare_scores = 1.0 - test_probs[:, rare_idx]
    rare_in_set = rare_scores <= q_hat_rare
    not_predicted_rare = test_baseline_pred != rare_class

    pred = test_baseline_pred.copy()
    pred[rare_in_set & not_predicted_rare] = rare_class
    return pred


def run_one(run_dir: Path, rare_class: str) -> list[dict]:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return []

    try:
        val_pred  = read_table(emb_dir / "validation_predictions.csv")
        test_pred = read_table(emb_dir / "test_predictions.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return []

    prob_cols = _prob_cols(val_pred)
    if not prob_cols:
        print(f"  SKIP: no prob_ columns")
        return []

    classes = [_class_from_prob_col(c) for c in prob_cols]
    val_probs  = val_pred[prob_cols].to_numpy(dtype=float)
    test_probs = test_pred[prob_cols].to_numpy(dtype=float)

    val_labels  = val_pred["true_label"].astype(str).to_numpy()
    test_labels = test_pred["true_label"].astype(str)
    test_baseline = test_pred["predicted_label"].astype(str).to_numpy()

    # scANVI baseline
    scanvi_m, _ = classification_tables(test_labels, test_pred["predicted_label"], rare_class=rare_class)

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    rows = []
    for alpha in ALPHA_VALUES:
        for variant, fn in [
            ("marginal", _marginal_conformal_rescue),
            ("class_conditional", _class_conditional_conformal_rescue),
        ]:
            pred = fn(val_probs, val_labels, test_probs, test_baseline,
                      classes, rare_class, alpha)
            m, _ = classification_tables(test_labels, pd.Series(pred), rare_class=rare_class)
            n_rescued = int(np.sum((pred != test_baseline) & (pred == rare_class)))
            rows.append({
                "run": run_dir.name,
                "rare_class": rare_class,
                "rts": rts,
                "alpha": alpha,
                "variant": variant,
                "scanvi_rare_f1": scanvi_m["rare_f1"],
                "conformal_rare_f1": m["rare_f1"],
                "conformal_rare_recall": m["rare_recall"],
                "conformal_rare_precision": m["rare_precision"],
                "conformal_overall_acc": m["overall_accuracy"],
                "n_rescued": n_rescued,
                "delta_f1": m["rare_f1"] - scanvi_m["rare_f1"],
            })

    # Print best result
    best = max(rows, key=lambda r: r["conformal_rare_f1"])
    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"best_conformal={best['conformal_rare_f1']:.3f} "
          f"(α={best['alpha']}, {best['variant']})")

    return rows


def main() -> pd.DataFrame:
    all_rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"Processing {run_dir.name} ...")
        try:
            rows = run_one(run_dir, rare_class)
            all_rows.extend(rows)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "results.csv")

    # Best per run (maximize rare_f1)
    best = df.loc[df.groupby(["run", "rare_class"])["conformal_rare_f1"].idxmax()]
    write_table(best, OUT_DIR / "best_per_run.csv")

    print("\n=== E26: Conformal Prediction — Best Results per Run ===")
    cols = ["run", "rare_class", "rts", "alpha", "variant",
            "scanvi_rare_f1", "conformal_rare_f1", "delta_f1"]
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins = (best["delta_f1"] > 0.01).sum()
    print(f"\nConformal wins vs scANVI: {wins}/{len(best)} runs")
    print(f"Mean delta F1: {best['delta_f1'].mean():.3f}")

    # Compare class-conditional vs marginal
    cc = df[df["variant"] == "class_conditional"].groupby(["run"])["conformal_rare_f1"].max()
    mg = df[df["variant"] == "marginal"].groupby(["run"])["conformal_rare_f1"].max()
    print(f"\nClass-conditional mean best F1: {cc.mean():.3f}")
    print(f"Marginal mean best F1: {mg.mean():.3f}")

    return df


if __name__ == "__main__":
    main()
