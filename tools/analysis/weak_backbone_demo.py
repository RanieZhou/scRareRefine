"""Weak-backbone rescue demo using kNN base predictions.

This cache-only supplementary experiment asks whether the rescue mechanism can
operate on a weaker base predictor than scANVI. It keeps the scANVI latent space
and training-set prototypes fixed, but replaces the base predictions with a
validation-selected kNN classifier. The result is a scope-limited demonstration,
not a new headline claim.

Outputs:
  results/weak_backbone/weak_backbone_summary.csv
  results/weak_backbone/weak_backbone_agg.csv
  results/weak_backbone/weak_backbone_report.md
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rescue import PrototypeRescuer, conformal_rescue  # noqa: E402
from src.utils import classification_tables, load_config, make_run_dir, parse_rare_train_size  # noqa: E402

CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
]
SEEDS = [42, 43, 44]
RTS = ["0.01", "0.05", "0.10", "all"]
SCARCE = {"0.01", "0.05", "0.10"}
K_GRID = [3, 5, 10, 15]
OUT_DIR = ROOT / "results" / "weak_backbone"


def _lat(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith("latent_")]
    return df[cols].to_numpy(np.float32)


def _knn_predict(train_lat: np.ndarray, train_labels: np.ndarray, query_lat: np.ndarray, k: int) -> pd.Series:
    clf = KNeighborsClassifier(n_neighbors=min(k, len(train_labels)), weights="uniform", n_jobs=1)
    clf.fit(train_lat, train_labels)
    return pd.Series(clf.predict(query_lat)).astype(str)


def _metrics(y_true, pred, base_pred, rare_class: str) -> dict[str, float | int]:
    metrics, _ = classification_tables(y_true, pred, rare_class=rare_class)
    y = np.asarray(y_true).astype(str)
    p = np.asarray(pred).astype(str)
    b = np.asarray(base_pred).astype(str)
    n_nonrare = int((y != rare_class).sum())
    n_false = int(((p != b) & (p == rare_class) & (y != rare_class)).sum())
    n_rescued = int(((p != b) & (p == rare_class)).sum())
    n_fp = int(((p == rare_class) & (y != rare_class)).sum())
    return {
        "rare_f1": round(metrics["rare_f1"], 4),
        "rare_recall": round(metrics["rare_recall"], 4),
        "rare_precision": round(metrics["rare_precision"], 4),
        "rare_fp_rate": round(n_fp / max(n_nonrare, 1), 6),
        "rescue_ffr": round(n_false / max(n_nonrare, 1), 6),
        "n_rescued": n_rescued,
        "n_false_rescue": n_false,
    }


def run() -> pd.DataFrame:
    rows = []
    for cfg_path in CONFIGS:
        cfg = load_config(cfg_path)
        exp = cfg.get("experiment", {})
        rare = exp["rare_class"]
        split_mode = exp.get("split_mode", "batch_heldout")
        dataset = cfg["dataset"]["name"]
        print(f"[dataset] {dataset}")

        for seed in SEEDS:
            for rts in RTS:
                run_dir = make_run_dir(cfg, split_mode, seed, rare, parse_rare_train_size(rts))
                emb = run_dir / "embeddings"
                if not (emb / "test_latent.csv").exists():
                    print(f"[skip] no cache: {dataset} seed={seed} rts={rts}")
                    continue

                train_pred = pd.read_csv(emb / "train_predictions.csv")
                val_pred = pd.read_csv(emb / "validation_predictions.csv")
                test_pred = pd.read_csv(emb / "test_predictions.csv", low_memory=False)
                train_lat = _lat(pd.read_csv(emb / "train_latent.csv"))
                val_lat = _lat(pd.read_csv(emb / "validation_latent.csv"))
                test_lat = _lat(pd.read_csv(emb / "test_latent.csv"))

                is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
                train_labels = train_pred["true_label"].astype(str)
                lab_lat = train_lat[is_labeled]
                lab_labels = train_labels[is_labeled].to_numpy()

                proto = PrototypeRescuer(rare)
                proto.fit(train_lat, train_labels, is_labeled)

                y_val = val_pred["true_label"].astype(str).to_numpy()
                y_test = test_pred["true_label"].astype(str).to_numpy()

                best_k = K_GRID[0]
                best_val_f1 = -1.0
                best_val_pred = None
                for k in K_GRID:
                    pred_val = _knn_predict(lab_lat, lab_labels, val_lat, k)
                    val_metrics, _ = classification_tables(y_val, pred_val, rare_class=rare)
                    if val_metrics["rare_f1"] > best_val_f1:
                        best_val_f1 = val_metrics["rare_f1"]
                        best_k = k
                        best_val_pred = pred_val

                assert best_val_pred is not None
                base_val = best_val_pred.astype(str)
                base_test = _knn_predict(lab_lat, lab_labels, test_lat, best_k)
                rescued, summary = conformal_rescue(
                    proto,
                    base_test,
                    base_val,
                    val_pred["true_label"].astype(str),
                    val_lat,
                    test_lat,
                )

                common = {
                    "dataset": dataset,
                    "seed": seed,
                    "rts": rts,
                    "rare_class": rare,
                    "k": best_k,
                    "sep": round(proto.separability_ratio, 4),
                }
                rows.append({
                    **common,
                    "variant": "kNN",
                    "abstain": False,
                    "abstain_reason": "",
                    "chosen_rank": 0,
                    **_metrics(y_test, base_test, base_test, rare),
                })
                rows.append({
                    **common,
                    "variant": "kNN+scRareRefine",
                    "abstain": bool(summary.get("abstain", False)),
                    "abstain_reason": summary.get("reason", ""),
                    "chosen_rank": int(summary.get("chosen_rank", 0)),
                    **_metrics(y_test, rescued, base_test, rare),
                })

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, sub in [("ALL", df), ("SCARCE", df[df["rts"].isin(SCARCE)])]:
        for variant, group in sub.groupby("variant"):
            rows.append({
                "region": region,
                "variant": variant,
                "n": len(group),
                "f1_mean": round(group["rare_f1"].mean(), 4),
                "recall_mean": round(group["rare_recall"].mean(), 4),
                "precision_mean": round(group["rare_precision"].mean(), 4),
                "rare_fp_rate_max": round(group["rare_fp_rate"].max(), 6),
                "rescue_ffr_max": round(group["rescue_ffr"].max(), 6),
                "n_abstain": int(group["abstain"].sum()),
            })

        pivot = sub.pivot_table(index=["dataset", "seed", "rts"], columns="variant", values="rare_f1")
        if {"kNN", "kNN+scRareRefine"}.issubset(pivot.columns):
            delta = pivot["kNN+scRareRefine"] - pivot["kNN"]
            rows.append({
                "region": region,
                "variant": "paired_gain",
                "n": len(delta),
                "f1_mean": round(delta.mean(), 4),
                "recall_mean": np.nan,
                "precision_mean": np.nan,
                "rare_fp_rate_max": np.nan,
                "rescue_ffr_max": np.nan,
                "n_abstain": np.nan,
                "wins": int((delta > 1e-9).sum()),
                "ties": int((delta.abs() <= 1e-9).sum()),
                "losses": int((delta < -1e-9).sum()),
                "worst_delta": round(delta.min(), 4),
                "best_delta": round(delta.max(), 4),
            })
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "None"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(display.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in display.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def write_report(df: pd.DataFrame, agg: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "weak_backbone_summary.csv", index=False)
    agg.to_csv(OUT_DIR / "weak_backbone_agg.csv", index=False)

    scarce = agg[(agg["region"] == "SCARCE") & (agg["variant"].isin(["kNN", "kNN+scRareRefine", "paired_gain"]))]
    paired = (
        df.pivot_table(index=["dataset", "seed", "rts"], columns="variant", values="rare_f1")
        .assign(delta=lambda x: x["kNN+scRareRefine"] - x["kNN"])
        .reset_index()
    )
    scarce_loss_rows = paired[paired["rts"].isin(SCARCE) & (paired["delta"] < -1e-9)]
    all_loss_rows = paired[paired["delta"] < -1e-9]
    lines = [
        "# Weak-Backbone Rescue Demo",
        "",
        "Scope: kNN base predictions on the same scANVI latent space, followed by the unchanged validation-calibrated scRareRefine rescue.",
        "",
        "Scarce-region summary:",
        "",
        _markdown_table(scarce),
        "",
        "Negative paired cells in scarce region:",
        "",
        _markdown_table(scarce_loss_rows),
        "",
        "Negative paired cells across all rare_train_size settings:",
        "",
        _markdown_table(all_loss_rows),
        "",
        "Interpretation: the rescue mechanism transfers to a weaker predictor in aggregate, but this demo has one negative scarce-region cell and two negative cells overall. It should not be claimed as no-regression evidence.",
    ]
    (OUT_DIR / "weak_backbone_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    df = run()
    agg = summarize(df)
    write_report(df, agg)


if __name__ == "__main__":
    main()
