from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def classification_tables(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    rare_class: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    labels = sorted(set(y_true) | set(y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )
    rare_row = per_class[per_class["label"] == rare_class]
    if rare_row.empty:
        rare_precision = rare_recall = rare_f1 = 0.0
    else:
        rare_precision = float(rare_row["precision"].iloc[0])
        rare_recall = float(rare_row["recall"].iloc[0])
        rare_f1 = float(rare_row["f1"].iloc[0])

    overall = {
        "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "rare_precision": rare_precision,
        "rare_recall": rare_recall,
        "rare_f1": rare_f1,
    }
    return overall, per_class


def compute_uncertainty(probabilities: pd.DataFrame, *, rare_class: str) -> pd.DataFrame:
    probs = probabilities.astype(float)
    arr = probs.to_numpy()
    classes = probs.columns.to_numpy()
    order = np.argsort(-arr, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1] if arr.shape[1] > 1 else order[:, 0]
    top1 = arr[np.arange(arr.shape[0]), top1_idx]
    top2 = arr[np.arange(arr.shape[0]), top2_idx]
    entropy = -(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum(axis=1)

    return pd.DataFrame(
        {
            "max_prob": top1,
            "entropy": entropy,
            "margin": top1 - top2,
            "top1_label": classes[top1_idx],
            "top2_label": classes[top2_idx],
            "top2_is_" + rare_class: classes[top2_idx] == rare_class,
        },
        index=probabilities.index,
    )


def topk_review_recall(
    events: np.ndarray | pd.Series,
    risk_scores: np.ndarray | pd.Series,
    *,
    ks: list[float],
) -> pd.DataFrame:
    events = np.asarray(events, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=float)
    n = len(events)
    total_events = int(events.sum())
    order = np.argsort(-risk_scores)
    rows = []
    for k in ks:
        n_review = max(1, int(np.ceil(n * k)))
        selected = order[:n_review]
        covered = int(events[selected].sum())
        rows.append(
            {
                "k_fraction": float(k),
                "n_review": int(n_review),
                "events_total": total_events,
                "events_covered": covered,
                "event_recall": float(covered / total_events) if total_events else 0.0,
            }
        )
    return pd.DataFrame(rows)
