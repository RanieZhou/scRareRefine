"""Mahalanobis prototype PoC — does NOT touch the main pipeline.

Motivation
----------
The current main method uses Euclidean distance to a sample-mean prototype:
    d_c(z) = || z - mean(z for cells in class c) ||_2

This is a point estimate of the class center and ignores two things:
  (1) the anisotropic shape of each class cluster in latent space, and
  (2) the uncertainty of the prototype itself — which is very high when
      n_rare is tiny (e.g. 5).

A natural Bayesian upgrade is:

    d_c(z) = (z - mu_c)^T Sigma_c^{-1} (z - mu_c)           # Mahalanobis
                + tr(Sigma_c^{-1}) / n_c                    # posterior variance penalty

The second term grows as n_c shrinks, so rare-class distances are automatically
inflated — the model is "less confident" about small classes without any
heuristic.

For small n_c we cannot invert the empirical covariance directly, so we use
Ledoit-Wolf shrinkage towards a scaled identity. When n_c <= n_dim we fall back
to the diagonal of the shrunk covariance.

This script reads the already-cached scANVI latent and predictions for a single
run, computes four distance variants, and reports rare-class F1 by simply
assigning each test cell to its nearest prototype. No gate, no marker, no
threshold — pure geometry. If Mahalanobis + posterior penalty is already
competitive on rare classes, we have evidence that the direction is worth
pursuing for the full Bayesian framework.

Usage
-----
    python src/experimental/mahalanobis_poc.py \\
        --run_dir outputs/immune_dc/batch_heldout_seed42_cdc1_rare5 \\
        --rare_class cDC1

Outputs (under {run_dir}/experimental/mahalanobis_poc/):
    comparison.csv           per-variant rare-class F1, precision, recall
    distance_stats.csv       per-class intra-spread and inter-prototype distances
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from utils import classification_tables, read_table, write_table


# ── Distance variants ────────────────────────────────────────────────────────

def _latent(df: pd.DataFrame) -> np.ndarray:
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy(dtype=float)


def _class_prototypes(
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    is_labeled: np.ndarray,
) -> tuple[list[str], np.ndarray, dict[str, int]]:
    classes = sorted(reference_labels[is_labeled].unique())
    protos = []
    counts = {}
    for c in classes:
        mask = is_labeled & reference_labels.eq(c).to_numpy()
        protos.append(reference_latent[mask].mean(axis=0))
        counts[c] = int(mask.sum())
    return classes, np.vstack(protos), counts


def _euclidean(query: np.ndarray, protos: np.ndarray) -> np.ndarray:
    diff = query[:, None, :] - protos[None, :, :]
    return np.sqrt((diff * diff).sum(axis=2))


def _class_covariance_shrunk(
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    is_labeled: np.ndarray,
    classes: list[str],
    *,
    min_cells: int = 3,
) -> dict[str, np.ndarray]:
    """Per-class covariance with Ledoit-Wolf shrinkage.

    For classes with very few labeled cells (< min_cells) we shrink aggressively
    towards the diagonal to keep the covariance invertible.
    """
    d = reference_latent.shape[1]
    out: dict[str, np.ndarray] = {}
    for c in classes:
        mask = is_labeled & reference_labels.eq(c).to_numpy()
        n_c = int(mask.sum())
        X_c = reference_latent[mask]
        if n_c < min_cells:
            # Not enough cells: identity covariance scaled by pooled variance
            pooled_var = float(np.var(reference_latent[is_labeled], axis=0).mean())
            out[c] = pooled_var * np.eye(d)
            continue
        try:
            lw = LedoitWolf().fit(X_c)
            out[c] = lw.covariance_
        except Exception:
            # fallback: diagonal
            var = np.var(X_c, axis=0)
            var[var < 1e-8] = 1e-8
            out[c] = np.diag(var)
    return out


def _pooled_covariance_shrunk(
    reference_latent: np.ndarray,
    reference_labels: pd.Series,
    is_labeled: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    """Single pooled within-class covariance, shrunk with Ledoit-Wolf.

    This is the LDA-style assumption that all classes share a common
    within-cluster scale / shape. It is far more reliable than per-class
    covariances when some classes have very few cells (e.g. n_rare = 5).
    """
    parts = []
    for c in classes:
        mask = is_labeled & reference_labels.eq(c).to_numpy()
        X_c = reference_latent[mask]
        if X_c.shape[0] < 2:
            continue
        parts.append(X_c - X_c.mean(axis=0))
    if not parts:
        return np.eye(reference_latent.shape[1])
    centered = np.vstack(parts)
    try:
        return LedoitWolf().fit(centered).covariance_
    except Exception:
        var = np.var(centered, axis=0)
        var[var < 1e-8] = 1e-8
        return np.diag(var)


def _mahalanobis(query: np.ndarray, protos: np.ndarray, covs: list[np.ndarray]) -> np.ndarray:
    """Per-class Mahalanobis distance.

    Returns (n_query, n_classes) matrix where d[i, c] = sqrt((z-mu_c)^T Sigma_c^-1 (z-mu_c)).
    Uses per-class covariance (each class gets its own metric).
    """
    n_q = query.shape[0]
    n_c = protos.shape[0]
    dists = np.zeros((n_q, n_c), dtype=float)
    for c in range(n_c):
        diff = query - protos[c]
        try:
            inv = np.linalg.inv(covs[c])
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(covs[c])
        # (diff @ inv @ diff^T).diagonal() in batched form:
        quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
        quad = np.clip(quad, 0.0, None)  # numerical safety
        dists[:, c] = np.sqrt(quad)
    return dists


def _mahalanobis_with_posterior_penalty(
    query: np.ndarray,
    protos: np.ndarray,
    covs: list[np.ndarray],
    counts: list[int],
) -> np.ndarray:
    """Mahalanobis distance plus a posterior-variance penalty.

    Under a conjugate Normal model for the prototype mean, the posterior
    covariance of mu_c given the data is Sigma_c / n_c. The posterior
    predictive distance then includes an extra tr(Sigma_c^-1)/n_c term that
    grows when n_c is small. This automatically down-weights confidence in
    prototypes estimated from very few labeled cells.
    """
    base = _mahalanobis(query, protos, covs)
    penalty = np.zeros(len(covs))
    for c, (cov, n_c) in enumerate(zip(covs, counts)):
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv = np.linalg.pinv(cov)
        penalty[c] = float(np.trace(inv)) / max(n_c, 1)
    # Add as squared-distance penalty, then convert back via sqrt for comparability
    return np.sqrt(base * base + penalty[None, :])


# ── Evaluation ───────────────────────────────────────────────────────────────

def _predict_nearest(distances: np.ndarray, classes: list[str]) -> np.ndarray:
    return np.array(classes)[distances.argmin(axis=1)]


def _evaluate(y_true: pd.Series, y_pred: np.ndarray, *, rare_class: str, method: str) -> dict:
    metrics, _ = classification_tables(y_true, pd.Series(y_pred), rare_class=rare_class)
    return {"method": method, **metrics}


def _distance_diagnostics(
    protos: np.ndarray, covs: list[np.ndarray], counts: list[int], classes: list[str],
    rare_class: str,
) -> pd.DataFrame:
    rows = []
    r = classes.index(rare_class)
    try:
        inv_r = np.linalg.inv(covs[r])
    except np.linalg.LinAlgError:
        inv_r = np.linalg.pinv(covs[r])
    for c, cls in enumerate(classes):
        if c == r:
            continue
        diff = protos[r] - protos[c]
        mahal_dist = float(np.sqrt(max(diff @ inv_r @ diff, 0.0)))
        rows.append({
            "rare_class": rare_class,
            "other_class": cls,
            "n_rare_train": counts[r],
            "n_other_train": counts[c],
            "euclidean_between_prototypes": float(np.linalg.norm(diff)),
            "mahalanobis_between_prototypes": mahal_dist,
            "posterior_penalty_rare": float(np.trace(inv_r)) / max(counts[r], 1),
        })
    return pd.DataFrame(rows).sort_values("mahalanobis_between_prototypes")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mahalanobis prototype proof of concept")
    parser.add_argument("--run_dir", required=True, help="e.g. outputs/immune_dc/batch_heldout_seed42_cdc1_rare5")
    parser.add_argument("--rare_class", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    emb_dir = run_dir / "embeddings"
    out_dir = run_dir / "experimental" / "mahalanobis_poc"

    train_pred = read_table(emb_dir / "train_predictions.csv")
    train_lat = read_table(emb_dir / "train_latent.csv")
    test_pred = read_table(emb_dir / "test_predictions.csv")
    test_lat = read_table(emb_dir / "test_latent.csv")

    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    classes, protos, counts_map = _class_prototypes(
        _latent(train_lat), train_pred["true_label"], is_labeled,
    )
    counts = [counts_map[c] for c in classes]
    print(f"\nRun: {run_dir}")
    print(f"Rare class: {args.rare_class}  (n_rare_train={counts_map.get(args.rare_class, 0)})")
    print(f"Total classes: {len(classes)}")
    print(f"Class counts: {counts_map}\n")

    covs_map = _class_covariance_shrunk(_latent(train_lat), train_pred["true_label"], is_labeled, classes)
    covs = [covs_map[c] for c in classes]

    pooled = _pooled_covariance_shrunk(_latent(train_lat), train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)

    q_latent = _latent(test_lat)
    y_true = test_pred["true_label"].astype(str)

    variants = {
        "euclidean (current method)":
            _euclidean(q_latent, protos),
        "mahalanobis (per-class Sigma_c)":
            _mahalanobis(q_latent, protos, covs),
        "mahalanobis (pooled Sigma, LDA-style)":
            _mahalanobis(q_latent, protos, pooled_covs),
        "mahalanobis per-class + posterior penalty":
            _mahalanobis_with_posterior_penalty(q_latent, protos, covs, counts),
        "mahalanobis pooled + posterior penalty":
            _mahalanobis_with_posterior_penalty(q_latent, protos, pooled_covs, counts),
    }

    rows = []
    # scANVI baseline already in test_pred
    rows.append(_evaluate(y_true, test_pred["predicted_label"], rare_class=args.rare_class, method="scANVI baseline"))
    for name, dists in variants.items():
        pred = _predict_nearest(dists, classes)
        rows.append(_evaluate(y_true, pred, rare_class=args.rare_class, method=name))

    df = pd.DataFrame(rows)
    write_table(df, out_dir / "comparison.csv")
    print("Test-set comparison (nearest-prototype only, no gate, no marker):\n")
    cols = ["method", "rare_f1", "rare_recall", "rare_precision", "overall_accuracy"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    diag = _distance_diagnostics(protos, covs, counts, classes, args.rare_class)
    write_table(diag, out_dir / "distance_stats.csv")
    print(f"\nDistance diagnostics saved to: {out_dir / 'distance_stats.csv'}")
    print(f"Top 3 nearest majority classes to {args.rare_class} (Mahalanobis):")
    print(diag.head(3)[["other_class", "euclidean_between_prototypes", "mahalanobis_between_prototypes"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
