"""E19: Contrastive fine-tuning of latent space (lightweight).

Algorithm:
1. Take 30-dim scANVI latent embeddings
2. Train a 2-layer MLP projection head: 30 → 16 → 8 dims
3. Loss: Supervised Contrastive Loss (Khosla 2020)
   - Positive pairs: same class
   - Negative pairs: different class
   - Temperature τ=0.1
4. Class-balanced re-weighting: weight_c = 1 / sqrt(n_c)
5. Project ALL cells through fine-tuned head
6. Run Euclidean nearest-prototype in new 8-dim space

Run on: cDC1 rare5, ASDC rare5, epsilon rare20 (seed42).
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e19_contrastive_finetune"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",    "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",    "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
]

PROJ_DIM = 8
HIDDEN_DIM = 16
N_EPOCHS = 200
LR = 0.001
TEMPERATURE = 0.1


def _separability_ratio(
    train_z: np.ndarray,
    train_labels: np.ndarray,
    rare_class: str,
    classes: list[str],
    protos: np.ndarray,
) -> float:
    rare_idx = classes.index(rare_class)
    rare_mask = train_labels == rare_class
    rare_cells = train_z[rare_mask]
    if len(rare_cells) < 2:
        d_intra = 1e-6
    else:
        diffs = rare_cells[:, None, :] - rare_cells[None, :, :]
        pairwise = np.sqrt((diffs * diffs).sum(axis=2))
        n = len(rare_cells)
        idx = np.triu_indices(n, k=1)
        d_intra = float(pairwise[idx].mean()) if len(idx[0]) > 0 else 1e-6
    rare_proto = protos[rare_idx]
    majority_protos = np.delete(protos, rare_idx, axis=0)
    diffs_inter = majority_protos - rare_proto[None, :]
    d_inter = float(np.sqrt((diffs_inter * diffs_inter).sum(axis=1)).min())
    return d_inter / max(d_intra, 1e-10)


def _train_contrastive_head(
    X: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    in_dim: int,
    hidden_dim: int,
    proj_dim: int,
    n_epochs: int,
    lr: float,
    temperature: float,
    batch_size: int,
) -> "torch.nn.Module":
    """Train a 2-layer MLP with supervised contrastive loss."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProjectionHead(nn.Module):
        def __init__(self, in_dim, hidden_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return F.normalize(self.net(x), dim=1)

    # Class-balanced weights: weight_c = 1 / sqrt(n_c)
    class_counts = {c: int((labels == c).sum()) for c in classes}
    label_to_idx = {c: i for i, c in enumerate(classes)}
    label_idx = np.array([label_to_idx[l] for l in labels])
    weights = np.array([1.0 / np.sqrt(max(class_counts[c], 1)) for c in classes])
    sample_weights = weights[label_idx]
    sample_weights = sample_weights / sample_weights.sum()

    device = torch.device("cpu")
    model = ProjectionHead(in_dim, hidden_dim, proj_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    labels_t = torch.tensor(label_idx, dtype=torch.long)
    weights_t = torch.tensor(sample_weights, dtype=torch.float32)

    n = len(X)
    actual_batch = min(batch_size, n)

    for epoch in range(n_epochs):
        model.train()
        # Sample a batch with class-balanced weights
        idx = np.random.choice(n, size=actual_batch, replace=True, p=sample_weights)
        x_batch = X_t[idx]
        y_batch = labels_t[idx]

        z = model(x_batch)  # (B, proj_dim), already L2-normalized

        # Supervised contrastive loss
        # sim matrix: (B, B)
        sim = torch.mm(z, z.T) / temperature

        # Mask: positive pairs = same class (excluding self)
        y_eq = y_batch.unsqueeze(0) == y_batch.unsqueeze(1)  # (B, B)
        self_mask = torch.eye(actual_batch, dtype=torch.bool)
        pos_mask = y_eq & ~self_mask

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim)
        # Exclude self from denominator
        exp_sim_no_self = exp_sim * (~self_mask).float()
        log_prob = sim - torch.log(exp_sim_no_self.sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs
        n_pos = pos_mask.float().sum(dim=1)
        loss_per_sample = -(pos_mask.float() * log_prob).sum(dim=1) / (n_pos + 1e-8)
        # Only include samples that have at least one positive
        has_pos = n_pos > 0
        if has_pos.sum() == 0:
            continue
        loss = loss_per_sample[has_pos].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}: loss={loss.item():.4f}")

    model.eval()
    return model


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

    try:
        import torch
    except ImportError:
        print("  WARNING: PyTorch not available, skipping contrastive fine-tuning.")
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

    in_dim = labeled_z.shape[1]
    batch_size = min(256, len(labeled_z))

    # ── Baseline: Euclidean in original 30-dim space ───────────────────────
    euc_dists = _euclidean(test_z, protos)
    euc_pred  = _predict_nearest(euc_dists, classes)
    euc_m, _  = classification_tables(y_test, pd.Series(euc_pred), rare_class=rare_class)

    # ── Baseline: Mahal-pooled in original 30-dim space ───────────────────
    pooled = _pooled_covariance_shrunk(train_z, train_pred["true_label"], is_labeled, classes)
    pooled_covs = [pooled] * len(classes)
    mahal_dists = _mahalanobis(test_z, protos, pooled_covs)
    mahal_pred  = _predict_nearest(mahal_dists, classes)
    mahal_m, _  = classification_tables(y_test, pd.Series(mahal_pred), rare_class=rare_class)

    # ── Separability ratio in original space ──────────────────────────────
    S_orig = _separability_ratio(labeled_z, labeled_labels, rare_class, classes, protos)

    # ── Train contrastive projection head ─────────────────────────────────
    print(f"  Training contrastive head for {run_dir.name} ({in_dim}→{HIDDEN_DIM}→{PROJ_DIM})...")
    import torch
    torch.manual_seed(42)
    np.random.seed(42)

    model = _train_contrastive_head(
        labeled_z, labeled_labels, classes,
        in_dim=in_dim, hidden_dim=HIDDEN_DIM, proj_dim=PROJ_DIM,
        n_epochs=N_EPOCHS, lr=LR, temperature=TEMPERATURE,
        batch_size=batch_size,
    )

    # Project all cells
    with torch.no_grad():
        labeled_proj = model(torch.tensor(labeled_z, dtype=torch.float32)).numpy()
        test_proj    = model(torch.tensor(test_z, dtype=torch.float32)).numpy()

    # ── Euclidean in projected 8-dim space ────────────────────────────────
    _, protos_proj, _ = _class_prototypes(
        np.vstack([labeled_proj, np.zeros((train_z.shape[0] - labeled_z.shape[0], PROJ_DIM))]),
        train_pred["true_label"],
        is_labeled,
    )
    # Recompute prototypes from projected labeled cells only
    protos_proj_list = []
    for c in classes:
        mask = labeled_labels == c
        if mask.sum() > 0:
            protos_proj_list.append(labeled_proj[mask].mean(axis=0))
        else:
            protos_proj_list.append(np.zeros(PROJ_DIM))
    protos_proj = np.vstack(protos_proj_list)

    cont_dists = _euclidean(test_proj, protos_proj)
    cont_pred  = _predict_nearest(cont_dists, classes)
    cont_m, _  = classification_tables(y_test, pd.Series(cont_pred), rare_class=rare_class)

    # ── Separability ratio in projected space ─────────────────────────────
    S_proj = _separability_ratio(labeled_proj, labeled_labels, rare_class, classes, protos_proj)

    # ── scANVI baseline ────────────────────────────────────────────────────
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    print(f"  {run_dir.name}: S_orig={S_orig:.3f}  S_proj={S_proj:.3f}  "
          f"scanvi={scanvi_m['rare_f1']:.3f}  euc={euc_m['rare_f1']:.3f}  "
          f"mahal={mahal_m['rare_f1']:.3f}  contrastive={cont_m['rare_f1']:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "n_rare_train": counts_map.get(rare_class, 0),
        "separability_ratio_original": S_orig,
        "separability_ratio_projected": S_proj,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "euclidean_orig_rare_f1": euc_m["rare_f1"],
        "mahal_pooled_orig_rare_f1": mahal_m["rare_f1"],
        "contrastive_proj_rare_f1": cont_m["rare_f1"],
        "contrastive_recall": cont_m["rare_recall"],
        "contrastive_precision": cont_m["rare_precision"],
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
            print(f"  ERROR in {run_dir.name}: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E19 Results: Contrastive Fine-tuning ===")
    cols = ["run", "rare_class", "n_rare_train",
            "separability_ratio_original", "separability_ratio_projected",
            "scanvi_rare_f1", "euclidean_orig_rare_f1",
            "mahal_pooled_orig_rare_f1", "contrastive_proj_rare_f1"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    return df


if __name__ == "__main__":
    main()
