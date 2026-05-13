"""E28: Contrastive Learning + Logit Adjustment (combined).

Literature basis:
- "Long-Tailed Recognition by Mutual Information Maximization between
  Latent Features and Ground-Truth Labels", ICML 2023
- Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
- Logit Adjustment (Menon et al., ICLR 2021)

Core idea: Combine two complementary approaches:
1. SupCon loss improves the latent representation (pulls same-class cells
   together, pushes different-class cells apart) — especially important for
   rare class with few examples
2. Logit adjustment corrects the classifier's bias toward majority classes

The combination addresses BOTH the representation problem AND the
decision boundary problem simultaneously.

Paradigm: Representation learning + Statistical calibration
Unique advantage: Addresses the root cause (bad representation) AND the
symptom (biased classifier) at the same time.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import _latent

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e28_contrastive_logit_adj"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",    "innate lymphoid cell"),
]


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 16, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class LinearClassifier(nn.Module):
    def __init__(self, proj_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(proj_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    device = features.device
    n = features.shape[0]
    # Similarity matrix
    sim = torch.mm(features, features.T) / temperature
    # Mask: same class pairs (excluding self)
    labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(n, dtype=torch.bool, device=device)
    pos_mask = labels_eq & ~self_mask

    # Log-softmax over all pairs (excluding self)
    sim_exp = torch.exp(sim)
    sim_exp_no_self = sim_exp * (~self_mask).float()
    log_prob = sim - torch.log(sim_exp_no_self.sum(dim=1, keepdim=True) + 1e-12)

    # Mean over positive pairs
    n_pos = pos_mask.float().sum(dim=1)
    loss = -(log_prob * pos_mask.float()).sum(dim=1) / (n_pos + 1e-12)
    return loss[n_pos > 0].mean() if (n_pos > 0).any() else torch.tensor(0.0, device=device)


def train_contrastive_classifier(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    log_pi: np.ndarray,
    *,
    proj_dim: int = 16,
    hidden: int = 32,
    con_epochs: int = 80,
    cls_epochs: int = 40,
    lr: float = 1e-3,
    tau_adj: float = 1.0,
    temperature: float = 0.1,
    seed: int = 42,
) -> tuple:
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y_enc)

    proj = ProjectionHead(X.shape[1], proj_dim, hidden)
    cls  = LinearClassifier(proj_dim, n_classes)

    # Class-balanced sampling
    class_indices = [np.where(y_enc == c)[0] for c in range(n_classes)]
    batch_size = min(64, len(X))
    n_per_class = max(2, batch_size // n_classes)  # at least 2 per class for SupCon

    # Phase 1: Contrastive pre-training
    opt_proj = optim.Adam(proj.parameters(), lr=lr)
    proj.train()
    for _ in range(con_epochs):
        idx = []
        for ci in class_indices:
            if len(ci) > 0:
                idx.extend(np.random.choice(ci, size=min(n_per_class, len(ci)), replace=True).tolist())
        idx = np.array(idx[:batch_size])
        x_b = X_t[idx]
        y_b = y_t[idx]
        feats = proj(x_b)
        loss = supcon_loss(feats, y_b, temperature)
        opt_proj.zero_grad()
        loss.backward()
        opt_proj.step()

    # Phase 2: Linear classifier with logit adjustment
    opt_cls = optim.Adam(cls.parameters(), lr=lr)
    log_pi_t = torch.FloatTensor(log_pi)
    proj.eval()
    cls.train()
    for _ in range(cls_epochs):
        idx = []
        for ci in class_indices:
            if len(ci) > 0:
                idx.extend(np.random.choice(ci, size=min(n_per_class, len(ci)), replace=True).tolist())
        idx = np.array(idx[:batch_size])
        x_b = X_t[idx]
        y_b = y_t[idx]
        with torch.no_grad():
            feats = proj(x_b)
        logits = cls(feats)
        # Logit adjustment: subtract τ * log(π_c)
        adjusted_logits = logits - tau_adj * log_pi_t.unsqueeze(0)
        loss = F.cross_entropy(adjusted_logits, y_b)
        opt_cls.zero_grad()
        loss.backward()
        opt_cls.step()

    return proj, cls


def predict(proj, cls, X: np.ndarray, le) -> np.ndarray:
    proj.eval(); cls.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X)
        feats = proj(X_t)
        logits = cls(feats)
        pred_enc = logits.argmax(dim=1).numpy()
    return le.inverse_transform(pred_enc)


def run_one(run_dir: Path, rare_class: str) -> dict | None:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return None

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return None

    from sklearn.preprocessing import LabelEncoder
    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    test_z  = _latent(test_lat)
    y_test  = test_pred["true_label"].astype(str)

    X_train = train_z[is_labeled]
    y_train = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    le = LabelEncoder()
    le.fit(y_train)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    if rare_class not in le.classes_:
        return None

    # Class frequencies for logit adjustment
    class_counts = np.array([(y_train == c).sum() for c in le.classes_])
    pi = class_counts / class_counts.sum()
    log_pi = np.log(pi + 1e-12)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Logit adjustment only (E24 result for comparison)
    prob_cols = [c for c in test_pred.columns if c.startswith("prob_")]
    if prob_cols:
        classes_prob = [c[len("prob_"):] for c in prob_cols]
        test_probs = test_pred[prob_cols].to_numpy(dtype=float)
        log_probs = np.log(test_probs + 1e-12)
        # Use best τ=1.0 as default
        adj = log_probs - 1.0 * np.array([log_pi[list(le.classes_).index(c)] if c in le.classes_ else 0.0
                                           for c in classes_prob])[None, :]
        la_pred = np.array(classes_prob)[adj.argmax(axis=1)]
        la_m, _ = classification_tables(y_test, pd.Series(la_pred), rare_class=rare_class)
        la_f1 = la_m["rare_f1"]
    else:
        la_f1 = float("nan")

    # Contrastive + Logit Adjustment
    print(f"  Training SupCon+LogitAdj (n_rare={(y_train==rare_class).sum()}) ...")
    try:
        proj, cls = train_contrastive_classifier(
            X_train, y_enc, n_classes, log_pi,
            con_epochs=80, cls_epochs=40, tau_adj=1.0,
        )
        con_pred = predict(proj, cls, test_z, le)
        con_m, _ = classification_tables(y_test, pd.Series(con_pred), rare_class=rare_class)
        con_f1 = con_m["rare_f1"]
        con_recall = con_m["rare_recall"]
        con_precision = con_m["rare_precision"]
    except Exception as ex:
        import traceback
        print(f"  SupCon failed: {ex}")
        traceback.print_exc()
        con_f1 = con_recall = con_precision = float("nan")

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"LogitAdj={la_f1:.3f}  SupCon+LA={con_f1:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": int((y_train == rare_class).sum()),
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "logit_adj_rare_f1": la_f1,
        "supcon_logitadj_rare_f1": con_f1,
        "supcon_logitadj_recall": con_recall,
        "supcon_logitadj_precision": con_precision,
        "delta_supcon_vs_logitadj": con_f1 - la_f1,
        "delta_supcon_vs_scanvi": con_f1 - scanvi_m["rare_f1"],
    }


def main() -> pd.DataFrame:
    rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"\nProcessing {run_dir.name} ...")
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

    print("\n=== E28: SupCon + Logit Adjustment Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train",
            "scanvi_rare_f1", "logit_adj_rare_f1",
            "supcon_logitadj_rare_f1", "delta_supcon_vs_logitadj"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins = (df["delta_supcon_vs_logitadj"] > 0.01).sum()
    print(f"\nSupCon+LA beats LogitAdj alone: {wins}/{len(df)} runs")
    print(f"Mean delta SupCon vs LogitAdj: {df['delta_supcon_vs_logitadj'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
