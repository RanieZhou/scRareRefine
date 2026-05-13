"""E29: MC Dropout Bayesian uncertainty for rare cell rescue.

Literature basis:
- Gal & Ghahramani, "Dropout as a Bayesian Approximation", ICML 2016
- "Uncertainty in Deep Learning" (Gal, PhD thesis 2016)

Core idea: Train a MLP with dropout. At test time, keep dropout ACTIVE
and run T=50 forward passes. The variance of predictions across passes
estimates epistemic uncertainty (model uncertainty due to limited data).

Rescue rule: rescue cells where:
  - mean_p(rare) > threshold_mean (model thinks it might be rare)
  - var_p(rare) < threshold_var (model is consistently uncertain, not just noisy)

This is fundamentally different from all previous methods:
- Not a distance method (no prototype computation)
- Not a probability calibration (not adjusting existing probabilities)
- Uses UNCERTAINTY as the rescue signal

Paradigm: Bayesian deep learning / Uncertainty quantification
Unique advantage: Identifies cells where the model is "consistently uncertain"
about the rare class — these are the most likely rescue candidates.
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
OUT_DIR = ROOT / "outputs" / "_experimental" / "e29_mc_dropout"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare20",  "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
    ("outputs/tabula_spleen/batch_heldout_seed42_innate_lymphoid_cell_rare20",    "innate lymphoid cell"),
]


class MCDropoutMLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden: int = 64, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout_p),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mc_dropout(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    class_weights: np.ndarray,
    *,
    hidden: int = 64,
    dropout_p: float = 0.3,
    epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
) -> MCDropoutMLP:
    torch.manual_seed(seed)
    model = MCDropoutMLP(X.shape[1], n_classes, hidden, dropout_p)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    weight_t = torch.FloatTensor(class_weights)

    X_t = torch.FloatTensor(X)
    y_t = torch.LongTensor(y_enc)

    # Class-balanced sampling
    class_indices = [np.where(y_enc == c)[0] for c in range(n_classes)]
    batch_size = min(64, len(X))
    n_per_class = max(2, batch_size // n_classes)

    model.train()
    for _ in range(epochs):
        idx = []
        for ci in class_indices:
            if len(ci) > 0:
                idx.extend(np.random.choice(ci, size=min(n_per_class, len(ci)), replace=True).tolist())
        idx = np.array(idx[:batch_size])
        x_b = X_t[idx]
        y_b = y_t[idx]
        logits = model(x_b)
        loss = F.cross_entropy(logits, y_b, weight=weight_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def mc_predict(
    model: MCDropoutMLP,
    X: np.ndarray,
    T: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Run T stochastic forward passes. Returns mean and variance of softmax probs."""
    torch.manual_seed(seed)
    model.train()  # keep dropout active
    X_t = torch.FloatTensor(X)
    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(X_t)
            probs = F.softmax(logits, dim=1).numpy()
            all_probs.append(probs)
    all_probs = np.stack(all_probs, axis=0)  # (T, n_cells, n_classes)
    mean_probs = all_probs.mean(axis=0)
    var_probs  = all_probs.var(axis=0)
    return mean_probs, var_probs


def run_one(run_dir: Path, rare_class: str) -> list[dict]:
    emb_dir = run_dir / "embeddings"
    if not emb_dir.exists():
        return []

    try:
        train_pred = read_table(emb_dir / "train_predictions.csv")
        train_lat  = read_table(emb_dir / "train_latent.csv")
        val_pred   = read_table(emb_dir / "validation_predictions.csv")
        val_lat    = read_table(emb_dir / "validation_latent.csv")
        test_pred  = read_table(emb_dir / "test_predictions.csv")
        test_lat   = read_table(emb_dir / "test_latent.csv")
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return []

    from sklearn.preprocessing import LabelEncoder
    is_labeled = train_pred["is_labeled_for_scanvi"].astype(bool).to_numpy()
    train_z = _latent(train_lat)
    val_z   = _latent(val_lat)
    test_z  = _latent(test_lat)
    y_val   = val_pred["true_label"].astype(str)
    y_test  = test_pred["true_label"].astype(str)

    X_train = train_z[is_labeled]
    y_train = train_pred["true_label"].astype(str).to_numpy()[is_labeled]

    le = LabelEncoder()
    le.fit(y_train)
    y_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    if rare_class not in le.classes_:
        return []

    rare_idx = list(le.classes_).index(rare_class)

    # Class weights: inverse frequency
    class_counts = np.array([(y_train == c).sum() for c in le.classes_], dtype=float)
    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * n_classes

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Train MC Dropout model
    print(f"  Training MC Dropout (n_rare={(y_train==rare_class).sum()}) ...")
    model = train_mc_dropout(X_train, y_enc, n_classes, class_weights, epochs=100)

    # MC predictions on validation (for threshold tuning)
    val_mean, val_var = mc_predict(model, val_z, T=50)
    val_rare_mean = val_mean[:, rare_idx]
    val_rare_var  = val_var[:, rare_idx]
    val_baseline  = val_pred["predicted_label"].astype(str).to_numpy()
    y_val_np      = y_val.to_numpy()

    # MC predictions on test
    test_mean, test_var = mc_predict(model, test_z, T=50)
    test_rare_mean = test_mean[:, rare_idx]
    test_rare_var  = test_var[:, rare_idx]
    test_baseline  = test_pred["predicted_label"].astype(str).to_numpy()

    # Grid search thresholds on validation
    mean_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    var_thresholds  = [0.01, 0.02, 0.05, 0.10, 0.20, 1.0]  # 1.0 = no var constraint

    best_val_f1 = -1.0
    best_params = (0.3, 1.0)

    for mt in mean_thresholds:
        for vt in var_thresholds:
            rescue_mask = (val_rare_mean > mt) & (val_rare_var < vt) & (val_baseline != rare_class)
            pred = val_baseline.copy()
            pred[rescue_mask] = rare_class
            m, _ = classification_tables(pd.Series(y_val_np), pd.Series(pred), rare_class=rare_class)
            if m["rare_f1"] > best_val_f1:
                best_val_f1 = m["rare_f1"]
                best_params = (mt, vt)

    # Apply best params to test
    mt, vt = best_params
    rescue_mask = (test_rare_mean > mt) & (test_rare_var < vt) & (test_baseline != rare_class)
    test_pred_mc = test_baseline.copy()
    test_pred_mc[rescue_mask] = rare_class
    mc_m, _ = classification_tables(y_test, pd.Series(test_pred_mc), rare_class=rare_class)

    # Also: standard MLP prediction (argmax of mean)
    std_pred = le.inverse_transform(test_mean.argmax(axis=1))
    std_m, _ = classification_tables(y_test, pd.Series(std_pred), rare_class=rare_class)

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    n_rescued = int(rescue_mask.sum())
    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"MLP={std_m['rare_f1']:.3f}  MC-rescue={mc_m['rare_f1']:.3f}  "
          f"(mean_t={mt}, var_t={vt}, n_rescued={n_rescued})")

    return [{
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": int((y_train == rare_class).sum()),
        "best_mean_threshold": mt,
        "best_var_threshold": vt,
        "n_rescued": n_rescued,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "mlp_rare_f1": std_m["rare_f1"],
        "mc_dropout_rare_f1": mc_m["rare_f1"],
        "mc_dropout_recall": mc_m["rare_recall"],
        "mc_dropout_precision": mc_m["rare_precision"],
        "delta_mc_vs_scanvi": mc_m["rare_f1"] - scanvi_m["rare_f1"],
        "delta_mc_vs_mlp": mc_m["rare_f1"] - std_m["rare_f1"],
    }]


def main() -> pd.DataFrame:
    all_rows = []
    for rel_path, rare_class in RUNS:
        run_dir = ROOT / rel_path
        print(f"\nProcessing {run_dir.name} ...")
        try:
            rows = run_one(run_dir, rare_class)
            all_rows.extend(rows)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()

    df = pd.DataFrame(all_rows)
    write_table(df, OUT_DIR / "results.csv")

    print("\n=== E29: MC Dropout Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train",
            "scanvi_rare_f1", "mlp_rare_f1", "mc_dropout_rare_f1",
            "delta_mc_vs_scanvi"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    wins = (df["delta_mc_vs_scanvi"] > 0.01).sum()
    print(f"\nMC Dropout wins vs scANVI: {wins}/{len(df)} runs")
    print(f"Mean delta MC vs scANVI: {df['delta_mc_vs_scanvi'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
