"""E25: Conditional VAE for rare cell synthesis in latent space.

Literature basis: scDiffusion (Bioinformatics 2024), CVAE for imbalanced
classification (arxiv 2024).

Core idea: Train a conditional VAE on scANVI latent embeddings. Condition on
class label to generate synthetic rare cells. Use real + synthetic cells to
train a logistic regression classifier.

This is a GENERATIVE method — it creates new data points rather than
adjusting distances or probabilities.

Paradigm: Generative
Unique advantage: Learns the true latent distribution of rare cells (not just
linear interpolation like SMOTE), enabling better coverage of the rare class
manifold.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from utils import classification_tables, read_table, write_table
from experimental.mahalanobis_poc import _latent

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "_experimental" / "e25_conditional_vae"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare5",   "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_cdc1_rare20",  "cDC1"),
    ("outputs/immune_dc/batch_heldout_seed42_asdc_rare5",   "ASDC"),
    ("outputs/pancreas/batch_heldout_seed42_epsilon_rare20", "epsilon"),
    ("outputs/pancreas/batch_heldout_seed42_gamma_rare5",   "gamma"),
    ("outputs/tabula_liver/cell_stratified_seed42_non-classical_monocyte_rare20", "non-classical monocyte"),
]


class CVAE(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, latent_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        cond_dim = input_dim + n_classes
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mu  = nn.Linear(hidden, latent_dim)
        self.fc_var = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + n_classes, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def train_cvae(
    X: np.ndarray,
    y_enc: np.ndarray,
    n_classes: int,
    *,
    latent_dim: int = 16,
    hidden: int = 64,
    epochs: int = 150,
    lr: float = 1e-3,
    beta: float = 0.001,
    seed: int = 42,
) -> CVAE:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    model = CVAE(X.shape[1], n_classes, latent_dim, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    X_t = torch.FloatTensor(X).to(device)
    y_t = torch.LongTensor(y_enc).to(device)
    C_t = torch.zeros(len(y_enc), n_classes).to(device)
    C_t.scatter_(1, y_t.unsqueeze(1), 1.0)

    # Class-balanced sampling indices
    class_indices = [np.where(y_enc == c)[0] for c in range(n_classes)]
    batch_size = min(64, len(X))

    model.train()
    for epoch in range(epochs):
        # Sample balanced batch
        n_per_class = max(1, batch_size // n_classes)
        idx = []
        for ci in class_indices:
            if len(ci) > 0:
                idx.extend(np.random.choice(ci, size=min(n_per_class, len(ci)), replace=True).tolist())
        idx = np.array(idx[:batch_size])

        x_b = X_t[idx]
        c_b = C_t[idx]

        recon, mu, logvar = model(x_b, c_b)
        recon_loss = nn.functional.mse_loss(recon, x_b)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + beta * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


def synthesize_rare(
    model: CVAE,
    rare_class_idx: int,
    n_classes: int,
    n_samples: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim)
        c = torch.zeros(n_samples, n_classes)
        c[:, rare_class_idx] = 1.0
        synthetic = model.decode(z, c).numpy()
    return synthetic


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
    rare_idx = list(le.classes_).index(rare_class) if rare_class in le.classes_ else -1

    if rare_idx < 0:
        print(f"  SKIP: rare_class '{rare_class}' not in training labels")
        return None

    n_rare = int((y_train == rare_class).sum())
    n_aug = max(100, 5 * n_rare)

    # scANVI baseline
    scanvi_m, _ = classification_tables(y_test, test_pred["predicted_label"], rare_class=rare_class)

    # Standard LR (no augmentation)
    lr_std = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_std.fit(X_train, y_enc)
    lr_pred = le.inverse_transform(lr_std.predict(test_z))
    lr_m, _ = classification_tables(y_test, pd.Series(lr_pred), rare_class=rare_class)

    # SMOTE-LR baseline
    try:
        from collections import Counter
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        min_cnt = min(Counter(y_enc).values())
        k_nb = min(5, min_cnt - 2)
        sampler = SMOTE(k_neighbors=max(1, k_nb), random_state=42) if k_nb >= 1 else RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_enc)
        lr_smote = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_smote.fit(X_res, y_res)
        smote_pred = le.inverse_transform(lr_smote.predict(test_z))
        smote_m, _ = classification_tables(y_test, pd.Series(smote_pred), rare_class=rare_class)
        smote_f1 = smote_m["rare_f1"]
    except Exception as ex:
        print(f"  SMOTE failed: {ex}")
        smote_f1 = float("nan")

    # CVAE-LR
    print(f"  Training CVAE (n_rare={n_rare}, n_aug={n_aug}) ...")
    try:
        # Normalize latents for stable training
        X_mean = X_train.mean(axis=0)
        X_std  = X_train.std(axis=0) + 1e-8
        X_norm = (X_train - X_mean) / X_std

        model = train_cvae(X_norm, y_enc, n_classes, epochs=150, beta=0.001)
        synthetic_norm = synthesize_rare(model, rare_idx, n_classes, n_aug)
        synthetic = synthetic_norm * X_std + X_mean

        # Augmented training set
        X_aug = np.vstack([X_train, synthetic])
        y_aug = np.concatenate([y_enc, np.full(n_aug, rare_idx, dtype=int)])

        lr_cvae = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_cvae.fit(X_aug, y_aug)
        cvae_pred = le.inverse_transform(lr_cvae.predict(test_z))
        cvae_m, _ = classification_tables(y_test, pd.Series(cvae_pred), rare_class=rare_class)
        cvae_f1 = cvae_m["rare_f1"]
        cvae_recall = cvae_m["rare_recall"]
        cvae_precision = cvae_m["rare_precision"]
    except Exception as ex:
        import traceback
        print(f"  CVAE failed: {ex}")
        traceback.print_exc()
        cvae_f1 = cvae_recall = cvae_precision = float("nan")

    rts = "unknown"
    for part in run_dir.name.split("_"):
        if part.startswith("rare") and part != "rareall":
            rts = part.replace("rare", "")

    print(f"  {run_dir.name}: scANVI={scanvi_m['rare_f1']:.3f}  "
          f"LR={lr_m['rare_f1']:.3f}  SMOTE={smote_f1:.3f}  CVAE={cvae_f1:.3f}")

    return {
        "run": run_dir.name,
        "rare_class": rare_class,
        "rts": rts,
        "n_rare_train": n_rare,
        "n_aug": n_aug,
        "scanvi_rare_f1": scanvi_m["rare_f1"],
        "lr_rare_f1": lr_m["rare_f1"],
        "smote_lr_rare_f1": smote_f1,
        "cvae_lr_rare_f1": cvae_f1,
        "cvae_lr_recall": cvae_recall,
        "cvae_lr_precision": cvae_precision,
        "delta_cvae_vs_smote": cvae_f1 - smote_f1 if not np.isnan(smote_f1) else float("nan"),
        "delta_cvae_vs_scanvi": cvae_f1 - scanvi_m["rare_f1"],
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

    print("\n=== E25: Conditional VAE Results ===")
    cols = ["run", "rare_class", "rts", "n_rare_train",
            "scanvi_rare_f1", "lr_rare_f1", "smote_lr_rare_f1",
            "cvae_lr_rare_f1", "delta_cvae_vs_smote"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cvae_beats_smote = (df["delta_cvae_vs_smote"] > 0.01).sum()
    print(f"\nCVAE beats SMOTE: {cvae_beats_smote}/{len(df)} runs")
    print(f"Mean delta CVAE vs SMOTE: {df['delta_cvae_vs_smote'].mean():.3f}")

    return df


if __name__ == "__main__":
    main()
