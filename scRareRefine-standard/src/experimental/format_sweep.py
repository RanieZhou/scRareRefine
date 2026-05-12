"""Compact tabular view of mahalanobis_sweep.csv for slide/report use."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / "outputs" / "_experimental" / "mahalanobis_sweep.csv"


METHOD_ORDER = [
    "scANVI baseline",
    "euclidean (current method)",
    "mahalanobis (per-class Sigma_c)",
    "mahalanobis (pooled Sigma, LDA-style)",
    "mahalanobis per-class + posterior penalty",
    "mahalanobis pooled + posterior penalty",
]

SHORT = {
    "scANVI baseline":                               "scANVI",
    "euclidean (current method)":                    "Eucl(cur)",
    "mahalanobis (per-class Sigma_c)":               "Mahal-pc",
    "mahalanobis (pooled Sigma, LDA-style)":         "Mahal-pool",
    "mahalanobis per-class + posterior penalty":     "Mahal-pc+post",
    "mahalanobis pooled + posterior penalty":        "Mahal-pool+post",
}


def tag(run_id: str) -> str:
    # Keep only the informative suffix for display
    parts = run_id.split("_")
    rts = next((p for p in parts if p.startswith("rare")), "")
    return rts


def main() -> None:
    df = pd.read_csv(SWEEP)
    df["rts"] = df["run_id"].map(tag)
    df["short"] = df["method"].map(SHORT)

    table = df.pivot_table(
        index=["regime", "dataset", "rare_class", "rts"],
        columns="short",
        values="rare_f1",
    ).round(3)

    short_order = [SHORT[m] for m in METHOD_ORDER]
    cols = [c for c in short_order if c in table.columns]
    table = table[cols].reset_index()

    print("\nRare-class F1 on test set (nearest-prototype only, no gate / no marker):\n")
    print(table.to_markdown(index=False))

    out_path = ROOT / "outputs" / "_experimental" / "mahalanobis_sweep_summary.md"
    out_path.write_text("# Mahalanobis PoC sweep\n\n" + table.to_markdown(index=False) + "\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
