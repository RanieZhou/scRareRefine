"""审计正式缓存中是否触发 no_feasible_rank 严格弃权。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.rescue import PrototypeRescuer, conformal_rescue  # noqa: E402
from src.utils import load_config, make_run_dir, parse_rare_train_size  # noqa: E402


CONFIGS = [
    "configs/immune_dc.yaml",
    "configs/pancreas_baron.yaml",
    "configs/pancreas_integrated.yaml",
    "configs/tabula_lung_endo.yaml",
    "configs/tabula_sapiens_stomach.yaml",
    "configs/tabula_small_intestine.yaml",
]
SEEDS = (42, 43, 44)
RARE_TRAIN_SIZES = ("0.01", "0.05", "0.10", "all")


def _latent(df: pd.DataFrame):
    return df[[c for c in df.columns if c.startswith("latent_")]].to_numpy()


def main() -> None:
    rows = []
    for config_path in CONFIGS:
        config = load_config(ROOT / config_path)
        experiment = config["experiment"]
        rare = experiment["rare_class"]
        split_mode = experiment.get("split_mode", "batch_heldout")
        dataset = config["dataset"]["name"]

        for seed in SEEDS:
            for rts in RARE_TRAIN_SIZES:
                run_dir = ROOT / make_run_dir(
                    config, split_mode, seed, rare, parse_rare_train_size(rts)
                )
                embeddings = run_dir / "embeddings"
                required = [
                    embeddings / f"{split}_{kind}.csv"
                    for split in ("train", "validation", "test")
                    for kind in ("predictions", "latent")
                ]
                if not all(path.exists() for path in required):
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "rare_train_size": rts,
                            "status": "missing_cache",
                            "abstain": True,
                            "reason": "missing_cache",
                            "chosen_rank": 0,
                        }
                    )
                    continue

                predictions = {
                    split: pd.read_csv(
                        embeddings / f"{split}_predictions.csv", low_memory=False
                    )
                    for split in ("train", "validation", "test")
                }
                latents = {
                    split: pd.read_csv(embeddings / f"{split}_latent.csv")
                    for split in ("train", "validation", "test")
                }
                train_predictions = predictions["train"]
                proto = PrototypeRescuer(rare)
                proto.fit(
                    _latent(latents["train"]),
                    train_predictions["true_label"].astype(str),
                    train_predictions["is_labeled_for_scanvi"].astype(bool).to_numpy(),
                )
                _, summary = conformal_rescue(
                    proto,
                    predictions["test"]["predicted_label"],
                    predictions["validation"]["predicted_label"],
                    predictions["validation"]["true_label"],
                    _latent(latents["validation"]),
                    _latent(latents["test"]),
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "rare_train_size": rts,
                        "status": "ok",
                        "separability": proto.separability_ratio,
                        "val_missed": summary.get("val_missed"),
                        "abstain": summary["abstain"],
                        "reason": summary["reason"],
                        "chosen_rank": summary["chosen_rank"],
                        "tau": summary["tau"],
                        "n_candidate": summary["n_candidate"],
                        "n_rescued": summary["n_rescued"],
                    }
                )

    output_dir = ROOT / "results" / "provenance"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(output_dir / "no_feasible_rank_audit.csv", index=False)

    ok = audit[audit["status"] == "ok"]
    no_feasible = ok[ok["reason"] == "no_feasible_rank"]
    missing = audit[audit["status"] == "missing_cache"]
    reason_counts = ok["reason"].replace("", "rescue_applied").value_counts()
    report = [
        "# No-feasible-rank audit",
        "",
        f"- Audited configurations: {len(audit)}",
        f"- Cache-complete configurations: {len(ok)}",
        f"- Missing caches: {len(missing)}",
        f"- Newly strict no-feasible-rank abstentions: {len(no_feasible)}",
        "",
        "## Decision outcomes",
        "",
        "| outcome | count |",
        "|---|---:|",
    ]
    report.extend(f"| {reason} | {count} |" for reason, count in reason_counts.items())
    if not no_feasible.empty:
        report.extend(
            [
                "",
                "## Affected configurations",
                "",
                no_feasible[
                    ["dataset", "seed", "rare_train_size", "separability", "val_missed"]
                ].to_markdown(index=False),
            ]
        )
    (output_dir / "no_feasible_rank_audit.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
