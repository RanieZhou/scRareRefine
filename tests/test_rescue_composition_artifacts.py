from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "rescue_composition" / "v1"


def test_rescue_composition_artifact_ledger_and_invariants():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})

    assert len(runs) == 96
    assert not runs.duplicated(["dataset", "seed", "rare_train_size"]).any()
    assert runs["status"].eq("success").all()
    assert (
        runs["baseline_missed_rare"]
        == runs["true_rescues"] + runs["remaining_missed_rare"]
    ).all()
    assert (runs["all_rescues"] == runs["true_rescues"] + runs["false_rescues"]).all()
    rescued = runs[runs["all_rescues"] > 0]
    assert np.allclose(rescued["rescue_precision"] + rescued["rescue_fdp"], 1.0)
    assert np.allclose(
        runs["incremental_fpr"], runs["false_rescues"] / runs["true_nonrare"]
    )
    assert np.allclose(runs["incremental_fpr"], runs["rescue_ffr"])


def test_historical_rows_do_not_invent_decision_metadata():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    historical = runs[runs["reconstruction_basis"] == "historical_counts_only"]

    assert len(historical) == 11
    assert (
        historical[["abstain", "chosen_rank", "tau", "raw_candidates"]]
        .isna()
        .all()
        .all()
    )
    assert (
        historical["abstain_reason"]
        .eq("unavailable_historical_decision_metadata")
        .all()
    )


def test_notes_numbers_reconstruct_from_run_ledger():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    scarce = runs[runs["rare_train_size"].isin(["0.01", "0.05", "0.10"])]
    true_rescues = int(scarce["true_rescues"].sum())
    false_rescues = int(scarce["false_rescues"].sum())
    precision = true_rescues / (true_rescues + false_rescues)
    notes = (OUT / "analysis_notes.md").read_text(encoding="utf-8")

    assert f"true rescues={true_rescues}, false rescues={false_rescues}" in notes
    assert f"={precision:.4f}." in notes
    assert f"={scarce['incremental_fpr'].max():.6f}." in notes
