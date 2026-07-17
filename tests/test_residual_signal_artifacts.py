from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "residual_signal" / "v1"


def test_residual_signal_run_ledger_and_provenance_boundaries():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})

    assert len(runs) == 96
    assert not runs.duplicated(["dataset", "seed", "rare_train_size"]).any()
    assert runs["status"].eq("success").all()
    assert runs["current_replay_available"].eq(True).all()
    assert (~runs["historical_cell_identity_available"].astype(bool)).sum() == 11
    assert runs["scANVI_probability_available"].eq(True).all()
    assert (
        runs[
            [
                "n_baseline_correct_rare",
                "n_true_rescued_rare",
                "n_unrescued_rare",
                "n_non_target",
            ]
        ].sum(axis=1)
        == runs["n_test"]
    ).all()


def test_cell_level_groups_metrics_and_false_rescue_subset():
    cells = pd.read_parquet(OUT / "cell_level.parquet")

    assert len(cells) > 0
    assert cells["cell_id"].notna().all()
    assert not cells.duplicated(["dataset", "seed", "rare_train_size", "cell_id"]).any()
    assert (
        cells["primary_group"]
        .isin(
            [
                "baseline_correct_rare",
                "true_rescued_rare",
                "unrescued_rare",
                "non_target",
            ]
        )
        .all()
    )
    assert cells.loc[cells["false_rescue"], "primary_group"].eq("non_target").all()
    metric_columns = [
        "rare_membership_score",
        "rare_rank",
        "rare_standardized_distance",
        "standardized_prototype_margin",
        "rare_prototype_distance",
        "nearest_nonrare_distance",
        "prototype_margin",
        "scANVI_rare_probability",
    ]
    assert np.isfinite(cells[metric_columns].to_numpy(dtype=float)).all()
    assert cells["scANVI_rare_probability"].between(0, 1).all()


def test_contrast_artifacts_have_frozen_grid_and_valid_effects():
    contrasts = pd.read_csv(OUT / "tables" / "group_contrasts.csv")
    summary = pd.read_csv(OUT / "summary.csv", dtype={"rare_train_size": str})

    assert set(contrasts["contrast"]) == {
        "H1_baseline_correct_vs_true_rescued",
        "H2_true_rescued_vs_unrescued",
        "H3a_true_rescued_vs_non_target",
        "H3b_true_rescued_vs_closest_competitor",
    }
    finite_delta = contrasts["raw_cliffs_delta"].dropna()
    assert finite_delta.between(-1, 1).all()
    oriented = contrasts["oriented_cliffs_delta"].dropna()
    assert oriented.between(-1, 1).all()
    assert summary["dataset_direction_rate"].between(0, 1).all()
    assert (OUT / "figures" / "selection_pathway_distributions.png").exists()
    assert (OUT / "figures" / "prototype_margin_contrasts.pdf").exists()
