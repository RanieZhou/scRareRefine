import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "supplementary_ablation" / "v1"


def test_supplementary_ablation_closed_ledger_and_metrics():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    assert len(runs) == 960
    assert not runs.duplicated(["dataset", "seed", "rare_train_size", "variant"]).any()
    assert runs["status"].eq("success").all()
    assert np.allclose(runs["incremental_fpr"], runs["rescue_ffr"])
    assert (runs["all_rescues"] == runs["true_rescues"] + runs["false_rescues"]).all()
    rescued = runs[runs["all_rescues"] > 0]
    assert np.allclose(
        rescued["rescue_precision"],
        rescued["true_rescues"] / rescued["all_rescues"],
    )
    assert runs.loc[runs["all_rescues"].eq(0), "rescue_precision"].isna().all()
    assert (runs["alpha_violation"] == (runs["incremental_fpr"] > runs["alpha"])).all()


def test_duplicate_variants_are_exactly_consistent():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    keys = ["dataset", "seed", "rare_train_size"]
    columns = [
        "rare_f1",
        "rare_recall",
        "true_rescues",
        "false_rescues",
        "incremental_fpr",
        "abstain",
        "chosen_rank",
    ]
    for left, right in (
        ("fixed_rank_1", "rank_1"),
        ("full_method", "adaptive_rank"),
    ):
        a = runs[runs["variant"].eq(left)].set_index(keys)[columns].sort_index()
        b = runs[runs["variant"].eq(right)].set_index(keys)[columns].sort_index()
        pd.testing.assert_frame_equal(a, b)


def test_effective_budget_tables_and_manifest():
    identity = pd.read_csv(OUT / "tables" / "identity_collapsed.csv")
    seed_units = pd.read_csv(OUT / "tables" / "seed_count_variant_units.csv")
    summary = pd.read_csv(OUT / "summary.csv")
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert len(identity) == 870
    assert identity["n_folded_nominal_budgets"].sum() == 960
    assert len(seed_units) == 870
    assert len(summary) == 290
    assert manifest["expected_run_variant_rows"] == 960
    assert manifest["status_counts"] == {"success": 960}
    for suffix in ("png", "pdf"):
        figure = OUT / "figures" / f"supplementary_ablation.{suffix}"
        assert figure.exists() and figure.stat().st_size > 0
