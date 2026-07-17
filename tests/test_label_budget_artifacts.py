import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "label_budget" / "v1"


def test_label_budget_closed_ledger_and_ratios():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    assert len(runs) == 96
    assert not runs.duplicated(["dataset", "seed", "rare_train_size"]).any()
    assert runs["status"].eq("success").all()
    assert runs["identity_status"].eq("verified").all()
    assert (
        runs["labeled_rare_id_sha256"]
        .map(lambda value: bool(re.fullmatch(r"[0-9a-f]{64}", str(value))))
        .all()
    )
    assert (
        runs["actual_training_labeled_rare_count"]
        == runs["expected_training_labeled_rare_count"]
    ).all()
    assert np.allclose(
        runs["training_rare_label_fraction"],
        runs["actual_training_labeled_rare_count"] / runs["train_rare_pool"],
    )
    assert np.allclose(
        runs["training_rare_share_of_all_split_rare"],
        runs["actual_training_labeled_rare_count"] / runs["all_split_rare"],
    )
    assert np.allclose(
        runs["total_supervised_rare_share_of_all_split_rare"],
        runs["total_rare_supervision"] / runs["all_split_rare"],
    )
    assert np.allclose(
        runs["training_rare_label_share_of_training_cells"],
        runs["actual_training_labeled_rare_count"] / runs["all_training_cells"],
    )
    assert (
        runs["all_split_rare"]
        == runs["train_rare_pool"] + runs["validation_rare"] + runs["test_rare"]
    ).all()


def test_identity_hash_reconstructs_from_exact_serialized_ids():
    runs = pd.read_csv(OUT / "run_level.csv", dtype={"rare_train_size": str})
    for row in runs.itertuples(index=False):
        expected = hashlib.sha256(row.labeled_rare_id_json.encode("utf-8")).hexdigest()
        assert expected == row.labeled_rare_id_sha256
        assert (
            len(json.loads(row.labeled_rare_id_json))
            == row.actual_training_labeled_rare_count
        )


def test_collapsed_tables_preserve_frozen_uniqueness_rules():
    identity = pd.read_csv(OUT / "tables" / "identity_collapsed.csv")
    units = pd.read_csv(OUT / "tables" / "seed_count_units.csv")
    identity_keys = [
        "dataset",
        "rare_class",
        "seed",
        "split_hash",
        "actual_training_labeled_rare_count",
        "labeled_rare_id_sha256",
    ]
    unit_keys = [
        "dataset",
        "rare_class",
        "actual_training_labeled_rare_count",
        "seed",
    ]
    assert not identity.duplicated(identity_keys).any()
    assert not units.duplicated(unit_keys).any()
    assert identity["n_folded_nominal_budgets"].sum() == 96
    assert (units["n_identity_runs"] >= 1).all()


def test_manifest_and_data_figures_exist():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_configurations"] == 96
    assert manifest["status_counts"] == {"success": 96}
    for name in (
        "rare_label_budget_accounting.png",
        "rare_label_budget_accounting.pdf",
    ):
        path = OUT / "figures" / name
        assert path.exists() and path.stat().st_size > 0
