import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.analysis.label_budget import (
    canonical_id_identity,
    classify_collapses,
    normalize_bool,
)


def test_canonical_id_identity_uses_frozen_serialization():
    ordered, serialized, digest = canonical_id_identity(pd.Series(["细胞-2", "cell-1"]))
    assert ordered == ["cell-1", "细胞-2"]
    assert serialized == '["cell-1","细胞-2"]'
    assert digest == hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_canonical_id_identity_rejects_missing_and_duplicates():
    with pytest.raises(ValueError, match="missing"):
        canonical_id_identity(pd.Series(["a", None]))
    with pytest.raises(ValueError, match="duplicates"):
        canonical_id_identity(pd.Series(["a", "a"]))


def test_normalize_bool_is_strict():
    assert normalize_bool(pd.Series([True, False])).tolist() == [True, False]
    assert normalize_bool(pd.Series(["True", "0", "1", "false"])).tolist() == [
        True,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="invalid"):
        normalize_bool(pd.Series(["yes"]))


def test_collapse_classification_distinguishes_identity_from_count():
    base = {
        "dataset": "d",
        "rare_class": "r",
        "seed": 42,
        "actual_training_labeled_rare_count": 5,
        "split_hash": "s",
        "status": "success",
    }
    frame = pd.DataFrame(
        [
            {**base, "rare_train_size": "0.01", "labeled_rare_id_sha256": "a"},
            {**base, "rare_train_size": "0.05", "labeled_rare_id_sha256": "a"},
            {**base, "rare_train_size": "0.10", "labeled_rare_id_sha256": "b"},
        ]
    )
    classified, count_table = classify_collapses(frame)
    assert classified["count_collapse"].all()
    assert classified["identity_collapse"].tolist() == [True, True, False]
    assert classified["collapse_class"].tolist() == [
        "identity_collapse",
        "identity_collapse",
        "count_only_collision",
    ]
    assert json.loads(count_table.iloc[0]["identity_hashes"]) == ["a", "b"]
    assert bool(count_table.iloc[0]["count_only_collision"])
