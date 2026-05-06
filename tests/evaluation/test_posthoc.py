from scrare.evaluation.posthoc import METHOD_LABELS, METHOD_ORDER


def test_method_order_matches_design() -> None:
    assert METHOD_ORDER == [
        "baseline",
        "baseline_plus_prototype",
        "baseline_plus_prototype_gate",
        "baseline_plus_prototype_gate_plus_marker",
        "baseline_plus_fusion",
    ]


def test_method_labels_match_design() -> None:
    assert METHOD_LABELS == {
        "baseline": "scANVI baseline",
        "baseline_plus_prototype": "prototype candidate",
        "baseline_plus_prototype_gate": "prototype rank1 gate",
        "baseline_plus_prototype_gate_plus_marker": "validation-tuned marker",
        "baseline_plus_fusion": "fusion",
    }
