from scrare.evaluation.posthoc import METHOD_ORDER


def test_method_order_matches_design() -> None:
    assert METHOD_ORDER == [
        "baseline",
        "baseline_plus_prototype",
        "baseline_plus_prototype_plus_marker",
        "baseline_plus_fusion",
    ]
