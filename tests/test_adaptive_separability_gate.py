import numpy as np
import pandas as pd

from src.rescue import (
    AdaptiveGatePolicy as CoreAdaptiveGatePolicy,
    adaptive_conformal_rescue,
    rescue_with_separability_gate,
    stable_adaptive_decision_seed,
)
from tools.analysis.adaptive_separability_gate import (
    AdaptiveGatePolicy,
    _paired_stratified_bootstrap_delta_f1,
    _stable_decision_seed,
    adaptive_separability_rescue,
    wilson_upper,
)


class AnalyticPrototype:
    """Latent column 0 is membership score; column 1 is rare rank."""

    rare_class = "rare"

    def __init__(self, separability=1.1):
        self.separability_ratio = separability

    def rare_membership_score(self, latent):
        return np.asarray(latent)[:, 0].astype(float)

    def rare_rank(self, latent):
        return np.asarray(latent)[:, 1].astype(int)

    def rank_candidate(self, latent, predicted_labels, max_rank=1):
        return (predicted_labels.to_numpy() != self.rare_class) & (
            self.rare_rank(latent) <= max_rank
        )


def _safe_case(n_nonrare=500, n_rare=50):
    # Non-rare rank 4 never enters the {1,2,3} candidate pool.  Rare misses are
    # rank 1 with high score and should be recovered in every OOF fold.
    val_true = pd.Series(["major"] * n_nonrare + ["rare"] * n_rare)
    val_pred = pd.Series(["major"] * (n_nonrare + n_rare))
    val_latent = np.column_stack(
        [
            np.r_[np.zeros(n_nonrare), np.ones(n_rare)],
            np.r_[np.full(n_nonrare, 4), np.ones(n_rare)],
        ]
    )
    test_pred = pd.Series(["major", "major"])
    test_latent = np.array([[1.0, 1.0], [0.0, 4.0]])
    return test_pred, val_pred, val_true, val_latent, test_latent


def test_wilson_upper_matches_safety_direction():
    assert wilson_upper(0, 500) < 0.01
    assert wilson_upper(10, 500) > 0.01


def test_paired_bootstrap_perfect_rescue_has_positive_lcb():
    y = pd.Series(["major"] * 100 + ["rare"] * 20)
    base = pd.Series(["major"] * 120)
    refined = pd.Series(["major"] * 100 + ["rare"] * 20)
    lcb, median, ucb = _paired_stratified_bootstrap_delta_f1(
        y,
        base,
        refined,
        rare="rare",
        reps=200,
        lower_alpha=0.05,
        seed=42,
    )
    assert lcb == median == ucb == 1.0


def test_low_s_safe_case_passes_and_is_deterministic():
    args = _safe_case()
    policy = AdaptiveGatePolicy(bootstrap_reps=200)
    first_pred, first = adaptive_separability_rescue(
        AnalyticPrototype(), *args, policy=policy, decision_seed=42
    )
    second_pred, second = adaptive_separability_rescue(
        AnalyticPrototype(), *args, policy=policy, decision_seed=42
    )
    assert first_pred.tolist() == ["rare", "major"]
    assert first_pred.equals(second_pred)
    assert first["adaptive_pass"] is True
    assert first["active_folds"] == 5
    assert first["oof_delta_f1_lcb"] > 0
    assert first["oof_ffr_wilson_upper"] <= policy.alpha
    assert first == second


def test_low_validation_support_abstains():
    args = _safe_case(n_rare=2)
    pred, summary = adaptive_separability_rescue(
        AnalyticPrototype(),
        *args,
        policy=AdaptiveGatePolicy(bootstrap_reps=100),
        decision_seed=42,
    )
    assert pred.equals(args[0])
    assert summary["adaptive_pass"] is False
    assert summary["reason"] == "adaptive_val_support"


def test_oof_unsafe_case_is_rejected():
    test_pred, val_pred, val_true, val_latent, test_latent = _safe_case()
    # Put 20 validation non-rare cells into the same high-score/rank-1 region.
    # Fold calibration should reject candidate ranks or the OOF Wilson check.
    val_latent[:20, 0] = 1.0
    val_latent[:20, 1] = 1.0
    pred, summary = adaptive_separability_rescue(
        AnalyticPrototype(),
        test_pred,
        val_pred,
        val_true,
        val_latent,
        test_latent,
        policy=AdaptiveGatePolicy(bootstrap_reps=200),
        decision_seed=42,
    )
    assert pred.equals(test_pred)
    assert summary["adaptive_pass"] is False
    assert summary["reason"] == "adaptive_audit_failed"


def test_gate_decision_does_not_depend_on_test_features():
    args = _safe_case()
    policy = AdaptiveGatePolicy(bootstrap_reps=200)
    _, first = adaptive_separability_rescue(
        AnalyticPrototype(), *args, policy=policy, decision_seed=42
    )
    changed_test_latent = np.array([[0.0, 4.0], [1.0, 1.0]])
    _, second = adaptive_separability_rescue(
        AnalyticPrototype(),
        args[0],
        args[1],
        args[2],
        args[3],
        changed_test_latent,
        policy=policy,
        decision_seed=42,
    )
    audit_keys = [
        "adaptive_pass",
        "active_folds",
        "oof_delta_f1_lcb",
        "oof_ffr_wilson_upper",
        "adaptive_reason",
    ]
    assert {k: first[k] for k in audit_keys} == {k: second[k] for k in audit_keys}


def test_core_adaptive_implementation_matches_frozen_experiment():
    args = _safe_case()
    frozen_pred, frozen = adaptive_separability_rescue(
        AnalyticPrototype(),
        *args,
        policy=AdaptiveGatePolicy(bootstrap_reps=200),
        decision_seed=42,
    )
    core_pred, core = adaptive_conformal_rescue(
        AnalyticPrototype(),
        *args,
        policy=CoreAdaptiveGatePolicy(bootstrap_reps=200),
        decision_seed=42,
    )
    assert core_pred.equals(frozen_pred)
    keys = [
        "adaptive_pass",
        "active_folds",
        "oof_delta_f1",
        "oof_delta_f1_lcb",
        "oof_ffr_wilson_upper",
        "chosen_rank",
        "n_rescued",
    ]
    assert {key: core[key] for key in keys} == {
        key: frozen[key] for key in keys
    }


def test_dispatcher_keeps_fixed_gate_as_explicit_control():
    args = _safe_case()
    pred, summary = rescue_with_separability_gate(
        AnalyticPrototype(separability=1.1), *args, gate_mode="fixed"
    )
    assert pred.equals(args[0])
    assert summary["separability_gate_mode"] == "fixed"
    assert summary["reason"] == "sep<1.3"


def test_core_decision_seed_matches_frozen_v1_convention():
    for rts in ["0.01", "0.05", "0.10", "all"]:
        assert stable_adaptive_decision_seed("dataset", 42, rts) == (
            _stable_decision_seed("dataset", 42, rts)
        )
    assert stable_adaptive_decision_seed("dataset", 42, 0.1) == (
        _stable_decision_seed("dataset", 42, "0.10")
    )
