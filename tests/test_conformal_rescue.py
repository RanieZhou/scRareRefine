import numpy as np
import pandas as pd

from src.rescue import conformal_rescue
from tools.analysis.ablation import _conformal_with_overrides


class FakePrototype:
    def __init__(
        self, val_scores, test_scores, val_ranks, test_ranks, separability=2.0
    ):
        self.rare_class = "rare"
        self.separability_ratio = separability
        self._scores = {
            0: np.asarray(val_scores, dtype=float),
            1: np.asarray(test_scores, dtype=float),
        }
        self._ranks = {
            0: np.asarray(val_ranks, dtype=int),
            1: np.asarray(test_ranks, dtype=int),
        }

    @staticmethod
    def _split(query_latent):
        return int(np.asarray(query_latent)[0, 0])

    def rare_membership_score(self, query_latent):
        return self._scores[self._split(query_latent)].copy()

    def rare_rank(self, query_latent):
        return self._ranks[self._split(query_latent)].copy()

    def rank_candidate(self, query_latent, predicted_labels, max_rank=1):
        ranks = self.rare_rank(query_latent)
        return (predicted_labels.to_numpy() != self.rare_class) & (ranks <= max_rank)


def _case(*, rare_rank=1, nonrare_rank=4, separability=2.0):
    n_nonrare = 500
    n_rare = 3
    val_true = pd.Series(["major"] * n_nonrare + ["rare"] * n_rare)
    val_pred = pd.Series(["major"] * (n_nonrare + n_rare))
    val_scores = np.array([0.0] * n_nonrare + [1.0] * n_rare)
    val_ranks = np.array([nonrare_rank] * n_nonrare + [rare_rank] * n_rare)
    test_pred = pd.Series(["major", "major"])
    proto = FakePrototype(
        val_scores,
        test_scores=[1.0, 0.0],
        val_ranks=val_ranks,
        test_ranks=[rare_rank, nonrare_rank],
        separability=separability,
    )
    val_latent = np.zeros((len(val_true), 1))
    test_latent = np.ones((len(test_pred), 1))
    return proto, test_pred, val_pred, val_true, val_latent, test_latent


def test_low_separability_abstains():
    args = _case(separability=1.2)
    final, summary = conformal_rescue(*args)
    assert final.equals(args[1])
    assert summary["abstain"] is True
    assert summary["reason"] == "sep<1.3"
    assert summary["chosen_rank"] == 0


def test_zero_validation_misses_abstains():
    proto, test_pred, val_pred, val_true, val_latent, test_latent = _case()
    val_pred.iloc[-3:] = "rare"
    final, summary = conformal_rescue(
        proto, test_pred, val_pred, val_true, val_latent, test_latent
    )
    assert final.equals(test_pred)
    assert summary["abstain"] is True
    assert summary["reason"] == "val baseline 零漏判稀有"


def test_no_feasible_rank_strictly_abstains():
    proto, test_pred, val_pred, val_true, val_latent, test_latent = _case(
        rare_rank=1, nonrare_rank=1
    )
    final, summary = conformal_rescue(
        proto, test_pred, val_pred, val_true, val_latent, test_latent
    )
    assert final.equals(test_pred)
    assert summary["abstain"] is True
    assert summary["reason"] == "no_feasible_rank"
    assert summary["chosen_rank"] == 0
    assert summary["n_candidate"] == 0
    assert summary["n_rescued"] == 0


def test_ablation_mirror_no_feasible_rank_strictly_abstains():
    proto, test_pred, val_pred, val_true, val_latent, test_latent = _case(
        rare_rank=1, nonrare_rank=1
    )
    final, summary = _conformal_with_overrides(
        proto,
        test_pred,
        val_pred,
        val_true,
        val_latent,
        test_latent,
        rank_grid=(1, 2, 3),
    )
    assert final.equals(test_pred)
    assert summary["abstain"] is True
    assert summary["reason"] == "no_feasible_rank"
    assert summary["chosen_rank"] == 0


def test_adaptive_rank_maximizes_validation_f1():
    args = _case(rare_rank=2)
    final, summary = conformal_rescue(*args)
    assert summary["abstain"] is False
    assert summary["chosen_rank"] == 2
    assert final.tolist() == ["rare", "major"]


def test_rank_tie_prefers_smaller_rank():
    args = _case(rare_rank=1)
    _, summary = conformal_rescue(*args)
    assert summary["abstain"] is False
    assert summary["chosen_rank"] == 1


def test_tau_uses_validation_nonrare_scores_only():
    proto, test_pred, val_pred, val_true, val_latent, test_latent = _case()
    _, first = conformal_rescue(
        proto, test_pred, val_pred, val_true, val_latent, test_latent
    )
    proto._scores[1] = np.array([1000.0, -1000.0])
    _, second = conformal_rescue(
        proto, test_pred, val_pred, val_true, val_latent, test_latent
    )
    assert first["tau"] == second["tau"] == 0.0


def test_external_test_labels_cannot_change_rescue_decisions():
    args = _case(rare_rank=2)
    first_pred, first_summary = conformal_rescue(*args)
    test_true = pd.Series(["rare", "major"])
    shuffled_test_true = test_true.iloc[::-1].reset_index(drop=True)
    second_pred, second_summary = conformal_rescue(*args)
    assert not test_true.equals(shuffled_test_true)
    assert first_pred.equals(second_pred)
    assert first_summary == second_summary
