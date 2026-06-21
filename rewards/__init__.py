"""Reward specs, predicates, and composable reward programs for hudathon VLA tasks."""

from .engine import RewardComputation, compute_pick_reward
from .predicates import Baseline, SimView
from .program import RewardProgram, SuccessClause, Term, to_program
from .spec import RewardSpec, RewardWeights

__all__ = [
    "Baseline",
    "RewardComputation",
    "RewardProgram",
    "RewardSpec",
    "RewardWeights",
    "SimView",
    "SuccessClause",
    "Term",
    "compute_pick_reward",
    "to_program",
]
