"""Offline (rule-based) algorithm selection and default hyperparameters.

Used as fallback when LLM-based selection (Filter → Reranker) is unavailable
or fails.  Pure heuristic — no API calls.
"""
from __future__ import annotations


def select_algorithm_offline(statistics) -> str:
    """Pick a sensible algorithm based on dataset statistics alone.

    Decision tree (simplified):
      time-series → PCMCI+
      non-Gaussian errors → DirectLiNGAM
      non-linear → GES (robust default)
      linear + Gaussian + missing → PC (mvpc mode)
      linear + Gaussian → PC
      else → PC
    """
    ts = getattr(statistics, "time_series", False)
    if ts:
        return "PCMCI+"

    gaussian = getattr(statistics, "gaussian_error", True)
    linear = getattr(statistics, "linearity", True)

    if not gaussian and linear:
        return "DirectLiNGAM"

    if not linear:
        return "GES"

    return "PC"


def get_default_hp(algorithm: str, statistics) -> dict:
    """Return conservative default hyperparameters for *algorithm*."""
    missing = getattr(statistics, "missingness", False)

    defaults: dict = {
        "PC": {"alpha": 0.05, "indep_test": "fisherz", "stable": True},
        "GES": {"score_func": "local_score_BIC"},
        "DirectLiNGAM": {"measure": "pwling"},
        "PCMCI+": {"tau_max": 3, "pc_alpha": 0.05},
        "FCI": {"alpha": 0.05, "indep_test": "fisherz"},
    }

    hp = dict(defaults.get(algorithm, {}))

    if algorithm == "PC" and missing:
        hp["mvpc"] = True

    return hp
