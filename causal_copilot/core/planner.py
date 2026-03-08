"""Rule-based algorithm selection planner (offline, deterministic)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PlannerDecision:
    """Output of algorithm selection — typed, inspectable, serializable."""

    algorithm: str
    hyperparams: dict[str, Any]
    reason: str


def detect_data_properties(data: pd.DataFrame) -> dict[str, Any]:
    """Detect basic statistical properties of input data for algorithm selection."""
    n_samples, n_features = data.shape

    # Check for time-series indicators
    is_time_series = False
    if isinstance(data.index, pd.DatetimeIndex):
        is_time_series = True

    # Check linearity heuristic: correlation between feature pairs
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.shape[1] >= 2:
        corr = numeric_data.corr().abs().to_numpy(copy=True)
        np.fill_diagonal(corr, 0)
        avg_corr = float(corr.mean())
        likely_linear = avg_corr > 0.3  # heuristic
    else:
        likely_linear = True

    # Check for missing values
    has_missing = data.isnull().any().any()
    missing_ratio = data.isnull().sum().sum() / (n_samples * n_features) if n_samples > 0 else 0

    # Check gaussianity heuristic (simple: skewness-based)
    try:
        skewness = numeric_data.skew().abs().mean()
        likely_gaussian = skewness < 1.0
    except Exception:
        likely_gaussian = True

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "is_time_series": is_time_series,
        "likely_linear": likely_linear,
        "likely_gaussian": likely_gaussian,
        "has_missing": has_missing,
        "missing_ratio": missing_ratio,
    }


def _get_defaults(name: str) -> dict[str, Any]:
    """Read default hyperparams from the canonical registry."""
    from causal_copilot.algorithms.registry import REGISTRY

    return dict(REGISTRY[name].default_params)


def _is_available(name: str) -> bool:
    """Check whether an algorithm's deps are importable."""
    from causal_copilot.algorithms.registry import is_algorithm_available

    return is_algorithm_available(name)


def _make_decision(name: str, reason: str) -> PlannerDecision:
    return PlannerDecision(algorithm=name, hyperparams=_get_defaults(name), reason=reason)


def rule_based_select(properties: dict[str, Any]) -> PlannerDecision:
    """
    Select algorithm based on data properties using a simple decision tree.

    Each candidate is checked for dependency availability.  When the preferred
    algorithm is not importable the planner falls back to the next best choice,
    ultimately landing on GES (whose only dep, causal-learn, is bundled).

    Decision logic:
    1. Time-series -> PCMCI  (fallback: GrangerCausality -> GES)
    2. Non-gaussian errors -> DirectLiNGAM  (fallback: ICALiNGAM -> GES)
    3. Small-medium data + likely linear -> PC  (fallback: GES)
    4. Large data or many features -> NOTEARSLinear  (fallback: GES)
    5. Default fallback -> GES
    """
    n = properties["n_samples"]
    p = properties["n_features"]

    if properties["is_time_series"]:
        if _is_available("PCMCI"):
            return _make_decision(
                "PCMCI",
                "Time-series data detected — PCMCI handles temporal causal discovery.",
            )
        if _is_available("GrangerCausality"):
            return _make_decision(
                "GrangerCausality",
                "Time-series data detected — PCMCI unavailable, falling back to Granger causality.",
            )
        return _make_decision(
            "GES",
            "Time-series data but no time-series algorithm available — falling back to GES.",
        )

    if not properties["likely_gaussian"]:
        if p <= 50:
            if _is_available("DirectLiNGAM"):
                return _make_decision(
                    "DirectLiNGAM",
                    "Non-gaussian errors detected — DirectLiNGAM exploits non-gaussianity for identifiability.",
                )
            if _is_available("ICALiNGAM"):
                return _make_decision(
                    "ICALiNGAM",
                    "Non-gaussian errors detected — DirectLiNGAM unavailable, falling back to ICALiNGAM.",
                )
            # fall through to general rules below

    if properties["likely_linear"] and p <= 30 and n <= 5000:
        if _is_available("PC"):
            return _make_decision(
                "PC",
                "Small-medium linear data — PC is well-understood with strong theoretical guarantees.",
            )
        return _make_decision(
            "GES",
            "Small-medium linear data — PC unavailable, falling back to GES.",
        )

    if p > 30 or n > 5000:
        if _is_available("NOTEARSLinear"):
            return _make_decision(
                "NOTEARSLinear",
                "Large data or many features — NOTEARSLinear scales via continuous optimization.",
            )
        if _is_available("GES"):
            return _make_decision(
                "GES",
                "Large data — NOTEARSLinear unavailable, falling back to GES.",
            )

    # Default fallback
    return _make_decision(
        "GES",
        "General-purpose score-based method — GES is a robust default for medium-sized data.",
    )
