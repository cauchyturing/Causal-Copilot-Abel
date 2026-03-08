"""Tests for rule-based planner and data guards."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_copilot.core.guards import DataValidationError, validate_data
from causal_copilot.core.planner import detect_data_properties, rule_based_select


class TestDataProperties:
    def test_basic(self):
        df = pd.DataFrame(np.random.randn(100, 5), columns=list("ABCDE"))
        props = detect_data_properties(df)
        assert props["n_samples"] == 100
        assert props["n_features"] == 5
        assert props["is_time_series"] is False

    def test_time_series(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        df = pd.DataFrame(np.random.randn(100, 3), index=idx, columns=list("ABC"))
        props = detect_data_properties(df)
        assert props["is_time_series"] is True


def _all_available(name):
    """Mock: pretend every algorithm is available."""
    return True


class TestRuleBasedPlanner:
    """Tests for the decision tree logic (all algorithms mocked as available)."""

    def test_small_linear_selects_pc(self):
        props = {
            "n_samples": 500,
            "n_features": 10,
            "is_time_series": False,
            "likely_linear": True,
            "likely_gaussian": True,
            "has_missing": False,
            "missing_ratio": 0,
        }
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert decision.algorithm == "PC"

    def test_time_series_selects_pcmci(self):
        props = {
            "n_samples": 500,
            "n_features": 10,
            "is_time_series": True,
            "likely_linear": True,
            "likely_gaussian": True,
            "has_missing": False,
            "missing_ratio": 0,
        }
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert decision.algorithm == "PCMCI"

    def test_large_data_selects_notears(self):
        props = {
            "n_samples": 10000,
            "n_features": 50,
            "is_time_series": False,
            "likely_linear": True,
            "likely_gaussian": True,
            "has_missing": False,
            "missing_ratio": 0,
        }
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert decision.algorithm == "NOTEARSLinear"

    def test_non_gaussian_selects_lingam(self):
        props = {
            "n_samples": 500,
            "n_features": 10,
            "is_time_series": False,
            "likely_linear": True,
            "likely_gaussian": False,
            "has_missing": False,
            "missing_ratio": 0,
        }
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert decision.algorithm == "DirectLiNGAM"

    def test_decision_has_reason(self):
        props = {
            "n_samples": 500,
            "n_features": 10,
            "is_time_series": False,
            "likely_linear": False,
            "likely_gaussian": True,
            "has_missing": False,
            "missing_ratio": 0,
        }
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert len(decision.reason) > 0
        assert isinstance(decision.hyperparams, dict)


class TestPlannerFallbacks:
    """Verify planner falls back when an algorithm's deps are missing."""

    def _props(self, **overrides):
        base = {
            "n_samples": 500,
            "n_features": 10,
            "is_time_series": False,
            "likely_linear": True,
            "likely_gaussian": True,
            "has_missing": False,
            "missing_ratio": 0,
        }
        base.update(overrides)
        return base

    def test_notears_unavailable_falls_back_to_ges(self):
        """Large data should get NOTEARSLinear; if unavailable, fall back to GES."""
        props = self._props(n_samples=10000, n_features=50)
        with patch(
            "causal_copilot.core.planner._is_available",
            side_effect=lambda name: name != "NOTEARSLinear",
        ):
            decision = rule_based_select(props)
        assert decision.algorithm == "GES"
        assert "unavailable" in decision.reason.lower() or "falling back" in decision.reason.lower()

    def test_pcmci_unavailable_falls_back_to_granger(self):
        props = self._props(is_time_series=True)
        with patch(
            "causal_copilot.core.planner._is_available",
            side_effect=lambda name: name not in ("PCMCI",),
        ):
            decision = rule_based_select(props)
        assert decision.algorithm == "GrangerCausality"

    def test_all_timeseries_unavailable_falls_back_to_ges(self):
        props = self._props(is_time_series=True)
        with patch(
            "causal_copilot.core.planner._is_available",
            side_effect=lambda name: name not in ("PCMCI", "GrangerCausality"),
        ):
            decision = rule_based_select(props)
        assert decision.algorithm == "GES"

    def test_directlingam_unavailable_falls_back_to_icalingam(self):
        props = self._props(likely_gaussian=False)
        with patch(
            "causal_copilot.core.planner._is_available",
            side_effect=lambda name: name != "DirectLiNGAM",
        ):
            decision = rule_based_select(props)
        assert decision.algorithm == "ICALiNGAM"

    def test_all_available_no_fallback(self):
        """When all deps are present, original choices are preserved."""
        props = self._props(n_samples=10000, n_features=50)
        with patch("causal_copilot.core.planner._is_available", side_effect=_all_available):
            decision = rule_based_select(props)
        assert decision.algorithm == "NOTEARSLinear"


class TestDataValidation:
    def test_valid_data(self):
        df = pd.DataFrame(np.random.randn(100, 5))
        warnings = validate_data(df)
        assert isinstance(warnings, list)

    def test_empty_data(self):
        with pytest.raises(DataValidationError, match="empty"):
            validate_data(pd.DataFrame())

    def test_too_few_samples(self):
        with pytest.raises(DataValidationError, match="Too few samples"):
            validate_data(pd.DataFrame(np.random.randn(5, 3)))

    def test_too_few_features(self):
        with pytest.raises(DataValidationError, match="Too few features"):
            validate_data(pd.DataFrame(np.random.randn(100, 1)))

    def test_too_many_samples(self):
        with pytest.raises(DataValidationError, match="Too many samples"):
            validate_data(pd.DataFrame(np.random.randn(200_000, 3)))

    def test_non_numeric_warning(self):
        df = pd.DataFrame({"a": range(100), "b": range(100), "c": ["x"] * 100})
        warnings = validate_data(df)
        assert any("Non-numeric" in w for w in warnings)
