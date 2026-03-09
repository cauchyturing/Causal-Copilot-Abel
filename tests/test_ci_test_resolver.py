"""Tests for deterministic CI test resolution."""

from unittest.mock import MagicMock


def _make_stats(**kwargs):
    """Create a mock Statistics object with given attributes."""
    defaults = {
        "missingness": False,
        "data_type": "Continuous",
        "linearity": True,
        "gaussian_error": True,
        "sample_size": 500,
        "feature_number": 10,
        "time_series": False,
    }
    defaults.update(kwargs)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestCITestResolver:
    def test_missing_data_returns_mv_fisherz(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(missingness=True)
        assert resolve_ci_test(stats) == "mv_fisherz"

    def test_discrete_returns_chisq(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(data_type="Discrete")
        assert resolve_ci_test(stats) == "chisq"

    def test_linear_continuous_returns_fisherz(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(linearity=True, data_type="Continuous")
        assert resolve_ci_test(stats) == "fisherz"

    def test_nonlinear_small_returns_kci(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(linearity=False, sample_size=500, feature_number=8)
        assert resolve_ci_test(stats) == "kci"

    def test_nonlinear_medium_returns_rcit(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(linearity=False, sample_size=5000, feature_number=50)
        assert resolve_ci_test(stats) == "rcit"

    def test_nonlinear_large_returns_fastkci(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(linearity=False, sample_size=20000, feature_number=150)
        assert resolve_ci_test(stats) == "fastkci"

    def test_binary_returns_chisq(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(data_type="Binary")
        assert resolve_ci_test(stats) == "chisq"

    def test_mixture_returns_kci(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(data_type="Mixture")
        assert resolve_ci_test(stats) == "kci"

    def test_mixed_returns_kci(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test

        stats = _make_stats(data_type="Mixed")
        assert resolve_ci_test(stats) == "kci"
