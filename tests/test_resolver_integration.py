"""Integration tests for resolver wiring into HP selector."""
from unittest.mock import MagicMock
import pytest


def _make_global_state(linearity=True, data_type="Continuous", missingness=False,
                       sample_size=500, feature_number=10, algorithm="PC"):
    gs = MagicMock()
    gs.statistics.linearity = linearity
    gs.statistics.data_type = data_type
    gs.statistics.missingness = missingness
    gs.statistics.sample_size = sample_size
    gs.statistics.feature_number = feature_number
    gs.statistics.gaussian_error = linearity
    gs.statistics.time_series = False
    gs.algorithm.selected_algorithm = algorithm
    gs.algorithm.algorithm_arguments = {"indep_test": "fisherz"}
    return gs


class TestResolverOverride:
    def test_nonlinear_pc_overrides_to_kci(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test
        stats = _make_global_state(linearity=False, sample_size=500, feature_number=5).statistics
        result = resolve_ci_test(stats)
        assert result == "kci"

    def test_discrete_ges_overrides_to_bdeu(self):
        from causal_discovery.score_resolver import resolve_score_func
        stats = _make_global_state(data_type="Discrete").statistics
        result = resolve_score_func(stats, "GES")
        assert result == "local_score_BDeu"

    def test_missing_data_overrides_to_mv_fisherz(self):
        from causal_discovery.ci_test_resolver import resolve_ci_test
        stats = _make_global_state(missingness=True).statistics
        result = resolve_ci_test(stats)
        assert result == "mv_fisherz"
