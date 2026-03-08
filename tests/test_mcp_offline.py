"""Tests for offline (rule-based) algorithm selection and default HP."""
from unittest.mock import MagicMock


def _make_stats(**kwargs):
    defaults = {
        "time_series": False,
        "gaussian_error": True,
        "linearity": True,
        "missingness": False,
        "data_type": "Continuous",
        "sample_size": 500,
        "feature_number": 10,
    }
    defaults.update(kwargs)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestSelectAlgorithmOffline:
    def test_timeseries_returns_pcmci(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        assert select_algorithm_offline(_make_stats(time_series=True)) == "PCMCI"

    def test_non_gaussian_linear_returns_directlingam(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        stats = _make_stats(gaussian_error=False, linearity=True)
        assert select_algorithm_offline(stats) == "DirectLiNGAM"

    def test_nonlinear_returns_ges(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        stats = _make_stats(linearity=False)
        assert select_algorithm_offline(stats) == "GES"

    def test_linear_gaussian_returns_pc(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        stats = _make_stats(linearity=True, gaussian_error=True)
        assert select_algorithm_offline(stats) == "PC"

    def test_default_returns_pc(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        # All defaults → PC
        assert select_algorithm_offline(_make_stats()) == "PC"

    def test_timeseries_takes_priority(self):
        from causal_copilot.mcp.offline import select_algorithm_offline

        # Even if non-linear + non-Gaussian, time-series wins
        stats = _make_stats(time_series=True, linearity=False, gaussian_error=False)
        assert select_algorithm_offline(stats) == "PCMCI"


class TestGetDefaultHP:
    def test_pc_defaults(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("PC", _make_stats())
        assert hp["alpha"] == 0.05
        assert hp["indep_test"] == "fisherz"
        assert hp["stable"] is True

    def test_ges_defaults(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("GES", _make_stats())
        assert hp["score_func"] == "local_score_BIC"

    def test_directlingam_defaults(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("DirectLiNGAM", _make_stats())
        assert hp["measure"] == "pwling"

    def test_pcmci_defaults(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("PCMCI", _make_stats())
        assert hp["tau_max"] == 3
        assert hp["pc_alpha"] == 0.05

    def test_fci_defaults(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("FCI", _make_stats())
        assert hp["alpha"] == 0.05

    def test_pc_with_missing_adds_mvpc(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("PC", _make_stats(missingness=True))
        assert hp["mvpc"] is True

    def test_unknown_algorithm_returns_empty(self):
        from causal_copilot.mcp.offline import get_default_hp

        hp = get_default_hp("UnknownAlgo", _make_stats())
        assert hp == {}
