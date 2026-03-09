"""Tests for deterministic score function resolution."""

from unittest.mock import MagicMock


def _make_stats(**kwargs):
    defaults = {
        "data_type": "Continuous",
        "linearity": True,
    }
    defaults.update(kwargs)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestScoreResolver:
    def test_discrete_returns_bdeu(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(data_type="Discrete")
        assert resolve_score_func(stats, "GES") == "local_score_BDeu"

    def test_nonlinear_returns_cv_general(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(linearity=False, data_type="Continuous")
        assert resolve_score_func(stats, "GES") == "local_score_CV_general"

    def test_linear_ges_returns_bic(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(linearity=True)
        assert resolve_score_func(stats, "GES") == "local_score_BIC"

    def test_linear_grasp_returns_bic_from_cov(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(linearity=True)
        assert resolve_score_func(stats, "GRaSP") == "local_score_BIC_from_cov"

    def test_category_returns_bdeu(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(data_type="Category")
        assert resolve_score_func(stats, "FGES") == "local_score_BDeu"

    def test_mixture_returns_cv_general(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(data_type="Mixture")
        assert resolve_score_func(stats, "GES") == "local_score_CV_general"

    def test_mixed_returns_cv_general(self):
        from causal_discovery.score_resolver import resolve_score_func

        stats = _make_stats(data_type="Mixed")
        assert resolve_score_func(stats, "GES") == "local_score_CV_general"
