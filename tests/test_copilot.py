"""Integration tests for CausalCopilot.analyze() — the main entry point."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_copilot.copilot import CausalCopilot, _build_graph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    """50-row, 4-column numeric DataFrame — well within validation limits."""
    rng = np.random.default_rng(0)
    n = 50
    x = rng.normal(size=n)
    y = 2 * x + rng.normal(size=n) * 0.3
    z = x + y + rng.normal(size=n) * 0.5
    w = rng.normal(size=n)
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": w})


@pytest.fixture
def csv_path(simple_df, tmp_path):
    """Write simple_df to a CSV and return its path."""
    p = tmp_path / "test_data.csv"
    simple_df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_planner(self):
        c = CausalCopilot()
        assert c.planner == "rule"

    def test_invalid_planner_raises(self):
        with pytest.raises(ValueError, match="Unknown planner"):
            CausalCopilot(planner="bogus")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


class TestDataLoading:
    def test_file_not_found(self):
        result = CausalCopilot().analyze("/nonexistent/path.csv")
        assert result.status == "failed"
        assert "not found" in result.summary.lower()

    def test_csv_loading(self, csv_path):
        # Mock the algorithm to avoid needing real wrappers
        with _mock_algorithm():
            result = CausalCopilot().analyze(csv_path, seed=0)
        assert result.status == "ok"

    def test_dataframe_input(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"

    def test_unsupported_type(self):
        result = CausalCopilot().analyze(42)
        assert result.status == "failed"
        assert "Unsupported" in result.summary

    def test_malformed_csv(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02\x03")
        result = CausalCopilot().analyze(bad)
        # Should fail gracefully, not raise
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Validation & cleaning
# ---------------------------------------------------------------------------


class TestValidationAndCleaning:
    def test_too_few_numeric_columns(self):
        df = pd.DataFrame({"a": [1] * 20, "b": ["x"] * 20})
        result = CausalCopilot().analyze(df)
        assert result.status == "failed"

    def test_constant_columns_dropped(self, simple_df):
        simple_df["const"] = 7.0
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert any("constant" in w.lower() for w in result.warnings)

    def test_nan_rows_dropped(self, simple_df):
        simple_df.iloc[0, 0] = np.nan
        simple_df.iloc[1, 1] = np.nan
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert any("missing" in w.lower() for w in result.warnings)

    def test_all_nan_fails(self):
        df = pd.DataFrame({"a": [np.nan] * 20, "b": [np.nan] * 20})
        result = CausalCopilot().analyze(df)
        assert result.status == "failed"

    def test_missing_values_imputed(self):
        # 20 rows, 12 have NaN — pipeline imputes via MICE instead of dropping
        vals = [float(i) for i in range(8)] + [np.nan] * 12
        df = pd.DataFrame(
            {
                "a": vals,
                "b": [float(i) for i in range(20)],
            }
        )
        with _mock_algorithm():
            result = CausalCopilot().analyze(df)
        assert result.status == "ok"
        assert any("missing" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Algorithm selection & execution
# ---------------------------------------------------------------------------


class TestAlgorithmSelection:
    def test_forced_algorithm(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", seed=0)
        assert result.status == "ok"
        assert result.provenance.algorithm == "PC"
        assert "User specified" in result.algorithm_selection_reason

    def test_unknown_algorithm(self, simple_df):
        result = CausalCopilot().analyze(simple_df, algorithm="FakeAlgo")
        assert result.status == "failed"
        assert "FakeAlgo" in result.summary

    def test_planner_override(self, simple_df):
        result = CausalCopilot().analyze(simple_df, planner="bogus")
        assert result.status == "failed"
        assert "Unknown planner" in result.summary

    def test_algorithm_failure_captured(self, simple_df):
        """If algorithm.fit() raises, result is failed with message."""

        def _failing_fit(data, **kwargs):
            raise RuntimeError("algo exploded")

        with _mock_algorithm(fit_side_effect=_failing_fit):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "failed"
        assert "exploded" in result.summary
        assert result.provenance is not None


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestResultStructure:
    def test_ok_result_has_all_fields(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert result.adjacency_matrix is not None
        assert result.adjacency_matrix.shape[0] == result.adjacency_matrix.shape[1]
        assert result.provenance is not None
        assert result.provenance.seed == 0
        assert result.provenance.runtime_seconds > 0
        assert result.provenance.dataset_hash
        assert result.provenance.planner in ("rule", "rule-based-fallback", "llm")
        assert len(result.assumptions) >= 1
        assert result.summary

    def test_provenance_uses_frozen_params(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert isinstance(result.provenance.hyperparams, tuple)

    def test_node_names_populated(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.node_names == ["x", "y", "z", "w"]

    def test_discovery_metadata_populated(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.discovery_metadata == {"mock": True}

    def test_forced_algo_has_effective_hyperparams(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", seed=0)
        # Should have default_params, not empty {}
        params = dict(result.provenance.hyperparams)
        assert params  # non-empty
        assert "mock_param" in params

    def test_matrix_smaller_than_data_trims_node_names(self, simple_df):
        # Simulate a wrapper (e.g. CDNOD) that drops domain_index internally,
        # producing a 3x3 matrix for 4-column data.
        small_adj = np.zeros((3, 3))
        with _mock_algorithm(adj_matrix=small_adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "ok"
        assert result.node_names == ["x", "y", "z"]  # trimmed to match matrix
        assert any("trimmed" in w for w in result.warnings)

    def test_matrix_larger_than_data_fails(self, simple_df):
        # 5x5 matrix for 4-column data should fail, not silently pass
        big_adj = np.zeros((5, 5))
        with _mock_algorithm(adj_matrix=big_adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "failed"
        assert "does not match" in result.summary

    def test_non_square_matrix_fails(self, simple_df):
        bad_adj = np.zeros((3, 4))
        with _mock_algorithm(adj_matrix=bad_adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert result.status == "failed"
        assert "does not match" in result.summary

    def test_to_dict_roundtrip(self, simple_df):
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, seed=0)
        d = result.to_dict()
        assert d["status"] == "ok"
        assert "adjacency_matrix" in d
        assert "node_names" in d
        assert d["node_names"] == ["x", "y", "z", "w"]
        assert "discovery_metadata" in d
        assert "provenance" in d
        assert d["provenance"]["seed"] == 0


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_directed_edges(self):
        # mat[i,j]=1 means j→i
        mat = np.array([[0, 1], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        assert g is not None
        assert g.has_edge("B", "A")
        assert not g.has_edge("A", "B")

    def test_undirected_edges(self):
        mat = np.array([[0, 2], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        # Undirected: both directions
        assert g.has_edge("B", "A")
        assert g.has_edge("A", "B")
        assert g["B"]["A"]["edge_type"] == "undirected"

    def test_bidirected_edges(self):
        mat = np.array([[0, 3], [0, 0]])
        g = _build_graph(mat, ["A", "B"])
        assert g.has_edge("B", "A")
        assert g.has_edge("A", "B")
        assert g["B"]["A"]["edge_type"] == "bidirected"

    def test_empty_graph(self):
        mat = np.zeros((3, 3))
        g = _build_graph(mat, ["A", "B", "C"])
        assert g is not None
        assert g.number_of_edges() == 0
        assert g.number_of_nodes() == 3

    def test_mixed_edges(self):
        mat = np.array(
            [
                [0, 1, 0],
                [0, 0, 2],
                [0, 0, 0],
            ]
        )
        g = _build_graph(mat, ["A", "B", "C"])
        assert g.has_edge("B", "A")  # directed
        assert g["B"]["A"]["edge_type"] == "directed"
        assert g.has_edge("C", "B")  # undirected
        assert g.has_edge("B", "C")  # undirected (both)


# ---------------------------------------------------------------------------
# Edge counting in summary
# ---------------------------------------------------------------------------


class TestEdgeCounting:
    def test_directed_count(self, simple_df):
        adj = np.array(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "2 directed" in result.summary

    def test_undirected_count_symmetric(self, simple_df):
        """Undirected edges encoded symmetrically: both (i,j) and (j,i) = 2."""
        adj = np.array(
            [
                [0, 2, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "1 undirected" in result.summary

    def test_undirected_count_one_sided(self, simple_df):
        """Undirected edges encoded one-sided: only (i,j) = 2 (PC wrapper style)."""
        adj = np.array(
            [
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            result = CausalCopilot().analyze(simple_df, seed=0)
        assert "1 undirected" in result.summary


# ---------------------------------------------------------------------------
# Custom algorithm_params
# ---------------------------------------------------------------------------


class TestAlgorithmParams:
    def test_custom_params_reach_provenance(self, simple_df):
        """Custom params via algorithm_params= must appear in provenance."""
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", algorithm_params={"alpha": 0.01}, seed=0)
        assert result.status == "ok"
        assert result.provenance is not None
        hp_dict = dict(result.provenance.hyperparams)
        assert hp_dict.get("alpha") == 0.01

    def test_custom_params_merge_with_defaults(self, simple_df):
        """algorithm_params should merge on top of defaults."""
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", algorithm_params={"alpha": 0.01}, seed=0)
        hp_dict = dict(result.provenance.hyperparams)
        # Should have both the mock default and the override
        assert hp_dict.get("mock_param") is True  # from default_params()
        assert hp_dict.get("alpha") == 0.01  # from algorithm_params

    def test_algorithm_params_default_is_none(self, simple_df):
        """Without algorithm_params, defaults should be used (existing behavior)."""
        with _mock_algorithm():
            result = CausalCopilot().analyze(simple_df, algorithm="PC", seed=0)
        assert result.status == "ok"
        hp_dict = dict(result.provenance.hyperparams)
        assert hp_dict.get("mock_param") is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockAlgo:
    """Minimal mock that satisfies the algorithm interface."""

    def __init__(self, adj_matrix=None, fit_side_effect=None):
        self._adj = adj_matrix
        self._side_effect = fit_side_effect

    def default_params(self):
        return {"mock_param": True}

    def fit(self, data, **kwargs):
        if self._side_effect:
            return self._side_effect(data, **kwargs)
        n = data.shape[1]
        adj = self._adj if self._adj is not None else np.zeros((n, n))
        return adj, {"mock": True}, None


class _mock_algorithm:
    """Context manager that patches _load_algorithm to return a mock."""

    def __init__(self, adj_matrix=None, fit_side_effect=None):
        self._adj = adj_matrix
        self._side_effect = fit_side_effect
        self._patcher = None

    def __enter__(self):
        mock_algo = _MockAlgo(self._adj, self._side_effect)
        self._patcher = patch(
            "causal_copilot.copilot._load_algorithm",
            return_value=mock_algo,
        )
        self._patcher.start()
        return mock_algo

    def __exit__(self, *args):
        self._patcher.stop()


# ---------------------------------------------------------------------------
# Estimation (estimate_effect / refute_estimate)
# ---------------------------------------------------------------------------

# Check if inference extras are available
try:
    import dowhy  # noqa: F401
    import econml  # noqa: F401

    _HAS_INFERENCE = True
except ImportError:
    _HAS_INFERENCE = False

needs_inference = pytest.mark.skipif(not _HAS_INFERENCE, reason="needs dowhy + econml")


class TestEstimateEffect:
    """Tests for CausalCopilot.estimate_effect()."""

    @pytest.fixture
    def dag_result(self, simple_df):
        """analyze() result with a DAG (all directed edges): x→y, x→z, y→z."""
        adj = np.array(
            [
                [0, 0, 0, 0],  # x: no parents
                [1, 0, 0, 0],  # y: x→y
                [1, 1, 0, 0],  # z: x→z, y→z
                [0, 0, 0, 0],  # w: no parents
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)
        return copilot, result

    @pytest.fixture
    def cpdag_result(self, simple_df):
        """analyze() result with a CPDAG (has undirected edge)."""
        adj = np.array(
            [
                [0, 2, 0, 0],  # x -- y (undirected)
                [2, 0, 0, 0],
                [1, 1, 0, 0],  # z: x→z, y→z
                [0, 0, 0, 0],  # w: no parents
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)
        return copilot, result

    def test_no_adjacency_matrix(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        copilot._last_data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = CausalResult(status="failed")
        with pytest.raises(ValueError, match="no adjacency matrix"):
            copilot.estimate_effect(result, "x", "y")

    def test_no_data(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        result = CausalResult(
            status="ok",
            adjacency_matrix=np.eye(2),
            node_names=["x", "y"],
        )
        with pytest.raises(ValueError, match="No data"):
            copilot.estimate_effect(result, "x", "y")

    def test_treatment_not_in_graph(self, dag_result):
        copilot, result = dag_result
        with pytest.raises(ValueError, match="not in graph"):
            copilot.estimate_effect(result, "NONEXISTENT", "y")

    def test_same_treatment_outcome(self, dag_result):
        copilot, result = dag_result
        with pytest.raises(ValueError, match="must be different"):
            copilot.estimate_effect(result, "x", "x")

    def test_pag_rejected(self, simple_df):
        """PAG graph should reject inference."""
        adj = np.array(
            [
                [0, 3, 0, 0],  # bidirected = PAG
                [3, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)
        result = copilot.estimate_effect(result, "x", "y")
        assert "rejected" in " ".join(result.warnings).lower()

    def test_cpdag_nonlinear_rejected(self, simple_df):
        """CPDAG with non-linear/non-Gaussian data should reject if policy says no."""
        adj = np.array(
            [
                [0, 2, 0, 0],
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)
        # Force non-linear non-gaussian
        copilot._last_properties = {"likely_linear": False, "likely_gaussian": False}
        result = copilot.estimate_effect(result, "x", "y")
        assert "rejected" in " ".join(result.warnings).lower()

    def test_unknown_method(self, dag_result):
        copilot, result = dag_result
        with pytest.raises(ValueError, match="Unknown method"):
            copilot.estimate_effect(result, "x", "y", method="bogus")

    @needs_inference
    def test_linear_estimation(self, dag_result):
        copilot, result = dag_result
        result = copilot.estimate_effect(
            result, "x", "y", method="linear",
            control_value=0, treatment_value=1,
        )
        assert "x->y" in result.effects
        eff = result.effects["x->y"]
        assert eff.method == "linear"
        assert eff.ate is not None
        # x→y with coefficient ~2 in the fixture
        assert abs(eff.ate - 2.0) < 1.0, f"ATE={eff.ate}, expected ~2.0"

    @needs_inference
    def test_matching_estimation(self, simple_df):
        """Matching with binary treatment."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        t = (x + rng.normal(size=n) * 0.5 > 0).astype(float)
        y = 3.0 * t + x + rng.normal(size=n) * 0.3
        df = pd.DataFrame({"x": x, "T": t, "Y": y})

        adj = np.array(
            [
                [0, 0, 0],  # x: no parents
                [1, 0, 0],  # T: x→T
                [1, 1, 0],  # Y: x→Y, T→Y
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(df, seed=42)

        result = copilot.estimate_effect(result, "T", "Y", method="matching")
        assert "T->Y" in result.effects
        eff = result.effects["T->Y"]
        assert eff.method == "matching"
        assert eff.ate is not None
        assert abs(eff.ate - 3.0) < 2.0, f"ATE={eff.ate}, expected ~3.0"

    @needs_inference
    def test_dml_estimation(self, dag_result):
        copilot, result = dag_result
        result = copilot.estimate_effect(
            result, "x", "y", method="dml",
        )
        assert "x->y" in result.effects
        eff = result.effects["x->y"]
        assert eff.method == "dml"
        assert eff.ate is not None

    @needs_inference
    def test_auto_method_selection(self, dag_result):
        """When method=None, auto-select should pick a valid method."""
        copilot, result = dag_result
        result = copilot.estimate_effect(result, "x", "y")
        assert "x->y" in result.effects
        eff = result.effects["x->y"]
        assert eff.method in ("linear", "matching", "dml", "drl", "metalearner", "iv")
        assert eff.ate is not None

    @needs_inference
    def test_effects_in_to_dict(self, dag_result):
        """Effects should serialize via to_dict()."""
        copilot, result = dag_result
        result = copilot.estimate_effect(result, "x", "y", method="linear")
        d = result.to_dict()
        assert "effects" in d
        assert "x->y" in d["effects"]
        assert d["effects"]["x->y"]["method"] == "linear"

    @needs_inference
    def test_multiple_effects(self, dag_result):
        """Can estimate multiple treatment→outcome pairs on same result."""
        copilot, result = dag_result
        result = copilot.estimate_effect(result, "x", "y", method="linear")
        result = copilot.estimate_effect(result, "x", "z", method="linear")
        assert len(result.effects) == 2
        assert "x->y" in result.effects
        assert "x->z" in result.effects

    @needs_inference
    def test_explicit_data_param(self, simple_df):
        """Can pass data= explicitly instead of relying on stored data."""
        adj = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)

        copilot._last_data = None  # clear stored data
        result = copilot.estimate_effect(
            result, "x", "y", data=simple_df, method="linear",
        )
        assert "x->y" in result.effects


class TestRefuteEstimate:
    """Tests for CausalCopilot.refute_estimate()."""

    @needs_inference
    def test_refutation_runs(self, simple_df):
        adj = np.array(
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(simple_df, seed=0)

        refutation = copilot.refute_estimate(result, "x", "y")
        assert "original_estimate" in refutation
        assert "refutations" in refutation


# ---------------------------------------------------------------------------
# Shared DAG fixture for GCM-based tests
# ---------------------------------------------------------------------------


@pytest.fixture
def dag_copilot_and_result():
    """CausalCopilot + CausalResult with a DAG: x→y→z, x→z, w independent.

    Uses 200 samples for stable GCM fitting.
    """
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=n)
    y = 2.0 * x + rng.normal(size=n) * 0.3
    z = x + 0.6 * y + rng.normal(size=n) * 0.5
    w = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})

    adj = np.array(
        [
            [0, 0, 0, 0],  # x: no parents
            [1, 0, 0, 0],  # y: x→y
            [1, 1, 0, 0],  # z: x→z, y→z
            [0, 0, 0, 0],  # w: no parents
        ]
    )
    with _mock_algorithm(adj_matrix=adj):
        copilot = CausalCopilot()
        result = copilot.analyze(df, seed=42)
    return copilot, result, df


# ---------------------------------------------------------------------------
# inspect_graph
# ---------------------------------------------------------------------------


class TestInspectGraph:
    """Tests for CausalCopilot.inspect_graph()."""

    def test_dag_classification(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        info = copilot.inspect_graph(result)
        assert info["graph_kind"] == "dag"
        assert info["inference_policy"]["eligibility"] is True
        assert info["graph_stats"]["n_directed"] > 0

    def test_cpdag_classification(self):
        adj = np.array([[0, 2, 0], [2, 0, 0], [1, 1, 0]])
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(df, seed=0)

        info = copilot.inspect_graph(result)
        assert info["graph_kind"] == "cpdag"
        assert info["graph_stats"]["n_undirected"] > 0

    def test_identifiable_edges(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        info = copilot.inspect_graph(result)
        assert "identifiable" in info["identifiability"]
        assert len(info["identifiability"]["identifiable"]) > 0

    def test_query_assessment(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        info = copilot.inspect_graph(result, treatment="x", outcome="y")
        assert "query_assessment" in info
        qa = info["query_assessment"]
        assert qa["treatment"] == "x"
        assert qa["outcome"] == "y"
        assert qa["directed_path_exists"] is True
        assert qa["effect_identifiable"] is True

    def test_query_no_path(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        info = copilot.inspect_graph(result, treatment="w", outcome="x")
        qa = info["query_assessment"]
        assert qa["directed_path_exists"] is False
        assert qa["effect_identifiable"] is False

    def test_no_adjacency_matrix(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        result = CausalResult(status="failed")
        with pytest.raises(ValueError, match="no adjacency matrix"):
            copilot.inspect_graph(result)

    def test_invalid_treatment_node(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        with pytest.raises(ValueError, match="not in graph"):
            copilot.inspect_graph(result, treatment="NONEXISTENT", outcome="y")


# ---------------------------------------------------------------------------
# estimate_counterfactual
# ---------------------------------------------------------------------------


class TestEstimateCounterfactual:
    """Tests for CausalCopilot.estimate_counterfactual()."""

    @needs_inference
    def test_counterfactual_basic(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        cf = copilot.estimate_counterfactual(result, "x", "y", intervention_value=5.0)
        assert "observed" in cf
        assert "counterfactual" in cf
        assert "effect" in cf
        assert cf["counterfactual"]["x"] == 5.0

    @needs_inference
    def test_counterfactual_specific_row(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        cf = copilot.estimate_counterfactual(
            result, "x", "y", intervention_value=0.0, observed_row_index=0,
        )
        assert cf["observed_row_index"] == 0

    def test_pag_rejected(self):
        adj = np.array([[0, 3, 0], [3, 0, 0], [0, 0, 0]])
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50), "c": rng.normal(size=50)})
        with _mock_algorithm(adj_matrix=adj):
            copilot = CausalCopilot()
            result = copilot.analyze(df, seed=0)
        with pytest.raises(ValueError, match="PAG"):
            copilot.estimate_counterfactual(result, "a", "b", intervention_value=1.0)


# ---------------------------------------------------------------------------
# simulate_intervention
# ---------------------------------------------------------------------------


class TestSimulateIntervention:
    """Tests for CausalCopilot.simulate_intervention()."""

    @needs_inference
    def test_shift_intervention(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        sim = copilot.simulate_intervention(
            result, "x", "y", intervention_value=2.0, shift=True,
        )
        assert sim["intervention_type"] == "shift"
        assert sim["intervention_value"] == 2.0
        assert "original_distribution" in sim
        assert "intervention_distribution" in sim
        assert "mean_change" in sim
        # Shifting x by +2 should increase y (coefficient ~2)
        assert sim["mean_change"] is not None

    @needs_inference
    def test_atomic_intervention(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        sim = copilot.simulate_intervention(
            result, "x", "y", intervention_value=0.0, shift=False,
        )
        assert sim["intervention_type"] == "atomic"

    def test_no_data(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        result = CausalResult(status="ok", adjacency_matrix=np.eye(2), node_names=["a", "b"])
        with pytest.raises(ValueError, match="No data"):
            copilot.simulate_intervention(result, "a", "b")


# ---------------------------------------------------------------------------
# attribute_anomaly
# ---------------------------------------------------------------------------


class TestAttributeAnomaly:
    """Tests for CausalCopilot.attribute_anomaly()."""

    @needs_inference
    def test_anomaly_attribution(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        attr = copilot.attribute_anomaly(result, "z")
        assert attr["target_node"] == "z"
        assert "attributions" in attr
        assert isinstance(attr["attributions"], dict)
        assert attr["n_anomaly_samples"] > 0

    @needs_inference
    def test_custom_threshold(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        attr = copilot.attribute_anomaly(result, "z", threshold_percentile=90.0)
        assert attr["threshold_percentile"] == 90.0


# ---------------------------------------------------------------------------
# attribute_distribution_change
# ---------------------------------------------------------------------------


class TestAttributeDistributionChange:
    """Tests for CausalCopilot.attribute_distribution_change()."""

    @needs_inference
    def test_distribution_change(self, dag_copilot_and_result):
        copilot, result, df_old = dag_copilot_and_result
        # Create shifted data
        rng = np.random.default_rng(99)
        n = 500
        x = rng.normal(loc=2.0, size=n)  # shifted mean
        y = 2.0 * x + rng.normal(size=n) * 0.3
        z = x + 0.6 * y + rng.normal(size=n) * 0.5
        w = rng.normal(size=n)
        df_new = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})

        attr = copilot.attribute_distribution_change(result, "z", data_new=df_new)
        assert attr["target_node"] == "z"
        assert "attributions" in attr
        assert attr["n_old"] == len(df_old)
        assert attr["n_new"] == len(df_new)

    def test_no_baseline_data(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        result = CausalResult(status="ok", adjacency_matrix=np.eye(2), node_names=["a", "b"])
        df_new = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="No baseline data"):
            copilot.attribute_distribution_change(result, "a", data_new=df_new)


# ---------------------------------------------------------------------------
# compute_feature_importance
# ---------------------------------------------------------------------------

try:
    import shap  # noqa: F401

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

needs_shap = pytest.mark.skipif(not _HAS_SHAP, reason="needs shap")


class TestComputeFeatureImportance:
    """Tests for CausalCopilot.compute_feature_importance()."""

    @needs_shap
    def test_feature_importance(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        fi = copilot.compute_feature_importance(result, "z")
        assert fi["target_node"] == "z"
        assert "feature_importance" in fi
        assert "top_features" in fi
        assert isinstance(fi["feature_importance"], dict)
        assert len(fi["top_features"]) > 0

    def test_invalid_target(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        with pytest.raises(ValueError, match="not in data"):
            copilot.compute_feature_importance(result, "NONEXISTENT")

    def test_no_data(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        result = CausalResult(status="ok")
        with pytest.raises(ValueError, match="No data"):
            copilot.compute_feature_importance(result, "x")


# ---------------------------------------------------------------------------
# validate_graph
# ---------------------------------------------------------------------------


class TestValidateGraph:
    """Tests for CausalCopilot.validate_graph()."""

    @needs_inference
    def test_graph_validation(self, dag_copilot_and_result):
        copilot, result, _ = dag_copilot_and_result
        val = copilot.validate_graph(result, n_permutations=5)
        assert "falsification_result" in val
        assert val["n_nodes"] > 0
        assert val["n_edges"] >= 0

    def test_no_adjacency_matrix(self):
        from causal_copilot.core.result import CausalResult

        copilot = CausalCopilot()
        copilot._last_data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = CausalResult(status="failed")
        with pytest.raises(ValueError, match="no adjacency matrix"):
            copilot.validate_graph(result)
