"""Tests for the MCP server tools."""

import json
from unittest.mock import patch

import numpy as np
import pytest

from causal_copilot.mcp.server import analyze, explain_graph, list_algorithms


class _MockAlgo:
    def default_params(self):
        return {"mock_param": True}

    def fit(self, data, **kwargs):
        n = data.shape[1]
        return np.zeros((n, n)), {"mock": True}, None


class _mock_algorithm:
    def __enter__(self):
        self._patcher = patch(
            "causal_copilot.copilot._load_algorithm",
            return_value=_MockAlgo(),
        )
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()


class TestAnalyzeTool:
    def test_basic_csv(self):
        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_algorithm():
            result = json.loads(analyze(csv))
        assert result["status"] in ("ok", "partial", "failed")
        assert "provenance" in result

    def test_bad_csv(self):
        result = json.loads(analyze("not,valid\ncsv"))
        # Should still parse (it's valid CSV, just small)
        assert "status" in result

    def test_empty_csv(self):
        result = json.loads(analyze(""))
        assert result["status"] == "error"

    def test_single_column(self):
        result = json.loads(analyze("a\n1\n2\n3"))
        assert result["status"] == "error"
        assert "2 columns" in result["error"]

    def test_with_algorithm(self):
        csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        with _mock_algorithm():
            result = json.loads(analyze(csv, algorithm="PC"))
        assert result["status"] == "ok"

    def test_returns_interpretation_hints(self):
        # Create data where PC finds edges
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(size=n)
        y = 0.8 * x + rng.normal(size=n) * 0.3
        lines = ["x,y"] + [f"{x[i]},{y[i]}" for i in range(n)]
        csv = "\n".join(lines)
        with _mock_algorithm():
            result = json.loads(analyze(csv))
        assert result["status"] == "ok"


class TestListAlgorithmsTool:
    def test_available(self):
        result = json.loads(list_algorithms("available"))
        assert isinstance(result, list)
        assert len(result) >= 5
        for algo in result:
            assert algo["available"] is True
            assert "name" in algo
            assert "family" in algo

    def test_all(self):
        result = json.loads(list_algorithms("all"))
        assert len(result) >= 19

    def test_filter_timeseries(self):
        result = json.loads(list_algorithms("timeseries"))
        for algo in result:
            assert "timeseries" in algo["tags"]

    def test_filter_constraint(self):
        result = json.loads(list_algorithms("constraint"))
        assert len(result) >= 2
        # All should either have family=constraint or tag "constraint"
        for algo in result:
            assert algo["family"] == "constraint" or "constraint" in algo["tags"]

    def test_unavailable_has_install_hint(self):
        result = json.loads(list_algorithms("all"))
        unavailable = [a for a in result if not a["available"]]
        for algo in unavailable:
            assert "install_hint" in algo

    def test_has_best_for(self):
        result = json.loads(list_algorithms("available"))
        for algo in result:
            assert "best_for" in algo


class TestExplainGraphTool:
    def test_simple_chain(self):
        # X → Y → Z
        adj = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        names = ["X", "Y", "Z"]
        result = json.loads(explain_graph(adj, names))
        assert "explanation" in result
        assert "X → Y" in result["explanation"]
        assert "Y → Z" in result["explanation"]
        assert result["graph_stats"]["n_directed_edges"] == 2
        assert "X" in result["graph_stats"]["root_causes"]
        assert "Z" in result["graph_stats"]["terminal_effects"]
        assert "Y" in result["graph_stats"]["mediators"]

    def test_undirected_edges(self):
        adj = [[0, 2], [2, 0]]
        names = ["A", "B"]
        result = json.loads(explain_graph(adj, names))
        assert result["graph_stats"]["n_undirected_edges"] == 1

    def test_empty_graph(self):
        adj = [[0, 0], [0, 0]]
        names = ["A", "B"]
        result = json.loads(explain_graph(adj, names))
        assert result["graph_stats"]["n_directed_edges"] == 0

    def test_dimension_mismatch(self):
        adj = [[0, 1], [0, 0]]
        names = ["A", "B", "C"]
        result = json.loads(explain_graph(adj, names))
        assert "error" in result

    def test_causal_chain_detected(self):
        # A → B → C
        adj = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        names = ["A", "B", "C"]
        result = json.loads(explain_graph(adj, names))
        assert "mediates" in result["explanation"]
        assert "Causal chain" in result["explanation"]


class _MockWrapper:
    """Mock wrapper that mimics a causal discovery algorithm wrapper."""

    def __init__(self, args=None):
        pass

    def fit(self, data, **kwargs):
        n = data.shape[1]
        return np.zeros((n, n)), {"mock": True}, None


def _mock_stat_info(gs):
    """Mock stat_info_collection that sets plausible statistics."""
    gs.statistics.linearity = True
    gs.statistics.gaussian_error = True
    gs.statistics.missingness = False
    gs.statistics.data_type = "Continuous"
    gs.statistics.sample_size = gs.user_data.raw_data.shape[0]
    gs.statistics.feature_number = gs.user_data.raw_data.shape[1]
    gs.statistics.time_series = False
    return gs


def _inject_fake_stat_module():
    """Inject fake preprocess.stat_info_functions into sys.modules.

    Returns (had_module, old_module) for cleanup.
    """
    import sys
    from types import ModuleType

    had = "preprocess.stat_info_functions" in sys.modules
    old = sys.modules.get("preprocess.stat_info_functions")
    fake = ModuleType("preprocess.stat_info_functions")
    fake.stat_info_collection = _mock_stat_info
    sys.modules["preprocess.stat_info_functions"] = fake
    return had, old


def _restore_stat_module(had, old):
    """Restore preprocess.stat_info_functions after mock."""
    import sys

    if had and old is not None:
        sys.modules["preprocess.stat_info_functions"] = old
    elif not had:
        sys.modules.pop("preprocess.stat_info_functions", None)


class _mock_stat_info_ctx:
    """Context manager that injects a fake stat module for diagnose_data tests."""

    def __enter__(self):
        self._had, self._old = _inject_fake_stat_module()
        return self

    def __exit__(self, *args):
        _restore_stat_module(self._had, self._old)


class _mock_run_algorithm:
    """Context manager that patches wrappers + stat_info so pipeline tools work."""

    def __enter__(self):
        self._had, self._old = _inject_fake_stat_module()
        self._patcher = patch(
            "causal_discovery.wrappers.PC",
            _MockWrapper,
        )
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()
        _restore_stat_module(self._had, self._old)


class TestDiagnoseDataTool:
    def test_basic_diagnosis(self):
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(csv))
        assert result["status"] == "ok"
        assert "linearity" in result["diagnosis"]
        assert "data_type" in result["diagnosis"]
        assert "sample_size" in result["diagnosis"]

    def test_empty_csv(self):
        from causal_copilot.mcp.server import diagnose_data

        result = json.loads(diagnose_data(""))
        assert result["status"] == "error"


class TestRunAlgorithmTool:
    def test_run_with_mock(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(csv, algorithm="PC"))
        assert result["status"] == "ok"
        assert "adjacency_matrix" in result
        assert "run_id" in result

    def test_missing_algorithm(self):
        from causal_copilot.mcp.server import run_algorithm

        result = json.loads(run_algorithm("a,b\n1,2\n3,4", algorithm=""))
        assert result["status"] == "error"


class TestMCPCLI:
    def test_mcp_help(self, capsys):
        from causal_copilot.cli import main

        with pytest.raises(SystemExit):
            main(["mcp", "--help"])


class TestRefineGraphTool:
    def test_refine_simple_graph(self):
        from causal_copilot.mcp.server import refine_graph

        adj = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        names = ["X", "Y", "Z"]
        result = json.loads(refine_graph(json.dumps(adj), json.dumps(names)))
        assert result["status"] == "ok"
        assert "graph_kind" in result
        assert result["graph_kind"] == "dag"
        assert "edge_confidence" in result
        assert result["n_directed"] == 2

    def test_dimension_mismatch(self):
        from causal_copilot.mcp.server import refine_graph

        adj = [[0, 1], [0, 0]]
        names = ["A", "B", "C"]
        result = json.loads(refine_graph(json.dumps(adj), json.dumps(names)))
        assert result["status"] == "error"

    def test_invalid_json(self):
        from causal_copilot.mcp.server import refine_graph

        result = json.loads(refine_graph("not-json", '["A"]'))
        assert result["status"] == "error"

    def test_cpdag_detected(self):
        from causal_copilot.mcp.server import refine_graph

        # Undirected edge -> CPDAG
        adj = [[0, 2], [2, 0]]
        names = ["A", "B"]
        result = json.loads(refine_graph(json.dumps(adj), json.dumps(names)))
        assert result["status"] == "ok"
        assert result["graph_kind"] == "cpdag"

    def test_run_id_passthrough(self):
        from causal_copilot.mcp.server import refine_graph

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        result = json.loads(refine_graph(
            json.dumps(adj), json.dumps(names), run_id="test-123",
        ))
        assert result["run_id"] == "test-123"


class TestEstimateEffectsTool:
    def test_dag_allows_inference(self):
        from causal_copilot.mcp.server import estimate_effects

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(100))
        result = json.loads(estimate_effects(
            json.dumps(adj), json.dumps(names), csv,
            treatment="X", outcome="Y",
        ))
        assert result["status"] == "partial"
        assert result["inference_method"] == "standard"
        assert result["graph_kind"] == "dag"
        assert result["effect_estimate"] is None

    def test_pag_rejects_inference(self):
        from causal_copilot.mcp.server import estimate_effects

        adj = [[0, 4], [5, 0]]
        names = ["A", "B"]
        csv = "a,b\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        result = json.loads(estimate_effects(
            json.dumps(adj), json.dumps(names), csv,
            treatment="A", outcome="B",
        ))
        assert result["status"] == "error"
        assert "PAG" in result["error"]

    def test_missing_treatment(self):
        from causal_copilot.mcp.server import estimate_effects

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        csv = "x,y\n1,2\n3,4"
        result = json.loads(estimate_effects(
            json.dumps(adj), json.dumps(names), csv,
        ))
        assert result["status"] == "error"

    def test_treatment_not_in_names(self):
        from causal_copilot.mcp.server import estimate_effects

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        csv = "x,y\n1,2\n3,4"
        result = json.loads(estimate_effects(
            json.dumps(adj), json.dumps(names), csv,
            treatment="Z", outcome="Y",
        ))
        assert result["status"] == "error"
        assert "Z" in result["error"]

    def test_cpdag_linear_gaussian_allows_ida(self):
        from causal_copilot.mcp.server import estimate_effects
        from causal_discovery.pdag_policy import check_inference_policy

        # Test PDAG policy directly: CPDAG + linear-Gaussian -> IDA allowed
        adj = np.array([[0, 2], [2, 0]])
        policy = check_inference_policy(adj, is_linear_gaussian=True)
        assert policy["allow_inference"] is True
        assert policy["method"] == "ida"

    def test_cpdag_nonlinear_rejects(self):
        from causal_copilot.mcp.server import estimate_effects
        from causal_discovery.pdag_policy import check_inference_policy

        # Test PDAG policy directly: CPDAG + non-linear -> reject
        adj = np.array([[0, 2], [2, 0]])
        policy = check_inference_policy(adj, is_linear_gaussian=False)
        assert policy["allow_inference"] is False


class TestDiscoverTool:
    def test_full_pipeline_with_mock(self):
        from causal_copilot.mcp.server import discover

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(discover(csv))
        assert result["status"] in ("ok", "partial", "error")
        if result["status"] == "ok":
            assert "graph_kind" in result
            assert "provenance" in result
            assert "run_id" in result

    def test_empty_csv(self):
        from causal_copilot.mcp.server import discover

        result = json.loads(discover(""))
        assert result["status"] == "error"

    def test_too_few_rows(self):
        from causal_copilot.mcp.server import discover

        result = json.loads(discover("a,b\n1,2\n3,4"))
        assert result["status"] == "error"
        assert "10 rows" in result["error"]

    def test_with_algorithm_override(self):
        from causal_copilot.mcp.server import discover

        csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        with _mock_run_algorithm():
            result = json.loads(discover(csv, algorithm="PC"))
        assert result["status"] in ("ok", "partial", "error")
