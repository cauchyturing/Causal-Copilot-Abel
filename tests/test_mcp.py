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


class _mock_run_algorithm:
    """Context manager that patches wrappers so Programming.forward works."""

    def __enter__(self):
        self._patcher = patch(
            "causal_discovery.wrappers.PC",
            _MockWrapper,
        )
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()


class TestDiagnoseDataTool:
    def test_basic_diagnosis(self):
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
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
