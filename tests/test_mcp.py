"""Tests for the MCP server tools (4-tool contract: discover, inspect_graph, diagnose_data, run_algorithm)."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastmcp.exceptions import ToolError


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


# ── inspect_graph ──────────────────────────────────────────────────────


class TestInspectGraphTool:
    def test_dag_with_adj(self):
        from causal_copilot.mcp.server import inspect_graph

        # A->B: adj[1,0]=1
        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,0],[1,0]]",
            node_names='["A","B"]',
        ))
        assert result["status"] == "ok"
        assert result["graph_kind"] == "dag"
        assert result["inference_policy"]["eligibility"] is True
        assert result["inference_policy"]["method"] == "standard"
        assert "summary" in result
        assert "key_findings" in result

    def test_cpdag_needs_more_input(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,2],[2,0]]",
            node_names='["A","B"]',
        ))
        assert result["status"] == "needs_more_input"
        assert "data_diagnosis" in result["missing_inputs"]
        assert "next_step" in result

    def test_cpdag_with_diagnosis_allows_ida(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,2],[2,0]]",
            node_names='["A","B"]',
            data_diagnosis='{"linearity": true, "gaussian_error": true}',
        ))
        assert result["status"] == "ok"
        assert result["inference_policy"]["eligibility"] is True
        assert result["inference_policy"]["method"] == "ida"

    def test_cpdag_nonlinear_rejects(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,2],[2,0]]",
            node_names='["A","B"]',
            data_diagnosis='{"linearity": false, "gaussian_error": true}',
        ))
        assert result["status"] == "ok"
        assert result["inference_policy"]["eligibility"] is False

    def test_pag_rejects(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,3],[3,0]]",
            node_names='["A","B"]',
        ))
        assert result["status"] == "ok"
        assert result["graph_kind"] == "pag"
        assert result["inference_policy"]["eligibility"] is False

    def test_query_assessment(self):
        from causal_copilot.mcp.server import inspect_graph

        # A->B->C: adj[1,0]=1, adj[2,1]=1
        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,0,0],[1,0,0],[0,1,0]]",
            node_names='["A","B","C"]',
            treatment="A",
            outcome="C",
        ))
        assert result["status"] == "ok"
        qa = result["query_assessment"]
        assert qa["directed_path_exists"] is True
        assert qa["effect_identifiable"] is True
        assert qa["directly_connected"] is False

    def test_query_no_path(self):
        from causal_copilot.mcp.server import inspect_graph

        # B->A: adj[0,1]=1. Query A->B has no directed path.
        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,1],[0,0]]",
            node_names='["A","B"]',
            treatment="A",
            outcome="B",
        ))
        assert result["status"] == "ok"
        qa = result["query_assessment"]
        assert qa["directed_path_exists"] is False
        assert qa["effect_identifiable"] is False

    def test_run_id_mode(self):
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import inspect_graph

        rid = get_store().save({
            "adjacency_matrix": [[0, 0], [1, 0]],
            "node_names": ["X", "Y"],
            "data_diagnosis": {"linearity": True, "gaussian_error": True},
        })
        result = json.loads(inspect_graph(run_id=rid))
        assert result["status"] == "ok"
        assert result["graph_kind"] == "dag"

    def test_run_id_not_found(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="not found"):
            inspect_graph(run_id="nonexistent")

    def test_mutual_exclusion(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="mutually exclusive"):
            inspect_graph(run_id="abc", adjacency_matrix="[[0]]")

    def test_no_input(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError):
            inspect_graph()

    def test_missing_node_names(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="node_names"):
            inspect_graph(adjacency_matrix="[[0,1],[0,0]]")

    def test_treatment_outcome_all_or_none(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="[Bb]oth"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="A",
            )

    def test_treatment_equals_outcome(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="different"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="A",
                outcome="A",
            )

    def test_treatment_not_in_names(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="Z"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="Z",
                outcome="B",
            )

    def test_graph_stats(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(inspect_graph(
            adjacency_matrix="[[0,0],[1,0]]",
            node_names='["A","B"]',
        ))
        gs = result["graph_stats"]
        assert gs["n_nodes"] == 2
        assert gs["n_edges"] == 1
        assert gs["n_directed"] == 1
        assert gs["density"] == 1.0


# ── diagnose_data ──────────────────────────────────────────────────────


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

        with pytest.raises(ToolError):
            diagnose_data("")


# ── run_algorithm ──────────────────────────────────────────────────────


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

    def test_hp_transparency(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(
                csv,
                algorithm="PC",
                hyperparameters='{"alpha": 0.01}',
            ))
        assert result["status"] == "ok"
        prov = result["provenance"]
        assert "requested_hyperparameters" in prov
        assert "effective_hyperparameters" in prov
        assert "resolver_adjustments" in prov
        assert prov["requested_hyperparameters"]["alpha"] == 0.01

    def test_no_resolver_overrides(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(
                csv,
                algorithm="PC",
                hyperparameters='{"alpha": 0.01, "indep_test": "kci"}',
                allow_resolver_overrides=False,
            ))
        assert result["status"] == "ok"
        prov = result["provenance"]
        # With overrides disabled, effective should match requested
        assert prov["effective_hyperparameters"]["indep_test"] == "kci"
        assert prov["resolver_adjustments"] == {}

    def test_artifact_stores_data(self):
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(csv, algorithm="PC"))
        assert result["status"] == "ok"
        cached = get_store().get(result["run_id"])
        assert cached is not None
        assert "_processed_data" in cached
        assert "_statistics" in cached
        import pandas as pd
        assert isinstance(cached["_processed_data"], pd.DataFrame)

    def test_missing_algorithm(self):
        from causal_copilot.mcp.server import run_algorithm

        with pytest.raises(ToolError):
            run_algorithm("a,b\n1,2\n3,4", algorithm="")


# ── discover ───────────────────────────────────────────────────────────


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
            # Enriched output fields
            assert "summary" in result
            assert "key_findings" in result
            assert "limitations" in result
            assert "algorithm_rationale" in result

    def test_empty_csv(self):
        from causal_copilot.mcp.server import discover

        with pytest.raises(ToolError):
            discover("")

    def test_too_few_rows(self):
        from causal_copilot.mcp.server import discover

        with pytest.raises(ToolError, match="10 rows"):
            discover("a,b\n1,2\n3,4")

    def test_with_algorithm_override(self):
        from causal_copilot.mcp.server import discover

        csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        with _mock_run_algorithm():
            result = json.loads(discover(csv, algorithm="PC"))
        assert result["status"] in ("ok", "partial", "error")


# ── MCP CLI ────────────────────────────────────────────────────────────


# ── estimation helpers ────────────────────────────────────────────────


class TestEstimationLinear:
    def test_linear_basic(self):
        from causal_copilot.mcp.estimation import estimate_linear

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        y = 2.0 * x + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"X": x, "Y": y})
        dot_graph = "digraph { X -> Y; }"
        result = estimate_linear(data, dot_graph, "X", "Y", 0.0, 1.0)
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert 1.5 < ate < 2.5, f"ATE should be ~2.0, got {ate}"
        assert result["ate"]["p_value"] < 0.05


class TestEstimationMatching:
    def test_matching_binary(self):
        from causal_copilot.mcp.estimation import estimate_matching

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)  # binary treatment
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_matching(data, "T", "Y", ["Z"], 0, 1)
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert 2.0 < ate < 4.0, f"ATE should be ~3.0, got {ate}"


# ── MCP CLI ────────────────────────────────────────────────────────────


class TestMCPCLI:
    def test_mcp_help(self, capsys):
        from causal_copilot.cli import main

        with pytest.raises(SystemExit):
            main(["mcp", "--help"])
