"""Golden tests: verify tool output schemas remain stable.

If these fail, it means a tool's output contract changed.
Update the golden snapshot ONLY after verifying the change is intentional.
"""
import json
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest


def _mock_stat_info(gs):
    gs.statistics.linearity = True
    gs.statistics.gaussian_error = True
    gs.statistics.missingness = False
    gs.statistics.data_type = "Continuous"
    gs.statistics.sample_size = gs.user_data.raw_data.shape[0]
    gs.statistics.feature_number = gs.user_data.raw_data.shape[1]
    gs.statistics.time_series = False
    return gs


def _ensure_stat_module():
    """Inject fake stat module if real one unavailable."""
    if "preprocess.stat_info_functions" not in sys.modules:
        fake = ModuleType("preprocess.stat_info_functions")
        fake.stat_info_collection = _mock_stat_info
        sys.modules["preprocess.stat_info_functions"] = fake


def _check_keys(result, required_keys):
    """Verify all required keys are present in result."""
    for key in required_keys:
        assert key in result, f"Missing required key: {key}"


class TestOutputSchemaStability:
    """Each tool's output must contain these keys. Adding keys is OK.
    Removing or renaming keys is a breaking change."""

    def test_diagnose_data_schema(self):
        _ensure_stat_module()
        from causal_copilot.mcp.server import diagnose_data

        csv = "x,y\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        result = json.loads(diagnose_data(csv))
        if result["status"] == "ok":
            _check_keys(result, ["status", "diagnosis"])
            _check_keys(result["diagnosis"], [
                "linearity", "data_type", "sample_size", "feature_number",
            ])

    def test_explain_graph_schema(self):
        from causal_copilot.mcp.server import explain_graph

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        result = json.loads(explain_graph(adj, names))
        _check_keys(result, ["explanation", "graph_stats"])
        _check_keys(result["graph_stats"], [
            "n_directed_edges", "n_undirected_edges",
            "root_causes", "terminal_effects",
        ])

    def test_explain_result_schema(self):
        from causal_copilot.mcp.server import explain_result

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        result = json.loads(explain_result(adj, names))
        _check_keys(result, ["explanation", "graph_stats", "graph_kind", "identifiability"])

    def test_estimate_effects_rejection_schema(self):
        from causal_copilot.mcp.server import estimate_effects

        # PAG → should reject
        adj = [[0, 4], [5, 0]]
        names = ["A", "B"]
        csv = "a,b\n" + "\n".join(f"{i},{i*2}" for i in range(50))
        result = json.loads(estimate_effects(
            json.dumps(adj), json.dumps(names), csv,
            treatment="A", outcome="B",
        ))
        _check_keys(result, ["status", "error", "graph_kind"])

    def test_refine_graph_schema(self):
        from causal_copilot.mcp.server import refine_graph

        adj = [[0, 0], [1, 0]]
        names = ["X", "Y"]
        result = json.loads(refine_graph(
            json.dumps(adj), json.dumps(names),
        ))
        _check_keys(result, ["status", "adjacency_matrix", "edges", "graph_kind"])

    def test_list_algorithms_schema(self):
        from causal_copilot.mcp.server import list_algorithms

        result = json.loads(list_algorithms())
        # Result is a list of algorithm dicts
        assert isinstance(result, list)
        assert len(result) > 0
        _check_keys(result[0], ["name", "family", "available"])

    def test_analyze_schema(self):
        from causal_copilot.mcp.server import analyze

        # Use non-degenerate data (add noise to avoid singular matrix)
        rng = np.random.default_rng(0)
        csv = "x,y\n" + "\n".join(
            f"{rng.normal()},{rng.normal()}" for _ in range(60)
        )
        result = json.loads(analyze(csv))
        _check_keys(result, ["summary"])
        # edges present on success, not on failure
        if result.get("status") != "failed":
            _check_keys(result, ["edges"])


class TestResourceSchemaStability:
    def test_algorithm_resource_has_required_fields(self):
        from causal_copilot.mcp.resources import get_algorithm_resources

        resources = get_algorithm_resources()
        for r in resources:
            _check_keys(r, ["name", "uri", "description"])

    def test_guide_resources_exist(self):
        from causal_copilot.mcp.resources import get_all_guide_names, get_guide_content

        guides = get_all_guide_names()
        assert "ci-tests" in guides
        assert "score-functions" in guides
        assert "interpreting-graphs" in guides
        for name in guides:
            content = get_guide_content(name)
            assert content is not None
            assert len(content) > 50
