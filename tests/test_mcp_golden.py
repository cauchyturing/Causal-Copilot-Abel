"""Golden tests: verify tool output schemas remain stable (4-tool contract).

If these fail, it means a tool's output contract changed.
Update the golden snapshot ONLY after verifying the change is intentional.
"""

import json
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np


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

        csv = "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(50))
        result = json.loads(diagnose_data(csv))
        if result["status"] == "ok":
            _check_keys(result, ["status", "diagnosis"])
            _check_keys(
                result["diagnosis"],
                [
                    "linearity",
                    "data_type",
                    "sample_size",
                    "feature_number",
                ],
            )

    def test_inspect_graph_ok_schema(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["X","Y"]',
            )
        )
        _check_keys(
            result,
            [
                "status",
                "graph_kind",
                "graph_stats",
                "identifiability",
                "inference_policy",
                "summary",
                "key_findings",
                "limitations",
            ],
        )
        _check_keys(
            result["graph_stats"],
            [
                "n_nodes",
                "n_edges",
                "n_directed",
                "n_undirected",
                "n_bidirected",
                "density",
            ],
        )
        _check_keys(
            result["inference_policy"],
            [
                "eligibility",
                "method",
                "reason",
                "assumptions_used",
            ],
        )

    def test_inspect_graph_needs_more_input_schema(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,2],[2,0]]",
                node_names='["A","B"]',
            )
        )
        _check_keys(
            result,
            [
                "status",
                "graph_kind",
                "graph_stats",
                "missing_inputs",
                "next_step",
            ],
        )
        assert result["status"] == "needs_more_input"

    def test_inspect_graph_query_assessment_schema(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="A",
                outcome="B",
            )
        )
        assert "query_assessment" in result
        _check_keys(
            result["query_assessment"],
            [
                "treatment",
                "outcome",
                "directly_connected",
                "directed_path_exists",
                "effect_identifiable",
                "method",
            ],
        )

    def test_run_algorithm_schema(self):
        _ensure_stat_module()
        from causal_copilot.mcp.server import run_algorithm

        class _MockWrapper:
            def __init__(self, args=None):
                pass

            def fit(self, data, **kwargs):
                n = data.shape[1]
                return np.zeros((n, n)), {}, None

        rng = np.random.default_rng(0)
        csv = "a,b\n" + "\n".join(f"{rng.normal()},{rng.normal()}" for _ in range(60))
        with patch("causal_discovery.wrappers.PC", _MockWrapper):
            result = json.loads(run_algorithm(csv, algorithm="PC"))
        if result["status"] == "ok":
            _check_keys(
                result,
                [
                    "status",
                    "adjacency_matrix",
                    "edges",
                    "graph_kind",
                    "identifiability",
                    "run_id",
                    "provenance",
                ],
            )
            _check_keys(
                result["provenance"],
                [
                    "algorithm",
                    "requested_hyperparameters",
                    "effective_hyperparameters",
                    "resolver_adjustments",
                    "seed",
                    "planner",
                ],
            )


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
