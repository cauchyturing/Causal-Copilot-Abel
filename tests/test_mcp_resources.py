"""Tests for MCP resources and prompts."""
import json
import pytest


class TestAlgorithmResources:
    def test_list_algorithm_resources(self):
        from causal_copilot.mcp.resources import get_algorithm_resources
        resources = get_algorithm_resources()
        assert len(resources) >= 5
        for r in resources:
            assert "name" in r
            assert "uri" in r
            assert r["uri"].startswith("causal://algorithms/")

    def test_algorithm_resource_content(self):
        from causal_copilot.mcp.resources import get_algorithm_content
        content = get_algorithm_content("PC")
        assert content is not None
        assert len(content) > 10


class TestGuideResources:
    def test_ci_test_guide(self):
        from causal_copilot.mcp.resources import get_guide_content
        content = get_guide_content("ci-tests")
        assert "fisherz" in content
        assert "kci" in content

    def test_score_function_guide(self):
        from causal_copilot.mcp.resources import get_guide_content
        content = get_guide_content("score-functions")
        assert "BIC" in content
        assert "BDeu" in content

    def test_graph_guide(self):
        from causal_copilot.mcp.resources import get_guide_content
        content = get_guide_content("interpreting-graphs")
        assert "DAG" in content


class TestPrompts:
    def test_causal_expert_prompt(self):
        from causal_copilot.mcp.prompts import PROMPTS
        assert "causal-expert" in PROMPTS
        prompt = PROMPTS["causal-expert"]
        assert len(prompt) > 200
        assert "causal" in prompt.lower()

    def test_analyze_dataset_prompt(self):
        from causal_copilot.mcp.prompts import PROMPTS
        assert "analyze-dataset" in PROMPTS
        prompt = PROMPTS["analyze-dataset"]
        assert "diagnose" in prompt.lower()


class TestExplainResultTool:
    def test_basic(self):
        from causal_copilot.mcp.server import explain_result
        adj = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        names = ["X", "Y", "Z"]
        result = json.loads(explain_result(adj, names))
        assert "graph_kind" in result
        assert result["graph_kind"] == "dag"
        assert "identifiability" in result

    def test_cpdag(self):
        from causal_copilot.mcp.server import explain_result
        adj = [[0, 2], [2, 0]]
        names = ["A", "B"]
        result = json.loads(explain_result(adj, names))
        assert result["graph_kind"] == "cpdag"
