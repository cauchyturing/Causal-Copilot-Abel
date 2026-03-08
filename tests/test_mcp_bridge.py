"""Tests for the MCP-to-pipeline bridge."""
import pytest
import numpy as np
import pandas as pd


class TestBridgeImports:
    def test_setup_pipeline_adds_paths(self):
        from causal_copilot.mcp.bridge import PIPELINE_ROOT
        import sys
        assert str(PIPELINE_ROOT) in sys.path

    def test_can_import_global_state(self):
        from causal_copilot.mcp.bridge import make_global_state
        assert callable(make_global_state)


class TestMakeGlobalState:
    def test_from_dataframe(self):
        from causal_copilot.mcp.bridge import make_global_state
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        gs = make_global_state(df)
        assert gs.user_data.raw_data is not None
        assert gs.user_data.selected_features == ["x", "y"]

    def test_with_query(self):
        from causal_copilot.mcp.bridge import make_global_state
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        gs = make_global_state(df, query="What causes y?")
        assert "y" in gs.user_data.initial_query


class TestAdjToEdges:
    def test_directed_chain(self):
        from causal_copilot.mcp.bridge import adj_to_edges
        # X->Y->Z: adj[1,0]=1 (X->Y), adj[2,1]=1 (Y->Z)
        adj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        edges = adj_to_edges(adj, ["X", "Y", "Z"])
        assert {"from": "X", "to": "Y", "type": "directed"} in edges
        assert {"from": "Y", "to": "Z", "type": "directed"} in edges

    def test_undirected(self):
        from causal_copilot.mcp.bridge import adj_to_edges
        adj = np.array([[0, 2], [2, 0]])
        edges = adj_to_edges(adj, ["A", "B"])
        assert len(edges) == 1
        assert edges[0]["type"] == "undirected"
