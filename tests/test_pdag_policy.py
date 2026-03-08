"""Tests for PDAG->inference policy."""
import numpy as np
import pytest


class TestClassifyGraphKind:
    def test_dag_all_directed(self):
        from causal_discovery.pdag_policy import classify_graph_kind
        adj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        assert classify_graph_kind(adj) == "dag"

    def test_cpdag_has_undirected(self):
        from causal_discovery.pdag_policy import classify_graph_kind
        adj = np.array([[0, 0, 0], [1, 0, 2], [0, 2, 0]])
        assert classify_graph_kind(adj) == "cpdag"

    def test_pag_has_circle_marks(self):
        from causal_discovery.pdag_policy import classify_graph_kind
        adj = np.array([[0, 4], [5, 0]])
        assert classify_graph_kind(adj) == "pag"

    def test_empty_graph(self):
        from causal_discovery.pdag_policy import classify_graph_kind
        adj = np.zeros((3, 3))
        assert classify_graph_kind(adj) == "dag"

    def test_bidirected_classifies_as_pag(self):
        """Bidirected edges (value 3) imply latent confounders → PAG-like."""
        from causal_discovery.pdag_policy import classify_graph_kind
        adj = np.array([[0, 3], [0, 0]])
        assert classify_graph_kind(adj) == "pag"


class TestInferencePolicy:
    def test_dag_allows_full_inference(self):
        from causal_discovery.pdag_policy import check_inference_policy
        adj = np.array([[0, 0], [1, 0]])
        result = check_inference_policy(adj, is_linear_gaussian=True)
        assert result["allow_inference"] is True
        assert result["method"] == "standard"

    def test_cpdag_linear_gaussian_allows_ida(self):
        from causal_discovery.pdag_policy import check_inference_policy
        adj = np.array([[0, 2], [2, 0]])
        result = check_inference_policy(adj, is_linear_gaussian=True)
        assert result["allow_inference"] is True
        assert result["method"] == "ida"

    def test_cpdag_nonlinear_rejects(self):
        from causal_discovery.pdag_policy import check_inference_policy
        adj = np.array([[0, 2], [2, 0]])
        result = check_inference_policy(adj, is_linear_gaussian=False)
        assert result["allow_inference"] is False
        assert "ambiguous" in result["reason"]

    def test_pag_always_rejects(self):
        from causal_discovery.pdag_policy import check_inference_policy
        adj = np.array([[0, 4], [5, 0]])
        result = check_inference_policy(adj, is_linear_gaussian=True)
        assert result["allow_inference"] is False
        assert "PAG" in result["reason"]

    def test_bidirected_rejects_inference(self):
        """Bidirected edges → latent confounders → reject inference."""
        from causal_discovery.pdag_policy import check_inference_policy
        adj = np.array([[0, 3], [3, 0]])
        result = check_inference_policy(adj, is_linear_gaussian=True)
        assert result["allow_inference"] is False
        assert result["graph_kind"] == "pag"

    def test_identifiable_edges(self):
        from causal_discovery.pdag_policy import get_identifiable_edges
        adj = np.array([[0, 0, 0], [1, 0, 2], [0, 2, 0]])
        names = ["X", "Y", "Z"]
        ident = get_identifiable_edges(adj, names)
        assert {"from": "X", "to": "Y"} in ident["identifiable"]
        assert any("Y" in e["nodes"] and "Z" in e["nodes"] for e in ident["ambiguous"])

    def test_bidirected_edge_labeled_explicitly(self):
        """Bidirected edges should be labeled as 'bidirected', not 'pag'."""
        from causal_discovery.pdag_policy import get_identifiable_edges
        adj = np.array([[0, 3], [3, 0]])
        names = ["A", "B"]
        ident = get_identifiable_edges(adj, names)
        assert len(ident["ambiguous"]) == 1
        assert ident["ambiguous"][0]["type"] == "bidirected"
