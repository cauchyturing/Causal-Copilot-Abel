"""PDAG policy: what inference is valid for each graph type.

- DAG: full inference (all edges identified)
- CPDAG: IDA when linear-Gaussian, else reject ambiguous edges
- PAG: never coerce to DAG, reject inference on ambiguous edges

This prevents the scientific error of treating undirected edges as directed.
"""
import numpy as np

_DIRECTED = 1
_UNDIRECTED = 2
_BIDIRECTED = 3
_PAG_TYPES = {4, 5, 6, 7}


def classify_graph_kind(adj):
    """Classify adjacency matrix as dag, cpdag, or pag.

    Args:
        adj: numpy array, adjacency matrix (mat[i,j] convention: j->i)

    Returns:
        str: 'dag', 'cpdag', or 'pag'
    """
    values = set(np.unique(adj).astype(int)) - {0}
    if values & _PAG_TYPES:
        return "pag"
    if _BIDIRECTED in values:
        return "pag"  # bidirected (↔) implies latent confounders
    if _UNDIRECTED in values:
        return "cpdag"
    return "dag"


def check_inference_policy(adj, is_linear_gaussian=False):
    """Determine what causal inference is valid for this graph.

    Args:
        adj: adjacency matrix
        is_linear_gaussian: True if data is linear with Gaussian errors

    Returns:
        dict with keys:
            allow_inference (bool): whether inference is valid
            method (str): 'standard', 'ida', or None
            reason (str): explanation
            graph_kind (str): 'dag', 'cpdag', 'pag'
    """
    kind = classify_graph_kind(adj)

    if kind == "dag":
        return {
            "allow_inference": True,
            "method": "standard",
            "reason": "All edges directed — full causal inference valid.",
            "graph_kind": kind,
        }

    if kind == "cpdag":
        if is_linear_gaussian:
            return {
                "allow_inference": True,
                "method": "ida",
                "reason": "CPDAG with linear-Gaussian data — IDA provides valid bounds.",
                "graph_kind": kind,
            }
        return {
            "allow_inference": False,
            "method": None,
            "reason": "CPDAG with ambiguous edges — cannot identify causal effects "
                      "without linear-Gaussian assumption.",
            "graph_kind": kind,
        }

    # PAG
    return {
        "allow_inference": False,
        "method": None,
        "reason": "PAG output — latent confounders possible. "
                  "Cannot coerce to DAG for inference.",
        "graph_kind": kind,
    }


def get_identifiable_edges(adj, node_names):
    """Partition edges into identifiable and ambiguous.

    Args:
        adj: adjacency matrix
        node_names: list of variable names

    Returns:
        dict with 'identifiable' (directed edges) and 'ambiguous' (undirected/PAG)
    """
    n = len(node_names)
    identifiable = []
    ambiguous = []

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and adj[j, i] == 0:
                continue
            if adj[i, j] == _DIRECTED and adj[j, i] == 0:
                identifiable.append({"from": node_names[j], "to": node_names[i]})
            elif adj[j, i] == _DIRECTED and adj[i, j] == 0:
                identifiable.append({"from": node_names[i], "to": node_names[j]})
            else:
                if adj[i, j] == _BIDIRECTED or adj[j, i] == _BIDIRECTED:
                    edge_type = "bidirected"
                elif adj[i, j] == _UNDIRECTED or adj[j, i] == _UNDIRECTED:
                    edge_type = "undirected"
                else:
                    edge_type = "pag"
                ambiguous.append({"nodes": [node_names[i], node_names[j]],
                                  "type": edge_type})

    return {"identifiable": identifiable, "ambiguous": ambiguous}
