"""Bridge between MCP server and legacy Causal-Copilot pipeline.

Handles sys.path setup, CWD management, and GlobalState construction
so the MCP server can call pipeline modules directly.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

# Pipeline root = project root (two levels up from this file)
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent

# Add pipeline modules to sys.path
_paths = [
    str(PIPELINE_ROOT),
    str(PIPELINE_ROOT / "externals"),
    str(PIPELINE_ROOT / "externals" / "causal-learn"),
]
for p in _paths:
    if p not in sys.path:
        sys.path.insert(0, p)

# Now pipeline imports work
from global_setting.state import GlobalState  # noqa: E402


def make_global_state(df, query="", algorithm=None, seed=42):
    """Construct GlobalState from a DataFrame."""
    output_dir = tempfile.mkdtemp(prefix="cc_mcp_")

    gs = GlobalState()
    gs.user_data.raw_data = df.copy()
    gs.user_data.processed_data = df.copy()
    gs.user_data.initial_query = query or "Discover causal relationships in this dataset."
    gs.user_data.selected_features = df.columns.tolist()
    gs.user_data.output_report_dir = output_dir
    gs.user_data.output_graph_dir = output_dir

    if algorithm:
        gs.algorithm.selected_algorithm = algorithm

    return gs


def make_args(query="", seed=42, debug=False):
    """Create args namespace compatible with pipeline modules."""
    return SimpleNamespace(
        data_file="<mcp>",
        output_report_dir=tempfile.mkdtemp(prefix="cc_mcp_"),
        output_graph_dir=tempfile.mkdtemp(prefix="cc_mcp_"),
        simulation_mode="offline",
        data_mode="real",
        debug=debug,
        initial_query=query or "Discover causal relationships.",
        parallel=False,
        demo_mode=False,
    )


def adj_to_edges(adj, node_names):
    """Convert adjacency matrix to edge list.

    Args:
        adj: numpy array, mat[i,j]=1 means j->i
        node_names: list of variable names

    Returns:
        list of dicts with from, to, type
    """
    edges = []
    n = len(node_names)

    for i in range(n):
        for j in range(n):
            v = int(adj[i, j])
            if v == 0:
                continue
            if v == 1:  # j->i
                edges.append({
                    "from": node_names[j],
                    "to": node_names[i],
                    "type": "directed",
                })
            elif v == 2 and i < j:  # undirected, emit once
                edges.append({
                    "from": node_names[i],
                    "to": node_names[j],
                    "type": "undirected",
                })
            elif v == 3 and i < j:  # bidirected, emit once
                edges.append({
                    "from": node_names[i],
                    "to": node_names[j],
                    "type": "bidirected",
                })

    return edges


def serialize_result(gs, node_names=None, provenance=None):
    """Serialize GlobalState results to JSON-compatible dict."""
    adj = gs.results.converted_graph
    if adj is None:
        adj = gs.results.raw_result
    if adj is None:
        return {"status": "error", "error": "No graph produced"}

    if node_names is None:
        node_names = gs.user_data.selected_features or [
            f"V{i}" for i in range(adj.shape[0])
        ]

    from causal_discovery.pdag_policy import classify_graph_kind, get_identifiable_edges

    graph_kind = classify_graph_kind(adj)
    identifiability = get_identifiable_edges(adj, node_names)
    edges = adj_to_edges(adj, node_names)

    result = {
        "status": "ok",
        "adjacency_matrix": adj.tolist(),
        "edges": edges,
        "node_names": node_names,
        "graph_kind": graph_kind,
        "identifiability": identifiability,
        "n_directed": sum(1 for e in edges if e["type"] == "directed"),
        "n_undirected": sum(1 for e in edges if e["type"] == "undirected"),
        "n_bidirected": sum(1 for e in edges if e["type"] == "bidirected"),
    }

    if provenance:
        result["provenance"] = provenance

    stats = gs.statistics
    result["data_diagnosis"] = {
        "linearity": getattr(stats, "linearity", None),
        "gaussian_error": getattr(stats, "gaussian_error", None),
        "missingness": getattr(stats, "missingness", None),
        "data_type": getattr(stats, "data_type", None),
        "sample_size": getattr(stats, "sample_size", None),
        "feature_number": getattr(stats, "feature_number", None),
        "time_series": getattr(stats, "time_series", None),
    }

    return result
