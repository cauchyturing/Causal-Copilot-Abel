"""Causal-Copilot MCP Server.

Exposes causal discovery as tools that any LLM/Agent can call.
Install: pip install causal-copilot[mcp]
Run:     python -m causal_copilot.mcp
"""

from __future__ import annotations

import io
import json
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from causal_copilot import CausalCopilot, __version__
from causal_copilot.algorithms.registry import REGISTRY, available_algorithms
from causal_copilot.mcp.artifacts import get_store
from causal_copilot.mcp.bridge import (
    PIPELINE_ROOT,
    adj_to_edges,
    make_args,
    make_global_state,
    serialize_result,
)
from causal_discovery.pdag_policy import (
    classify_graph_kind,
    check_inference_policy,
    get_identifiable_edges,
)

mcp = FastMCP(
    "Causal-Copilot",
    instructions=(
        "Causal discovery expert. Use the `analyze` tool to discover causal "
        "relationships in tabular data. Use `list_algorithms` to see available "
        "methods and their strengths. Use `explain_graph` to interpret results."
    ),
)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@contextmanager
def _pipeline_cwd():
    """Temporarily set CWD to pipeline root (legacy code assumes it)."""
    prev = os.getcwd()
    os.chdir(PIPELINE_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _adj_to_edges(adj: np.ndarray, columns: list[str]) -> list[dict[str, str]]:
    """Convert adjacency matrix to a list of edge dicts for LLM consumption."""
    edges = []
    n = adj.shape[0]
    seen = set()
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 0:
                continue
            key = (min(i, j), max(i, j))
            if adj[i, j] == 1:
                # j → i (directed)
                edges.append({
                    "from": columns[j],
                    "to": columns[i],
                    "type": "directed",
                })
            elif adj[i, j] == 2 and key not in seen:
                edges.append({
                    "from": columns[i],
                    "to": columns[j],
                    "type": "undirected",
                })
                seen.add(key)
            elif adj[i, j] == 3 and key not in seen:
                edges.append({
                    "from": columns[i],
                    "to": columns[j],
                    "type": "bidirected",
                })
                seen.add(key)
    return edges


def _format_result(result) -> dict[str, Any]:
    """Format CausalResult into an LLM-friendly dict."""
    output: dict[str, Any] = {
        "status": result.status,
        "summary": result.summary,
    }

    if result.adjacency_matrix is not None:
        columns = result.node_names or [f"V{i}" for i in range(result.adjacency_matrix.shape[0])]
        output["edges"] = _adj_to_edges(result.adjacency_matrix, columns)
        output["node_names"] = columns
        output["n_edges"] = len(output["edges"])
        output["adjacency_matrix"] = result.adjacency_matrix.tolist()

    if result.assumptions:
        output["assumptions"] = result.assumptions

    if result.warnings:
        output["warnings"] = result.warnings

    if result.provenance:
        p = result.provenance
        output["provenance"] = {
            "algorithm": p.algorithm,
            "planner": p.planner,
            "seed": p.seed,
            "runtime_seconds": round(p.runtime_seconds, 2),
        }

    return output


@mcp.tool()
def analyze(
    csv_data: str,
    query: str = "",
    algorithm: str | None = None,
    seed: int = 42,
    timeout: int = 300,
) -> str:
    """Run causal discovery on tabular CSV data.

    Discovers causal relationships (X causes Y) from observational data.
    Returns the causal graph as directed edges, a natural language summary,
    and full provenance for reproducibility.

    Args:
        csv_data: CSV-formatted data as a string (with header row).
        query: Optional causal question to guide analysis (e.g. "What causes churn?").
        algorithm: Force a specific algorithm (e.g. "PC", "GES", "FCI").
                   If omitted, the best algorithm is selected automatically.
        seed: Random seed for reproducibility (default: 42).
        timeout: Maximum seconds for the algorithm to run (default: 300).

    Returns:
        JSON with edges, summary, assumptions, and provenance.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Failed to parse CSV: {e}"})

    if df.empty or df.shape[1] < 2:
        return json.dumps({"status": "error", "error": "Need at least 2 columns of data."})

    copilot = CausalCopilot()

    try:
        result = copilot.analyze(
            df,
            algorithm=algorithm,
            seed=seed,
            timeout=timeout,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Analysis failed: {e}"})

    output = _format_result(result)

    # Add interpretation hints for the LLM
    if output.get("edges"):
        directed = [e for e in output["edges"] if e["type"] == "directed"]
        undirected = [e for e in output["edges"] if e["type"] == "undirected"]
        hints = []
        if directed:
            hints.append(
                f"{len(directed)} directed edge(s) found — these represent "
                f"likely causal relationships."
            )
        if undirected:
            hints.append(
                f"{len(undirected)} undirected edge(s) — causal direction "
                f"could not be determined."
            )
        output["interpretation_hints"] = hints

    return json.dumps(output, indent=2)


@mcp.tool()
def list_algorithms(
    filter: str = "available",
) -> str:
    """List causal discovery algorithms with their capabilities.

    Args:
        filter: Which algorithms to show.
            - "available": Only algorithms whose dependencies are installed (default).
            - "all": All 19 registered algorithms.
            - "timeseries": Time-series capable algorithms.
            - "constraint": Constraint-based algorithms (PC, FCI, etc.).
            - "score": Score-based algorithms (GES, GRaSP, etc.).
            - "functional": Functional model algorithms (LiNGAM variants).
            - "latent": Algorithms that handle latent confounders (FCI).

    Returns:
        JSON list of algorithms with name, family, tags, and availability.
    """
    if filter == "available":
        algos = available_algorithms()
    elif filter == "all":
        algos = REGISTRY
    else:
        # Filter by family or tag
        algos = {}
        for name, spec in REGISTRY.items():
            if filter == spec.family:
                algos[name] = spec
            elif filter in spec.tags:
                algos[name] = spec

    avail_set = set(available_algorithms().keys())

    result = []
    for name, spec in sorted(algos.items()):
        entry = {
            "name": name,
            "family": spec.family,
            "tags": list(spec.tags),
            "available": name in avail_set,
            "upstream_packages": spec.upstream_packages,
        }
        # Add human-readable guidance
        if "timeseries" in spec.tags:
            entry["best_for"] = "Time-series / temporal data"
        elif "non-gaussian" in spec.tags:
            entry["best_for"] = "Non-Gaussian continuous data (can identify unique DAG)"
        elif "latent-confounders" in spec.tags:
            entry["best_for"] = "Data with possible unmeasured confounders"
        elif "nonlinear" in spec.tags:
            entry["best_for"] = "Nonlinear relationships"
        elif spec.family == "constraint":
            entry["best_for"] = "General-purpose, works well with Gaussian data"
        elif spec.family == "score":
            entry["best_for"] = "General-purpose, score-based search"
        else:
            entry["best_for"] = "Specialized use case"

        if not entry["available"]:
            entry["install_hint"] = f"pip install {' '.join(spec.upstream_packages)}"

        result.append(entry)

    return json.dumps(result, indent=2)


@mcp.tool()
def explain_graph(
    adjacency_matrix: list[list[int]],
    node_names: list[str],
) -> str:
    """Explain a causal graph in natural language.

    Converts an adjacency matrix into a human-readable description of
    causal relationships, including direct causes, effects, and potential
    confounders.

    Args:
        adjacency_matrix: Square matrix where mat[i][j]=1 means j causes i.
        node_names: Names for each node (must match matrix dimensions).

    Returns:
        Natural language explanation of the causal structure.
    """
    adj = np.array(adjacency_matrix)
    n = adj.shape[0]

    if adj.shape[0] != adj.shape[1]:
        return json.dumps({"error": "Adjacency matrix must be square."})
    if len(node_names) != n:
        return json.dumps({"error": f"Expected {n} node names, got {len(node_names)}."})

    edges = _adj_to_edges(adj, node_names)

    # Build per-node analysis
    causes_of: dict[str, list[str]] = {name: [] for name in node_names}
    effects_of: dict[str, list[str]] = {name: [] for name in node_names}
    undirected_neighbors: dict[str, list[str]] = {name: [] for name in node_names}

    for e in edges:
        if e["type"] == "directed":
            causes_of[e["to"]].append(e["from"])
            effects_of[e["from"]].append(e["to"])
        elif e["type"] == "undirected":
            undirected_neighbors[e["from"]].append(e["to"])
            undirected_neighbors[e["to"]].append(e["from"])

    # Find roots (no causes) and leaves (no effects)
    roots = [name for name in node_names if not causes_of[name] and effects_of[name]]
    leaves = [name for name in node_names if causes_of[name] and not effects_of[name]]
    mediators = [
        name for name in node_names
        if causes_of[name] and effects_of[name]
    ]

    # Build explanation
    lines = []
    directed = [e for e in edges if e["type"] == "directed"]
    undirected = [e for e in edges if e["type"] == "undirected"]
    bidirected = [e for e in edges if e["type"] == "bidirected"]

    lines.append(f"Causal graph: {n} variables, {len(edges)} edges.")
    lines.append("")

    if directed:
        lines.append("Causal relationships:")
        for e in directed:
            lines.append(f"  {e['from']} → {e['to']}")

    if undirected:
        lines.append("")
        lines.append("Associated but direction unknown:")
        for e in undirected:
            lines.append(f"  {e['from']} — {e['to']}")

    if bidirected:
        lines.append("")
        lines.append("Bidirected (possible hidden common cause):")
        for e in bidirected:
            lines.append(f"  {e['from']} ↔ {e['to']}")

    lines.append("")
    if roots:
        lines.append(f"Root causes (no parents): {', '.join(roots)}")
    if leaves:
        lines.append(f"Terminal effects (no children): {', '.join(leaves)}")
    if mediators:
        lines.append(f"Mediators (both cause and effect): {', '.join(mediators)}")

    # Identify chains
    for name in mediators:
        for cause in causes_of[name]:
            for effect in effects_of[name]:
                lines.append(
                    f"  Causal chain: {cause} → {name} → {effect} "
                    f"({name} mediates the effect of {cause} on {effect})"
                )

    explanation = {
        "explanation": "\n".join(lines),
        "graph_stats": {
            "n_nodes": n,
            "n_directed_edges": len(directed),
            "n_undirected_edges": len(undirected),
            "n_bidirected_edges": len(bidirected),
            "root_causes": roots,
            "terminal_effects": leaves,
            "mediators": mediators,
        },
    }
    return json.dumps(explanation, indent=2)


@mcp.tool()
def explain_result(
    adjacency_matrix: list[list[int]],
    node_names: list[str],
    run_id: str = "",
) -> str:
    """Explain a causal graph in natural language with identifiability info.

    Enhanced version of explain_graph with graph_kind and identifiability.

    Args:
        adjacency_matrix: 2D array (mat[i,j]=1 means j->i)
        node_names: Variable names
        run_id: Optional run_id from previous call

    Returns:
        JSON with explanation, graph_stats, graph_kind, identifiability
    """
    base_result = json.loads(explain_graph(adjacency_matrix, node_names))
    if "error" in base_result:
        return json.dumps(base_result)

    adj = np.array(adjacency_matrix)
    base_result["graph_kind"] = classify_graph_kind(adj)
    base_result["identifiability"] = get_identifiable_edges(adj, node_names)

    return json.dumps(base_result)


@mcp.tool()
def diagnose_data(csv_data: str) -> str:
    """Analyze dataset statistical characteristics for causal discovery.

    Returns linearity, gaussianity, missingness, data type, sample size,
    feature count, and time-series detection. Use this to understand your
    data before choosing an algorithm.

    Args:
        csv_data: CSV string with header row

    Returns:
        JSON with diagnosis dict + status
    """
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Failed to parse CSV: {e}"})

    if df.empty or df.shape[1] < 2:
        return json.dumps({"status": "error", "error": "Need at least 2 columns of data."})

    try:
        from preprocess.stat_info_functions import stat_info_collection

        gs = make_global_state(df)
        with _pipeline_cwd():
            gs = stat_info_collection(gs)

        stats = gs.statistics

        def _jsonable(v):
            """Coerce numpy types to Python builtins for JSON."""
            if isinstance(v, (np.bool_, np.integer)):
                return v.item()
            if isinstance(v, np.floating):
                return float(v)
            return v

        diagnosis = {
            "linearity": _jsonable(getattr(stats, "linearity", None)),
            "gaussian_error": _jsonable(getattr(stats, "gaussian_error", None)),
            "missingness": _jsonable(getattr(stats, "missingness", None)),
            "data_type": getattr(stats, "data_type", None),
            "sample_size": _jsonable(getattr(stats, "sample_size", None)),
            "feature_number": _jsonable(getattr(stats, "feature_number", None)),
            "time_series": _jsonable(getattr(stats, "time_series", None)),
            "features": gs.user_data.selected_features,
        }
        return json.dumps({"status": "ok", "diagnosis": diagnosis}, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Diagnosis failed: {e}"})


@mcp.tool()
def run_algorithm(
    csv_data: str,
    algorithm: str,
    hyperparameters: str = "{}",
    seed: int = 42,
) -> str:
    """Run a specific causal discovery algorithm with given hyperparameters.

    No automatic selection, no postprocessing. Returns the raw graph.

    Args:
        csv_data: CSV string with header row
        algorithm: Algorithm name (e.g., "PC", "GES", "DirectLiNGAM")
        hyperparameters: JSON string of algorithm hyperparameters
        seed: Random seed

    Returns:
        JSON with adjacency_matrix, edges, graph_kind, run_id, provenance
    """
    if not algorithm or not algorithm.strip():
        return json.dumps({"status": "error", "error": "algorithm must not be empty."})

    try:
        hp = json.loads(hyperparameters)
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "error": f"Invalid hyperparameters JSON: {e}"})

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Failed to parse CSV: {e}"})

    if df.empty or df.shape[1] < 2:
        return json.dumps({"status": "error", "error": "Need at least 2 columns of data."})

    try:
        from causal_discovery.ci_test_resolver import resolve_ci_test
        from causal_discovery.program import Programming
        from causal_discovery.score_resolver import resolve_score_func
        from preprocess.stat_info_functions import stat_info_collection

        gs = make_global_state(df, algorithm=algorithm, seed=seed)
        args = make_args(seed=seed)

        with _pipeline_cwd():
            gs = stat_info_collection(gs)

            # Apply user hyperparameters
            algo_args = dict(hp)

            # Resolver overrides: correct CI test / score func for the data
            ci_test_algos = {
                "PC", "FCI", "CDNOD", "PCParallel", "InterIAMB",
                "BAMB", "HITONMB", "IAMBnPC", "MBOR",
            }
            if algorithm in ci_test_algos:
                algo_args["indep_test"] = resolve_ci_test(gs.statistics)

            score_algos = {"GES", "FGES", "XGES", "GRaSP", "ExactSearch", "BOSS"}
            if algorithm in score_algos:
                algo_args["score_func"] = resolve_score_func(gs.statistics, algorithm)

            if algorithm == "PC" and gs.statistics.missingness:
                algo_args["mvpc"] = True

            gs.algorithm.algorithm_arguments = algo_args

            gs = Programming(args).forward(gs)

        node_names = gs.user_data.selected_features
        provenance = {
            "algorithm": algorithm,
            "hyperparameters": algo_args,
            "seed": seed,
            "planner": "user-specified",
        }
        result = serialize_result(gs, node_names=node_names, provenance=provenance)

        # Save to artifact store
        run_id = get_store().save(result)
        result["run_id"] = run_id

        return json.dumps(result, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Algorithm execution failed: {e}"})
