"""Bridge between MCP server and legacy Causal-Copilot pipeline.

Handles sys.path setup, CWD management, and GlobalState construction
so the MCP server can call pipeline modules directly.
"""

from __future__ import annotations

import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

# Pipeline root = project root (two levels up from this file)
PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_pipeline_importable():
    """Add pipeline modules to sys.path when running from source checkout.

    When pip-installed, all modules are already importable via the wheel.
    This fallback only activates if global_setting can't be imported normally.
    """
    try:
        import global_setting.state  # noqa: F401

        return  # Already importable (pip-installed or previously patched)
    except ImportError:
        pass

    _paths = [
        str(PIPELINE_ROOT),
        str(PIPELINE_ROOT / "externals"),
        str(PIPELINE_ROOT / "externals" / "causal-learn"),
    ]
    for p in _paths:
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_pipeline_importable()

from global_setting.state import GlobalState  # noqa: E402


def make_global_state(df, query="", algorithm=None, seed=42):
    """Construct GlobalState from a DataFrame."""
    output_dir = tempfile.mkdtemp(prefix="cc_mcp_")

    gs = GlobalState()
    gs.user_data.raw_data = df.copy()
    gs.user_data.processed_data = df.copy()
    gs.user_data.initial_query = query or "Discover causal relationships in this dataset."
    cols = df.columns.tolist()
    gs.user_data.selected_features = cols
    # visual_selected_features: used by stat_info_functions linearity_check/
    # gaussian_check for datasets with ≥10 features (selects subset for plots).
    # Without this, stat_info_collection crashes with TypeError: df_raw[None].
    gs.user_data.visual_selected_features = cols
    # knowledge_docs: downstream code (HP Selector, Filter, Reranker) calls
    # '\n'.join(knowledge_docs) — None causes TypeError. Default to empty list.
    gs.user_data.knowledge_docs = []
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
                edges.append(
                    {
                        "from": node_names[j],
                        "to": node_names[i],
                        "type": "directed",
                    }
                )
            elif v == 2 and i < j:  # undirected, emit once
                edges.append(
                    {
                        "from": node_names[i],
                        "to": node_names[j],
                        "type": "undirected",
                    }
                )
            elif v == 3 and i < j:  # bidirected, emit once
                edges.append(
                    {
                        "from": node_names[i],
                        "to": node_names[j],
                        "type": "bidirected",
                    }
                )
            elif v in (4, 5, 6, 7) and i < j:  # PAG edge types, emit once
                # 4=circle-tail, 5=circle-arrow, 6=tail-tail, 7=arrow-arrow
                pag_labels = {4: "circle-tail", 5: "circle-arrow", 6: "tail-tail", 7: "arrow-arrow"}
                edges.append(
                    {
                        "from": node_names[i],
                        "to": node_names[j],
                        "type": "pag",
                        "pag_detail": pag_labels.get(v, f"pag-{v}"),
                    }
                )

    return edges


def serialize_result(gs, node_names=None, provenance=None):
    """Serialize GlobalState results to JSON-compatible dict.

    Prefers revised_graph (post-Judge refinement) over converted_graph.
    Includes bootstrap edge confidence when available.
    """
    # Prefer refined graph (post-Judge) over raw converted graph
    adj = getattr(gs.results, "revised_graph", None)
    used_revised = adj is not None
    if adj is None:
        adj = gs.results.converted_graph
    if adj is None:
        adj = gs.results.raw_result

    # Handle time-series lagged_graph (3D array: [n_lags, n_vars, n_vars])
    # Collapse into a 2D summary graph for serialization.
    lagged = getattr(gs.results, "lagged_graph", None)
    if adj is None and lagged is not None:
        import numpy as _np

        adj = _np.any(lagged, axis=0).astype(int)
        used_revised = False

    if adj is None:
        return {"status": "error", "error": "No graph produced"}

    if node_names is None:
        node_names = gs.user_data.selected_features or [f"V{i}" for i in range(adj.shape[0])]

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
        "graph_refined": used_revised,
    }

    # Include bootstrap edge confidence when available
    # bootstrap_probability is a dict of ndarrays:
    #   certain_edges, uncertain_edges, bi_edges,
    #   half_certain_edges, half_uncertain_edges, none_edges, none_existence
    boot_prob = getattr(gs.results, "bootstrap_probability", None)
    if boot_prob is not None and isinstance(boot_prob, dict):
        edge_confidence = {}
        # Map edge types to bootstrap probability layers
        layer_map = {
            "directed": "certain_edges",  # j→i
            "undirected": "uncertain_edges",  # j-i
            "bidirected": "bi_edges",  # j↔i
        }
        for e in edges:
            etype = e["type"]
            layer_key = layer_map.get(etype)
            if layer_key is None:
                continue
            prob_mat = boot_prob.get(layer_key)
            if prob_mat is None:
                continue
            src, tgt = e["from"], e["to"]
            try:
                si = node_names.index(src)
                ti = node_names.index(tgt)
                # mat[i,j] = P(j→i) for directed; symmetric for undirected/bi
                conf = float(prob_mat[ti, si])
                arrow = "->" if etype == "directed" else ("-" if etype == "undirected" else "<->")
                edge_confidence[f"{src}{arrow}{tgt}"] = round(conf, 3)
            except (ValueError, IndexError):
                pass
        if edge_confidence:
            result["edge_confidence"] = edge_confidence

    # Include LLM pruning decisions when available (Judge transparency)
    llm_decisions = getattr(gs.results, "llm_errors", None)
    if llm_decisions and isinstance(llm_decisions, dict):
        pruning = {}
        for key, label in [("direct_record", "confirmed"), ("forbid_record", "rejected")]:
            record = llm_decisions.get(key)
            if record:
                named = []
                for pair in record:
                    try:
                        j, i = int(pair[0]), int(pair[1])
                        named.append(f"{node_names[j]}->{node_names[i]}")
                    except (ValueError, IndexError):
                        named.append(str(pair))
                pruning[label] = named
        if pruning:
            result["llm_pruning"] = pruning

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


def generate_discovery_summary(result):
    """Generate deterministic summary from structured discovery result.

    Returns dict with summary, key_findings, limitations, algorithm_rationale.
    All fields derived from structured state — no LLM calls.
    """
    graph_kind = result.get("graph_kind", "unknown")
    node_names = result.get("node_names", [])
    edges = result.get("edges", [])
    n_directed = result.get("n_directed", 0)
    n_undirected = result.get("n_undirected", 0)
    n_bidirected = result.get("n_bidirected", 0)
    provenance = result.get("provenance", {})
    algo = provenance.get("algorithm", "unknown")
    planner = provenance.get("planner", "unknown")
    diagnosis = result.get("data_diagnosis", {})
    n_nodes = len(node_names)
    n_edges = len(edges)

    # Summary
    summary = (
        f"Causal discovery on {n_nodes} variables using {algo}. "
        f"{n_edges} edges ({n_directed} directed, {n_undirected} undirected, "
        f"{n_bidirected} bidirected). Graph type: {graph_kind.upper()}."
    )

    # Key findings
    key_findings = []
    if graph_kind == "dag":
        key_findings.append("Fully oriented DAG — all causal directions determined")
    elif graph_kind == "cpdag":
        key_findings.append(f"CPDAG — {n_directed} edges oriented, {n_undirected} ambiguous")
    elif graph_kind == "pag":
        key_findings.append("PAG — possible latent confounders detected")

    if n_directed > 0:
        sources = Counter(e["from"] for e in edges if e["type"] == "directed")
        if sources:
            top_name, top_count = sources.most_common(1)[0]
            key_findings.append(f"Most influential variable: {top_name} ({top_count} outgoing edges)")

    if n_edges == 0:
        key_findings.append("No edges discovered — variables appear independent")

    # Limitations
    limitations = []
    if graph_kind == "cpdag":
        limitations.append("Some edge directions ambiguous — consider LiNGAM for unique DAG if data is non-Gaussian")
    elif graph_kind == "pag":
        limitations.append("Latent confounders possible — causal effect estimation unreliable")

    if diagnosis:
        sample_size = diagnosis.get("sample_size")
        if sample_size and sample_size < 100:
            limitations.append(f"Small sample ({sample_size} rows) — results may be unstable")

    limitations.append("Observational data cannot prove causation — validate with domain knowledge")

    # Algorithm rationale
    if planner == "llm":
        rationale = f"LLM pipeline selected {algo} based on data characteristics"
    elif planner == "rule-based-fallback":
        rationale = f"Rule-based fallback selected {algo} (LLM unavailable)"
    elif planner == "user-specified":
        rationale = f"User specified {algo}"
    else:
        rationale = f"{algo} selected by {planner}"

    if diagnosis:
        reasons = []
        if diagnosis.get("linearity") is False:
            reasons.append("nonlinear data")
        if diagnosis.get("gaussian_error") is False:
            reasons.append("non-Gaussian errors")
        if diagnosis.get("time_series"):
            reasons.append("time-series structure")
        if diagnosis.get("missingness"):
            reasons.append("missing data")
        if reasons:
            rationale += f" ({', '.join(reasons)})"

    return {
        "summary": summary,
        "key_findings": key_findings,
        "limitations": limitations,
        "algorithm_rationale": rationale,
    }
