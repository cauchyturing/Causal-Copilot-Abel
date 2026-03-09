"""Causal-Copilot MCP Server.

Exposes causal discovery as tools that any LLM/Agent can call.
Install: pip install causal-copilot[mcp]
Run:     python -m causal_copilot.mcp
"""

from __future__ import annotations

import io
import json
import os
import threading
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from causal_copilot.mcp.artifacts import get_store
from causal_copilot.mcp.bridge import (
    PIPELINE_ROOT,
    adj_to_edges,
    generate_discovery_summary,
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
    instructions="""\
Causal discovery expert — turns any dataset into a causal graph.

## Tools (4)
1. **discover** — autonomous pipeline. Handles everything: data diagnosis, algorithm
   selection, hyperparameter tuning, execution, postprocessing. Use for 90% of cases.
2. **inspect_graph** — analyze a causal graph: classify (DAG/CPDAG/PAG), check if
   causal effects are identifiable, assess specific treatment→outcome queries.
   Always use after discover to answer follow-up causal questions.
3. **diagnose_data** — get data statistics (linearity, gaussianity, missingness).
   Expert mode only — discover does this automatically.
4. **run_algorithm** — run a named algorithm with explicit hyperparameters.
   Expert mode only — discover selects the best algorithm automatically.

## Workflow
- Default: discover(csv) → inspect_graph(run_id, treatment, outcome)
- Expert: diagnose_data(csv) → run_algorithm(csv, algo) → inspect_graph(run_id)
- If inspect_graph returns status="needs_more_input", follow its next_step field.

## Resources (reference material)
- causal://algorithms — list of all algorithms with descriptions
- causal://algorithms/{name} — detailed profile for one algorithm
- causal://hyperparameters/{name} — valid hyperparameters for an algorithm
- causal://guides/ci-tests — how to choose CI tests
- causal://guides/score-functions — how to choose score functions
- causal://guides/interpreting-graphs — how to read DAG/CPDAG/PAG

## Key Rules
- CPDAG/PAG edges are NOT definitive directions — say so.
- Always check inference_policy before claiming effects are identifiable.
- discover already handles algorithm selection — don't manually select unless asked.
""",
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


_pipeline_lock = threading.Lock()


@contextmanager
def _pipeline_cwd():
    """Temporarily set CWD to pipeline root (legacy code assumes it).

    Uses a process-level lock to prevent concurrent CWD changes
    from racing in FastMCP's threadpool.
    """
    with _pipeline_lock:
        prev = os.getcwd()
        os.chdir(PIPELINE_ROOT)
        try:
            yield
        finally:
            os.chdir(prev)


def _has_directed_path(adj: np.ndarray, src_idx: int, tgt_idx: int) -> bool:
    """BFS for directed path from src to tgt. adj[i,j]=1 means j→i."""
    n = adj.shape[0]
    visited = {src_idx}
    queue = [src_idx]
    while queue:
        current = queue.pop(0)
        if current == tgt_idx:
            return True
        for i in range(n):
            if adj[i, current] == 1 and i not in visited:
                visited.add(i)
                queue.append(i)
    return False


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
        raise ToolError(f"Failed to parse CSV: {e}")

    if df.empty or df.shape[1] < 2:
        raise ToolError("Need at least 2 columns of data.")

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

        # Recommend algorithm families based on diagnosis
        recommendations = []
        if diagnosis.get("time_series"):
            recommendations.append("Time-series detected → PCMCI, VARLiNGAM")
        elif diagnosis.get("linearity") is False:
            recommendations.append("Nonlinear data → consider KCI-based tests")
        elif diagnosis.get("gaussian_error") is False:
            recommendations.append("Non-Gaussian → LiNGAM family gives unique DAG")
        else:
            recommendations.append("Linear + Gaussian → PC or GES (fast, standard)")

        return json.dumps({
            "status": "ok",
            "diagnosis": diagnosis,
            "recommendations": recommendations,
            "resources": {
                "ci_test_guide": "causal://guides/ci-tests",
                "score_function_guide": "causal://guides/score-functions",
                "algorithms": "causal://algorithms",
            },
        }, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Diagnosis failed: {e}"})


@mcp.tool()
def run_algorithm(
    csv_data: str,
    algorithm: str,
    hyperparameters: str = "{}",
    seed: int = 42,
    allow_resolver_overrides: bool = True,
) -> str:
    """Run a specific causal discovery algorithm with given hyperparameters.

    No automatic selection, no postprocessing. Returns the raw graph.

    Args:
        csv_data: CSV string with header row
        algorithm: Algorithm name (e.g., "PC", "GES", "DirectLiNGAM")
        hyperparameters: JSON string of algorithm hyperparameters
        seed: Random seed
        allow_resolver_overrides: If true (default), resolvers may override CI test
            and score function based on data characteristics. Set to false to use
            your exact hyperparameters without any automatic adjustments.

    Returns:
        JSON with adjacency_matrix, edges, graph_kind, run_id, provenance
        (including requested_hyperparameters, effective_hyperparameters,
        resolver_adjustments for full transparency).
    """
    if not algorithm or not algorithm.strip():
        raise ToolError("algorithm must not be empty.")

    try:
        hp = json.loads(hyperparameters)
    except json.JSONDecodeError as e:
        raise ToolError(f"Invalid hyperparameters JSON: {e}")

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        raise ToolError(f"Failed to parse CSV: {e}")

    if df.empty or df.shape[1] < 2:
        raise ToolError("Need at least 2 columns of data.")

    try:
        from causal_discovery.ci_test_resolver import resolve_ci_test
        from causal_discovery.program import Programming
        from causal_discovery.score_resolver import resolve_score_func
        from preprocess.stat_info_functions import stat_info_collection

        gs = make_global_state(df, algorithm=algorithm, seed=seed)
        args = make_args(seed=seed)

        with _pipeline_cwd():
            gs = stat_info_collection(gs)

            # Start from user's exact hyperparameters
            requested_hp = dict(hp)
            algo_args = dict(hp)
            resolver_adjustments = {}

            if allow_resolver_overrides:
                # Resolver overrides: correct CI test / score func for the data
                ci_test_algos = {
                    "PC", "FCI", "CDNOD", "PCParallel", "InterIAMB",
                    "BAMB", "HITONMB", "IAMBnPC", "MBOR",
                }
                if algorithm in ci_test_algos:
                    resolved = resolve_ci_test(gs.statistics)
                    if resolved != algo_args.get("indep_test"):
                        resolver_adjustments["indep_test"] = {
                            "requested": algo_args.get("indep_test"),
                            "effective": resolved,
                            "reason": "data-adaptive CI test selection",
                        }
                    algo_args["indep_test"] = resolved

                score_algos = {"GES", "FGES", "XGES", "GRaSP", "ExactSearch", "BOSS"}
                if algorithm in score_algos:
                    resolved = resolve_score_func(gs.statistics, algorithm)
                    if resolved != algo_args.get("score_func"):
                        resolver_adjustments["score_func"] = {
                            "requested": algo_args.get("score_func"),
                            "effective": resolved,
                            "reason": "data-adaptive score function selection",
                        }
                    algo_args["score_func"] = resolved

                if algorithm == "PC" and gs.statistics.missingness:
                    if not algo_args.get("mvpc"):
                        resolver_adjustments["mvpc"] = {
                            "requested": algo_args.get("mvpc"),
                            "effective": True,
                            "reason": "missing data detected",
                        }
                    algo_args["mvpc"] = True

            gs.algorithm.algorithm_arguments = algo_args

            gs = Programming(args).forward(gs)

        node_names = gs.user_data.selected_features
        provenance = {
            "algorithm": algorithm,
            "requested_hyperparameters": requested_hp,
            "effective_hyperparameters": algo_args,
            "resolver_adjustments": resolver_adjustments,
            "seed": seed,
            "planner": "user-specified",
        }
        result = serialize_result(gs, node_names=node_names, provenance=provenance)

        # Save to artifact store (include data for downstream estimate_effect)
        store_payload = dict(result)
        store_payload["_processed_data"] = gs.user_data.processed_data
        store_payload["_statistics"] = gs.statistics
        run_id = get_store().save(store_payload)
        result["run_id"] = run_id
        result["resources"] = {
            "algorithm_profile": f"causal://algorithms/{algorithm}",
            "hyperparameter_spec": f"causal://hyperparameters/{algorithm}",
            "graph_guide": "causal://guides/interpreting-graphs",
        }
        result["next_steps"] = [
            f"inspect_graph(run_id='{run_id}') to classify graph and check inference eligibility",
            f"inspect_graph(run_id='{run_id}', treatment='X', outcome='Y') to assess specific causal query",
        ]

        return json.dumps(result, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Algorithm execution failed: {e}"})


@mcp.tool()
def discover(
    csv_data: str,
    query: str = "",
    algorithm: str = "",
    seed: int = 42,
    timeout: int = 300,
) -> str:
    """Full autonomous causal discovery pipeline.

    Analyzes data characteristics, selects the best algorithm (or uses specified one),
    tunes hyperparameters, executes, and refines the result. This is the primary tool —
    use it when you want the system to handle everything.

    Args:
        csv_data: CSV string with header row
        query: Optional causal question (helps LLM select algorithm)
        algorithm: Optional algorithm override (skips LLM selection)
        seed: Random seed
        timeout: Timeout in seconds

    Returns:
        JSON with adjacency_matrix, edges, graph_kind, identifiability,
        data_diagnosis, provenance, run_id, warnings
    """
    # -- Parse & validate ------------------------------------------------
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        raise ToolError(f"Failed to parse CSV: {e}")

    if df.empty or df.shape[1] < 2:
        raise ToolError("Need at least 2 columns of data.")

    if len(df) < 10:
        raise ToolError(f"Need at least 10 rows of data (got {len(df)}).")

    warnings: list[str] = []

    try:
        from causal_discovery.ci_test_resolver import resolve_ci_test
        from causal_discovery.hyperparameter_selector import HyperparameterSelector
        from causal_discovery.program import Programming
        from causal_discovery.score_resolver import resolve_score_func
        from preprocess.stat_info_functions import (
            convert_stat_info_to_text,
            stat_info_collection,
        )

        gs = make_global_state(df, query=query, algorithm=algorithm or None, seed=seed)
        args = make_args(query=query, seed=seed)

        with _pipeline_cwd():
            # 1. Statistical analysis
            gs = stat_info_collection(gs)
            stat_text = convert_stat_info_to_text(gs.statistics)
            _ = stat_text  # available for LLM prompts; unused in offline path

            # 2. Algorithm selection
            used_planner = "user-specified"
            if algorithm:
                # User override — skip LLM selection
                gs.algorithm.selected_algorithm = algorithm
                try:
                    gs = HyperparameterSelector(args).forward(gs)
                except Exception as hp_err:
                    warnings.append(f"HP selector failed, using defaults: {hp_err}")
                    from causal_copilot.mcp.offline import get_default_hp
                    gs.algorithm.algorithm_arguments = get_default_hp(
                        algorithm, gs.statistics,
                    )
            else:
                # Full LLM path: Filter → Reranker → HP
                try:
                    from causal_discovery.filter import Filter
                    from causal_discovery.rerank import Reranker

                    gs = Filter(args).forward(gs)
                    gs = Reranker(args).forward(gs)
                    gs = HyperparameterSelector(args).forward(gs)
                    used_planner = "llm"
                except Exception as llm_err:
                    warnings.append(
                        f"LLM selection failed, using rule-based: {llm_err}"
                    )
                    from causal_copilot.mcp.offline import (
                        get_default_hp,
                        select_algorithm_offline,
                    )
                    gs.algorithm.selected_algorithm = select_algorithm_offline(
                        gs.statistics,
                    )
                    gs.algorithm.algorithm_arguments = get_default_hp(
                        gs.algorithm.selected_algorithm, gs.statistics,
                    )
                    used_planner = "rule-based-fallback"

            # 3. Resolver overrides (CI test / score func)
            algo_name = gs.algorithm.selected_algorithm
            algo_args = dict(gs.algorithm.algorithm_arguments or {})

            ci_test_algos = {
                "PC", "FCI", "CDNOD", "PCParallel", "InterIAMB",
                "BAMB", "HITONMB", "IAMBnPC", "MBOR",
            }
            if algo_name in ci_test_algos:
                algo_args["indep_test"] = resolve_ci_test(gs.statistics)

            score_algos = {"GES", "FGES", "XGES", "GRaSP", "ExactSearch", "BOSS"}
            if algo_name in score_algos:
                algo_args["score_func"] = resolve_score_func(
                    gs.statistics, algo_name,
                )

            if algo_name == "PC" and gs.statistics.missingness:
                algo_args["mvpc"] = True

            gs.algorithm.algorithm_arguments = algo_args

            # 4. Execute
            gs = Programming(args).forward(gs)

            # 5. Postprocess (skip for time-series)
            is_ts = getattr(gs.statistics, "time_series", False)
            if not is_ts:
                try:
                    from postprocess.judge import Judge
                    gs = Judge(gs, args).forward(gs, "cot_all_relation", 1)
                except Exception as pp_err:
                    warnings.append(f"Postprocessing skipped: {pp_err}")

        # 6. Serialize
        node_names = gs.user_data.selected_features
        provenance = {
            "algorithm": gs.algorithm.selected_algorithm,
            "hyperparameters": gs.algorithm.algorithm_arguments,
            "seed": seed,
            "planner": used_planner,
        }
        result = serialize_result(gs, node_names=node_names, provenance=provenance)

        # Enrich with human-ready summary (deterministic, no LLM call)
        result.update(generate_discovery_summary(result))

        if warnings:
            result["warnings"] = warnings

        # Save to artifact store (include data for downstream estimate_effect)
        store_payload = dict(result)
        store_payload["_processed_data"] = gs.user_data.processed_data
        store_payload["_statistics"] = gs.statistics
        run_id = get_store().save(store_payload)
        result["run_id"] = run_id

        algo = gs.algorithm.selected_algorithm or "unknown"
        result["resources"] = {
            "algorithm_profile": f"causal://algorithms/{algo}",
            "graph_guide": "causal://guides/interpreting-graphs",
        }
        result["next_steps"] = [
            f"inspect_graph(run_id='{run_id}') to classify graph and check inference eligibility",
            f"inspect_graph(run_id='{run_id}', treatment='X', outcome='Y') to assess a specific causal query",
        ]

        return json.dumps(result, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        payload: dict[str, Any] = {
            "status": "error",
            "error": f"Pipeline failed: {e}",
        }
        if warnings:
            payload["warnings"] = warnings
        return json.dumps(payload)


@mcp.tool()
def inspect_graph(
    run_id: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    data_diagnosis: str = "",
    treatment: str = "",
    outcome: str = "",
) -> str:
    """Analyze a causal graph: classification, inference policy, query assessment.

    Two input modes (mutually exclusive):
    1. run_id from discover/run_algorithm (preferred — includes cached diagnosis)
    2. adjacency_matrix + node_names (+ optional data_diagnosis for CPDAG inference)

    Optional: treatment + outcome triggers query_assessment for a specific causal query.

    Args:
        run_id: Run ID from a previous discover or run_algorithm call.
        adjacency_matrix: JSON 2D array (mat[i][j]=1 means j causes i). Use with node_names.
        node_names: JSON array of variable names. Required with adjacency_matrix.
        data_diagnosis: JSON with linearity/gaussian_error fields. Needed for CPDAG
                        inference policy when not using run_id.
        treatment: Treatment variable name (triggers query_assessment).
        outcome: Outcome variable name (triggers query_assessment).

    Returns:
        JSON with graph_kind, graph_stats, identifiability, inference_policy,
        query_assessment (if treatment/outcome), summary, key_findings, limitations.
    """
    # --- Resolve inputs ---
    adj = None
    names = None
    diagnosis = None

    if run_id and adjacency_matrix:
        raise ToolError("run_id and adjacency_matrix are mutually exclusive.")

    if run_id:
        cached = get_store().get(run_id)
        if cached is None:
            raise ToolError(f"run_id '{run_id}' not found or expired.")
        adj = np.array(cached["adjacency_matrix"])
        names = cached["node_names"]
        diagnosis = cached.get("data_diagnosis")
    elif adjacency_matrix:
        if not node_names:
            raise ToolError("node_names required when using adjacency_matrix.")
        try:
            adj_list = json.loads(adjacency_matrix)
            names = json.loads(node_names)
        except (json.JSONDecodeError, TypeError) as e:
            raise ToolError(f"Invalid JSON: {e}")
        adj = np.array(adj_list)
        if data_diagnosis:
            try:
                diagnosis = json.loads(data_diagnosis)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid data_diagnosis JSON: {e}")
    else:
        raise ToolError("Provide either run_id or adjacency_matrix + node_names.")

    # --- Validate ---
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ToolError("Adjacency matrix must be square.")
    if adj.shape[0] != len(names):
        raise ToolError(f"Matrix dimension {adj.shape[0]} != {len(names)} node names.")

    # --- Graph analysis ---
    graph_kind = classify_graph_kind(adj)
    identifiability = get_identifiable_edges(adj, names)
    edges = adj_to_edges(adj, names)
    n_directed = sum(1 for e in edges if e["type"] == "directed")
    n_undirected = sum(1 for e in edges if e["type"] == "undirected")
    n_bidirected = sum(1 for e in edges if e["type"] == "bidirected")
    n_edges = len(edges)
    n_nodes = len(names)

    graph_stats = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_directed": n_directed,
        "n_undirected": n_undirected,
        "n_bidirected": n_bidirected,
        "density": round(n_edges / max(n_nodes * (n_nodes - 1) / 2, 1), 3),
    }

    # --- Inference policy ---
    if graph_kind == "dag":
        inference_policy = {
            "eligibility": True,
            "method": "standard",
            "reason": "DAG — all causal effects identifiable",
            "assumptions_used": ["causal sufficiency", "faithfulness"],
        }
    elif graph_kind == "cpdag":
        if diagnosis is None:
            return json.dumps({
                "status": "needs_more_input",
                "graph_kind": graph_kind,
                "graph_stats": graph_stats,
                "missing_inputs": ["data_diagnosis"],
                "next_step": (
                    "CPDAG inference requires data diagnosis (linearity + gaussianity). "
                    "Use run_id from discover/run_algorithm, or provide data_diagnosis "
                    "with linearity and gaussian_error fields."
                ),
            })
        is_lg = bool(diagnosis.get("linearity")) and bool(diagnosis.get("gaussian_error"))
        policy = check_inference_policy(adj, is_linear_gaussian=is_lg)
        inference_policy = {
            "eligibility": policy["allow_inference"],
            "method": policy["method"],
            "reason": policy["reason"],
            "assumptions_used": (
                ["linearity", "Gaussian errors", "causal sufficiency"]
                if is_lg else ["causal sufficiency", "faithfulness"]
            ),
        }
    elif graph_kind == "pag":
        inference_policy = {
            "eligibility": False,
            "method": None,
            "reason": "PAG — latent confounders possible, effects not identifiable",
            "assumptions_used": [],
        }
    else:
        inference_policy = {
            "eligibility": False,
            "method": None,
            "reason": f"Unknown graph kind: {graph_kind}",
            "assumptions_used": [],
        }

    # --- Query assessment (optional) ---
    query_assessment = None
    if treatment or outcome:
        if not treatment or not outcome:
            raise ToolError("Both treatment and outcome must be provided together.")
        if treatment == outcome:
            raise ToolError("Treatment and outcome must be different variables.")
        if treatment not in names:
            raise ToolError(f"Treatment '{treatment}' not in node_names: {names}")
        if outcome not in names:
            raise ToolError(f"Outcome '{outcome}' not in node_names: {names}")

        src_idx = names.index(treatment)
        tgt_idx = names.index(outcome)
        path_exists = _has_directed_path(adj, src_idx, tgt_idx)
        directly_connected = bool(adj[tgt_idx, src_idx] == 1)

        query_assessment = {
            "treatment": treatment,
            "outcome": outcome,
            "directly_connected": directly_connected,
            "directed_path_exists": path_exists,
            "effect_identifiable": inference_policy["eligibility"] and path_exists,
            "method": inference_policy["method"] if path_exists else None,
        }

    # --- Summary ---
    summary = (
        f"{graph_kind.upper()} with {n_nodes} variables, {n_edges} edges "
        f"({n_directed} directed, {n_undirected} undirected, "
        f"{n_bidirected} bidirected)."
    )

    key_findings = []
    if graph_kind == "dag":
        key_findings.append("Fully oriented DAG — all effects identifiable")
    elif graph_kind == "cpdag":
        key_findings.append(f"CPDAG — {n_undirected} edge directions ambiguous")
    elif graph_kind == "pag":
        key_findings.append("PAG — latent confounders possible")

    if inference_policy["eligibility"]:
        key_findings.append(
            f"Causal inference possible via {inference_policy['method']}"
        )
    else:
        key_findings.append(
            f"Causal inference blocked: {inference_policy['reason']}"
        )

    if query_assessment:
        if query_assessment["effect_identifiable"]:
            key_findings.append(
                f"Effect of {treatment} on {outcome} is identifiable"
            )
        elif query_assessment["directed_path_exists"]:
            key_findings.append(
                f"Path {treatment} -> {outcome} exists but effect not identifiable"
            )
        else:
            key_findings.append(
                f"No directed path from {treatment} to {outcome}"
            )

    limitations = []
    if graph_kind != "dag":
        limitations.append(
            f"Graph is {graph_kind.upper()} — some causal directions uncertain"
        )
    if n_bidirected > 0:
        limitations.append(
            f"{n_bidirected} bidirected edges suggest latent confounders"
        )

    # --- Result ---
    result: dict[str, Any] = {
        "status": "ok",
        "graph_kind": graph_kind,
        "graph_stats": graph_stats,
        "identifiability": identifiability,
        "inference_policy": inference_policy,
        "summary": summary,
        "key_findings": key_findings,
        "limitations": limitations,
        "resources": {
            "graph_guide": "causal://guides/interpreting-graphs",
        },
    }

    if query_assessment:
        result["query_assessment"] = query_assessment

    # Contextual next steps based on graph state
    next_steps = []
    if graph_kind == "cpdag" and not inference_policy["eligibility"]:
        next_steps.append(
            "Try DirectLiNGAM via run_algorithm — LiNGAM gives unique DAG if errors are non-Gaussian"
        )
    if graph_kind == "pag":
        next_steps.append(
            "PAG detected — consider using PC (without latent variable assumption) for a CPDAG instead"
        )
    if not treatment and not outcome and inference_policy["eligibility"]:
        next_steps.append(
            "Specify treatment and outcome to assess a specific causal query"
        )
    if next_steps:
        result["next_steps"] = next_steps

    return json.dumps(result, indent=2, cls=_NumpyEncoder)


# ── MCP Resources ─────────────────────────────────────────────────────
from causal_copilot.mcp.resources import (
    get_algorithm_resources,
    get_algorithm_content,
    get_hp_content,
    get_guide_content,
    get_all_guide_names,
)


@mcp.resource("causal://algorithms")
def algorithms_index():
    """List all available causal discovery algorithms."""
    return json.dumps(get_algorithm_resources(), indent=2)


@mcp.resource("causal://algorithms/{name}")
def algorithm_profile(name: str):
    """Get detailed profile for a specific algorithm."""
    content = get_algorithm_content(name)
    if content is None:
        return json.dumps({"error": f"Algorithm '{name}' not found"})
    return content


@mcp.resource("causal://hyperparameters/{name}")
def hp_spec(name: str):
    """Get hyperparameter specification for an algorithm."""
    content = get_hp_content(name)
    if content is None:
        return json.dumps({"error": f"HP spec for '{name}' not found"})
    return content


@mcp.resource("causal://guides/{guide_name}")
def guide(guide_name: str):
    """Get a methodology guide document."""
    content = get_guide_content(guide_name)
    if content is None:
        return json.dumps({"error": f"Guide '{guide_name}' not found"})
    return content


# ── MCP Prompts ───────────────────────────────────────────────────────
from causal_copilot.mcp.prompts import PROMPTS


@mcp.prompt()
def causal_expert():
    """Expert system prompt for causal discovery."""
    return PROMPTS["causal-expert"]


@mcp.prompt()
def analyze_dataset():
    """Step-by-step workflow for analyzing a dataset."""
    return PROMPTS["analyze-dataset"]
