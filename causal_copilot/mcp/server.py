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
    check_inference_policy,
    classify_graph_kind,
    get_identifiable_edges,
)

mcp = FastMCP(
    "Causal-Copilot",
    instructions="""\
Causal discovery & inference expert — turns any dataset into a causal graph, estimates effects, and performs causal reasoning.

## Tools (13)
### Core Pipeline
1. **discover** — autonomous pipeline. Handles everything: data diagnosis, algorithm
   selection, hyperparameter tuning, execution, postprocessing. Use for 90% of cases.
2. **inspect_graph** — analyze a causal graph: classify (DAG/CPDAG/PAG), check if
   causal effects are identifiable, assess specific treatment→outcome queries.
3. **estimate_effect** — estimate causal effect of treatment on outcome (ATE/ATT with CIs).
   Methods: linear, matching, dml, drl, metalearner, iv. Checks inference eligibility first.
4. **diagnose_data** — get data statistics (linearity, gaussianity, missingness). Expert mode.
5. **run_algorithm** — run a named algorithm with explicit hyperparameters. Expert mode.

### Causal Reasoning
6. **refute_estimate** — sensitivity analysis: test robustness of a causal effect estimate.
   Runs 3 refutation methods (data subset, random cause, placebo).
7. **estimate_counterfactual** — answer "what if?": what would outcome be if treatment
   were set to a specific value? Uses DoWhy GCM.
8. **attribute_anomaly** — root cause analysis: which causal parents drive anomalous
   values of a target variable? Uses DoWhy GCM.
9. **attribute_distribution_change** — explain distribution shifts: which causal mechanisms
   changed between two time periods? Uses DoWhy GCM.
10. **simulate_intervention** — interventional what-if: simulate shifting or setting a
    treatment value and see the outcome distribution change. Uses DoWhy GCM.

### Analysis & Validation
11. **compute_feature_importance** — SHAP-based feature importance: which variables most
    influence a target? Uses linear or tree SHAP.
12. **validate_graph** — graph falsification: test if the discovered causal graph is
    consistent with the data. Uses DoWhy GCM LMC testing.

### Reporting
13. **generate_report** — generate a comprehensive PDF report from a discovery run.
    Covers EDA, algorithm selection, graph analysis, bootstrap confidence, and inference.
    Requires LLM access + LaTeX. Returns PDF path.

## Workflow
- **Best**: diagnose_data(csv) → generate domain knowledge from knowledge_prompt →
  discover(csv, domain_knowledge=...) → inspect_graph → estimate_effect
  → generate_report(run_id) for comprehensive PDF report
  (Use the causal_analysis prompt for the full workflow)
- Quick: discover(csv) → estimate_effect(run_id, T, Y)
- Validate: estimate_effect(…) → refute_estimate(run_id, T, Y) for robustness
- Graph check: discover(csv) → validate_graph(run_id) to test graph-data consistency
- Feature drivers: discover(csv) → compute_feature_importance(run_id, target) for SHAP
- What-if: discover(csv) → estimate_counterfactual(run_id, T, Y, value)
- Root cause: discover(csv) → attribute_anomaly(run_id, target_node)
- Expert: diagnose_data(csv) → run_algorithm(csv, algo) → estimate_effect(run_id, T, Y)

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
- estimate_effect checks this automatically — trust its "rejected" status.
- discover already handles algorithm selection — don't manually select unless asked.
- GCM tools (counterfactual, anomaly, distribution_change, intervention) require a DAG.
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


def _build_knowledge_prompt(diagnosis: dict) -> str:
    """Generate a data-adaptive prompt asking the agent for domain knowledge.

    The prompt focuses on what matters for THIS dataset — not a generic template.
    The agent generates knowledge from this prompt, then passes it to discover().
    """
    features = diagnosis.get("features", [])
    n_features = len(features)
    sample_size = diagnosis.get("sample_size", 0)

    # Determine if variable names look meaningful (not just X1, X2, V1, V2)
    import re

    generic_pattern = re.compile(r"^[VXvx]\d+$")
    meaningful_names = not all(generic_pattern.match(str(f)) for f in features)

    sections = []
    sections.append(f"The dataset has {n_features} variables and {sample_size} observations.")

    if meaningful_names:
        var_list = ", ".join(str(f) for f in features[:30])
        if n_features > 30:
            var_list += f", ... ({n_features - 30} more)"
        sections.append(
            f"Variables: {var_list}\n\n"
            "Based on these variable names, please provide domain knowledge:\n"
            "1. **Variable descriptions**: What does each variable measure? Units and typical ranges.\n"
            "2. **Known causal relationships**: Which variables are known causes/effects of others?\n"
            "3. **Forbidden edges**: Are there pairs where causation is impossible "
            "(e.g., 'age cannot be caused by income')?\n"
            "4. **Potential confounders**: What unmeasured variables might confound observed relationships?"
        )
    else:
        sections.append(
            "Variable names are generic (X1, X2, ...). "
            "If you know the domain context, describe what each variable represents "
            "and their expected causal relationships. "
            "If no domain knowledge is available, say 'No domain knowledge available.'"
        )

    # Data-specific follow-ups
    if diagnosis.get("time_series"):
        sections.append(
            "**Time-series detected**: What are the expected temporal lags between variables? "
            "Are there seasonal patterns or regime changes?"
        )
    if diagnosis.get("linearity") is False:
        sections.append(
            "**Nonlinear relationships detected**: What nonlinear mechanisms might exist? "
            "Thresholds, saturation effects, interactions?"
        )
    if diagnosis.get("gaussian_error") is False:
        sections.append(
            "**Non-Gaussian errors detected**: This enables unique DAG identification. "
            "Are there known asymmetric or heavy-tailed distributions in this domain?"
        )
    if diagnosis.get("missingness"):
        sections.append(
            "**Missing data detected**: What's the likely mechanism — "
            "Missing Completely At Random (MCAR), Missing At Random (MAR), "
            "or Missing Not At Random (MNAR)?"
        )
    if sample_size and sample_size < 200:
        sections.append(
            f"**Small sample ({sample_size} rows)**: Domain knowledge is especially "
            "valuable here. Any known structural constraints will improve results."
        )
    if n_features > 30:
        sections.append(
            f"**High-dimensional ({n_features} variables)**: Are there known variable "
            "groups or clusters? Which variables are most likely to be causally central?"
        )

    return "\n\n".join(sections)


def _parse_background_knowledge(
    forbidden_edges: str,
    required_edges: str,
    warnings: list[str],
) -> dict | None:
    """Parse forbidden/required edges into background_knowledge spec.

    Returns dict compatible with wrapper._create_background_knowledge(),
    or None if no constraints provided.
    """
    spec: dict = {}

    if forbidden_edges and forbidden_edges.strip():
        try:
            fe = json.loads(forbidden_edges)
            if isinstance(fe, list) and fe:
                spec["forbidden_edges"] = [[str(a), str(b)] for a, b in fe]
        except (json.JSONDecodeError, ValueError) as e:
            warnings.append(f"Invalid forbidden_edges JSON: {e}")

    if required_edges and required_edges.strip():
        try:
            re_edges = json.loads(required_edges)
            if isinstance(re_edges, list) and re_edges:
                spec["required_edges"] = [[str(a), str(b)] for a, b in re_edges]
        except (json.JSONDecodeError, ValueError) as e:
            warnings.append(f"Invalid required_edges JSON: {e}")

    return spec if spec else None


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


def _sanitize_for_estimation(adj: np.ndarray, node_names: list[str]):
    """Drop undirected/bidirected edges for clean DAG estimation.

    Returns (clean_adj, dropped_edges_list).
    """
    clean = adj.copy()
    dropped = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if clean[i, j] != 0 and clean[j, i] != 0:
                dropped.append(f"{node_names[i]} -- {node_names[j]}")
                clean[i, j] = 0
                clean[j, i] = 0
    return clean, dropped


def _identify_confounders(adj: np.ndarray, names: list[str], t_idx: int, o_idx: int) -> list[str]:
    """Find shared parents of treatment and outcome in the graph.

    adj[i,j]=1 means j→i. Parents of k = {j : adj[k,j] == 1}.
    """
    n = adj.shape[0]
    t_parents = {j for j in range(n) if adj[t_idx, j] == 1 and j != t_idx}
    o_parents = {j for j in range(n) if adj[o_idx, j] == 1 and j != o_idx}
    shared = sorted(t_parents & o_parents)
    return [names[j] for j in shared]


def _adj_to_dot(adj: np.ndarray, names: list[str]) -> str:
    """Convert adjacency matrix to DOT format for DoWhy. Only directed edges."""
    edges = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:  # j→i
                edges.append(f"{names[j]} -> {names[i]}")
    return "digraph { " + "; ".join(edges) + "; }"


def _build_interpretation(treatment, outcome, method, estimates, confounders) -> str:
    """Build human-readable interpretation string."""
    ate_info = estimates.get("ate", {})
    est = ate_info.get("estimate")
    if est is None:
        return f"Could not estimate the effect of {treatment} on {outcome}."

    parts = [f"A one-unit increase in {treatment} causes {outcome} to change by {est:.4f}"]
    ci_lo = ate_info.get("ci_lower")
    ci_hi = ate_info.get("ci_upper")
    if ci_lo is not None and ci_hi is not None:
        parts.append(f"(95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])")
    p_val = ate_info.get("p_value")
    if p_val is not None:
        parts.append(f"(p={p_val:.4f})")
    if confounders:
        parts.append(f"adjusting for {', '.join(confounders)}")
    return ", ".join(parts) + "."


def _resolve_data_and_graph(
    run_id: str,
    csv_data: str,
    adjacency_matrix: str,
    node_names: str,
    data_diagnosis: str = "",
    require_data: bool = True,
):
    """Resolve inputs from either run_id or explicit args.

    Returns (df, adj, names, diagnosis) tuple.
    """
    adj = None
    names = None
    diagnosis = None
    df = None

    if run_id and csv_data:
        raise ToolError("run_id and csv_data are mutually exclusive.")

    if run_id:
        cached = get_store().get(run_id)
        if cached is None:
            raise ToolError(f"run_id '{run_id}' not found or expired.")
        adj = np.array(cached["adjacency_matrix"])
        names = cached["node_names"]
        diagnosis = cached.get("data_diagnosis")
        if require_data:
            stored_data = cached.get("_processed_data")
            if stored_data is None:
                raise ToolError(f"run_id '{run_id}' has no stored data. Re-run discover or run_algorithm to populate.")
            df = stored_data if isinstance(stored_data, pd.DataFrame) else pd.DataFrame(stored_data)
    elif csv_data:
        if not adjacency_matrix:
            raise ToolError("adjacency_matrix required when using csv_data.")
        if not node_names:
            raise ToolError("node_names required when using csv_data.")
        try:
            df = pd.read_csv(io.StringIO(csv_data))
        except Exception as e:
            raise ToolError(f"Failed to parse CSV: {e}") from e
        try:
            adj = np.array(json.loads(adjacency_matrix))
            names = json.loads(node_names)
        except (json.JSONDecodeError, TypeError) as e:
            raise ToolError(f"Invalid JSON: {e}") from e
        if data_diagnosis:
            try:
                diagnosis = json.loads(data_diagnosis)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid data_diagnosis JSON: {e}") from e
    else:
        raise ToolError("Provide either run_id or csv_data + adjacency_matrix + node_names.")

    if adj is not None:
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ToolError("Adjacency matrix must be square.")
        if adj.shape[0] != len(names):
            raise ToolError(f"Matrix dimension {adj.shape[0]} != {len(names)} node names.")

    return df, adj, names, diagnosis


def _find_instrument(adj: np.ndarray, names: list[str], t_idx: int, o_idx: int) -> str | None:
    """Find a valid instrumental variable from the graph.

    An IV Z must:
    1. Directly cause T: adj[t_idx, z_idx] == 1
    2. NOT directly cause O: adj[o_idx, z_idx] != 1
    3. Have no parents (simplified independence check)
    """
    n = adj.shape[0]
    for z_idx in range(n):
        if z_idx in (t_idx, o_idx):
            continue
        if adj[t_idx, z_idx] != 1:
            continue
        if adj[o_idx, z_idx] == 1:
            continue
        has_parents = any(adj[z_idx, j] == 1 for j in range(n) if j != z_idx)
        if not has_parents:
            return names[z_idx]
    return None


@mcp.tool()
def estimate_effect(
    treatment: str,
    outcome: str,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    method: str = "",
    control_value: float = 0.0,
    treatment_value: float = 1.0,
    confounders: str = "",
    data_diagnosis: str = "",
    instrument: str = "",
) -> str:
    """Estimate the causal effect of treatment on outcome.

    Requires a causal graph (DAG preferred). Checks inference eligibility
    before estimating — rejects if effects are not identifiable.

    Two input modes (mutually exclusive):
    1. run_id from discover/run_algorithm (preferred — includes cached data + graph)
    2. csv_data + adjacency_matrix + node_names (standalone)

    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        run_id: Run ID from a previous discover or run_algorithm call
        csv_data: CSV string with header row
        adjacency_matrix: JSON 2D array (mat[i][j]=1 means j causes i)
        node_names: JSON array of variable names
        method: Estimation method ("linear", "matching", "dml", "drl",
                "metalearner", "iv", or "" for auto)
        control_value: Reference value for control group (default 0.0)
        treatment_value: Reference value for treatment group (default 1.0)
        confounders: JSON array of confounder names (default: auto-detect from graph)
        data_diagnosis: JSON with linearity/gaussian_error (needed for CPDAG)
        instrument: Instrument variable name for IV method (auto-detected from graph if empty)

    Returns:
        JSON with status, estimates (ATE/ATT with CIs), confounders_used,
        interpretation, provenance, run_id, next_steps
    """
    valid_methods = {"linear", "matching", "dml", "drl", "metalearner", "iv", ""}
    if method not in valid_methods:
        raise ToolError(
            f"Unknown method '{method}'. Valid: linear, matching, dml, drl, metalearner, iv (or empty for auto)."
        )

    # --- Resolve inputs ---
    adj = None
    names = None
    diagnosis = None
    df = None

    if run_id and csv_data:
        raise ToolError("run_id and csv_data are mutually exclusive.")

    if run_id:
        cached = get_store().get(run_id)
        if cached is None:
            raise ToolError(f"run_id '{run_id}' not found or expired.")
        adj = np.array(cached["adjacency_matrix"])
        names = cached["node_names"]
        diagnosis = cached.get("data_diagnosis")
        stored_data = cached.get("_processed_data")
        if stored_data is None:
            raise ToolError(f"run_id '{run_id}' has no stored data. Re-run discover or run_algorithm to populate.")
        df = stored_data if isinstance(stored_data, pd.DataFrame) else pd.DataFrame(stored_data)
    elif csv_data:
        if not adjacency_matrix:
            raise ToolError("adjacency_matrix required when using csv_data.")
        if not node_names:
            raise ToolError("node_names required when using csv_data.")
        try:
            df = pd.read_csv(io.StringIO(csv_data))
        except Exception as e:
            raise ToolError(f"Failed to parse CSV: {e}") from e
        try:
            adj = np.array(json.loads(adjacency_matrix))
            names = json.loads(node_names)
        except (json.JSONDecodeError, TypeError) as e:
            raise ToolError(f"Invalid JSON: {e}") from e
        if data_diagnosis:
            try:
                diagnosis = json.loads(data_diagnosis)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid data_diagnosis JSON: {e}") from e
    else:
        raise ToolError("Provide either run_id or csv_data + adjacency_matrix + node_names.")

    # --- Validate ---
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ToolError("Adjacency matrix must be square.")
    if adj.shape[0] != len(names):
        raise ToolError(f"Matrix dimension {adj.shape[0]} != {len(names)} node names.")
    if treatment not in names:
        raise ToolError(f"Treatment '{treatment}' not in node_names: {names}")
    if outcome not in names:
        raise ToolError(f"Outcome '{outcome}' not in node_names: {names}")
    if treatment == outcome:
        raise ToolError("Treatment and outcome must be different.")
    if treatment not in df.columns:
        raise ToolError(f"Treatment '{treatment}' not in data columns: {df.columns.tolist()}")
    if outcome not in df.columns:
        raise ToolError(f"Outcome '{outcome}' not in data columns: {df.columns.tolist()}")

    # --- Data properties for intelligent method/model selection ---
    from causal_copilot.mcp.offline import (
        get_default_estimation_config as _offline_select_method_config,
    )
    from causal_copilot.mcp.offline import (
        identify_confounders as _offline_confounders,
    )
    from causal_copilot.mcp.offline import (
        prepare_treatment,
    )
    from causal_copilot.mcp.offline import (
        select_estimation_method as _offline_select_method,
    )

    _, _, _, treatment_kind = prepare_treatment(df, treatment)
    is_linear = bool(diagnosis.get("linearity", True)) if diagnosis else True
    is_gaussian = bool(diagnosis.get("gaussian_error", True)) if diagnosis else True

    # --- Time-series warning (causal effect estimation assumes i.i.d.) ---
    if diagnosis and diagnosis.get("time_series"):
        warnings_list: list[str] = [
            "Time-series structure detected — causal effect estimation "
            "assumes i.i.d. samples. Results may be biased if temporal "
            "lag structure matters. Consider time-series-specific methods."
        ]
    else:
        warnings_list: list[str] = []

    # --- Inference policy (honest gate) ---
    graph_kind = classify_graph_kind(adj)

    if graph_kind == "dag":
        inference_policy = {
            "eligibility": True,
            "method": "standard",
            "reason": "DAG — all causal effects identifiable",
        }
    elif graph_kind == "cpdag":
        if diagnosis is None:
            return json.dumps(
                {
                    "status": "rejected",
                    "treatment": treatment,
                    "outcome": outcome,
                    "reason": "CPDAG requires data diagnosis to check inference eligibility. "
                    "Use run_id from discover, or provide data_diagnosis.",
                    "graph_kind": "cpdag",
                    "next_steps": [
                        "Use run_id from discover/run_algorithm (includes diagnosis)",
                        "Provide data_diagnosis with linearity and gaussian_error fields",
                    ],
                }
            )
        is_lg = bool(diagnosis.get("linearity")) and bool(diagnosis.get("gaussian_error"))
        policy = check_inference_policy(adj, is_linear_gaussian=is_lg)
        if not policy["allow_inference"]:
            return json.dumps(
                {
                    "status": "rejected",
                    "treatment": treatment,
                    "outcome": outcome,
                    "reason": policy["reason"],
                    "graph_kind": "cpdag",
                    "inference_policy": {
                        "eligibility": False,
                        "reason": policy["reason"],
                    },
                    "next_steps": [
                        "Try DirectLiNGAM via run_algorithm — gives unique DAG if errors are non-Gaussian",
                        "Collect experimental data to resolve edge directions",
                    ],
                }
            )
        inference_policy = {
            "eligibility": True,
            "method": policy["method"],
            "reason": policy["reason"],
        }
        warnings_list.append("CPDAG: undirected edges dropped for estimation")
    elif graph_kind == "pag":
        return json.dumps(
            {
                "status": "rejected",
                "treatment": treatment,
                "outcome": outcome,
                "reason": "PAG — latent confounders possible, effects not identifiable",
                "graph_kind": "pag",
                "inference_policy": {
                    "eligibility": False,
                    "reason": "PAG — latent confounders possible",
                },
                "next_steps": [
                    "Use PC (without latent variable assumption) for a CPDAG instead",
                    "Collect experimental data",
                ],
            }
        )
    else:
        return json.dumps(
            {
                "status": "rejected",
                "treatment": treatment,
                "outcome": outcome,
                "reason": f"Unknown graph kind: {graph_kind}",
                "graph_kind": graph_kind,
            }
        )

    # --- Sanitize graph ---
    clean_adj, dropped_edges = _sanitize_for_estimation(adj, names)
    if dropped_edges:
        warnings_list.append(
            f"Dropped {len(dropped_edges)} undirected/bidirected edges: "
            + ", ".join(dropped_edges[:5])
            + ("..." if len(dropped_edges) > 5 else "")
        )

    # --- Confounders ---
    t_idx = names.index(treatment)
    o_idx = names.index(outcome)
    if confounders:
        try:
            conf_list = json.loads(confounders)
        except json.JSONDecodeError as e:
            raise ToolError(f"Invalid confounders JSON: {e}") from e
        conf_source = "user-specified"
    else:
        conf_list, potential_conf = _offline_confounders(adj, names, treatment, outcome)
        conf_source = "auto-detected-from-graph"
        if potential_conf:
            warnings_list.append(f"Potential confounders (undirected edges): {', '.join(potential_conf)}")

    # --- Method selection ---
    if method:
        selected_method = method
    else:
        has_iv = bool(instrument) or (_find_instrument(clean_adj, names, t_idx, o_idx) is not None)
        selected_method = _offline_select_method(
            df,
            treatment,
            treatment_kind,
            is_linear=is_linear,
            is_gaussian=is_gaussian,
            n_features=len(names) - 1,
            has_instrument=has_iv and treatment_kind == "continuous",
        )

    # --- Run estimation ---
    try:
        from causal_copilot.mcp.estimation import (
            estimate_dml,
            estimate_drl,
            estimate_iv,
            estimate_linear,
            estimate_matching,
            estimate_metalearner,
        )

        if selected_method == "linear":
            dot_graph = _adj_to_dot(clean_adj, names)
            with _pipeline_cwd():
                estimates = estimate_linear(
                    df,
                    dot_graph,
                    treatment,
                    outcome,
                    control_value,
                    treatment_value,
                )
            method_detail = "DoWhy backdoor.linear_regression"

        elif selected_method == "matching":
            match_conf = conf_list if conf_list else [c for c in names if c != treatment and c != outcome]
            with _pipeline_cwd():
                estimates = estimate_matching(
                    df,
                    treatment,
                    outcome,
                    match_conf,
                    int(control_value),
                    int(treatment_value),
                )
            method_detail = "Propensity Score Matching (sklearn)"

        elif selected_method == "dml":
            X_col = [c for c in names if c != treatment and c != outcome and c not in conf_list]
            if not X_col:
                X_col = conf_list[:] if conf_list else [c for c in names if c != treatment and c != outcome]
            W_col = conf_list if conf_list else []
            with _pipeline_cwd():
                estimates = estimate_dml(
                    df,
                    treatment,
                    outcome,
                    X_col,
                    W_col,
                    control_value,
                    treatment_value,
                    is_linear=is_linear,
                    treatment_kind=treatment_kind,
                )
            algo_name = estimates.get("algo", "LinearDML")
            method_detail = f"Double Machine Learning (EconML {algo_name})"

        elif selected_method == "drl":
            X_col = [c for c in names if c != treatment and c != outcome and c not in conf_list]
            if not X_col:
                X_col = conf_list[:] if conf_list else [c for c in names if c != treatment and c != outcome]
            W_col = conf_list if conf_list else []
            with _pipeline_cwd():
                estimates = estimate_drl(
                    df,
                    treatment,
                    outcome,
                    X_col,
                    W_col,
                    control_value,
                    treatment_value,
                    is_linear=is_linear,
                    treatment_kind=treatment_kind,
                )
            algo_name = estimates.get("algo", "LinearDRL")
            method_detail = f"Doubly Robust Learning (EconML {algo_name})"

        elif selected_method == "metalearner":
            X_col = [c for c in names if c != treatment and c != outcome]
            ml_config = _offline_select_method_config(
                "metalearner",
                df,
                treatment,
                is_linear=is_linear,
            )
            learner_type = ml_config.get("learner", "t")
            with _pipeline_cwd():
                estimates = estimate_metalearner(
                    df,
                    treatment,
                    outcome,
                    X_col,
                    control_value,
                    treatment_value,
                    learner=learner_type,
                )
            algo_name = estimates.get("algo", f"{learner_type.upper()}Learner")
            method_detail = f"Meta-Learner {algo_name} (EconML)"

        elif selected_method == "iv":
            iv_var = instrument
            if not iv_var:
                iv_var = _find_instrument(clean_adj, names, t_idx, o_idx)
            if not iv_var:
                return json.dumps(
                    {
                        "status": "error",
                        "treatment": treatment,
                        "outcome": outcome,
                        "method": "iv",
                        "error": "No valid instrument variable found in graph. "
                        "Provide instrument parameter or use a different method.",
                        "next_steps": [
                            "Specify instrument variable explicitly",
                            "Use method='dml' or method='drl' instead",
                        ],
                    }
                )
            if iv_var not in names or iv_var not in df.columns:
                raise ToolError(f"Instrument '{iv_var}' not in data/graph.")
            X_col = [c for c in names if c not in (treatment, outcome, iv_var)]
            W_col = conf_list if conf_list else []
            with _pipeline_cwd():
                estimates = estimate_iv(
                    df,
                    treatment,
                    outcome,
                    iv_var,
                    X_col,
                    W_col,
                    control_value,
                    treatment_value,
                )
            method_detail = f"Instrumental Variables (EconML LinearDRIV, instrument={iv_var})"

        else:
            raise ToolError(f"Unknown method '{selected_method}'.")

    except ToolError:
        raise
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "treatment": treatment,
                "outcome": outcome,
                "method": selected_method,
                "error": f"Estimation failed: {e}",
                "next_steps": [
                    "Try a different method (linear, matching, dml, drl, metalearner, iv)",
                    "Check that treatment and outcome columns contain valid numeric data",
                ],
            }
        )

    # --- Build result ---
    interpretation = _build_interpretation(
        treatment,
        outcome,
        selected_method,
        estimates,
        conf_list,
    )

    result_payload: dict[str, Any] = {
        "status": "ok",
        "treatment": treatment,
        "outcome": outcome,
        "method": selected_method,
        "method_detail": method_detail,
        "estimates": estimates,
        "confounders_used": conf_list,
        "confounders_source": conf_source,
        "graph_kind": graph_kind,
        "interpretation": interpretation,
        "treatment_kind": treatment_kind,
        "control_value": control_value,
        "treatment_value": treatment_value,
        "provenance": {
            "method": selected_method,
            "algo": estimates.get("algo"),
            "inference_policy": inference_policy,
            "llm_used": False,
            "n_observations": len(df),
            "treatment_kind": treatment_kind,
            "is_linear": is_linear,
            "is_gaussian": is_gaussian,
            "graph_sanitization": {"edges_dropped": len(dropped_edges)},
        },
    }

    if warnings_list:
        result_payload["warnings"] = warnings_list

    # Resources + next steps
    result_payload["resources"] = {
        "graph_guide": "causal://guides/interpreting-graphs",
    }
    # Store data/graph in artifact for downstream tools (refute, counterfactual, etc.)
    store_for_downstream = dict(result_payload)
    store_for_downstream["_processed_data"] = df
    store_for_downstream["adjacency_matrix"] = clean_adj.tolist()
    store_for_downstream["node_names"] = names
    est_run_id_full = get_store().save(store_for_downstream)
    # Use the full run_id (with data) so downstream tools can use it
    result_payload["run_id"] = est_run_id_full

    next_steps = [
        f"refute_estimate(run_id='{est_run_id_full}', treatment='{treatment}', outcome='{outcome}') "
        "to test robustness of this estimate",
    ]
    if selected_method != "dml":
        next_steps.append(
            f"estimate_effect(treatment='{treatment}', outcome='{outcome}', method='dml') "
            "for heterogeneous treatment effects"
        )
    if selected_method not in ("metalearner",):
        next_steps.append(
            f"estimate_effect(treatment='{treatment}', outcome='{outcome}', method='metalearner') "
            "for meta-learner CATE estimation"
        )
    next_steps.append(
        f"estimate_counterfactual(run_id='{est_run_id_full}', treatment='{treatment}', "
        f"outcome='{outcome}', intervention_value=...) for 'what if?' analysis"
    )
    result_payload["next_steps"] = next_steps

    return json.dumps(result_payload, indent=2, cls=_NumpyEncoder)


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
        raise ToolError(f"Failed to parse CSV: {e}") from e

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

        # Per-column detail — gives agents full visibility into each feature
        column_detail = {}
        miss_ratio = getattr(stats, "miss_ratio", None)
        if miss_ratio and isinstance(miss_ratio, dict):
            for col, ratio in miss_ratio.items():
                column_detail.setdefault(col, {})["missing_ratio"] = _jsonable(ratio)
        dtc = getattr(stats, "data_type_column", None)
        if dtc and isinstance(dtc, dict):
            for col, dtype in dtc.items():
                column_detail.setdefault(col, {})["type"] = str(dtype)
        if column_detail:
            diagnosis["column_detail"] = column_detail

        # Time-series specific detail
        if diagnosis.get("time_series"):
            ts_detail = {}
            time_lag = getattr(stats, "time_lag", None)
            if time_lag is not None:
                ts_detail["estimated_lag"] = _jsonable(time_lag)
            stationary = getattr(stats, "stationary", None)
            if stationary is not None:
                ts_detail["stationary"] = _jsonable(stationary)
            time_index = getattr(stats, "time_index", None)
            if time_index is not None:
                ts_detail["time_index_column"] = str(time_index)
            if ts_detail:
                diagnosis["time_series_detail"] = ts_detail

        # Correlation groups (multicollinearity warning)
        high_corr = getattr(gs.user_data, "high_corr_feature_groups", None)
        if high_corr and isinstance(high_corr, dict):
            # Only include groups that actually have correlated partners
            corr_groups = {k: list(v) if not isinstance(v, list) else v for k, v in high_corr.items() if v}
            if corr_groups:
                diagnosis["high_correlation_groups"] = corr_groups

        # Descriptive statistics (EDA summary — numeric only, no images)
        try:
            desc = df[gs.user_data.selected_features].describe()
            desc_dict = {}
            for col in desc.columns:
                desc_dict[col] = {k: round(float(v), 4) for k, v in desc[col].items()}
            diagnosis["descriptive_stats"] = desc_dict
        except Exception:
            pass  # Non-numeric data; skip

        # Natural language summary (from pipeline's own converter)
        description = getattr(stats, "description", None)
        if description:
            diagnosis["description"] = str(description)

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

        if diagnosis.get("missingness"):
            recommendations.append("Missing data → use mv_fisherz CI test or MVPC mode")

        n = diagnosis.get("sample_size", 0)
        p = diagnosis.get("feature_number", 0)
        if n and p:
            if n < 100:
                recommendations.append(f"Small sample ({n} rows) → bootstrap validation critical")
            if p > 50:
                recommendations.append(f"High-dimensional ({p} features) → consider FGES or GRaSP")

        # Build data-adaptive knowledge prompt
        knowledge_prompt = _build_knowledge_prompt(diagnosis)

        return json.dumps(
            {
                "status": "ok",
                "diagnosis": diagnosis,
                "recommendations": recommendations,
                "knowledge_prompt": knowledge_prompt,
                "resources": {
                    "ci_test_guide": "causal://guides/ci-tests",
                    "score_function_guide": "causal://guides/score-functions",
                    "algorithms": "causal://algorithms",
                },
            },
            indent=2,
            cls=_NumpyEncoder,
        )
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Diagnosis failed: {e}"})


@mcp.tool()
def run_algorithm(
    csv_data: str,
    algorithm: str,
    hyperparameters: str = "{}",
    forbidden_edges: str = "",
    required_edges: str = "",
    seed: int = 42,
    allow_resolver_overrides: bool = True,
) -> str:
    """Run a specific causal discovery algorithm with given hyperparameters.

    No automatic selection, no postprocessing. Returns the raw graph.

    Args:
        csv_data: CSV string with header row
        algorithm: Algorithm name (e.g., "PC", "GES", "DirectLiNGAM")
        hyperparameters: JSON string of algorithm hyperparameters
        forbidden_edges: JSON array of [cause, effect] pairs that CANNOT exist.
            Example: '[["Age","Income"]]'. Supported by PC, FCI, CDNOD.
        required_edges: JSON array of [cause, effect] pairs that MUST exist.
            Example: '[["Education","Income"]]'. Supported by PC, FCI, CDNOD.
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
        raise ToolError(f"Invalid hyperparameters JSON: {e}") from e

    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        raise ToolError(f"Failed to parse CSV: {e}") from e

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
                    "PC",
                    "FCI",
                    "CDNOD",
                    "PCParallel",
                    "InterIAMB",
                    "BAMB",
                    "HITONMB",
                    "IAMBnPC",
                    "MBOR",
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

            # Inject structural constraints
            _ra_warnings: list[str] = []
            bk_spec = _parse_background_knowledge(
                forbidden_edges,
                required_edges,
                _ra_warnings,
            )
            if bk_spec:
                bk_algos = {"PC", "FCI", "CDNOD", "PCParallel"}
                if algorithm in bk_algos:
                    algo_args["background_knowledge"] = bk_spec
                elif _ra_warnings:
                    pass  # parse errors already recorded
                else:
                    _ra_warnings.append(f"{algorithm} does not support background_knowledge")
            if _ra_warnings:
                resolver_adjustments["background_knowledge"] = {
                    "warnings": _ra_warnings,
                }

            # Time-series mismatch warning
            is_ts = getattr(gs.statistics, "time_series", False)
            ts_algos = {"PCMCI", "VARLiNGAM", "GrangerCausality", "TiMINO", "DYNOTEARS"}
            if is_ts and algorithm not in ts_algos:
                resolver_adjustments["time_series_warning"] = (
                    f"Time-series data detected but '{algorithm}' is not temporal. "
                    f"Consider: {', '.join(sorted(ts_algos))}"
                )

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

        # Save to artifact store (include data for downstream tools)
        store_payload = dict(result)
        store_payload["_processed_data"] = gs.user_data.processed_data
        store_payload["_statistics"] = gs.statistics
        store_payload["_global_state"] = gs
        store_payload["_args"] = args
        run_id = get_store().save(store_payload)
        result["run_id"] = run_id
        result["resources"] = {
            "algorithm_profile": f"causal://algorithms/{algorithm}",
            "hyperparameter_spec": f"causal://hyperparameters/{algorithm}",
            "graph_guide": "causal://guides/interpreting-graphs",
        }
        result["next_steps"] = [
            f"inspect_graph(run_id='{run_id}') to classify graph and check inference eligibility",
            f"estimate_effect(run_id='{run_id}', treatment='X', outcome='Y') to estimate causal effects",
        ]

        return json.dumps(result, indent=2, cls=_NumpyEncoder)
    except Exception as e:
        return json.dumps({"status": "error", "error": f"Algorithm execution failed: {e}"})


@mcp.tool()
def discover(
    csv_data: str,
    query: str = "",
    domain_knowledge: str = "",
    forbidden_edges: str = "",
    required_edges: str = "",
    algorithm: str = "",
    seed: int = 42,
    timeout: int = 300,
) -> str:
    """Full autonomous causal discovery pipeline.

    Analyzes data characteristics, selects the best algorithm (or uses specified one),
    tunes hyperparameters, executes, and refines the result. This is the primary tool —
    use it when you want the system to handle everything.

    IMPORTANT: Pass domain_knowledge for dramatically better results. Call diagnose_data
    first — it returns a knowledge_prompt tailored to the dataset. Generate domain
    knowledge from that prompt and pass it here. This knowledge influences algorithm
    selection, hyperparameter tuning, AND graph refinement.

    Args:
        csv_data: CSV string with header row
        query: Optional causal question (helps LLM select algorithm)
        domain_knowledge: Domain knowledge about the variables and their relationships.
            Influences algorithm selection (Filter + Reranker), hyperparameter tuning,
            and graph refinement (Judge). Get a tailored prompt from diagnose_data's
            knowledge_prompt field.
        forbidden_edges: JSON array of [cause, effect] pairs that CANNOT exist.
            Example: '[["Age","Income"],["Gender","Height"]]'
            Supported by PC, FCI, CDNOD algorithms. Silently ignored by others.
        required_edges: JSON array of [cause, effect] pairs that MUST exist.
            Example: '[["Education","Income"]]'
            Supported by PC, FCI, CDNOD algorithms. Silently ignored by others.
        algorithm: Optional algorithm override (skips LLM selection)
        seed: Random seed
        timeout: Timeout in seconds

    Returns:
        JSON with adjacency_matrix, edges, edge_confidence, graph_kind,
        identifiability, data_diagnosis, provenance, run_id, warnings.
        graph_refined=true means bootstrap+LLM+KCI refinement was applied.
    """
    # -- Parse & validate ------------------------------------------------
    try:
        df = pd.read_csv(io.StringIO(csv_data))
    except Exception as e:
        raise ToolError(f"Failed to parse CSV: {e}") from e

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

        # Inject domain knowledge — feeds into Filter, Reranker, HP Selector, Judge
        # Wrap in list: downstream code (Filter, Reranker, Judge, Report) expects
        # knowledge_docs to be iterable (list), not a plain string.
        if domain_knowledge:
            gs.user_data.knowledge_docs = [domain_knowledge]

        with _pipeline_cwd():
            # 1. Statistical analysis
            gs = stat_info_collection(gs)
            gs.statistics.description = convert_stat_info_to_text(gs.statistics)

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
                        algorithm,
                        gs.statistics,
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
                    warnings.append(f"LLM selection failed, using rule-based: {llm_err}")
                    from causal_copilot.mcp.offline import (
                        get_default_hp,
                        select_algorithm_offline,
                    )

                    gs.algorithm.selected_algorithm = select_algorithm_offline(
                        gs.statistics,
                    )
                    gs.algorithm.algorithm_arguments = get_default_hp(
                        gs.algorithm.selected_algorithm,
                        gs.statistics,
                    )
                    used_planner = "rule-based-fallback"

            # 2b. Time-series algorithm alignment
            is_ts = getattr(gs.statistics, "time_series", False)
            ts_algos = {"PCMCI", "VARLiNGAM", "GrangerCausality", "TiMINO", "DYNOTEARS"}
            algo_name = gs.algorithm.selected_algorithm
            if is_ts and algo_name not in ts_algos:
                warnings.append(
                    f"Time-series data detected but non-temporal algorithm '{algo_name}' "
                    f"selected. Recommended: {', '.join(sorted(ts_algos))}. "
                    f"Results may miss temporal lag structure."
                )

            # 3. Resolver overrides (CI test / score func)
            algo_args = dict(gs.algorithm.algorithm_arguments or {})

            ci_test_algos = {
                "PC",
                "FCI",
                "CDNOD",
                "PCParallel",
                "InterIAMB",
                "BAMB",
                "HITONMB",
                "IAMBnPC",
                "MBOR",
            }
            if algo_name in ci_test_algos:
                algo_args["indep_test"] = resolve_ci_test(gs.statistics)

            score_algos = {"GES", "FGES", "XGES", "GRaSP", "ExactSearch", "BOSS"}
            if algo_name in score_algos:
                algo_args["score_func"] = resolve_score_func(
                    gs.statistics,
                    algo_name,
                )

            if algo_name == "PC" and gs.statistics.missingness:
                algo_args["mvpc"] = True

            # Inject structural constraints (forbidden/required edges)
            bk_spec = _parse_background_knowledge(
                forbidden_edges,
                required_edges,
                warnings,
            )
            if bk_spec:
                bk_algos = {"PC", "FCI", "CDNOD", "PCParallel"}
                if algo_name in bk_algos:
                    algo_args["background_knowledge"] = bk_spec
                else:
                    warnings.append(
                        f"Structural constraints ignored: {algo_name} "
                        f"does not support background_knowledge. "
                        f"Supported: {', '.join(sorted(bk_algos))}"
                    )

            gs.algorithm.algorithm_arguments = algo_args

            # 4. Execute
            gs = Programming(args).forward(gs)

            # 5. Postprocess (bootstrap stability + KCI pruning + LLM refinement)
            # Skip Judge for time-series data (main.py deliberately skips
            # bootstrap+KCI+LLM refinement for lagged graphs)
            is_ts = getattr(gs.statistics, "time_series", False)
            if is_ts:
                # TS: use converted_graph as revised_graph (no refinement)
                gs.results.revised_graph = gs.results.converted_graph
                warnings.append(
                    "Time-series data: graph refinement (bootstrap+KCI+LLM) "
                    "skipped — not applicable to lagged causal graphs."
                )
            else:
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

        # Build algorithm selection reasoning (transparency into LLM decisions)
        selection_reasoning = {}
        candidates = getattr(gs.algorithm, "algorithm_candidates", None)
        if candidates:
            selection_reasoning["candidates"] = candidates
        optimum = getattr(gs.algorithm, "algorithm_optimum", None)
        if optimum and isinstance(optimum, dict):
            selection_reasoning["ranking_reason"] = optimum.get("reason", "")
            score_calc = optimum.get("score_calculation")
            if score_calc:
                # Extract {algo: final_score} for concise view
                selection_reasoning["scores"] = {
                    k: v.get("final_score") if isinstance(v, dict) else v for k, v in score_calc.items()
                }
        hp_json = getattr(gs.algorithm, "algorithm_arguments_json", None)
        if hp_json and isinstance(hp_json, dict):
            hp_reasoning = {}
            hp_data = hp_json.get("hyperparameters", hp_json)
            for param, info in hp_data.items():
                if isinstance(info, dict) and "reasoning" in info:
                    hp_reasoning[param] = info["reasoning"]
            if hp_reasoning:
                selection_reasoning["hp_reasoning"] = hp_reasoning
        if selection_reasoning:
            provenance["selection_reasoning"] = selection_reasoning

        result = serialize_result(gs, node_names=node_names, provenance=provenance)

        # Enrich with human-ready summary (deterministic, no LLM call)
        result.update(generate_discovery_summary(result))

        if warnings:
            result["warnings"] = warnings

        # Save to artifact store (include data for downstream tools)
        store_payload = dict(result)
        store_payload["_processed_data"] = gs.user_data.processed_data
        store_payload["_statistics"] = gs.statistics
        store_payload["_global_state"] = gs
        store_payload["_args"] = args
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
            f"estimate_effect(run_id='{run_id}', treatment='X', outcome='Y') to estimate causal effects",
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
            raise ToolError(f"Invalid JSON: {e}") from e
        adj = np.array(adj_list)
        if data_diagnosis:
            try:
                diagnosis = json.loads(data_diagnosis)
            except json.JSONDecodeError as e:
                raise ToolError(f"Invalid data_diagnosis JSON: {e}") from e
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
            return json.dumps(
                {
                    "status": "needs_more_input",
                    "graph_kind": graph_kind,
                    "graph_stats": graph_stats,
                    "missing_inputs": ["data_diagnosis"],
                    "next_step": (
                        "CPDAG inference requires data diagnosis (linearity + gaussianity). "
                        "Use run_id from discover/run_algorithm, or provide data_diagnosis "
                        "with linearity and gaussian_error fields."
                    ),
                }
            )
        is_lg = bool(diagnosis.get("linearity")) and bool(diagnosis.get("gaussian_error"))
        policy = check_inference_policy(adj, is_linear_gaussian=is_lg)
        inference_policy = {
            "eligibility": policy["allow_inference"],
            "method": policy["method"],
            "reason": policy["reason"],
            "assumptions_used": (
                ["linearity", "Gaussian errors", "causal sufficiency"]
                if is_lg
                else ["causal sufficiency", "faithfulness"]
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
        key_findings.append(f"Causal inference possible via {inference_policy['method']}")
    else:
        key_findings.append(f"Causal inference blocked: {inference_policy['reason']}")

    if query_assessment:
        if query_assessment["effect_identifiable"]:
            key_findings.append(f"Effect of {treatment} on {outcome} is identifiable")
        elif query_assessment["directed_path_exists"]:
            key_findings.append(f"Path {treatment} -> {outcome} exists but effect not identifiable")
        else:
            key_findings.append(f"No directed path from {treatment} to {outcome}")

    limitations = []
    if graph_kind != "dag":
        limitations.append(f"Graph is {graph_kind.upper()} — some causal directions uncertain")
    if n_bidirected > 0:
        limitations.append(f"{n_bidirected} bidirected edges suggest latent confounders")

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
    if inference_policy["eligibility"] and treatment and outcome:
        if query_assessment and query_assessment["effect_identifiable"]:
            next_steps.append(
                f"estimate_effect(treatment='{treatment}', outcome='{outcome}'"
                + (f", run_id='{run_id}'" if run_id else "")
                + ") to estimate the causal effect"
            )
    if graph_kind == "cpdag" and not inference_policy["eligibility"]:
        next_steps.append("Try DirectLiNGAM via run_algorithm — LiNGAM gives unique DAG if errors are non-Gaussian")
    if graph_kind == "pag":
        next_steps.append("PAG detected — consider using PC (without latent variable assumption) for a CPDAG instead")
    if not treatment and not outcome and inference_policy["eligibility"]:
        next_steps.append("Specify treatment and outcome to assess a specific causal query")
    if inference_policy["eligibility"] and not treatment:
        next_steps.append("Call estimate_effect(treatment='X', outcome='Y') to estimate causal effects")
    if next_steps:
        result["next_steps"] = next_steps

    return json.dumps(result, indent=2, cls=_NumpyEncoder)


# ── Causal Reasoning Tools ───────────────────────────────────────────


@mcp.tool()
def refute_estimate(
    treatment: str,
    outcome: str,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    control_value: float = 0.0,
    treatment_value: float = 1.0,
) -> str:
    """Test robustness of a causal effect estimate with sensitivity analysis.

    Runs up to four refutation methods:
    - data_subset: re-estimate on 80% of data
    - random_common_cause: add a random variable as confounder
    - placebo_treatment: permute treatment column
    - unobserved_common_cause: partial-R2 sensitivity, benchmarked against
      SHAP top feature (only when common causes exist in graph)

    A robust estimate should remain stable across refutations.

    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        run_id: Run ID from estimate_effect or discover
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)
        control_value: Control group value (default 0.0)
        treatment_value: Treatment group value (default 1.0)

    Returns:
        JSON with original_estimate, refutation results, interpretation
    """
    df, adj, names, diagnosis = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
    )

    if treatment not in df.columns:
        raise ToolError(f"Treatment '{treatment}' not in data columns.")
    if outcome not in df.columns:
        raise ToolError(f"Outcome '{outcome}' not in data columns.")

    # Check graph kind for honest gate
    graph_kind = classify_graph_kind(adj)
    if graph_kind == "pag":
        return json.dumps(
            {
                "status": "rejected",
                "reason": "PAG — effects not identifiable, cannot refute",
                "graph_kind": "pag",
            }
        )

    clean_adj, _ = _sanitize_for_estimation(adj, names)
    dot_graph = _adj_to_dot(clean_adj, names)

    # Compute confounders + SHAP top feature for sensitivity benchmarking
    from causal_copilot.mcp.offline import identify_confounders as _offline_confounders

    conf_list, _ = _offline_confounders(clean_adj, names, treatment, outcome)

    shap_top = None
    try:
        from causal_copilot.mcp.estimation import compute_feature_importance

        is_linear = bool(diagnosis.get("linearity", True)) if diagnosis else True
        fi = compute_feature_importance(df, outcome, is_linear=is_linear)
        top_features = fi.get("top_features", [])
        if top_features:
            shap_top = top_features[0]
    except Exception:
        pass

    try:
        from causal_copilot.mcp.estimation import run_refutation

        with _pipeline_cwd():
            results = run_refutation(
                df,
                dot_graph,
                treatment,
                outcome,
                control_value,
                treatment_value,
                confounders=conf_list,
                shap_top_feature=shap_top,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Refutation failed: {e}",
            }
        )

    # Interpret robustness
    original = results["original_estimate"]
    robust = True
    issues = []
    for name, ref in results["refutations"].items():
        if "error" in ref:
            issues.append(f"{name}: failed ({ref['error']})")
            continue
        new_eff = ref.get("new_effect")
        if new_eff is not None and original is not None and original != 0:
            change_pct = abs(new_eff - original) / abs(original) * 100
            if change_pct > 20:
                robust = False
                issues.append(f"{name}: estimate changed by {change_pct:.1f}%")

    results["status"] = "ok"
    results["treatment"] = treatment
    results["outcome"] = outcome
    results["robust"] = robust
    if issues:
        results["robustness_issues"] = issues
    results["interpretation"] = f"The causal effect estimate of {treatment} on {outcome} " + (
        "appears robust across refutation tests." if robust else "shows sensitivity — interpret with caution."
    )
    results["next_steps"] = [
        "If robust: the estimate is reliable under standard assumptions",
        "If not robust: consider collecting more data or using stronger identification",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


@mcp.tool()
def estimate_counterfactual(
    treatment: str,
    outcome: str,
    intervention_value: float,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    observed_row_index: int = -1,
) -> str:
    """Answer 'what if?': what would outcome be if treatment were set to a value?

    Uses DoWhy Graphical Causal Model (GCM) for counterfactual reasoning.
    Requires a DAG (directed acyclic graph).

    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        intervention_value: The value to set treatment to in the counterfactual
        run_id: Run ID from discover/run_algorithm/estimate_effect
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)
        observed_row_index: Which data row to counterfactualize (-1 = auto, uses row with min treatment)

    Returns:
        JSON with observed values, counterfactual values, and causal effect
    """
    df, adj, names, _ = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
    )

    if treatment not in names or treatment not in df.columns:
        raise ToolError(f"Treatment '{treatment}' not in data/graph.")
    if outcome not in names or outcome not in df.columns:
        raise ToolError(f"Outcome '{outcome}' not in data/graph.")

    graph_kind = classify_graph_kind(adj)
    if graph_kind != "dag":
        clean_adj, _ = _sanitize_for_estimation(adj, names)
    else:
        clean_adj = adj

    try:
        from causal_copilot.mcp.estimation import run_counterfactual

        with _pipeline_cwd():
            results = run_counterfactual(
                df,
                clean_adj,
                names,
                treatment,
                outcome,
                intervention_value,
                observed_row_index,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Counterfactual estimation failed: {e}",
                "next_steps": ["Ensure graph is a DAG", "Check data has no missing values"],
            }
        )

    obs_y = results["observed"][outcome]
    cf_y = results["counterfactual"][outcome]
    effect = results["effect"]

    results["status"] = "ok"
    results["interpretation"] = (
        f"If {treatment} were set to {intervention_value} "
        f"(observed: {results['observed'][treatment]}), "
        f"{outcome} would change from {obs_y} to {cf_y} "
        f"(effect: {'+' if effect and effect > 0 else ''}{effect})."
    )
    results["next_steps"] = [
        f"simulate_intervention(treatment='{treatment}', outcome='{outcome}', "
        f"intervention_value={intervention_value}) for population-level simulation",
        f"refute_estimate(treatment='{treatment}', outcome='{outcome}') to validate the causal model",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


@mcp.tool()
def attribute_anomaly(
    target_node: str,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    anomaly_threshold_percentile: float = 95.0,
    num_samples: int = 5,
) -> str:
    """Identify root causes of anomalies in a target variable.

    Uses DoWhy GCM anomaly attribution to determine which parent nodes
    contribute most to anomalous values. Requires a DAG.

    Args:
        target_node: Variable whose anomalies to explain
        run_id: Run ID from discover/run_algorithm
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)
        anomaly_threshold_percentile: Percentile above which values are anomalous (default 95)
        num_samples: Max anomaly samples to analyze (default 5)

    Returns:
        JSON with attribution scores per parent node, sorted by impact
    """
    df, adj, names, _ = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
    )

    if target_node not in names or target_node not in df.columns:
        raise ToolError(f"Target node '{target_node}' not in data/graph.")

    graph_kind = classify_graph_kind(adj)
    if graph_kind != "dag":
        clean_adj, _ = _sanitize_for_estimation(adj, names)
    else:
        clean_adj = adj

    try:
        from causal_copilot.mcp.estimation import run_anomaly_attribution

        with _pipeline_cwd():
            results = run_anomaly_attribution(
                df,
                clean_adj,
                names,
                target_node,
                anomaly_threshold_percentile,
                num_samples,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Anomaly attribution failed: {e}",
                "next_steps": [
                    "Ensure target_node has parent nodes in the graph",
                    "Ensure graph is a DAG",
                ],
            }
        )

    results["status"] = "ok"

    # Build interpretation
    top_causes = []
    for node, scores in results.get("attributions", {}).items():
        ms = scores.get("mean_score")
        if ms is not None and abs(ms) > 0.01:
            top_causes.append(f"{node} (score={ms:.3f})")
    if top_causes:
        results["interpretation"] = f"Top root causes of anomalies in {target_node}: " + ", ".join(top_causes[:5])
    else:
        results["interpretation"] = f"No strong anomaly drivers found for {target_node}."

    results["next_steps"] = [
        f"estimate_effect(treatment='<top_cause>', outcome='{target_node}') to quantify the causal effect",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


@mcp.tool()
def attribute_distribution_change(
    target_node: str,
    csv_data_new: str,
    run_id: str = "",
    csv_data_old: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
) -> str:
    """Explain why a variable's distribution changed between two time periods.

    Uses DoWhy GCM distribution_change to identify which causal mechanisms shifted.
    Requires a DAG.

    Two input modes:
    1. run_id + csv_data_new: old data + graph from run_id, new data from csv
    2. csv_data_old + csv_data_new + adjacency_matrix + node_names

    Args:
        target_node: Variable whose distribution change to explain
        csv_data_new: CSV string of the new/current data
        run_id: Run ID from discover/run_algorithm (provides old data + graph)
        csv_data_old: CSV string of old/baseline data (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data_old)
        node_names: JSON array of variable names (needed with csv_data_old)

    Returns:
        JSON with attribution scores per node showing which mechanisms changed
    """
    try:
        df_new = pd.read_csv(io.StringIO(csv_data_new))
    except Exception as e:
        raise ToolError(f"Failed to parse csv_data_new: {e}") from e

    if run_id:
        cached = get_store().get(run_id)
        if cached is None:
            raise ToolError(f"run_id '{run_id}' not found or expired.")
        adj = np.array(cached["adjacency_matrix"])
        names = cached["node_names"]
        stored_data = cached.get("_processed_data")
        if stored_data is None:
            raise ToolError(f"run_id '{run_id}' has no stored data.")
        df_old = stored_data if isinstance(stored_data, pd.DataFrame) else pd.DataFrame(stored_data)
    elif csv_data_old:
        if not adjacency_matrix or not node_names:
            raise ToolError("adjacency_matrix and node_names required with csv_data_old.")
        try:
            df_old = pd.read_csv(io.StringIO(csv_data_old))
            adj = np.array(json.loads(adjacency_matrix))
            names = json.loads(node_names)
        except Exception as e:
            raise ToolError(f"Invalid input: {e}") from e
    else:
        raise ToolError("Provide either run_id or csv_data_old + adjacency_matrix + node_names.")

    if target_node not in names:
        raise ToolError(f"Target node '{target_node}' not in node_names.")

    graph_kind = classify_graph_kind(adj)
    if graph_kind != "dag":
        clean_adj, _ = _sanitize_for_estimation(adj, names)
    else:
        clean_adj = adj

    try:
        from causal_copilot.mcp.estimation import run_distribution_change

        with _pipeline_cwd():
            results = run_distribution_change(
                df_old,
                df_new,
                clean_adj,
                names,
                target_node,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Distribution change attribution failed: {e}",
                "next_steps": [
                    "Ensure both datasets have the same columns",
                    "Ensure graph is a DAG",
                ],
            }
        )

    results["status"] = "ok"

    top_changes = []
    for node, score in results.get("attributions", {}).items():
        if score is not None and abs(score) > 0.01:
            top_changes.append(f"{node} ({score:+.3f})")
    results["interpretation"] = f"Distribution of {target_node} changed. Top mechanism shifts: " + (
        ", ".join(top_changes[:5]) if top_changes else "no significant shifts detected"
    )

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


@mcp.tool()
def simulate_intervention(
    treatment: str,
    outcome: str,
    intervention_value: float = 1.0,
    shift: bool = True,
    num_samples: int = 1000,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
) -> str:
    """Simulate an intervention: what happens if we manipulate treatment?

    Uses DoWhy GCM interventional sampling. Requires a DAG.

    Two modes:
    - shift=True (default): treatment += intervention_value (e.g., +1.0)
    - shift=False: treatment := intervention_value (atomic, e.g., set to 5.0)

    Args:
        treatment: Treatment variable name
        outcome: Outcome variable name
        intervention_value: Value to shift/set treatment to (default 1.0)
        shift: If true, add value to treatment; if false, set treatment to value
        num_samples: Number of samples to draw (default 1000)
        run_id: Run ID from discover/run_algorithm
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)

    Returns:
        JSON with original and intervention outcome distributions, mean change
    """
    df, adj, names, _ = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
    )

    if treatment not in names or treatment not in df.columns:
        raise ToolError(f"Treatment '{treatment}' not in data/graph.")
    if outcome not in names or outcome not in df.columns:
        raise ToolError(f"Outcome '{outcome}' not in data/graph.")

    graph_kind = classify_graph_kind(adj)
    if graph_kind != "dag":
        clean_adj, _ = _sanitize_for_estimation(adj, names)
    else:
        clean_adj = adj

    try:
        from causal_copilot.mcp.estimation import run_intervention_simulation

        with _pipeline_cwd():
            results = run_intervention_simulation(
                df,
                clean_adj,
                names,
                treatment,
                outcome,
                intervention_value,
                shift,
                num_samples,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Intervention simulation failed: {e}",
                "next_steps": ["Ensure graph is a DAG"],
            }
        )

    results["status"] = "ok"

    orig_mean = results["original_distribution"]["mean"]
    intv_mean = results["intervention_distribution"]["mean"]
    change = results["mean_change"]
    mode = "shifting" if shift else "setting"
    results["interpretation"] = (
        f"Simulating {mode} {treatment} by {intervention_value}: "
        f"{outcome} mean changes from {orig_mean:.4f} to {intv_mean:.4f} "
        f"(change: {change:+.4f})."
    )

    results["next_steps"] = [
        f"estimate_effect(treatment='{treatment}', outcome='{outcome}') "
        "for formal causal effect estimate with confidence intervals",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


# ── Feature Importance ─────────────────────────────────────────────────


@mcp.tool()
def compute_feature_importance(
    target_node: str,
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    data_diagnosis: str = "",
) -> str:
    """Compute SHAP-based feature importance for a target variable.

    Shows which variables have the strongest predictive influence on the
    target. Uses linear SHAP for linear data, tree SHAP for nonlinear.

    Args:
        target_node: Variable to analyze (must be in data columns)
        run_id: Run ID from discover/run_algorithm
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)
        data_diagnosis: JSON with linearity info (optional)

    Returns:
        JSON with feature importance scores sorted by magnitude
    """
    df, adj, names, diagnosis = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
        data_diagnosis=data_diagnosis,
    )

    if target_node not in df.columns:
        raise ToolError(f"Target '{target_node}' not in data columns.")

    is_linear = True
    if diagnosis:
        is_linear = diagnosis.get("linearity", True)

    try:
        from causal_copilot.mcp.estimation import compute_feature_importance as _compute_fi

        with _pipeline_cwd():
            results = _compute_fi(df, target_node, is_linear)
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Feature importance failed: {e}",
                "next_steps": ["Check data has enough observations and variance."],
            }
        )

    results["status"] = "ok"

    # Interpretation
    top = results["top_features"][:3]
    results["interpretation"] = f"Top drivers of {target_node}: {', '.join(top)}. Method: {results['method']}."

    results["next_steps"] = [
        f"estimate_effect(treatment='{top[0]}', outcome='{target_node}') to quantify causal effect of the top feature"
        if top
        else "",
        "inspect_graph() to see causal structure between these variables",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


# ── Graph Validation ─────────────────────────────────────────────────


@mcp.tool()
def validate_graph(
    run_id: str = "",
    csv_data: str = "",
    adjacency_matrix: str = "",
    node_names: str = "",
    n_permutations: int = 20,
) -> str:
    """Test if a causal graph is consistent with data (falsification).

    Uses DoWhy GCM Local Markov Condition (LMC) testing. Compares the
    proposed graph against random permutations. Low p-value means the
    graph is significantly better than random.

    Args:
        run_id: Run ID from discover/run_algorithm
        csv_data: CSV string (alternative to run_id)
        adjacency_matrix: JSON 2D array (needed with csv_data)
        node_names: JSON array of variable names (needed with csv_data)
        n_permutations: Number of random graph permutations (default 20)

    Returns:
        JSON with falsification test results and interpretation
    """
    df, adj, names, _ = _resolve_data_and_graph(
        run_id,
        csv_data,
        adjacency_matrix,
        node_names,
    )

    graph_kind = classify_graph_kind(adj)
    if graph_kind != "dag":
        clean_adj, dropped = _sanitize_for_estimation(adj, names)
    else:
        clean_adj = adj
        dropped = []

    try:
        from causal_copilot.mcp.estimation import run_graph_falsification

        with _pipeline_cwd():
            results = run_graph_falsification(
                df,
                clean_adj,
                names,
                n_permutations,
            )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Graph falsification failed: {e}",
                "next_steps": ["Ensure graph is a DAG and data has enough observations."],
            }
        )

    results["status"] = "ok"
    results["graph_kind"] = graph_kind
    if dropped:
        results["dropped_edges"] = dropped

    # Interpretation
    p = results.get("p_value")
    if p is not None:
        if p < 0.05:
            results["interpretation"] = (
                f"Graph is significantly better than random (p={p:.4f}). "
                "The causal structure appears consistent with the data."
            )
        else:
            results["interpretation"] = (
                f"Graph is NOT significantly better than random (p={p:.4f}). "
                "The causal structure may not fit the data well."
            )
    else:
        results["interpretation"] = "Falsification test completed. See falsification_result for details."

    results["next_steps"] = [
        "discover() to re-run causal discovery if graph doesn't fit",
        "run_algorithm() with a different algorithm",
    ]

    return json.dumps(results, indent=2, cls=_NumpyEncoder)


def _prepare_gs_for_report(gs):
    """Fill in GlobalState fields needed by Report_generation that MCP may not populate.

    Report_generation (report/report_generation.py) was designed for the full
    main.py pipeline.  The MCP flow is leaner — it may skip LLM-based algorithm
    selection (using rule-based fallback) and never runs knowledge_info().
    This function fills gaps with sensible defaults so Report_generation.__init__
    doesn't crash.
    """
    import re as _re

    # meaningful_feature: detect from column names (generic = V0, X1, etc.)
    if gs.user_data.meaningful_feature is None:
        generic = _re.compile(r"^[VXvx]\d+$")
        names = gs.user_data.selected_features or gs.user_data.processed_data.columns.tolist()
        gs.user_data.meaningful_feature = not all(generic.match(str(f)) for f in names)

    # knowledge_docs_for_user: Report_generation accesses [0]
    if gs.user_data.knowledge_docs_for_user is None:
        if gs.user_data.knowledge_docs:
            gs.user_data.knowledge_docs_for_user = [gs.user_data.knowledge_docs]
        else:
            gs.user_data.knowledge_docs_for_user = ["No domain knowledge was provided for this analysis."]

    # statistics.description: convert_stat_info_to_text may not have stored it
    if gs.statistics.description is None:
        from preprocess.stat_info_functions import convert_stat_info_to_text

        gs.statistics.description = convert_stat_info_to_text(gs.statistics)

    # algorithm_candidates: Report_generation reads algo_can dict
    if gs.algorithm.algorithm_candidates is None:
        algo = gs.algorithm.selected_algorithm or "Unknown"
        gs.algorithm.algorithm_candidates = {
            algo: {
                "description": f"{algo} algorithm",
                "justification": "Selected based on data characteristics (rule-based)",
            }
        }

    # algorithm_arguments_json: Report_generation reads hyperparameter details
    if gs.algorithm.algorithm_arguments_json is None:
        raw_args = gs.algorithm.algorithm_arguments or {}
        hp_dict = {}
        for k, v in raw_args.items():
            hp_dict[k] = {
                "full_name": str(k),
                "value": str(v),
                "explanation": "Auto-configured based on data properties",
            }
        gs.algorithm.algorithm_arguments_json = {"hyperparameters": hp_dict}

    # select_conversation: Report_generation reads [0]['response']
    if not gs.logging.select_conversation:
        algo = gs.algorithm.selected_algorithm or "Unknown"
        gs.logging.select_conversation = [
            {"response": (f"Algorithm {algo} was selected based on dataset characteristics (rule-based selection).")}
        ]

    # argument_conversation: Report_generation reads [0]['response']
    if not gs.logging.argument_conversation:
        gs.logging.argument_conversation = [{"response": "Hyperparameters were configured based on data properties."}]

    # global_state_logging: Report_generation iterates this to load per-algo states
    if not gs.logging.global_state_logging:
        gs.logging.global_state_logging = [gs.algorithm.selected_algorithm]

    # graph_conversion: ensure dict exists
    if gs.logging.graph_conversion is None:
        gs.logging.graph_conversion = {}

    return gs


@mcp.tool()
def generate_report(run_id: str) -> str:
    """Generate a comprehensive PDF report from a causal discovery run.

    Creates a publication-quality LaTeX PDF covering:
    - Exploratory data analysis (distributions, correlations)
    - Algorithm selection rationale and hyperparameters
    - Causal graph visualizations (initial + refined)
    - Bootstrap confidence analysis with heatmaps
    - Graph interpretation (LLM-generated narrative)
    - Causal inference results (if estimate_effect was run on this run_id)
    - Conclusion and summary

    Prerequisites:
    - run_id from discover() or run_algorithm()
    - LLM access (LLM_PROVIDER env var) for narrative sections
    - LaTeX/latexmk installed for PDF compilation

    Args:
        run_id: Run ID from a previous discover() or run_algorithm() call.

    Returns:
        JSON with status, report_path (PDF location), tex_path, and output_dir.
        If PDF compilation fails, status is "partial" and tex_path is still usable.
    """
    cached = get_store().get(run_id)
    if not cached:
        raise ToolError(f"Run ID '{run_id}' not found or expired (1h TTL). Re-run discover() or run_algorithm() first.")

    gs = cached.get("_global_state")
    if gs is None:
        raise ToolError(
            "No GlobalState stored for this run. generate_report requires run_id from discover() or run_algorithm()."
        )

    args = cached.get("_args")
    if args is None:
        args = make_args(query=gs.user_data.initial_query or "")

    # Fill in fields Report_generation needs but MCP may not populate
    gs = _prepare_gs_for_report(gs)

    report_warnings: list[str] = []

    try:
        with _pipeline_cwd():
            # 1. EDA — generates distribution plots, correlation heatmaps
            try:
                from preprocess.eda_generation import EDA

                eda = EDA(gs)
                eda.generate_eda()
            except Exception as eda_err:
                report_warnings.append(f"EDA generation skipped: {eda_err}")
                # Set minimal eda to avoid crashes in report
                if not hasattr(gs.results, "eda") or gs.results.eda is None:
                    gs.results.eda = {}

            # 2. Visualizations — graph plots, heatmaps
            try:
                from postprocess.visualization import Visualization, convert_to_edges

                my_visual = Visualization(gs)
                algo_name = gs.algorithm.selected_algorithm

                if gs.statistics.time_series and gs.results.lagged_graph is not None:
                    converted = gs.results.lagged_graph
                    pos_est = my_visual.get_pos(converted[0])
                    for i in range(converted.shape[0]):
                        my_visual.plot_pdag(
                            converted[i],
                            f"{algo_name}_initial_graph_{i}.svg",
                            pos=pos_est,
                        )
                    summary_graph = np.any(converted, axis=0).astype(int)
                    my_visual.plot_pdag(
                        summary_graph,
                        f"{algo_name}_initial_graph_summary.svg",
                        pos=pos_est,
                    )
                else:
                    pos_est = my_visual.get_pos(gs.results.converted_graph)
                    my_visual.plot_pdag(
                        gs.results.converted_graph,
                        f"{algo_name}_initial_graph.pdf",
                        pos=pos_est,
                    )

                # Store layout for background_prompt() potential_relation.pdf
                gs.results.raw_pos = pos_est

                # Raw edges for report narrative
                gs.results.raw_edges = convert_to_edges(
                    algo_name,
                    gs.user_data.processed_data.columns,
                    gs.results.converted_graph,
                )

                # Revised graph visualization
                if gs.results.revised_graph is not None:
                    my_visual_rev = Visualization(gs)
                    my_visual_rev.plot_pdag(
                        gs.results.revised_graph,
                        f"{algo_name}_revised_graph.pdf",
                        pos=pos_est,
                    )
                    gs.results.revised_edges = convert_to_edges(
                        algo_name,
                        gs.user_data.processed_data.columns,
                        gs.results.revised_graph,
                    )
                    my_visual_rev.boot_heatmap_plot()
            except Exception as vis_err:
                report_warnings.append(f"Visualization generation failed: {vis_err}")

            # 3. Graph effect analysis (LLM call — produces narrative)
            from report.report_generation import Report_generation

            try:
                report_gen_pre = Report_generation(gs, args)
                gs.logging.graph_conversion["initial_graph_analysis"] = report_gen_pre.graph_effect_prompts()
            except Exception as ge_err:
                report_warnings.append(f"Graph effect analysis failed: {ge_err}")
                gs.logging.graph_conversion["initial_graph_analysis"] = (
                    "Graph effect analysis was not available for this run."
                )

            # 4. Generate full report (12+ LLM calls for narrative sections)
            report_gen = Report_generation(gs, args)
            report_tex = report_gen.generation()
            report_gen.save_report(report_tex)

            # 5. Check output
            report_path = os.path.join(gs.user_data.output_report_dir, "report.pdf")
            tex_path = os.path.join(gs.user_data.output_report_dir, "report.tex")

            result: dict[str, Any] = {
                "output_dir": gs.user_data.output_report_dir,
                "tex_path": tex_path,
            }

            if os.path.isfile(report_path):
                result["status"] = "ok"
                result["report_path"] = report_path
            else:
                result["status"] = "partial"
                result["error"] = (
                    "LaTeX compilation failed — report.tex was generated but "
                    "PDF was not produced. Check latexmk installation."
                )

            if report_warnings:
                result["warnings"] = report_warnings

            return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "error": f"Report generation failed: {e}",
                "output_dir": getattr(gs.user_data, "output_report_dir", None),
                "warnings": report_warnings,
            }
        )


# ── MCP Resources ─────────────────────────────────────────────────────
from causal_copilot.mcp.resources import (  # noqa: E402
    get_algorithm_content,
    get_algorithm_resources,
    get_guide_content,
    get_hp_content,
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
from causal_copilot.mcp.prompts import PROMPTS  # noqa: E402


@mcp.prompt()
def causal_expert():
    """Expert system prompt for causal discovery."""
    return PROMPTS["causal-expert"]


@mcp.prompt()
def analyze_dataset():
    """Step-by-step workflow for analyzing a dataset."""
    return PROMPTS["analyze-dataset"]


@mcp.prompt()
def causal_analysis():
    """Complete knowledge-first causal analysis workflow.

    Use this for the best results: diagnose → generate domain knowledge →
    discover with knowledge → interpret → deepen. Your domain knowledge
    directly influences algorithm selection and graph refinement.
    """
    return PROMPTS["causal-analysis"]
