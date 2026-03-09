"""CausalCopilot — the main entry point for causal analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from causal_copilot import __version__
from causal_copilot.core.base import CausalDiscoveryBase
from causal_copilot.core.guards import DataValidationError, validate_data
from causal_copilot.core.planner import PlannerDecision, detect_data_properties, rule_based_select
from causal_copilot.core.result import CausalResult, Provenance, TreatmentEffect
from causal_discovery.pdag_policy import (
    check_inference_policy,
    classify_graph_kind,
    get_identifiable_edges,
)

_VALID_PLANNERS = ("rule",)  # "llm" will be added in a future version

# Inline script executed in a fresh Python process by _run_in_subprocess.
# Reads JSON config from stdin, runs algo.fit(), writes JSON result to stdout.
_WORKER_SCRIPT = r"""
import io, json, sys
import numpy as np
import pandas as pd

cfg = json.load(sys.stdin)

# Reproduce parent's random seed in the child process
if cfg.get("seed") is not None:
    np.random.seed(cfg["seed"])

from causal_copilot.algorithms.registry import REGISTRY
from causal_copilot.core.result import _json_safe

spec = REGISTRY.get(cfg["algo_name"])
if spec is None:
    json.dump({"status": "error", "message": f"Unknown algorithm: {cfg['algo_name']}"}, sys.stdout)
    sys.exit(0)
adapter = spec.adapter_cls(params=cfg["algo_params"])
df = pd.read_json(io.StringIO(cfg["data_json"]))
try:
    adj, meta, _ = adapter.fit(df)
    json.dump({"status": "ok", "adj": adj.tolist(), "meta": _json_safe(meta)}, sys.stdout)
except Exception as e:
    json.dump({"status": "error", "message": str(e)}, sys.stdout)
"""


def _run_in_subprocess(
    algo: CausalDiscoveryBase, data: pd.DataFrame, timeout: int, seed: int | None = None
) -> tuple[np.ndarray, dict]:
    """Run algo.fit(data) in a fresh subprocess with kill-based timeout.

    Launches a new Python interpreter via subprocess.Popen, avoiding all
    fork-safety issues (deadlocks in multi-threaded parents, macOS spawn
    restrictions). Data crosses the process boundary as JSON over stdin/stdout.

    Returns (adj_matrix, metadata) or raises TimeoutError / RuntimeError.
    """
    cfg = json.dumps(
        {
            "algo_name": algo.name,
            "algo_params": algo.get_params(),
            "data_json": data.to_json(),
            "seed": seed,
        }
    )

    proc = subprocess.Popen(
        [sys.executable, "-c", _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = proc.communicate(input=cfg.encode(), timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise TimeoutError(f"Algorithm timed out after {timeout}s") from None

    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess exited with code {proc.returncode}: {stderr.decode()[:500]}")

    result = json.loads(stdout.decode())
    if result["status"] == "ok":
        return np.array(result["adj"]), result.get("meta", {})
    raise RuntimeError(result.get("message", "Unknown error in subprocess"))


def _load_algorithm(name: str, params: dict[str, Any]) -> CausalDiscoveryBase:
    """Instantiate an algorithm adapter from the canonical registry."""
    from causal_copilot.algorithms.registry import REGISTRY

    if name not in REGISTRY:
        raise ValueError(f"Unknown algorithm: {name!r}. Available: {sorted(REGISTRY)}")
    spec = REGISTRY[name]
    return spec.adapter_cls(params=params)


def _build_graph(adj_matrix: np.ndarray, columns: list):
    """Build networkx graph from adjacency matrix, handling all edge types.

    Edge encoding: 0=none, 1=directed, 2=undirected, 3=bidirected.
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    graph = nx.DiGraph()
    graph.add_nodes_from(columns)
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            val = adj_matrix[i, j]
            if val == 1:  # directed: j -> i
                graph.add_edge(columns[j], columns[i], edge_type="directed")
            elif val == 2:  # undirected: i -- j (add both directions)
                graph.add_edge(columns[j], columns[i], edge_type="undirected")
                graph.add_edge(columns[i], columns[j], edge_type="undirected")
            elif val == 3:  # bidirected: i <-> j
                graph.add_edge(columns[j], columns[i], edge_type="bidirected")
                graph.add_edge(columns[i], columns[j], edge_type="bidirected")
    return graph


def _make_provenance(data_hash, seed, decision, active_planner, elapsed):
    from causal_copilot.algorithms.registry import REGISTRY

    spec = REGISTRY.get(decision.algorithm)
    algo_version = spec.algorithm_version if spec else "unknown"
    return Provenance(
        dataset_hash=data_hash,
        seed=seed,
        algorithm=decision.algorithm,
        algorithm_version=algo_version,
        package_version=__version__,
        hyperparams=Provenance.freeze_params(decision.hyperparams),
        planner=active_planner,
        runtime_seconds=elapsed,
        timestamp=Provenance.now_utc(),
        environment=Provenance.get_environment(),
    )


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


def _adj_to_dot(adj: np.ndarray, names: list[str]) -> str:
    """Convert adjacency matrix to DOT format for DoWhy. Only directed edges."""
    parts = [f"{name}" for name in names]  # declare all nodes
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:  # j→i
                parts.append(f"{names[j]} -> {names[i]}")
    return "digraph { " + "; ".join(parts) + " }"


class CausalCopilot:
    """
    Main entry point for causal analysis.

    Usage:
        copilot = CausalCopilot()
        result = copilot.analyze("data.csv", seed=42)
    """

    def __init__(self, planner: str = "rule"):
        if planner not in _VALID_PLANNERS:
            raise ValueError(f"Unknown planner: {planner!r}. Available: {_VALID_PLANNERS}")
        self.planner = planner
        self._last_data: pd.DataFrame | None = None
        self._last_properties: dict | None = None
        self._last_gs: Any | None = None
        self._last_args: Any | None = None

    def analyze(
        self,
        data: str | Path | pd.DataFrame,
        *,
        planner: str | None = None,
        algorithm: str | None = None,
        algorithm_params: dict[str, Any] | None = None,
        timeout: int = 300,
        seed: int = 42,
    ) -> CausalResult:
        """
        Run causal discovery on data.

        Args:
            data: CSV file path or DataFrame.
            planner: Override instance planner (must be valid).
            algorithm: Force a specific algorithm (bypasses planner).
            algorithm_params: Custom hyperparameters (merged on top of defaults).
            timeout: Max seconds for algorithm execution (default 300).
            seed: Random seed for reproducibility (default 42).

        Returns:
            CausalResult with graph, provenance, assumptions, and warnings.
        """
        start_time = time.monotonic()
        warnings: list[str] = []

        # Validate planner
        active_planner = planner or self.planner
        if active_planner not in _VALID_PLANNERS:
            return CausalResult(
                status="failed",
                summary=f"Unknown planner: {active_planner!r}. Available: {_VALID_PLANNERS}",
            )

        # Set random seed (scoped — use Generator for future thread safety)
        np.random.seed(seed)

        # Load data
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                return CausalResult(
                    status="failed",
                    summary=f"File not found: {path}",
                    warnings=[f"File not found: {path}"],
                )
            try:
                df = pd.read_csv(path)
            except Exception as e:
                return CausalResult(
                    status="failed",
                    summary=f"Failed to read CSV: {e}",
                    warnings=[str(e)],
                )
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            return CausalResult(
                status="failed",
                summary=f"Unsupported data type: {type(data).__name__}",
            )

        # Validate raw data
        try:
            data_warnings = validate_data(df)
            warnings.extend(data_warnings)
        except DataValidationError as e:
            return CausalResult(
                status="failed",
                summary=str(e),
                warnings=[str(e)],
            )

        # --- Full pipeline preprocessing (MICE, encoding, normalization, stat tests) ---
        # Try the original pipeline's stat_info_collection; fall back to simple
        # preprocessing if pipeline modules aren't available (pip-installed mode).
        gs = None
        pipeline_available = False

        try:
            import os

            from causal_copilot.mcp.bridge import PIPELINE_ROOT, make_args, make_global_state

            gs = make_global_state(df, seed=seed)
            _args = make_args(seed=seed)

            # Detect missing values before preprocessing (imputation removes them)
            raw_has_missing = df.isnull().any().any()

            old_cwd = os.getcwd()
            os.chdir(str(PIPELINE_ROOT))
            try:
                from preprocess.stat_info_functions import stat_info_collection

                gs = stat_info_collection(gs)
                pipeline_available = True
            finally:
                os.chdir(old_cwd)

            # Use preprocessed data from the pipeline
            numeric_df = gs.user_data.processed_data
            if numeric_df is None or len(numeric_df) < 10:
                return CausalResult(
                    status="failed",
                    summary=f"After preprocessing, only {len(numeric_df) if numeric_df is not None else 0} samples remain.",
                    warnings=warnings,
                )

            # Report preprocessing actions
            has_missing = raw_has_missing or getattr(gs.statistics, "missingness", False)
            if has_missing:
                warnings.append("Missing values detected and imputed via MICE (IterativeImputer).")

            # Store real statistical test results (Ramsey RESET, Shapiro-Wilk)
            properties = {
                "n_samples": len(numeric_df),
                "n_features": numeric_df.shape[1],
                "likely_linear": getattr(gs.statistics, "linearity", True),
                "likely_gaussian": getattr(gs.statistics, "gaussian_error", True),
                "has_missing": has_missing,
                "is_time_series": getattr(gs.statistics, "time_series", False),
            }
        except ImportError:
            # Pipeline not available — fall back to simple preprocessing
            numeric_df = df.select_dtypes(include=[np.number])
            constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
            if constant_cols:
                warnings.append(f"Dropped constant columns: {constant_cols}")
                numeric_df = numeric_df.drop(columns=constant_cols)
            if numeric_df.shape[1] < 2:
                return CausalResult(
                    status="failed",
                    summary=f"After cleaning, only {numeric_df.shape[1]} numeric column(s) remain. Need at least 2.",
                    warnings=warnings,
                )
            n_before = len(numeric_df)
            numeric_df = numeric_df.dropna()
            n_dropped = n_before - len(numeric_df)
            if n_dropped > 0:
                warnings.append(f"Dropped {n_dropped} rows with missing values ({n_dropped / n_before:.1%}).")
            if len(numeric_df) < 10:
                return CausalResult(
                    status="failed",
                    summary=f"After dropping NaN rows, only {len(numeric_df)} samples remain. Need at least 10.",
                    warnings=warnings,
                )
            properties = detect_data_properties(numeric_df)

        # Drop constant columns (pipeline may not filter these)
        constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
        if constant_cols:
            warnings.append(f"Dropped constant columns: {constant_cols}")
            numeric_df = numeric_df.drop(columns=constant_cols)
        if numeric_df.shape[1] < 2:
            return CausalResult(
                status="failed",
                summary=f"After cleaning, only {numeric_df.shape[1]} numeric column(s) remain. Need at least 2.",
                warnings=warnings,
            )

        data_hash = Provenance.hash_data(numeric_df)
        # Store ORIGINAL-SCALE data for estimation (not normalized).
        # main.py's Analysis class uses raw_data for inference, not processed_data.
        # The normalized numeric_df is used for graph discovery only.
        if pipeline_available and gs is not None:
            raw_data = gs.user_data.raw_data
            # Keep only numeric columns that survived preprocessing
            estimation_df = raw_data[numeric_df.columns].copy()
            # Drop rows with NaN (imputed in numeric_df but we need clean rows)
            estimation_df = estimation_df.dropna()
            if len(estimation_df) < 10:
                estimation_df = numeric_df  # fallback to preprocessed
            self._last_data = estimation_df
        else:
            self._last_data = numeric_df
        self._last_properties = properties

        # --- Algorithm selection (LLM Filter+Reranker → rule-based fallback) ---
        if algorithm:
            decision = PlannerDecision(
                algorithm=algorithm,
                hyperparams=algorithm_params or {},
                reason=f"User specified algorithm: {algorithm}",
            )
            active_planner = "user-specified"
        elif pipeline_available and gs is not None:
            try:
                import os

                from causal_copilot.mcp.bridge import PIPELINE_ROOT

                old_cwd = os.getcwd()
                os.chdir(str(PIPELINE_ROOT))
                try:
                    from causal_discovery.filter import Filter
                    from causal_discovery.rerank import Reranker

                    gs = Filter(_args).forward(gs)
                    gs = Reranker(_args).forward(gs)
                    active_planner = "llm"
                finally:
                    os.chdir(old_cwd)

                algo_name = gs.algorithm.selected_algorithm
                decision = PlannerDecision(
                    algorithm=algo_name,
                    hyperparams=gs.algorithm.algorithm_arguments or {},
                    reason=f"LLM Filter+Reranker selected {algo_name}",
                )
            except Exception as llm_err:
                warnings.append(f"LLM selection failed ({llm_err}), using rule-based fallback")
                decision = rule_based_select(properties)
                active_planner = "rule-based-fallback"
        else:
            decision = rule_based_select(properties)

        # --- HP tuning (LLM HyperparameterSelector → defaults fallback) ---
        if not algorithm and pipeline_available and gs is not None:
            try:
                import os

                from causal_copilot.mcp.bridge import PIPELINE_ROOT

                old_cwd = os.getcwd()
                os.chdir(str(PIPELINE_ROOT))
                try:
                    from causal_discovery.hyperparameter_selector import HyperparameterSelector

                    gs.algorithm.selected_algorithm = decision.algorithm
                    gs = HyperparameterSelector(_args).forward(gs)
                    # Merge pipeline HP with decision
                    hp = gs.algorithm.algorithm_arguments or {}
                    if algorithm_params:
                        hp.update(algorithm_params)
                    decision = PlannerDecision(
                        algorithm=decision.algorithm,
                        hyperparams=hp,
                        reason=decision.reason,
                    )
                finally:
                    os.chdir(old_cwd)
            except Exception as hp_err:
                warnings.append(f"LLM HP tuning failed ({hp_err}), using defaults")

        # Load algorithm — merge defaults with overrides
        try:
            algo = _load_algorithm(decision.algorithm, decision.hyperparams)
            effective = algo.default_params()
            effective.update(decision.hyperparams)
            decision = PlannerDecision(
                algorithm=decision.algorithm,
                hyperparams=effective,
                reason=decision.reason,
            )
            algo = _load_algorithm(decision.algorithm, effective)
        except (ValueError, ImportError) as e:
            return CausalResult(
                status="failed",
                summary=f"Failed to load algorithm {decision.algorithm}: {e}",
                warnings=warnings + [str(e)],
            )

        # --- Execute algorithm (with process-based timeout) ---
        try:
            try:
                adj_matrix, metadata = _run_in_subprocess(algo, numeric_df, timeout, seed=seed)
            except (OSError, AttributeError, TypeError) as sub_err:
                if isinstance(sub_err, TimeoutError):
                    raise
                adj_matrix, metadata, _ = algo.fit(numeric_df)
        except TimeoutError:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} timed out after {timeout}s",
                warnings=warnings + [f"Timeout after {timeout}s"],
                provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                algorithm_selection_reason=decision.reason,
            )
        except Exception as e:
            elapsed = time.monotonic() - start_time
            return CausalResult(
                status="failed",
                summary=f"Algorithm {decision.algorithm} failed: {e}",
                warnings=warnings + [str(e)],
                provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                algorithm_selection_reason=decision.reason,
            )

        # --- Postprocess (Judge: bootstrap stability + KCI pruning + LLM refinement) ---
        if pipeline_available and gs is not None:
            try:
                import os

                from causal_copilot.mcp.bridge import PIPELINE_ROOT

                # Feed algorithm result into GlobalState for Judge
                gs.results.raw_result = adj_matrix
                gs.results.converted_graph = adj_matrix
                gs.algorithm.selected_algorithm = decision.algorithm
                gs.algorithm.algorithm_arguments = decision.hyperparams
                gs.user_data.processed_data = numeric_df
                gs.user_data.selected_features = list(numeric_df.columns)

                # Postprocess: skip Judge for time-series data (main.py behavior)
                is_ts = getattr(gs.statistics, "time_series", False)
                if is_ts:
                    gs.results.revised_graph = gs.results.converted_graph
                else:
                    old_cwd = os.getcwd()
                    os.chdir(str(PIPELINE_ROOT))
                    try:
                        from postprocess.judge import Judge

                        gs = Judge(gs, _args).forward(gs, "cot_all_relation", 1)
                    finally:
                        os.chdir(old_cwd)

                # Use refined graph if available
                refined = getattr(gs.results, "revised_graph", None)
                if refined is not None:
                    adj_matrix = refined
                    metadata = metadata if isinstance(metadata, dict) else {}
                    metadata["graph_refined"] = True

                    # Include bootstrap confidence
                    boot_prob = getattr(gs.results, "bootstrap_probability", None)
                    if boot_prob is not None:
                        metadata["bootstrap_probability"] = boot_prob

                    # Include LLM pruning decisions
                    llm_decisions = getattr(gs.results, "llm_errors", None)
                    if llm_decisions:
                        metadata["llm_pruning"] = llm_decisions
            except Exception as pp_err:
                warnings.append(f"Postprocessing skipped: {pp_err}")

        elapsed = time.monotonic() - start_time

        # Guard: adjacency matrix dimensions must match feature count
        cols = list(numeric_df.columns)
        n_vars = len(cols)
        if adj_matrix.shape != (n_vars, n_vars):
            if adj_matrix.shape[0] == adj_matrix.shape[1] and adj_matrix.shape[0] < n_vars:
                cols = cols[: adj_matrix.shape[0]]
                warnings.append(
                    f"Adjacency matrix is {adj_matrix.shape[0]}x{adj_matrix.shape[0]} "
                    f"but data had {n_vars} columns; node_names trimmed to match matrix."
                )
            else:
                return CausalResult(
                    status="failed",
                    summary=f"Adjacency matrix shape {adj_matrix.shape} does not match {n_vars} features",
                    warnings=warnings,
                    provenance=_make_provenance(data_hash, seed, decision, active_planner, elapsed),
                )

        graph = _build_graph(adj_matrix, cols)
        provenance = _make_provenance(data_hash, seed, decision, active_planner, elapsed)

        # Count edges by type
        n_directed = int(np.sum(adj_matrix == 1))
        n_undirected = sum(
            1
            for i in range(adj_matrix.shape[0])
            for j in range(i + 1, adj_matrix.shape[1])
            if adj_matrix[i, j] == 2 or adj_matrix[j, i] == 2
        )
        n_bidirected = sum(
            1
            for i in range(adj_matrix.shape[0])
            for j in range(i + 1, adj_matrix.shape[1])
            if adj_matrix[i, j] == 3 or adj_matrix[j, i] == 3
        )

        edge_parts = [f"{n_directed} directed"]
        if n_undirected:
            edge_parts.append(f"{n_undirected} undirected")
        if n_bidirected:
            edge_parts.append(f"{n_bidirected} bidirected")
        edge_summary = ", ".join(edge_parts)

        # Store GlobalState for generate_report
        if pipeline_available and gs is not None:
            self._last_gs = gs
            self._last_args = _args

        return CausalResult(
            status="ok",
            adjacency_matrix=adj_matrix,
            node_names=cols,
            graph=graph,
            discovery_metadata=metadata if isinstance(metadata, dict) else {},
            summary=f"Discovered {edge_summary} edges using {decision.algorithm} "
            f"on {numeric_df.shape[0]} samples × {numeric_df.shape[1]} features "
            f"in {elapsed:.1f}s.",
            warnings=warnings,
            provenance=provenance,
            algorithm_selection_reason=decision.reason,
        )

    def estimate_effect(
        self,
        result: CausalResult,
        treatment: str,
        outcome: str,
        *,
        data: pd.DataFrame | None = None,
        method: str | None = None,
        control_value: float | None = None,
        treatment_value: float | None = None,
        confounders: list[str] | None = None,
        instrument: str | None = None,
    ) -> CausalResult:
        """Estimate causal effect of treatment on outcome.

        Matches the scientific rigor of the original Analysis class pipeline:
        - 4-case treatment type dispatch (categorical/binary/discrete/continuous)
        - Graph-aware confounder detection (values 1/2/3/4)
        - Data-driven method + model selection (replaces LLM-based Filter/Reranker)
        - HTE (per-sample heterogeneous effects) for DML/DRL/MetaLearner/IV
        - Honest inference gate (PAG reject, CPDAG gate)

        Args:
            result: CausalResult from analyze() with adjacency_matrix.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            data: DataFrame (uses stored data from analyze() if None).
            method: Estimation method: linear, matching, dml, drl, metalearner, iv.
                    Auto-selected based on data properties if None.
            control_value: Control/baseline value (auto-detected from data if None).
            treatment_value: Treatment/active value (auto-detected from data if None).
            confounders: Confounder names (auto-detected from graph if None).
            instrument: Instrumental variable name (for IV; auto-detected if None).

        Returns:
            Updated CausalResult with populated effects dict.
        """
        from causal_copilot.mcp.offline import (
            get_default_estimation_config,
            identify_confounders,
            prepare_treatment,
            select_estimation_method,
        )

        # --- Resolve data ---
        df = data if data is not None else self._last_data
        if df is None:
            raise ValueError("No data available. Pass data= or call analyze() first.")

        if result.adjacency_matrix is None or result.node_names is None:
            raise ValueError("Result has no adjacency matrix. Run analyze() first.")

        adj = result.adjacency_matrix
        names = result.node_names

        if treatment not in names:
            raise ValueError(f"Treatment '{treatment}' not in graph nodes: {names}")
        if outcome not in names:
            raise ValueError(f"Outcome '{outcome}' not in graph nodes: {names}")
        if treatment == outcome:
            raise ValueError("Treatment and outcome must be different.")
        if treatment not in df.columns:
            raise ValueError(f"Treatment '{treatment}' not in data columns: {list(df.columns)}")
        if outcome not in df.columns:
            raise ValueError(f"Outcome '{outcome}' not in data columns: {list(df.columns)}")

        warnings_list = list(result.warnings)
        props = self._last_properties or {}
        is_linear = bool(props.get("likely_linear", True))
        is_gaussian = bool(props.get("likely_gaussian", True))

        # Time-series warning: causal effect estimation assumes i.i.d. samples
        if props.get("is_time_series", False):
            warnings_list.append(
                "Time-series structure detected — causal effect estimation "
                "assumes i.i.d. samples. Results may be biased if temporal "
                "lag structure matters. Consider time-series-specific methods."
            )

        # --- Treatment type dispatch (4 cases, matching original) ---
        _, T0, T1, treatment_kind = prepare_treatment(
            df,
            treatment,
            T0=control_value,
            T1=treatment_value,
        )
        control_value = T0
        treatment_value = T1

        # --- Inference policy (honest gate) ---
        graph_kind = classify_graph_kind(adj)

        if graph_kind == "dag":
            pass  # all effects identifiable
        elif graph_kind == "cpdag":
            is_lg = is_linear and is_gaussian
            policy = check_inference_policy(adj, is_linear_gaussian=is_lg)
            if not policy["allow_inference"]:
                result.warnings = warnings_list + [f"Inference rejected: {policy['reason']}"]
                result.summary += f" Inference rejected: {policy['reason']}"
                return result
            warnings_list.append("CPDAG: undirected edges dropped for estimation")
        elif graph_kind == "pag":
            result.warnings = warnings_list + [
                "Inference rejected: PAG — latent confounders possible, effects not identifiable"
            ]
            result.summary += " Inference rejected: PAG graph."
            return result
        else:
            result.warnings = warnings_list + [f"Inference rejected: unknown graph kind '{graph_kind}'"]
            return result

        # --- Sanitize graph: keep only directed edges (value=1) ---
        clean_adj = np.where(adj == 1, 1, 0).astype(adj.dtype)
        n = adj.shape[0]
        t_idx = names.index(treatment)
        o_idx = names.index(outcome)

        # If T→O edge was undirected and got dropped, restore as directed T→O.
        had_edge = adj[o_idx, t_idx] != 0 or adj[t_idx, o_idx] != 0
        if had_edge and clean_adj[o_idx, t_idx] == 0:
            clean_adj[o_idx, t_idx] = 1
            warnings_list.append(
                f"Restored {treatment}->{outcome} as directed for estimation (was undirected in CPDAG)"
            )

        # --- Determine confounders (full adj matrix type check) ---
        if confounders is not None:
            conf_list = confounders
            potential_conf = []
        else:
            conf_list, potential_conf = identify_confounders(
                adj,
                names,
                treatment,
                outcome,
            )
            if potential_conf and not conf_list:
                # Use potential confounders when no confirmed ones exist
                conf_list = potential_conf
                warnings_list.append(f"Using potential confounders from undirected edges: {conf_list}")

        # --- Check for IV ---
        has_iv = bool(instrument)
        if not has_iv:
            for z_idx in range(n):
                if z_idx in (t_idx, o_idx):
                    continue
                if clean_adj[t_idx, z_idx] != 1:
                    continue
                if clean_adj[o_idx, z_idx] == 1:
                    continue
                has_parents = any(clean_adj[z_idx, j] == 1 for j in range(n) if j != z_idx)
                if not has_parents:
                    has_iv = True
                    if not instrument:
                        instrument = names[z_idx]
                    break

        # --- Auto-select method (data-driven, replaces LLM Filter) ---
        if method is None:
            selected_method = select_estimation_method(
                df,
                treatment,
                treatment_kind,
                is_linear=is_linear,
                is_gaussian=is_gaussian,
                n_features=len(names) - 1,
                has_instrument=has_iv and treatment_kind == "continuous",
            )
        else:
            selected_method = method

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
                estimates = estimate_linear(
                    df,
                    dot_graph,
                    treatment,
                    outcome,
                    control_value,
                    treatment_value,
                )
            elif selected_method == "matching":
                match_conf = conf_list if conf_list else [c for c in names if c != treatment and c != outcome]
                estimates = estimate_matching(
                    df,
                    treatment,
                    outcome,
                    match_conf,
                    int(control_value),
                    int(treatment_value),
                )
            elif selected_method == "dml":
                X_col = [c for c in names if c != treatment and c != outcome and c not in conf_list]
                if not X_col:
                    X_col = conf_list[:] if conf_list else [c for c in names if c != treatment and c != outcome]
                W_col = conf_list if conf_list else []
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
            elif selected_method == "drl":
                X_col = [c for c in names if c != treatment and c != outcome and c not in conf_list]
                if not X_col:
                    X_col = conf_list[:] if conf_list else [c for c in names if c != treatment and c != outcome]
                W_col = conf_list if conf_list else []
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
            elif selected_method == "metalearner":
                X_col = [c for c in names if c != treatment and c != outcome]
                # Data-driven learner selection (S/T/X/DA based on balance + linearity)
                ml_config = get_default_estimation_config(
                    "metalearner",
                    df,
                    treatment,
                    is_linear=is_linear,
                )
                learner = ml_config.get("learner", "t")
                estimates = estimate_metalearner(
                    df,
                    treatment,
                    outcome,
                    X_col,
                    control_value,
                    treatment_value,
                    learner=learner,
                )
            elif selected_method == "iv":
                if not instrument:
                    raise ValueError(
                        "No valid instrument found in graph. Provide instrument= or use a different method."
                    )
                X_col = [c for c in names if c not in (treatment, outcome, instrument)]
                W_col = conf_list if conf_list else []
                estimates = estimate_iv(
                    df,
                    treatment,
                    outcome,
                    instrument,
                    X_col,
                    W_col,
                    control_value,
                    treatment_value,
                )
            else:
                raise ValueError(
                    f"Unknown method '{selected_method}'. Use: linear, matching, dml, drl, metalearner, iv."
                )
        except ImportError as e:
            raise ImportError("Estimation requires inference extras: pip install causal-copilot[inference]") from e

        # --- Build TreatmentEffect ---
        ate_info = estimates.get("ate", {})
        att_info = estimates.get("att", {})
        hte_arr = estimates.get("hte")

        effect = TreatmentEffect(
            treatment=treatment,
            outcome=outcome,
            method=selected_method,
            ate=ate_info.get("estimate"),
            ate_ci=(ate_info.get("ci_lower"), ate_info.get("ci_upper"))
            if ate_info.get("ci_lower") is not None
            else None,
            att=att_info.get("estimate"),
            att_ci=(att_info.get("ci_lower"), att_info.get("ci_upper"))
            if att_info.get("ci_lower") is not None
            else None,
            cate=hte_arr,
            metadata={
                "confounders": conf_list,
                "potential_confounders": potential_conf if not confounders else [],
                "graph_kind": graph_kind,
                "treatment_kind": treatment_kind,
                "control_value": control_value,
                "treatment_value": treatment_value,
                "p_value": ate_info.get("p_value"),
                "algo": estimates.get("algo"),
            },
        )

        key = f"{treatment}->{outcome}"
        result.effects[key] = effect

        # --- Update summary ---
        ate_val = ate_info.get("estimate")
        if ate_val is not None:
            effect_desc = f"Effect of {treatment} on {outcome}: ATE={ate_val:.4f}"
            ci_lo, ci_hi = ate_info.get("ci_lower"), ate_info.get("ci_upper")
            if ci_lo is not None and ci_hi is not None:
                effect_desc += f" (95% CI: [{ci_lo:.4f}, {ci_hi:.4f}])"
            algo_name = estimates.get("algo", selected_method)
            effect_desc += f" via {selected_method}"
            if algo_name and algo_name != selected_method:
                effect_desc += f" ({algo_name})"
            effect_desc += "."
            if hte_arr is not None:
                effect_desc += f" HTE: {len(hte_arr)} per-sample effects computed."
        else:
            effect_desc = f"Could not estimate effect of {treatment} on {outcome}."

        result.summary += f" {effect_desc}"
        result.warnings = warnings_list
        return result

    def refute_estimate(
        self,
        result: CausalResult,
        treatment: str,
        outcome: str,
        *,
        data: pd.DataFrame | None = None,
        control_value: float = 0,
        treatment_value: float = 1,
    ) -> dict[str, Any]:
        """Run sensitivity/refutation analysis on a causal effect estimate.

        Tests robustness via up to four methods (matching original pipeline):
        1. data_subset_refuter — stability under subsampling
        2. random_common_cause — robustness to random confounders
        3. placebo_treatment_refuter — null check via permutation
        4. add_unobserved_common_cause — partial-R2 sensitivity
           (only when graph has common causes)

        Args:
            result: CausalResult with adjacency_matrix.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            data: DataFrame (uses stored data from analyze() if None).
            control_value: Control value for treatment.
            treatment_value: Treatment value.

        Returns:
            Dict with original_estimate and refutation results.
        """
        df = data if data is not None else self._last_data
        if df is None:
            raise ValueError("No data available. Pass data= or call analyze() first.")
        if result.adjacency_matrix is None or result.node_names is None:
            raise ValueError("Result has no adjacency matrix. Run analyze() first.")

        adj = result.adjacency_matrix
        names = result.node_names

        # Sanitize graph: keep only directed edges
        clean_adj = np.where(adj == 1, 1, 0).astype(adj.dtype)

        # Ensure T→O edge exists for refutation
        t_idx = names.index(treatment)
        o_idx = names.index(outcome)
        had_edge = adj[o_idx, t_idx] != 0 or adj[t_idx, o_idx] != 0
        if had_edge and clean_adj[o_idx, t_idx] == 0:
            clean_adj[o_idx, t_idx] = 1

        dot_graph = _adj_to_dot(clean_adj, names)

        # Identify confounders for partial-R2 sensitivity
        from causal_copilot.mcp.offline import identify_confounders

        conf_list, _ = identify_confounders(adj, names, treatment, outcome)

        # Find SHAP top feature for benchmarking (matches original pipeline)
        shap_top = None
        if conf_list:
            try:
                from causal_copilot.mcp.estimation import compute_feature_importance

                props = self._last_properties or {}
                is_linear = bool(props.get("likely_linear", True))
                fi = compute_feature_importance(df, outcome, is_linear)
                top_features = fi.get("top_features", [])
                if top_features:
                    shap_top = top_features[0]
            except Exception:
                pass

        from causal_copilot.mcp.estimation import run_refutation

        return run_refutation(
            df,
            dot_graph,
            treatment,
            outcome,
            control_value,
            treatment_value,
            confounders=conf_list,
            shap_top_feature=shap_top,
        )

    # --- Shared helpers for graph-based methods ---

    def _resolve_inputs(
        self, result: CausalResult, data: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Resolve data + adjacency matrix from result and stored state.

        Returns (df, adj, node_names). Raises ValueError on missing inputs.
        """
        df = data if data is not None else self._last_data
        if df is None:
            raise ValueError("No data available. Pass data= or call analyze() first.")
        if result.adjacency_matrix is None or result.node_names is None:
            raise ValueError("Result has no adjacency matrix. Run analyze() first.")
        return df, result.adjacency_matrix, result.node_names

    def _to_dag(self, adj: np.ndarray) -> np.ndarray:
        """Convert adjacency matrix to DAG for GCM tools.

        Keeps directed edges, orients undirected edges (value=2) using column
        order (lower index causes higher index) to produce a valid DAG extension.
        Raises ValueError for PAG graphs (latent confounders).
        """
        graph_kind = classify_graph_kind(adj)
        if graph_kind == "pag":
            raise ValueError(
                "PAG graph — cannot use GCM tools. "
                "Re-run discovery with an algorithm that produces DAGs (e.g. DirectLiNGAM)."
            )

        n = adj.shape[0]
        dag = np.where(adj == 1, 1, 0).astype(adj.dtype)

        # Orient undirected edges: lower column index → higher column index
        for i in range(n):
            for j in range(i + 1, n):
                has_undirected = (adj[i, j] == 2 or adj[j, i] == 2) and dag[i, j] == 0 and dag[j, i] == 0
                if has_undirected:
                    dag[j, i] = 1  # i→j (adj[j,i]=1 means i causes j)

        return dag

    # --- Graph inspection ---

    def inspect_graph(
        self,
        result: CausalResult,
        *,
        treatment: str | None = None,
        outcome: str | None = None,
    ) -> dict[str, Any]:
        """Analyze a causal graph: classify type, check inference policy, assess queries.

        Args:
            result: CausalResult with adjacency_matrix.
            treatment: Optional treatment variable for query assessment.
            outcome: Optional outcome variable for query assessment.

        Returns:
            Dict with graph_kind, graph_stats, identifiability, inference_policy,
            and optional query_assessment.
        """
        if result.adjacency_matrix is None or result.node_names is None:
            raise ValueError("Result has no adjacency matrix. Run analyze() first.")

        adj = result.adjacency_matrix
        names = result.node_names
        n = adj.shape[0]

        graph_kind = classify_graph_kind(adj)
        edges = get_identifiable_edges(adj, names)

        # Edge statistics
        n_directed = int(np.sum(adj == 1))
        n_undirected = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i, j] == 2 or adj[j, i] == 2)
        n_bidirected = sum(1 for i in range(n) for j in range(i + 1, n) if adj[i, j] == 3 or adj[j, i] == 3)

        # Inference policy — explicit handling per graph kind (matching server.py)
        if graph_kind == "dag":
            inference_policy = {
                "eligibility": True,
                "method": "standard",
                "reason": "DAG — all causal effects identifiable",
            }
        elif graph_kind == "pag":
            inference_policy = {
                "eligibility": False,
                "method": None,
                "reason": "PAG — latent confounders possible, effects not identifiable",
            }
        else:
            # CPDAG: check inference policy with data properties
            props = self._last_properties or {}
            is_lg = bool(props.get("likely_linear")) and bool(props.get("likely_gaussian"))
            policy = check_inference_policy(adj, is_linear_gaussian=is_lg)
            inference_policy = {
                "eligibility": policy["allow_inference"],
                "method": policy.get("method"),
                "reason": policy["reason"],
            }

        output: dict[str, Any] = {
            "graph_kind": graph_kind,
            "graph_stats": {
                "n_nodes": n,
                "n_directed": n_directed,
                "n_undirected": n_undirected,
                "n_bidirected": n_bidirected,
            },
            "identifiability": edges,
            "inference_policy": inference_policy,
        }

        # Query assessment
        if treatment and outcome:
            if treatment not in names:
                raise ValueError(f"Treatment '{treatment}' not in graph nodes: {names}")
            if outcome not in names:
                raise ValueError(f"Outcome '{outcome}' not in graph nodes: {names}")

            t_idx = names.index(treatment)
            o_idx = names.index(outcome)

            # Check directed path and direct connection
            path_exists = _has_directed_path(adj, t_idx, o_idx)
            directly_connected = bool(adj[o_idx, t_idx] == 1)  # T→O edge

            output["query_assessment"] = {
                "treatment": treatment,
                "outcome": outcome,
                "directed_path_exists": path_exists,
                "directly_connected": directly_connected,
                "effect_identifiable": inference_policy["eligibility"] and path_exists,
                "method": inference_policy["method"] if path_exists else None,
            }

        return output

    # --- Causal reasoning (GCM-based) ---

    def estimate_counterfactual(
        self,
        result: CausalResult,
        treatment: str,
        outcome: str,
        intervention_value: float,
        *,
        data: pd.DataFrame | None = None,
        observed_row_index: int = -1,
    ) -> dict[str, Any]:
        """Answer 'what if?': what would outcome be if treatment were set to a value?

        Uses DoWhy GCM counterfactual reasoning. Requires a DAG.

        Args:
            result: CausalResult with adjacency_matrix.
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            intervention_value: Value to set treatment to in the counterfactual.
            data: DataFrame (uses stored data from analyze() if None).
            observed_row_index: Row to counterfactualize (-1 = row with min treatment).

        Returns:
            Dict with observed values, counterfactual values, and estimated effect.
        """
        df, adj, names = self._resolve_inputs(result, data)
        dag_adj = self._to_dag(adj)

        from causal_copilot.mcp.estimation import run_counterfactual

        return run_counterfactual(
            df,
            dag_adj,
            names,
            treatment,
            outcome,
            intervention_value,
            observed_row_index,
        )

    def simulate_intervention(
        self,
        result: CausalResult,
        treatment: str,
        outcome: str,
        intervention_value: float = 1.0,
        *,
        data: pd.DataFrame | None = None,
        shift: bool = True,
        n_samples: int = 1000,
    ) -> dict[str, Any]:
        """Simulate an intervention on a variable and observe outcome distribution.

        Uses DoWhy GCM interventional sampling. Requires a DAG.

        Args:
            result: CausalResult with adjacency_matrix.
            treatment: Variable to intervene on.
            outcome: Variable to observe.
            intervention_value: Value for intervention (shift amount or set value).
            data: DataFrame (uses stored data from analyze() if None).
            shift: True = shift treatment by value, False = set treatment to value.
            n_samples: Number of interventional samples to draw.

        Returns:
            Dict with original and intervention distributions plus mean change.
        """
        df, adj, names = self._resolve_inputs(result, data)
        dag_adj = self._to_dag(adj)

        from causal_copilot.mcp.estimation import run_intervention_simulation

        return run_intervention_simulation(
            df,
            dag_adj,
            names,
            treatment,
            outcome,
            intervention_value,
            shift,
            n_samples,
        )

    def attribute_anomaly(
        self,
        result: CausalResult,
        target_node: str,
        *,
        data: pd.DataFrame | None = None,
        threshold_percentile: float = 95.0,
        n_samples: int = 5,
    ) -> dict[str, Any]:
        """Identify root causes of anomalies in a target variable.

        Uses DoWhy GCM anomaly attribution. Requires a DAG.

        Args:
            result: CausalResult with adjacency_matrix.
            target_node: Variable whose anomalies to explain.
            data: DataFrame (uses stored data from analyze() if None).
            threshold_percentile: Percentile above which values are anomalous.
            n_samples: Number of anomaly samples to analyze.

        Returns:
            Dict with attributions mapping parent nodes to anomaly contribution scores.
        """
        df, adj, names = self._resolve_inputs(result, data)
        dag_adj = self._to_dag(adj)

        from causal_copilot.mcp.estimation import run_anomaly_attribution

        return run_anomaly_attribution(
            df,
            dag_adj,
            names,
            target_node,
            threshold_percentile,
            n_samples,
        )

    def attribute_distribution_change(
        self,
        result: CausalResult,
        target_node: str,
        data_new: pd.DataFrame,
        *,
        data_old: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Explain why a variable's distribution changed between two datasets.

        Uses DoWhy GCM distribution_change attribution. Requires a DAG.

        Args:
            result: CausalResult with adjacency_matrix.
            target_node: Variable whose distribution change to explain.
            data_new: New/current dataset.
            data_old: Old/baseline dataset (uses stored data from analyze() if None).

        Returns:
            Dict with attributions mapping nodes to their contribution to the shift.
        """
        df_old = data_old if data_old is not None else self._last_data
        if df_old is None:
            raise ValueError("No baseline data. Pass data_old= or call analyze() first.")
        if result.adjacency_matrix is None or result.node_names is None:
            raise ValueError("Result has no adjacency matrix. Run analyze() first.")

        adj = result.adjacency_matrix
        names = result.node_names
        dag_adj = self._to_dag(adj)

        from causal_copilot.mcp.estimation import run_distribution_change

        return run_distribution_change(
            df_old,
            data_new,
            dag_adj,
            names,
            target_node,
        )

    # --- Analysis & validation ---

    def compute_feature_importance(
        self,
        result: CausalResult,
        target_node: str,
        *,
        data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Compute SHAP-based feature importance for a target variable.

        Args:
            result: CausalResult (used for data properties to choose SHAP method).
            target_node: Variable to analyze.
            data: DataFrame (uses stored data from analyze() if None).

        Returns:
            Dict with feature_importance mapping and top_features list.
        """
        df = data if data is not None else self._last_data
        if df is None:
            raise ValueError("No data available. Pass data= or call analyze() first.")
        if target_node not in df.columns:
            raise ValueError(f"Target '{target_node}' not in data columns: {list(df.columns)}")

        props = self._last_properties or {}
        is_linear = bool(props.get("likely_linear", True))

        from causal_copilot.mcp.estimation import compute_feature_importance

        return compute_feature_importance(df, target_node, is_linear)

    def validate_graph(
        self,
        result: CausalResult,
        *,
        data: pd.DataFrame | None = None,
        n_permutations: int = 20,
    ) -> dict[str, Any]:
        """Test if the discovered graph is consistent with data.

        Uses DoWhy GCM Local Markov Condition (LMC) falsification. Requires a DAG.

        Args:
            result: CausalResult with adjacency_matrix.
            data: DataFrame (uses stored data from analyze() if None).
            n_permutations: Number of random graph permutations (default 20).

        Returns:
            Dict with falsification result, p-value, and graph statistics.
        """
        df, adj, names = self._resolve_inputs(result, data)
        dag_adj = self._to_dag(adj)

        from causal_copilot.mcp.estimation import run_graph_falsification

        return run_graph_falsification(df, dag_adj, names, n_permutations)

    # --- Reporting ---

    def generate_report(
        self,
        result: CausalResult,
        *,
        data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Generate a comprehensive PDF report from a causal discovery run.

        Creates a publication-quality LaTeX PDF covering:
        - Exploratory data analysis (distributions, correlations)
        - Algorithm selection rationale and hyperparameters
        - Causal graph visualizations (initial + refined)
        - Bootstrap confidence analysis with heatmaps
        - Graph interpretation (LLM-generated narrative)
        - Conclusion and summary

        Prerequisites:
        - A successful analyze() call (pipeline must have been available)
        - LLM access (LLM_PROVIDER env var) for narrative sections
        - LaTeX/latexmk installed for PDF compilation

        Args:
            result: CausalResult from analyze() with adjacency_matrix.
            data: DataFrame (uses stored data from analyze() if None).

        Returns:
            Dict with status, report_path (PDF location), tex_path, and output_dir.
            If PDF compilation fails, status is "partial" and tex_path is still usable.
        """
        import os

        gs = self._last_gs
        if gs is None:
            raise ValueError(
                "No GlobalState available. generate_report requires a prior "
                "analyze() call with the full pipeline available."
            )

        args = self._last_args
        if args is None:
            from causal_copilot.mcp.bridge import make_args

            args = make_args(query="")

        # Fill in fields Report_generation needs
        from causal_copilot.mcp.bridge import PIPELINE_ROOT

        old_cwd = os.getcwd()
        os.chdir(str(PIPELINE_ROOT))
        try:
            from causal_copilot.mcp.server import _prepare_gs_for_report

            gs = _prepare_gs_for_report(gs)

            report_warnings: list[str] = []

            # 1. EDA
            try:
                from preprocess.eda_generation import EDA

                eda = EDA(gs)
                eda.generate_eda()
            except Exception as eda_err:
                report_warnings.append(f"EDA generation skipped: {eda_err}")
                if not hasattr(gs.results, "eda") or gs.results.eda is None:
                    gs.results.eda = {}

            # 2. Visualizations
            try:
                from postprocess.visualization import (
                    Visualization,
                    convert_to_edges,
                )

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

                gs.results.raw_edges = convert_to_edges(
                    algo_name,
                    gs.user_data.processed_data.columns,
                    gs.results.converted_graph,
                )

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

            # 3. Graph effect analysis (LLM call)
            from report.report_generation import Report_generation

            try:
                report_gen_pre = Report_generation(gs, args)
                gs.logging.graph_conversion["initial_graph_analysis"] = report_gen_pre.graph_effect_prompts()
            except Exception as ge_err:
                report_warnings.append(f"Graph effect analysis failed: {ge_err}")
                gs.logging.graph_conversion["initial_graph_analysis"] = (
                    "Graph effect analysis was not available for this run."
                )

            # 4. Generate full report
            report_gen = Report_generation(gs, args)
            report_tex = report_gen.generation()
            report_gen.save_report(report_tex)

            # 5. Check output
            report_path = os.path.join(gs.user_data.output_report_dir, "report.pdf")
            tex_path = os.path.join(gs.user_data.output_report_dir, "report.tex")

            output: dict[str, Any] = {
                "output_dir": gs.user_data.output_report_dir,
                "tex_path": tex_path,
            }

            if os.path.isfile(report_path):
                output["status"] = "ok"
                output["report_path"] = report_path
            else:
                output["status"] = "partial"
                output["error"] = (
                    "LaTeX compilation failed — report.tex was generated but "
                    "PDF was not produced. Check latexmk installation."
                )

            if report_warnings:
                output["warnings"] = report_warnings

            return output
        finally:
            os.chdir(old_cwd)
