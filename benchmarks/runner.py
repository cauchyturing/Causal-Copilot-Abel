"""Benchmark runner — runs algorithms against scenarios and datasets.

Consults the capability matrix to skip incompatible algorithm/scenario pairs
(e.g. time-series algorithms on IID data) and tags results with output_type
so CPDAG vs DAG metrics are not conflated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmarks.capability import CAPABILITY_MATRIX
from benchmarks.datasets import BENCHMARK_DATASETS
from benchmarks.evaluate import DiscoveryMetrics, evaluate_adjacency
from benchmarks.scenarios import ALL_SCENARIOS


@dataclass
class BenchmarkResult:
    algorithm: str
    scenario: str
    status: str  # "ok", "failed", "skipped"
    metrics: DiscoveryMetrics | None
    output_type: str  # "dag", "cpdag", "pag" — from capability matrix
    adjacency_matrix: np.ndarray | None
    provenance: Any  # Provenance from CausalResult
    runtime_seconds: float
    error: str | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "algorithm": self.algorithm,
            "scenario": self.scenario,
            "status": self.status,
            "output_type": self.output_type,
            "runtime_seconds": round(self.runtime_seconds, 3),
        }
        if self.metrics is not None:
            d["metrics"] = self.metrics.to_dict()
        if self.error is not None:
            d["error"] = self.error
        if self.adjacency_matrix is not None:
            d["adjacency_matrix"] = self.adjacency_matrix.tolist()
        if self.provenance is not None:
            d["provenance"] = {
                "seed": self.provenance.seed,
                "algorithm": self.provenance.algorithm,
                "runtime_seconds": self.provenance.runtime_seconds,
            }
        return d


def _is_compatible(algorithm: str, scenario_name: str) -> tuple[bool, str]:
    """Check if algorithm is compatible with scenario using capability matrix.

    A time-series-only algorithm (handles_time_series=True, no other special
    capabilities) is skipped on IID scenarios.
    """
    cap = CAPABILITY_MATRIX.get(algorithm)
    if cap is None:
        return True, ""  # unknown algorithm — let it try

    is_ts_only = (
        cap.handles_time_series
        and not cap.handles_nonlinear
        and not cap.handles_non_gaussian
        and not cap.handles_latent_confounders
    )
    if is_ts_only and scenario_name not in ("time_series",):
        return False, f"{algorithm} requires time-series data, but {scenario_name} is IID"

    return True, ""


def run_benchmark(
    algorithm: str,
    scenario_name: str,
    seed: int = 42,
    timeout: int = 120,
    skip_incompatible: bool = True,
) -> BenchmarkResult:
    """Run a single algorithm on a single scenario/dataset and evaluate.

    Args:
        algorithm: Algorithm name (must be in REGISTRY).
        scenario_name: Name from ALL_SCENARIOS or BENCHMARK_DATASETS.
        seed: Random seed for reproducibility.
        timeout: Timeout in seconds.
        skip_incompatible: If True, skip algorithm/scenario pairs that
            violate the capability matrix (e.g. Granger on IID data).

    Returns:
        BenchmarkResult with status "ok", "failed", or "skipped".

    Raises:
        ValueError: If scenario_name is not found.
    """
    from causal_copilot import CausalCopilot

    cap = CAPABILITY_MATRIX.get(algorithm)
    output_type = cap.output_type if cap else "dag"

    # Look up scenario or dataset (validate BEFORE compatibility check)
    if scenario_name in ALL_SCENARIOS:
        scenario = ALL_SCENARIOS[scenario_name]
        data = scenario.data
        ground_truth = scenario.ground_truth
    elif scenario_name in BENCHMARK_DATASETS:
        ds = BENCHMARK_DATASETS[scenario_name]
        data = ds.data
        ground_truth = ds.ground_truth
    else:
        raise ValueError(f"Unknown scenario/dataset: {scenario_name}")

    # Check compatibility (after scenario validated)
    if skip_incompatible:
        compatible, reason = _is_compatible(algorithm, scenario_name)
        if not compatible:
            return BenchmarkResult(
                algorithm=algorithm,
                scenario=scenario_name,
                status="skipped",
                metrics=None,
                output_type=output_type,
                adjacency_matrix=None,
                provenance=None,
                runtime_seconds=0.0,
                error=reason,
            )

    start = time.monotonic()
    try:
        copilot = CausalCopilot()
        result = copilot.analyze(data, algorithm=algorithm, seed=seed, timeout=timeout)
    except Exception as e:
        return BenchmarkResult(
            algorithm=algorithm,
            scenario=scenario_name,
            status="failed",
            metrics=None,
            output_type=output_type,
            adjacency_matrix=None,
            provenance=None,
            runtime_seconds=time.monotonic() - start,
            error=str(e),
        )
    elapsed = time.monotonic() - start

    if result.status != "ok" or result.adjacency_matrix is None:
        return BenchmarkResult(
            algorithm=algorithm,
            scenario=scenario_name,
            status="failed",
            metrics=None,
            output_type=output_type,
            adjacency_matrix=None,
            provenance=result.provenance,
            runtime_seconds=elapsed,
            error=f"status={result.status}: {result.summary or 'no details'}",
        )

    adj = result.adjacency_matrix
    gt = ground_truth
    if adj.shape != ground_truth.shape:
        gt = ground_truth[: adj.shape[0], : adj.shape[1]]

    metrics = evaluate_adjacency(adj, gt)

    return BenchmarkResult(
        algorithm=algorithm,
        scenario=scenario_name,
        status="ok",
        metrics=metrics,
        output_type=output_type,
        adjacency_matrix=adj,
        provenance=result.provenance,
        runtime_seconds=elapsed,
    )
