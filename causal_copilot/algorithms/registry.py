"""Canonical algorithm registry -- single source of truth.

Every algorithm's metadata, adapter class, defaults, and upstream info
lives here. copilot.py, planner.py, cli.py all read from this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    adapter_cls: type
    family: str  # constraint, score, functional, hybrid, timeseries
    default_params: dict[str, Any]
    upstream_packages: list[str]
    algorithm_version: str  # e.g. "pc-v1|causallearn=0.1.3"
    tags: tuple = ()
    optional_deps: tuple = ()


def _build_registry() -> dict[str, AlgorithmSpec]:
    from causal_copilot.algorithms.adapters import (
        CALMAdapter,
        CDNODAdapter,
        CORLAdapter,
        DirectLiNGAMAdapter,
        DYNOTEARSAdapter,
        FCIAdapter,
        GESAdapter,
        GOLEMAdapter,
        GrangerCausalityAdapter,
        GRaSPAdapter,
        HybridAdapter,
        ICALiNGAMAdapter,
        NOTEARSLinearAdapter,
        NOTEARSNonlinearAdapter,
        PCAdapter,
        PCMCIAdapter,
        PCParallelAdapter,
        VARLiNGAMAdapter,
        XGESAdapter,
    )

    return {
        # ── Tier 0: Original 7 algorithms ──────────────────────────────
        "PC": AlgorithmSpec(
            name="PC",
            adapter_cls=PCAdapter,
            family="constraint",
            default_params={"alpha": 0.05, "indep_test": "fisherz", "stable": True, "depth": 4},
            upstream_packages=["causal-learn"],
            algorithm_version="pc-v1",
            tags=("cpdag-output", "linear", "gaussian"),
        ),
        "GES": AlgorithmSpec(
            name="GES",
            adapter_cls=GESAdapter,
            family="score",
            default_params={"score_func": "local_score_BIC"},
            upstream_packages=["causal-learn"],
            algorithm_version="ges-v1",
            tags=("cpdag-output", "linear"),
        ),
        "NOTEARSLinear": AlgorithmSpec(
            name="NOTEARSLinear",
            adapter_cls=NOTEARSLinearAdapter,
            family="score",
            default_params={"lambda1": 0.1, "max_iter": 100, "h_tol": 1e-8, "w_threshold": 0.3},
            upstream_packages=["castle", "causal-learn"],
            algorithm_version="notears-linear-v1",
            tags=("dag-output", "linear", "continuous-optimization"),
            optional_deps=("algorithms",),
        ),
        "DirectLiNGAM": AlgorithmSpec(
            name="DirectLiNGAM",
            adapter_cls=DirectLiNGAMAdapter,
            family="functional",
            default_params={"measure": "pwling"},
            upstream_packages=["causal-learn"],
            algorithm_version="direct-lingam-v1",
            tags=("dag-output", "linear", "non-gaussian"),
        ),
        "PCMCI": AlgorithmSpec(
            name="PCMCI",
            adapter_cls=PCMCIAdapter,
            family="timeseries",
            default_params={"tau_min": 0, "tau_max": 2, "pc_alpha": 0.05, "alpha_level": 0.05},
            upstream_packages=["tigramite"],
            algorithm_version="pcmci-v1",
            tags=("timeseries", "constraint"),
            optional_deps=("algorithms",),
        ),
        "ICALiNGAM": AlgorithmSpec(
            name="ICALiNGAM",
            adapter_cls=ICALiNGAMAdapter,
            family="functional",
            default_params={"max_iter": 1000},
            upstream_packages=["causal-learn"],
            algorithm_version="ica-lingam-v1",
            tags=("dag-output", "linear", "non-gaussian"),
        ),
        "GrangerCausality": AlgorithmSpec(
            name="GrangerCausality",
            adapter_cls=GrangerCausalityAdapter,
            family="timeseries",
            default_params={"p": 10, "alpha": 0.05, "criterion": "ssr_ftest"},
            upstream_packages=["statsmodels"],
            algorithm_version="granger-v1",
            tags=("timeseries", "linear", "pairwise"),
        ),
        # ── Tier 1: causal-learn algorithms ─────────────────────────────
        "FCI": AlgorithmSpec(
            name="FCI",
            adapter_cls=FCIAdapter,
            family="constraint",
            default_params={"alpha": 0.05, "indep_test": "fisherz", "depth": 4, "max_path_length": -1},
            upstream_packages=["causal-learn"],
            algorithm_version="fci-v1",
            tags=("pag-output", "latent-confounders"),
        ),
        "GRaSP": AlgorithmSpec(
            name="GRaSP",
            adapter_cls=GRaSPAdapter,
            family="score",
            default_params={"score_func": "local_score_BIC_from_cov", "depth": 4},
            upstream_packages=["causal-learn"],
            algorithm_version="grasp-v1",
            tags=("cpdag-output", "permutation"),
        ),
        "CDNOD": AlgorithmSpec(
            name="CDNOD",
            adapter_cls=CDNODAdapter,
            family="constraint",
            default_params={
                "alpha": 0.05,
                "indep_test": "fisherz",
                "stable": True,
                "uc_rule": 0,
                "uc_priority": 2,
                "depth": 5,
            },
            upstream_packages=["causal-learn"],
            algorithm_version="cdnod-v1",
            tags=("cpdag-output", "multi-domain"),
        ),
        "VARLiNGAM": AlgorithmSpec(
            name="VARLiNGAM",
            adapter_cls=VARLiNGAMAdapter,
            family="timeseries",
            default_params={"lags": 10, "criterion": "bic", "prune": True},
            upstream_packages=["causal-learn"],
            algorithm_version="var-lingam-v1",
            tags=("timeseries", "linear", "non-gaussian", "dag-output"),
        ),
        "CALM": AlgorithmSpec(
            name="CALM",
            adapter_cls=CALMAdapter,
            family="score",
            default_params={"lambda1": 0.005, "alpha": 0.01, "tau": 0.5},
            upstream_packages=["causal-learn"],
            algorithm_version="calm-v1",
            tags=("dag-output", "continuous-optimization", "gpu"),
            optional_deps=("gpu",),
        ),
        # ── Tier 2: gcastle algorithms ──────────────────────────────────
        "GOLEM": AlgorithmSpec(
            name="GOLEM",
            adapter_cls=GOLEMAdapter,
            family="score",
            default_params={
                "lambda_1": 0.01,
                "lambda_2": 5.0,
                "equal_variances": True,
                "learning_rate": 1e-3,
                "num_iter": 10000,
                "graph_thres": 0.3,
            },
            upstream_packages=["castle"],
            algorithm_version="golem-v1",
            tags=("dag-output", "continuous-optimization"),
            optional_deps=("algorithms",),
        ),
        "NOTEARSNonlinear": AlgorithmSpec(
            name="NOTEARSNonlinear",
            adapter_cls=NOTEARSNonlinearAdapter,
            family="score",
            default_params={
                "lambda1": 0.01,
                "lambda2": 0.01,
                "max_iter": 100,
                "h_tol": 1e-8,
                "w_threshold": 0.3,
                "hidden_layers": (10, 1),
                "model_type": "mlp",
            },
            upstream_packages=["castle"],
            algorithm_version="notears-nonlinear-v1",
            tags=("dag-output", "nonlinear", "continuous-optimization"),
            optional_deps=("algorithms",),
        ),
        "CORL": AlgorithmSpec(
            name="CORL",
            adapter_cls=CORLAdapter,
            family="score",
            default_params={
                "batch_size": 64,
                "input_dim": 100,
                "embed_dim": 256,
                "encoder_name": "transformer",
                "decoder_name": "lstm",
                "reward_score_type": "BIC",
                "iteration": 500,
            },
            upstream_packages=["castle"],
            algorithm_version="corl-v1",
            tags=("dag-output", "reinforcement-learning"),
            optional_deps=("algorithms",),
        ),
        "PCParallel": AlgorithmSpec(
            name="PCParallel",
            adapter_cls=PCParallelAdapter,
            family="constraint",
            default_params={"alpha": 0.05, "indep_test": "fisherz", "cores": 8},
            upstream_packages=["castle"],
            algorithm_version="pc-parallel-v1",
            tags=("cpdag-output", "parallel"),
            optional_deps=("algorithms",),
        ),
        # ── Tier 3: Other packages ──────────────────────────────────────
        "XGES": AlgorithmSpec(
            name="XGES",
            adapter_cls=XGESAdapter,
            family="score",
            default_params={"alpha": 2.0},
            upstream_packages=["xges"],
            algorithm_version="xges-v1",
            tags=("cpdag-output", "fast"),
            optional_deps=("algorithms",),
        ),
        "DYNOTEARS": AlgorithmSpec(
            name="DYNOTEARS",
            adapter_cls=DYNOTEARSAdapter,
            family="timeseries",
            default_params={
                "p": 10,
                "lambda_w": 0.01,
                "lambda_a": 0.01,
                "max_iter": 100,
                "h_tol": 1e-8,
                "w_threshold": 0.05,
            },
            upstream_packages=["causalnex"],
            algorithm_version="dynotears-v1",
            tags=("timeseries", "continuous-optimization", "dag-output"),
            optional_deps=("algorithms",),
        ),
        "Hybrid": AlgorithmSpec(
            name="Hybrid",
            adapter_cls=HybridAdapter,
            family="hybrid",
            default_params={
                "first_stage_algo": "pc",
                "second_stage_method": "pnl",
                "alpha": 0.05,
                "m_max": 3,
            },
            upstream_packages=["causal-learn", "castle"],
            algorithm_version="hybrid-v1",
            tags=("dag-output", "nonlinear"),
            optional_deps=("algorithms",),
        ),
    }


REGISTRY: dict[str, AlgorithmSpec] = _build_registry()

# pip-name -> import-name for availability checks
_PKG_IMPORT_MAP: dict[str, str] = {
    "causal-learn": "causallearn",
    "castle": "castle",
    "tigramite": "tigramite",
    "statsmodels": "statsmodels",
    "lingam": "lingam",
    "xges": "xges",
    "causalnex": "causalnex",
}


def is_algorithm_available(name: str) -> bool:
    """Check whether *name*'s upstream packages are importable.

    Returns True if every package listed in the spec's ``upstream_packages``
    can be imported, False otherwise.  Unknown algorithms return False.
    """
    spec = REGISTRY.get(name)
    if spec is None:
        return False
    for pkg in spec.upstream_packages:
        import_name = _PKG_IMPORT_MAP.get(pkg, pkg)
        try:
            __import__(import_name)
        except ImportError:
            return False
    return True


def available_algorithms() -> dict[str, AlgorithmSpec]:
    """Return the subset of REGISTRY whose deps are importable."""
    return {name: spec for name, spec in REGISTRY.items() if is_algorithm_available(name)}
