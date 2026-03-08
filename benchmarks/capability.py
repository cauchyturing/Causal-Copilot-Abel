"""Algorithm capability matrix — documents what each algorithm can and cannot handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class AlgorithmCapability:
    name: str
    output_type: str  # "dag", "cpdag", or "pag"
    assumptions: list[str] = field(default_factory=list)
    handles_latent_confounders: bool = False
    handles_nonlinear: bool = False
    handles_non_gaussian: bool = False
    handles_time_series: bool = False
    max_variables: Optional[int] = None
    recommended_n_min: Optional[int] = None


CAPABILITY_MATRIX: dict[str, AlgorithmCapability] = {
    "PC": AlgorithmCapability(
        name="PC",
        output_type="cpdag",
        assumptions=[
            "Causal sufficiency",
            "Faithfulness",
            "Linear relationships (with Fisher-z test)",
        ],
    ),
    "GES": AlgorithmCapability(
        name="GES",
        output_type="cpdag",
        assumptions=[
            "Causal sufficiency",
            "Faithfulness",
            "BIC score assumes Gaussian noise",
        ],
    ),
    "NOTEARSLinear": AlgorithmCapability(
        name="NOTEARSLinear",
        output_type="dag",
        max_variables=50,
        assumptions=[
            "Causal sufficiency",
            "Linear structural equations",
            "Continuous optimization — may find local optima",
        ],
    ),
    "DirectLiNGAM": AlgorithmCapability(
        name="DirectLiNGAM",
        output_type="dag",
        handles_non_gaussian=True,
        assumptions=[
            "Causal sufficiency",
            "Linear structural equations",
            "Non-Gaussian noise (identifiability condition)",
        ],
    ),
    "PCMCI": AlgorithmCapability(
        name="PCMCI",
        output_type="dag",
        handles_time_series=True,
        assumptions=[
            "Causal sufficiency",
            "Stationarity",
            "Time-lagged causal relationships",
        ],
    ),
    "ICALiNGAM": AlgorithmCapability(
        name="ICALiNGAM",
        output_type="dag",
        handles_non_gaussian=True,
        recommended_n_min=500,
        assumptions=[
            "Causal sufficiency",
            "Linear structural equations",
            "Non-Gaussian noise",
            "ICA decomposition — sensitive to sample size",
        ],
    ),
    "GrangerCausality": AlgorithmCapability(
        name="GrangerCausality",
        output_type="dag",
        handles_time_series=True,
        assumptions=[
            "Stationarity",
            "Linear autoregressive model",
            "Pairwise testing — does not account for multivariate confounding",
            "Predictive, not structural causation",
        ],
    ),
    # ── Tier 1: causal-learn algorithms ─────────────────────────────
    "FCI": AlgorithmCapability(
        name="FCI",
        output_type="pag",
        handles_latent_confounders=True,
        assumptions=[
            "Faithfulness",
            "Allows latent confounders",
            "Linear relationships (with Fisher-z test)",
        ],
    ),
    "GRaSP": AlgorithmCapability(
        name="GRaSP",
        output_type="cpdag",
        assumptions=[
            "Causal sufficiency",
            "Faithfulness",
            "Permutation-based search over orderings",
        ],
    ),
    "CDNOD": AlgorithmCapability(
        name="CDNOD",
        output_type="cpdag",
        assumptions=[
            "Faithfulness",
            "Multi-domain / nonstationary data with domain_index",
            "Linear relationships (with Fisher-z test)",
        ],
    ),
    "VARLiNGAM": AlgorithmCapability(
        name="VARLiNGAM",
        output_type="dag",
        handles_non_gaussian=True,
        handles_time_series=True,
        assumptions=[
            "Linear structural equations",
            "Non-Gaussian noise (identifiability condition)",
            "Stationarity (VAR model)",
        ],
    ),
    "CALM": AlgorithmCapability(
        name="CALM",
        output_type="dag",
        assumptions=[
            "Causal sufficiency",
            "Continuous optimization with L0 penalty",
            "Acyclicity via augmented Lagrangian",
        ],
    ),
    # ── Tier 2: gcastle algorithms ──────────────────────────────────
    "GOLEM": AlgorithmCapability(
        name="GOLEM",
        output_type="dag",
        assumptions=[
            "Causal sufficiency",
            "Likelihood-based continuous optimization",
            "Equal or unequal noise variances",
        ],
    ),
    "NOTEARSNonlinear": AlgorithmCapability(
        name="NOTEARSNonlinear",
        output_type="dag",
        handles_nonlinear=True,
        assumptions=[
            "Causal sufficiency",
            "Nonlinear structural equations (MLP or Sobolev)",
            "Continuous optimization — may find local optima",
        ],
    ),
    "CORL": AlgorithmCapability(
        name="CORL",
        output_type="dag",
        assumptions=[
            "Causal sufficiency",
            "Reinforcement learning-based ordering search",
            "BIC reward — assumes Gaussian noise for scoring",
        ],
    ),
    "PCParallel": AlgorithmCapability(
        name="PCParallel",
        output_type="cpdag",
        assumptions=[
            "Causal sufficiency",
            "Faithfulness",
            "Parallel implementation of PC for scalability",
        ],
    ),
    # ── Tier 3: Other packages ──────────────────────────────────────
    "XGES": AlgorithmCapability(
        name="XGES",
        output_type="cpdag",
        assumptions=[
            "Causal sufficiency",
            "Faithfulness",
            "Fast GES variant with optimized search",
        ],
    ),
    "DYNOTEARS": AlgorithmCapability(
        name="DYNOTEARS",
        output_type="dag",
        handles_time_series=True,
        assumptions=[
            "Causal sufficiency",
            "Stationarity",
            "Continuous optimization (NOTEARS-style) for time-series",
        ],
    ),
    "Hybrid": AlgorithmCapability(
        name="Hybrid",
        output_type="dag",
        handles_nonlinear=True,
        assumptions=[
            "Two-stage: constraint-based (PC/GES) + functional (ANM/PNL)",
            "Nonlinear causal relationships",
            "Causal sufficiency",
        ],
    ),
}
