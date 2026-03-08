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
}
