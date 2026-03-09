"""Offline (rule-based) algorithm selection and default hyperparameters.

Used as fallback when LLM-based selection (Filter → Reranker) is unavailable
or fails.  Pure heuristic — no API calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_algorithm_offline(statistics) -> str:
    """Pick a sensible algorithm based on dataset statistics alone.

    Decision tree (simplified):
      time-series → PCMCI
      non-Gaussian errors → DirectLiNGAM
      non-linear → GES (robust default)
      linear + Gaussian + missing → PC (mvpc mode)
      linear + Gaussian → PC
      else → PC
    """
    ts = getattr(statistics, "time_series", False)
    if ts:
        return "PCMCI"

    gaussian = getattr(statistics, "gaussian_error", True)
    linear = getattr(statistics, "linearity", True)

    if not gaussian and linear:
        return "DirectLiNGAM"

    if not linear:
        return "GES"

    return "PC"


def get_default_hp(algorithm: str, statistics) -> dict:
    """Return conservative default hyperparameters for *algorithm*."""
    missing = getattr(statistics, "missingness", False)

    defaults: dict = {
        "PC": {"alpha": 0.05, "indep_test": "fisherz", "stable": True},
        "GES": {"score_func": "local_score_BIC"},
        "DirectLiNGAM": {"measure": "pwling"},
        "PCMCI": {"tau_max": 3, "pc_alpha": 0.05},
        "FCI": {"alpha": 0.05, "indep_test": "fisherz"},
    }

    hp = dict(defaults.get(algorithm, {}))

    if algorithm == "PC" and missing:
        hp["mvpc"] = True

    return hp


# ── Treatment preparation ────────────────────────────────────────────


def prepare_treatment(
    data: pd.DataFrame,
    treatment: str,
    T0: float | None = None,
    T1: float | None = None,
) -> tuple[pd.Series, float, float, str]:
    """Classify treatment type and prepare T0/T1 values.

    Mirrors the 4-case dispatch from the original inference.py
    ``prepare_treatment_column()``.

    Returns:
        (treatment_series, T0, T1, treatment_kind)
        treatment_kind is one of: "categorical", "binary", "discrete", "continuous"
    """
    treatment_col = data[treatment]

    # Case 1: String / object / category
    if treatment_col.dtype == "object" or treatment_col.dtype.name == "category":
        unique_vals = sorted(treatment_col.unique().tolist())
        if T0 is None or T1 is None:
            T0, T1 = unique_vals[0], unique_vals[1]
        return treatment_col, T0, T1, "categorical"

    # Case 2: Binary numeric
    if treatment_col.nunique() == 2:
        sorted_vals = sorted(treatment_col.unique())
        if T0 is None or T1 is None:
            T0, T1 = sorted_vals[0], sorted_vals[1]
        return treatment_col, T0, T1, "binary"

    # Case 3: Discrete numeric (≤ 10 unique values)
    unique_vals = sorted(treatment_col.unique())
    if pd.api.types.is_numeric_dtype(treatment_col) and len(unique_vals) <= 10:
        if T0 is None or T1 is None:
            T0, T1 = unique_vals[0], unique_vals[1]
        return treatment_col, T0, T1, "discrete"

    # Case 4: Continuous treatment — use 10th/90th percentile
    if T0 is None or T1 is None:
        T0 = float(treatment_col.quantile(0.1))
        T1 = float(treatment_col.quantile(0.9))
    return treatment_col, T0, T1, "continuous"


# ── Confounder identification ─────────────────────────────────────────


def identify_confounders(
    adj: np.ndarray,
    names: list[str],
    treatment: str,
    outcome: str,
) -> tuple[list[str], list[str]]:
    """Identify confounders from adjacency matrix, matching original inference.py.

    A confounder is a variable that causally influences BOTH treatment and outcome.
    Uses the full adjacency matrix encoding:
      value 1 = directed, 3 = bidirected, 4 = circle-arrow  → confirmed influence
      value 2 = undirected (either direction)                → potential influence

    Returns:
        (confirmed_confounders, potential_confounders)
    """
    t_idx = names.index(treatment)
    o_idx = names.index(outcome)
    n = adj.shape[0]

    confounders = []
    potential_confounders = []

    for idx in range(n):
        name = names[idx]
        if name == treatment or name == outcome:
            continue

        # Check if variable influences treatment (adj[t_idx, idx] means idx→treatment)
        treatment_influence = adj[t_idx, idx] in (1, 3, 4)
        potential_t = adj[t_idx, idx] == 2 or adj[idx, t_idx] == 2

        # Check if variable influences outcome
        outcome_influence = adj[o_idx, idx] in (1, 3, 4)
        potential_o = adj[o_idx, idx] == 2 or adj[idx, o_idx] == 2

        if treatment_influence and outcome_influence:
            confounders.append(name)
        elif potential_t and potential_o:
            potential_confounders.append(name)

    return confounders, potential_confounders


# ── Rule-based estimation configuration ───────────────────────────────


def select_estimation_method(
    data: pd.DataFrame,
    treatment: str,
    treatment_kind: str,
    is_linear: bool = True,
    is_gaussian: bool = True,
    n_features: int = 0,
    has_instrument: bool = False,
) -> str:
    """Data-driven estimation method selection (replaces LLM-based Filter).

    Decision tree:
      instrument available → iv
      binary/categorical treatment + linear data → metalearner (T-learner)
      binary/categorical treatment + non-linear → dml (CausalForestDML)
      continuous + linear + gaussian → linear
      continuous + many features (>20) → dml (SparseLinearDML)
      continuous + non-linear → dml (CausalForestDML fallback to LinearDML)
      else → dml
    """
    if has_instrument:
        return "iv"

    if treatment_kind in ("binary", "categorical"):
        if is_linear:
            return "metalearner"
        return "dml"

    # Continuous or discrete treatment
    if is_linear and is_gaussian:
        return "linear"

    return "dml"


def select_dml_variant(
    data: pd.DataFrame,
    treatment: str,
    treatment_kind: str,
    is_linear: bool = True,
    n_features: int = 0,
) -> str:
    """Pick DML sub-algorithm (replaces LLM DML_HTE_Filter).

    Decision tree:
      non-linear data + binary treatment → CausalForestDML
      many features (>20) → SparseLinearDML
      else → LinearDML
    """
    if not is_linear and treatment_kind in ("binary", "discrete"):
        return "CausalForestDML"

    if n_features > 20:
        return "SparseLinearDML"

    return "LinearDML"


def select_models_for_method(
    method: str,
    algo: str,
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    is_linear: bool = True,
) -> dict:
    """Data-driven model selection for nuisance models (replaces LLM Param_Selector).

    For DML/DRL: picks model_y, model_t (and model_final if applicable).
    For MetaLearners: picks base learner.

    Decision tree for each nuisance model:
      binary target → LogisticRegressionCV
      linear data → LinearRegression
      non-linear data → GradientBoostingRegressor (or RandomForestRegressor)
    """
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, LogisticRegressionCV

    binary_t = data[treatment].nunique() <= 2
    binary_y = data[outcome].nunique() <= 2

    def _pick_model(is_binary: bool) -> object:
        if is_binary:
            if is_linear:
                return LogisticRegressionCV(max_iter=1000)
            return GradientBoostingClassifier(n_estimators=100)
        if is_linear:
            return LinearRegression()
        return GradientBoostingRegressor(n_estimators=100)

    if method == "dml":
        config = {
            "algo": algo,
            "model_y": _pick_model(binary_y),
            "model_t": _pick_model(binary_t),
        }
        return config

    if method == "drl":
        config = {
            "algo": algo,
            "model_regression": _pick_model(is_binary=False),  # outcome model
            "model_propensity": _pick_model(is_binary=True),   # treatment model
        }
        return config

    if method == "metalearner":
        if is_linear:
            return {"algo": "TLearner", "learner": "t"}
        return {"algo": "XLearner", "learner": "x"}

    if method == "iv":
        return {"algo": "LinearDRIV"}

    raise ValueError(f"Unknown estimation method: {method}")


def get_default_estimation_config(
    method: str,
    data: pd.DataFrame,
    treatment: str,
    *,
    outcome: str | None = None,
    is_linear: bool = True,
    treatment_kind: str = "binary",
) -> dict:
    """Rule-based defaults for estimation methods.

    Enhanced version: uses data properties to select both algorithm variant
    and nuisance models (replaces LLM-based Filter + Param_Selector).
    """
    n_features = data.shape[1] - 1  # exclude treatment

    if method == "dml":
        algo = select_dml_variant(
            data, treatment, treatment_kind,
            is_linear=is_linear, n_features=n_features,
        )
    elif method == "drl":
        if not is_linear and treatment_kind in ("binary", "discrete"):
            algo = "ForestDRL"
        elif n_features > 20:
            algo = "SparseLinearDRL"
        else:
            algo = "LinearDRL"
    elif method in ("metalearner", "iv"):
        algo = "TLearner" if method == "metalearner" else "LinearDRIV"
    else:
        raise ValueError(f"Unknown estimation method: {method}")

    outcome_col = outcome or ([c for c in data.columns if c != treatment][0])

    return select_models_for_method(
        method, algo, data, treatment, outcome_col,
        is_linear=is_linear,
    )
