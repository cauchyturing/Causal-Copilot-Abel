"""Thin estimation wrappers for MCP effect estimation.

Each function has lazy imports and returns a normalized dict.
We bypass the heavy Analysis class (which imports shap, matplotlib, etc.)
and call the underlying libraries directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_linear(
    data: pd.DataFrame,
    dot_graph: str,
    treatment: str,
    outcome: str,
    control_value: float,
    treatment_value: float,
) -> dict:
    """Estimate ATE via DoWhy backdoor linear regression.

    Returns dict with 'ate' key containing estimate, ci_lower, ci_upper, p_value.
    """
    from dowhy import CausalModel

    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        graph=dot_graph,
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        estimand,
        method_name="backdoor.linear_regression",
        control_value=control_value,
        treatment_value=treatment_value,
        target_units="ate",
    )
    significance = estimate.estimator.test_significance(
        data, estimate.value,
    )
    p_value = significance["p_value"]
    if isinstance(p_value, (list, np.ndarray)):
        p_value = float(p_value[0])
    else:
        p_value = float(p_value)

    return {
        "ate": {
            "estimate": float(estimate.value),
            "ci_lower": None,
            "ci_upper": None,
            "p_value": p_value,
        },
    }


def estimate_matching(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    control_value: int,
    treatment_value: int,
) -> dict:
    """Estimate ATE via propensity score matching.

    Treatment must be binary. Confounders used for propensity model.
    Returns dict with 'ate' key.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import NearestNeighbors

    df = data.copy()

    # Ensure binary treatment
    if df[treatment].nunique() > 2:
        threshold = df[treatment].median()
        df[treatment] = (df[treatment] > threshold).astype(int)

    # Propensity score model
    X_conf = df[confounders].values
    t_vals = df[treatment].values
    ps_model = LogisticRegression(solver="liblinear", max_iter=1000)
    ps_model.fit(X_conf, t_vals)
    df["_ps"] = ps_model.predict_proba(X_conf)[:, 1]

    # Nearest-neighbor matching
    treated = df[df[treatment] == treatment_value]
    control = df[df[treatment] == control_value]

    if len(treated) == 0 or len(control) == 0:
        return {"ate": {"estimate": None, "ci_lower": None, "ci_upper": None, "p_value": None}}

    nbrs = NearestNeighbors(n_neighbors=1).fit(control[["_ps"]])
    _, indices = nbrs.kneighbors(treated[["_ps"]])
    matched_control = control.iloc[indices.flatten()]

    ate = float(treated[outcome].mean() - matched_control[outcome].mean())

    return {
        "ate": {
            "estimate": ate,
            "ci_lower": None,
            "ci_upper": None,
            "p_value": None,
        },
    }


def estimate_dml(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    X_col: list[str],
    W_col: list[str],
    T0: float,
    T1: float,
) -> dict:
    """Estimate ATE/ATT via Double Machine Learning (EconML).

    Uses LinearDML with LinearRegression defaults (no LLM needed).
    X_col = effect modifiers, W_col = confounders/controls.
    """
    from causal_copilot.mcp.offline import get_default_estimation_config

    config = get_default_estimation_config("dml", data, treatment)

    from causal_inference.DML.hte_program import HTE_Programming
    from causal_copilot.mcp.bridge import make_args, make_global_state

    gs = make_global_state(data)
    gs.user_data.processed_data = data.copy()
    gs.inference.hte_algo_json = {"name": config["algo"]}
    gs.inference.hte_model_param = {
        "model_y": config["model_y"],
        "model_t": config["model_t"],
    }
    args = make_args()

    # Ensure W_col is non-empty (DML requires controls)
    df = data.copy()
    actual_W = list(W_col)
    if len(actual_W) == 0:
        df["_W_dummy"] = 0.0
        actual_W = ["_W_dummy"]
        gs.user_data.processed_data = df

    programmer = HTE_Programming(
        args, y_col=outcome, T_col=treatment,
        T0=T0, T1=T1, X_col=X_col, W_col=actual_W,
    )
    programmer.fit_model(gs)

    ate, ate_lower, ate_upper = programmer.forward(gs, task="ate")
    att, att_lower, att_upper = programmer.forward(gs, task="att")

    def _safe_float(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)

    return {
        "ate": {
            "estimate": _safe_float(ate),
            "ci_lower": _safe_float(ate_lower),
            "ci_upper": _safe_float(ate_upper),
            "p_value": None,
        },
        "att": {
            "estimate": _safe_float(att),
            "ci_lower": _safe_float(att_lower),
            "ci_upper": _safe_float(att_upper),
            "p_value": None,
        },
    }
