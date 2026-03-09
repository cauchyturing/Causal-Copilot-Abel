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
