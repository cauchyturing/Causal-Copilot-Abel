"""Thin estimation wrappers for MCP effect estimation.

Each function has lazy imports and returns a normalized dict.
We bypass the heavy Analysis class (which imports shap, matplotlib, etc.)
and call the underlying libraries directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_float(v):
    """Convert numeric value to float, returning None for None/NaN."""
    if v is None:
        return None
    v = float(v)
    if np.isnan(v):
        return None
    return v


# ── Treatment Effect Estimation ──────────────────────────────────────


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
    try:
        significance = estimate.test_stat_significance()
        p_value = significance["p_value"]
        if isinstance(p_value, (list, np.ndarray)):
            p_value = float(p_value[0])
        else:
            p_value = float(p_value)
    except (AttributeError, TypeError):
        p_value = None

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
    *,
    is_linear: bool = True,
    treatment_kind: str = "binary",
    compute_hte: bool = True,
) -> dict:
    """Estimate ATE/ATT/HTE via Double Machine Learning (EconML).

    Uses data-driven model selection (offline heuristics, no LLM needed).
    X_col = effect modifiers, W_col = confounders/controls.
    """
    from causal_copilot.mcp.offline import get_default_estimation_config

    config = get_default_estimation_config(
        "dml",
        data,
        treatment,
        outcome=outcome,
        is_linear=is_linear,
        treatment_kind=treatment_kind,
    )

    from causal_copilot.mcp.bridge import make_args, make_global_state
    from causal_inference.DML.hte_program import HTE_Programming

    gs = make_global_state(data)
    gs.user_data.processed_data = data.copy()
    gs.inference.hte_algo_json = {"name": config["algo"]}
    gs.inference.hte_model_param = {
        "model_y": config.get("model_y"),
        "model_t": config.get("model_t"),
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
        args,
        y_col=outcome,
        T_col=treatment,
        T0=T0,
        T1=T1,
        X_col=X_col,
        W_col=actual_W,
    )
    programmer.fit_model(gs)

    ate, ate_lower, ate_upper = programmer.forward(gs, task="ate")
    att, att_lower, att_upper = programmer.forward(gs, task="att")

    result = {
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
        "algo": config["algo"],
    }

    # HTE: per-sample heterogeneous treatment effects
    if compute_hte:
        try:
            hte, hte_lower, hte_upper = programmer.forward(gs, task="hte")
            hte_arr = np.array(hte).flatten()
            result["hte"] = hte_arr
            result["hte_ci_lower"] = np.array(hte_lower).flatten() if hte_lower is not None else None
            result["hte_ci_upper"] = np.array(hte_upper).flatten() if hte_upper is not None else None
        except Exception:
            pass  # HTE not always available

    return result


def estimate_drl(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    X_col: list[str],
    W_col: list[str],
    T0: float,
    T1: float,
    *,
    is_linear: bool = True,
    treatment_kind: str = "binary",
    compute_hte: bool = True,
) -> dict:
    """Estimate ATE/ATT/HTE via Doubly Robust Learning (EconML).

    Uses data-driven variant selection (LinearDRL/SparseLinearDRL/ForestDRL)
    via offline heuristics, then calls EconML directly with numpy arrays to
    avoid DataFrame/numpy incompatibilities in upstream wrappers.
    """
    from econml.dr import ForestDRLearner, LinearDRLearner, SparseLinearDRLearner

    from causal_copilot.mcp.offline import get_default_estimation_config

    config = get_default_estimation_config(
        "drl",
        data,
        treatment,
        outcome=outcome,
        is_linear=is_linear,
        treatment_kind=treatment_kind,
    )

    df = data.copy()
    actual_W = list(W_col)
    if len(actual_W) == 0:
        df["_W_dummy"] = 0.0
        actual_W = ["_W_dummy"]

    Y = df[outcome].values
    T = df[treatment].values
    X = df[X_col].values
    W = df[actual_W].values

    # Dispatch to correct DRL variant based on offline selection
    algo = config["algo"]
    if algo == "ForestDRL":
        model = ForestDRLearner(
            model_regression=config.get("model_regression"),
            model_propensity=config.get("model_propensity"),
        )
    elif algo == "SparseLinearDRL":
        model = SparseLinearDRLearner(
            model_regression=config.get("model_regression"),
            model_propensity=config.get("model_propensity"),
            cv=5,
        )
    else:  # LinearDRL (default)
        model = LinearDRLearner(
            model_regression=config.get("model_regression"),
            model_propensity=config.get("model_propensity"),
            cv=5,
        )
    model.fit(Y, T, X=X, W=W)

    ate = float(model.ate(X=X, T0=T0, T1=T1))
    try:
        ate_lower, ate_upper = model.ate_interval(X=X, T0=T0, T1=T1)
        ate_lower, ate_upper = float(ate_lower), float(ate_upper)
    except Exception:
        ate_lower = ate_upper = None

    # ATT: average effect on treated units
    treated = np.isclose(T, T1)
    if treated.sum() > 0:
        effects = model.effect(X[treated], T0=T0, T1=T1)
        att = float(np.mean(effects))
        try:
            lb, ub = model.effect_interval(X[treated], T0=T0, T1=T1)
            att_lower, att_upper = float(np.mean(lb)), float(np.mean(ub))
        except Exception:
            att_lower = att_upper = None
    else:
        att = att_lower = att_upper = None

    result = {
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
        "algo": config["algo"],
    }

    # HTE: per-sample effects
    if compute_hte:
        try:
            hte = model.effect(X, T0=T0, T1=T1)
            hte_arr = np.array(hte).flatten()
            result["hte"] = hte_arr
            try:
                hte_lb, hte_ub = model.effect_interval(X, T0=T0, T1=T1)
                result["hte_ci_lower"] = np.array(hte_lb).flatten()
                result["hte_ci_upper"] = np.array(hte_ub).flatten()
            except Exception:
                pass
        except Exception:
            pass

    return result


def estimate_metalearner(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    X_col: list[str],
    T0: float,
    T1: float,
    learner: str = "t",
    *,
    compute_hte: bool = True,
) -> dict:
    """Estimate ATE/ATT/HTE via EconML Meta-Learners (S/T/X/DA).

    Binary treatment required. Auto-binarizes if needed.
    learner: "s" (SLearner), "t" (TLearner), "x" (XLearner),
             "da" (DomainAdaptationLearner).
    Uses BootstrapInference(n=100), matching the original pipeline.
    """
    from econml.inference import BootstrapInference
    from econml.metalearners import SLearner, TLearner, XLearner
    from sklearn.linear_model import LinearRegression, LogisticRegression

    df = data.copy()

    # Ensure binary treatment
    if df[treatment].nunique() > 2:
        threshold = df[treatment].median()
        df[treatment] = (df[treatment] > threshold).astype(int)

    Y = df[outcome].values
    T = df[treatment].values
    X = df[X_col].values if X_col else df.drop(columns=[treatment, outcome]).values

    if learner == "s":
        model = SLearner(overall_model=LinearRegression())
    elif learner == "t":
        model = TLearner(models=LinearRegression())
    elif learner == "x":
        try:
            from xgboost import XGBRegressor

            base_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            base_model = GradientBoostingRegressor(n_estimators=100)
        model = XLearner(
            models=base_model,
            propensity_model=LogisticRegression(max_iter=1000),
        )
    elif learner == "da":
        from econml.metalearners import DomainAdaptationLearner

        try:
            from xgboost import XGBRegressor

            base_model = XGBRegressor(objective="reg:squarederror", n_estimators=100)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor

            base_model = GradientBoostingRegressor(n_estimators=100)
        model = DomainAdaptationLearner(
            models=base_model,
            final_models=base_model,
            propensity_model=LogisticRegression(max_iter=1000),
        )
    else:
        raise ValueError(f"Unknown learner: '{learner}'. Use 's', 't', 'x', or 'da'.")

    model.fit(Y, T, X=X, inference=BootstrapInference(n_bootstrap_samples=100))

    ate = float(model.ate(X=X, T0=T0, T1=T1))
    try:
        ate_lower, ate_upper = model.ate_interval(X=X, T0=T0, T1=T1)
        ate_lower, ate_upper = float(ate_lower), float(ate_upper)
    except Exception:
        ate_lower = ate_upper = None

    treated_mask = np.isclose(T, T1)
    if treated_mask.sum() > 0:
        effects = model.effect(X[treated_mask], T0=T0, T1=T1)
        att = float(np.mean(effects))
        try:
            lb, ub = model.effect_interval(X[treated_mask], T0=T0, T1=T1)
            att_lower, att_upper = float(np.mean(lb)), float(np.mean(ub))
        except Exception:
            att_lower = att_upper = None
    else:
        att = att_lower = att_upper = None

    result = {
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
        "algo": f"{learner.upper()}Learner",
    }

    # HTE: per-sample effects
    if compute_hte:
        try:
            hte = model.effect(X, T0=T0, T1=T1)
            result["hte"] = np.array(hte).flatten()
            try:
                hte_lb, hte_ub = model.effect_interval(X, T0=T0, T1=T1)
                result["hte_ci_lower"] = np.array(hte_lb).flatten()
                result["hte_ci_upper"] = np.array(hte_ub).flatten()
            except Exception:
                pass
        except Exception:
            pass

    return result


def estimate_iv(
    data: pd.DataFrame,
    treatment: str,
    outcome: str,
    instrument: str,
    X_col: list[str],
    W_col: list[str],
    T0: float,
    T1: float,
) -> dict:
    """Estimate ATE/ATT via Instrumental Variables (EconML LinearDRIV).

    Requires a valid instrument variable that:
    1. Directly affects treatment
    2. Only affects outcome through treatment
    3. Is independent of confounders

    Returns dict with 'ate' and 'att' keys.
    """
    from econml.iv.dr import LinearDRIV

    df = data.copy()
    actual_W = list(W_col)
    if len(actual_W) == 0:
        df["_W_dummy"] = 0.0
        actual_W = ["_W_dummy"]

    Y = df[outcome].values
    T = df[treatment].values
    Z = df[[instrument]].values
    X = df[X_col].values if X_col else None
    W = df[actual_W].values

    model = LinearDRIV()
    if X is not None:
        model.fit(Y, T, X=X, Z=Z, W=W)
    else:
        model.fit(Y, T, Z=Z, W=W)

    X_for_pred = X if X is not None else None
    ate = float(model.ate(X=X_for_pred, T0=T0, T1=T1))
    try:
        ate_lower, ate_upper = model.ate_interval(X=X_for_pred, T0=T0, T1=T1)
        ate_lower, ate_upper = float(ate_lower), float(ate_upper)
    except Exception:
        ate_lower = ate_upper = None

    treated_mask = np.isclose(T, T1)
    if treated_mask.sum() > 0 and X is not None:
        effects = model.effect(X[treated_mask], T0=T0, T1=T1)
        att = float(np.mean(effects))
        try:
            lb, ub = model.effect_interval(X[treated_mask], T0=T0, T1=T1)
            att_lower, att_upper = float(np.mean(lb)), float(np.mean(ub))
        except Exception:
            att_lower = att_upper = None
    else:
        att = att_lower = att_upper = None

    result = {
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
        "algo": "LinearDRIV",
    }

    # HTE: per-sample effects
    if X is not None:
        try:
            hte = model.effect(X, T0=T0, T1=T1)
            result["hte"] = np.array(hte).flatten()
            try:
                hte_lb, hte_ub = model.effect_interval(X, T0=T0, T1=T1)
                result["hte_ci_lower"] = np.array(hte_lb).flatten()
                result["hte_ci_upper"] = np.array(hte_ub).flatten()
            except Exception:
                pass
        except Exception:
            pass

    return result


# ── Sensitivity / Refutation ─────────────────────────────────────────


def run_refutation(
    data: pd.DataFrame,
    dot_graph: str,
    treatment: str,
    outcome: str,
    control_value: float,
    treatment_value: float,
    *,
    confounders: list[str] | None = None,
    shap_top_feature: str | None = None,
) -> dict:
    """Run DoWhy refutation/sensitivity analysis.

    Estimates the causal effect via linear regression, then tests robustness
    with up to four methods:
    1. data_subset_refuter — stability under subsampling
    2. random_common_cause — robustness to random confounders
    3. placebo_treatment_refuter — effect disappears under permutation
    4. add_unobserved_common_cause — sensitivity to unobserved confounders
       (only when common causes exist, uses partial-R2 method from original pipeline)
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

    results = {
        "original_estimate": _safe_float(estimate.value),
        "refutations": {},
    }

    # Data subset refuter
    try:
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="data_subset_refuter",
            subset_fraction=0.8,
        )
        results["refutations"]["data_subset"] = {
            "new_effect": _safe_float(refute.new_effect),
            "refutation_result": str(refute),
        }
    except Exception as e:
        results["refutations"]["data_subset"] = {"error": str(e)}

    # Random common cause
    try:
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="random_common_cause",
        )
        results["refutations"]["random_common_cause"] = {
            "new_effect": _safe_float(refute.new_effect),
            "refutation_result": str(refute),
        }
    except Exception as e:
        results["refutations"]["random_common_cause"] = {"error": str(e)}

    # Placebo treatment
    try:
        refute = model.refute_estimate(
            estimand,
            estimate,
            method_name="placebo_treatment_refuter",
            placebo_type="permute",
        )
        results["refutations"]["placebo_treatment"] = {
            "new_effect": _safe_float(refute.new_effect),
            "refutation_result": str(refute),
        }
    except Exception as e:
        results["refutations"]["placebo_treatment"] = {"error": str(e)}

    # Unobserved common cause sensitivity (partial-R2, matches original)
    common_causes = model.get_common_causes()
    if common_causes:
        try:
            benchmark = [shap_top_feature] if shap_top_feature else common_causes[:1]
            refute = model.refute_estimate(
                estimand,
                estimate,
                method_name="add_unobserved_common_cause",
                simulation_method="non-parametric-partial-R2",
                benchmark_common_causes=benchmark,
                effect_fraction_on_outcome=[1, 2, 3],
            )
            results["refutations"]["unobserved_common_cause"] = {
                "refutation_result": str(refute),
            }
        except Exception as e:
            results["refutations"]["unobserved_common_cause"] = {"error": str(e)}

    return results


# ── DoWhy GCM Tools ──────────────────────────────────────────────────


def _build_gcm(data: pd.DataFrame, adj: np.ndarray, names: list[str]):
    """Build a fitted DoWhy GCM (InvertibleStructuralCausalModel).

    Args:
        data: DataFrame with columns matching names
        adj: adjacency matrix (adj[i,j]=1 means j->i)
        names: variable names

    Returns:
        (scm, G) — fitted model and NetworkX DiGraph
    """
    import networkx as nx
    from dowhy import gcm

    G = nx.DiGraph()
    G.add_nodes_from(names)
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                G.add_edge(names[j], names[i])

    # Ensure DAG — remove back-edges greedily if cycles exist
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = list(next(iter(nx.simple_cycles(G))))
            G.remove_edge(cycle[-1], cycle[0])
        except StopIteration:
            break

    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Cannot resolve cycles in graph for GCM model.")

    # Filter data to graph nodes
    graph_cols = [c for c in names if c in data.columns]
    df = data[graph_cols].copy()

    scm = gcm.InvertibleStructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm, df)
    gcm.fit(scm, df)

    return scm, G


def run_counterfactual(
    data: pd.DataFrame,
    adj: np.ndarray,
    names: list[str],
    treatment: str,
    outcome: str,
    intervention_value: float,
    observed_row_idx: int = -1,
) -> dict:
    """Estimate counterfactual: what would outcome be if treatment were set to value?

    Uses DoWhy GCM counterfactual_samples.
    observed_row_idx: row to counterfactualize (-1 = row with min treatment).
    """
    from dowhy import gcm

    scm, G = _build_gcm(data, adj, names)

    if observed_row_idx < 0:
        observed_row_idx = int(data[treatment].idxmin())

    observed = data.iloc[[observed_row_idx]].copy().reset_index(drop=True)

    cf_samples = gcm.counterfactual_samples(
        scm,
        {treatment: lambda x: intervention_value},
        observed_data=observed,
    )

    return {
        "observed": {
            treatment: _safe_float(observed[treatment].iloc[0]),
            outcome: _safe_float(observed[outcome].iloc[0]),
        },
        "counterfactual": {
            treatment: _safe_float(cf_samples[treatment].iloc[0]),
            outcome: _safe_float(cf_samples[outcome].iloc[0]),
        },
        "effect": _safe_float(cf_samples[outcome].iloc[0] - observed[outcome].iloc[0]),
        "observed_row_index": observed_row_idx,
    }


def run_anomaly_attribution(
    data: pd.DataFrame,
    adj: np.ndarray,
    names: list[str],
    target_node: str,
    threshold_percentile: float = 95.0,
    n_samples: int = 5,
) -> dict:
    """Identify root causes of anomalies via DoWhy GCM.

    Selects anomaly samples (values above threshold_percentile of target)
    and attributes the anomaly to parent nodes.
    """
    from dowhy import gcm

    scm, G = _build_gcm(data, adj, names)

    threshold = data[target_node].quantile(threshold_percentile / 100.0)
    anomaly_mask = data[target_node] >= threshold
    if anomaly_mask.sum() == 0:
        anomaly_samples = data.tail(n_samples)
    else:
        anomaly_samples = data[anomaly_mask].head(n_samples)

    attribution = gcm.attribute_anomalies(
        causal_model=scm,
        target_node=target_node,
        anomaly_samples=anomaly_samples,
    )

    results = {}
    for node, scores in attribution.items():
        scores_arr = np.array(scores).flatten()
        results[node] = {
            "mean_score": _safe_float(np.mean(scores_arr)),
            "ci_lower": _safe_float(np.percentile(scores_arr, 2.5)),
            "ci_upper": _safe_float(np.percentile(scores_arr, 97.5)),
        }

    # Sort by mean score descending
    results = dict(sorted(results.items(), key=lambda x: abs(x[1]["mean_score"] or 0), reverse=True))

    return {
        "target_node": target_node,
        "n_anomaly_samples": len(anomaly_samples),
        "threshold_percentile": threshold_percentile,
        "attributions": results,
    }


def run_distribution_change(
    data_old: pd.DataFrame,
    data_new: pd.DataFrame,
    adj: np.ndarray,
    names: list[str],
    target_node: str,
) -> dict:
    """Explain distribution shift via DoWhy GCM distribution_change.

    Identifies which causal mechanisms changed between old and new data.
    """
    from dowhy import gcm

    scm, G = _build_gcm(data_old, adj, names)

    attribution = gcm.distribution_change(
        causal_model=scm,
        old_data=data_old[names],
        new_data=data_new[names],
        target_node=target_node,
    )

    results = {}
    for node, score in attribution.items():
        results[node] = _safe_float(score)

    results = dict(sorted(results.items(), key=lambda x: abs(x[1] or 0), reverse=True))

    return {
        "target_node": target_node,
        "n_old": len(data_old),
        "n_new": len(data_new),
        "attributions": results,
    }


def compute_feature_importance(
    data: pd.DataFrame,
    target_node: str,
    is_linear: bool = True,
) -> dict:
    """Compute SHAP-based feature importance for a target variable.

    Uses linear model SHAP for linear data, tree SHAP for nonlinear.
    Returns dict mapping feature names to mean absolute SHAP values.
    """
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    X = data.drop(columns=[target_node])
    y = data[[target_node]]

    if is_linear:
        model = LinearRegression()
        model.fit(X, y)
        background = shap.utils.sample(X, min(int(len(X) * 0.2), 100))
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(X)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y.values.ravel())
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

    shap_df = pd.DataFrame(np.abs(shap_values.values), columns=X.columns)
    mean_shap = shap_df.mean().sort_values(ascending=False)

    return {
        "target_node": target_node,
        "method": "linear_shap" if is_linear else "tree_shap",
        "feature_importance": {col: _safe_float(val) for col, val in mean_shap.items()},
        "top_features": list(mean_shap.head(10).index),
    }


def run_graph_falsification(
    data: pd.DataFrame,
    adj: np.ndarray,
    names: list[str],
    n_permutations: int = 20,
) -> dict:
    """Test if a causal graph is consistent with data via DoWhy GCM falsification.

    Checks Local Markov Condition (LMC) violations. Returns test summary.
    """
    import networkx as nx
    from dowhy.gcm.falsify import falsify_graph

    G = nx.DiGraph()
    G.add_nodes_from(names)
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                G.add_edge(names[j], names[i])

    # Ensure DAG
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = list(next(iter(nx.simple_cycles(G))))
            G.remove_edge(cycle[-1], cycle[0])
        except StopIteration:
            break

    df = data[[c for c in names if c in data.columns]].copy()

    result = falsify_graph(
        G,
        df,
        n_permutations=n_permutations,
        plot_histogram=False,
        suggestions=True,
    )

    # Parse result string for structured output
    import re

    result_str = str(result)

    # Extract key metrics from the result string

    # Look for p-value and violation info
    p_value = None
    p_match = re.search(r"p_value\s*=?\s*([\d.]+)", result_str)
    if p_match:
        p_value = float(p_match.group(1))

    return {
        "falsification_result": result_str,
        "p_value": _safe_float(p_value) if p_value else None,
        "n_permutations": n_permutations,
        "n_nodes": len(G.nodes),
        "n_edges": len(G.edges),
    }


def run_intervention_simulation(
    data: pd.DataFrame,
    adj: np.ndarray,
    names: list[str],
    treatment: str,
    outcome: str,
    value: float,
    shift: bool = True,
    n_samples: int = 1000,
) -> dict:
    """Simulate an intervention via DoWhy GCM interventional_samples.

    shift=True: shift treatment by value (treatment += value).
    shift=False: set treatment to value (atomic intervention).
    """
    from dowhy import gcm

    scm, G = _build_gcm(data, adj, names)

    if shift:
        intervention_fn = lambda x: x + value  # noqa: E731
    else:
        intervention_fn = lambda x: value  # noqa: E731

    samples = gcm.interventional_samples(
        scm,
        {treatment: intervention_fn},
        num_samples_to_draw=n_samples,
    )

    return {
        "treatment": treatment,
        "outcome": outcome,
        "intervention_type": "shift" if shift else "atomic",
        "intervention_value": value,
        "original_distribution": {
            "mean": _safe_float(data[outcome].mean()),
            "std": _safe_float(data[outcome].std()),
            "median": _safe_float(data[outcome].median()),
            "n": len(data),
        },
        "intervention_distribution": {
            "mean": _safe_float(samples[outcome].mean()),
            "std": _safe_float(samples[outcome].std()),
            "median": _safe_float(samples[outcome].median()),
            "n": n_samples,
        },
        "mean_change": _safe_float(samples[outcome].mean() - data[outcome].mean()),
    }
