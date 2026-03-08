"""LLM-driven algorithm selection and hyperparameter tuning.

Reads algorithm profiles and HP specs from packaged context files.
Only selects from algorithms registered in the clean core REGISTRY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from causal_copilot.agent import _CONTEXT_DIR
from causal_copilot.agent.llm import AgentLLM
from causal_copilot.algorithms.registry import REGISTRY


def _load_text(rel_path: str) -> str:
    return (_CONTEXT_DIR / rel_path).read_text(encoding="utf-8")


def _load_json(rel_path: str) -> dict:
    return json.loads((_CONTEXT_DIR / rel_path).read_text(encoding="utf-8"))


def _build_algorithm_catalog() -> dict[str, dict]:
    """Build catalog of registered algorithms with context profiles and HP specs.

    Only includes algorithms that are BOTH in the REGISTRY and have context files.
    """
    catalog: dict[str, dict] = {}
    for name in REGISTRY:
        profile_path = _CONTEXT_DIR / "algos" / f"{name}.txt"
        hp_path = _CONTEXT_DIR / "hyperparameters" / f"{name}.json"
        if profile_path.exists():
            catalog[name] = {
                "profile": profile_path.read_text(encoding="utf-8"),
                "hyperparameters": (
                    json.loads(hp_path.read_text(encoding="utf-8"))
                    if hp_path.exists()
                    else {}
                ),
            }
    return catalog


@dataclass
class AgentDecision:
    """Output of LLM-driven algorithm selection."""

    algorithm: str
    hyperparams: dict
    reasoning: str
    source: str  # "llm" or "rule_fallback"


def select_algorithm(
    llm: AgentLLM,
    data_properties: dict,
    query: str = "",
) -> AgentDecision:
    """Use LLM to select the best algorithm for the given data.

    Wraps LLM call in try/except — falls back to rule_based_select()
    on ANY failure (auth, network, JSON parse, unknown algo).
    Returns AgentDecision with reasoning (algorithm guaranteed in REGISTRY).
    """
    from causal_copilot.core.planner import rule_based_select

    catalog = _build_algorithm_catalog()
    available = sorted(catalog.keys())

    # Build algorithm summaries for the prompt
    algo_summaries = ""
    for name in available:
        profile = catalog[name]["profile"][:600]
        algo_summaries += f"\n### {name}\n{profile}\n"

    prompt = f"""You are selecting a causal discovery algorithm.

## Data Properties
- Samples: {data_properties['n_samples']}
- Features: {data_properties['n_features']}
- Likely linear: {data_properties['likely_linear']}
- Likely Gaussian: {data_properties['likely_gaussian']}
- Time series: {data_properties['is_time_series']}

## User Query
{query or 'Discover causal relationships in this dataset.'}

## Available Algorithms
{algo_summaries}

Select the SINGLE best algorithm for this data. You MUST choose from: {available}

Respond in JSON: {{"algorithm": "<name>", "reasoning": "<1-2 sentences>"}}"""

    try:
        result = llm.complete(prompt, json_mode=True)
        selected = result.get("algorithm", "")
        reasoning = result.get("reasoning", "")

        if selected in REGISTRY:
            return AgentDecision(
                algorithm=selected,
                hyperparams={},
                reasoning=reasoning,
                source="llm",
            )
    except Exception:
        pass  # Fall through to rule-based fallback

    # Fallback: rule-based selection (always works, no LLM needed)
    decision = rule_based_select(data_properties)
    return AgentDecision(
        algorithm=decision.algorithm,
        hyperparams=decision.hyperparams,
        reasoning=f"[LLM fallback] {decision.reason}",
        source="rule_fallback",
    )


def tune_hyperparameters(
    llm: AgentLLM,
    algorithm: str,
    data_properties: dict,
) -> dict:
    """Use LLM to tune hyperparameters for the selected algorithm.

    Returns dict of hyperparameter name -> value. Returns {} on any failure.
    """
    catalog = _build_algorithm_catalog()
    info = catalog.get(algorithm)
    if info is None:
        return {}

    hp_spec = info["hyperparameters"]
    if not hp_spec:
        return {}

    # Convert HP spec to readable format
    hp_description = ""
    for param, details in hp_spec.items():
        if param == "algorithm_name":
            continue
        if isinstance(details, dict) and "meaning" in details:
            hp_description += f"\n**{param}**: {details['meaning']}\n"
            if "available_values" in details:
                hp_description += f"  Values: {details['available_values']}\n"
            if "expert_suggestion" in details:
                hp_description += f"  Expert suggestion: {details['expert_suggestion']}\n"

    prompt = f"""You are tuning hyperparameters for the {algorithm} causal discovery algorithm.

## Data Properties
- Samples: {data_properties['n_samples']}
- Features: {data_properties['n_features']}
- Likely linear: {data_properties['likely_linear']}
- Likely Gaussian: {data_properties['likely_gaussian']}

## Available Hyperparameters
{hp_description}

Select optimal values. Respond in JSON with parameter names as keys and values.
Only include parameters you want to change from defaults."""

    try:
        result = llm.complete(prompt, json_mode=True)
    except Exception:
        return {}  # LLM failure -> use defaults

    # Filter to only valid parameter names
    valid_params = {k: v for k, v in result.items() if k in hp_spec and k != "algorithm_name"}
    return valid_params
