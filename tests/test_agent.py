"""Tests for the agent pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Check if openai is available (needed for AgentLLM tests)
try:
    import openai as _openai

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

needs_openai = pytest.mark.skipif(not _HAS_OPENAI, reason="openai not installed")


# ---------------------------------------------------------------------------
# Task 12: Context files
# ---------------------------------------------------------------------------


class TestAgentContextFiles:
    def test_context_dir_exists(self):
        from causal_copilot.agent import _CONTEXT_DIR

        assert _CONTEXT_DIR.is_dir(), f"Context dir missing: {_CONTEXT_DIR}"

    def test_has_algorithm_profiles(self):
        from causal_copilot.agent import _CONTEXT_DIR

        algos_dir = _CONTEXT_DIR / "algos"
        txt_files = list(algos_dir.glob("*.txt"))
        assert len(txt_files) >= 7, f"Expected >=7 algo profiles, got {len(txt_files)}"

    def test_has_hyperparameter_specs(self):
        from causal_copilot.agent import _CONTEXT_DIR

        hp_dir = _CONTEXT_DIR / "hyperparameters"
        json_files = list(hp_dir.glob("*.json"))
        assert len(json_files) >= 7, f"Expected >=7 HP specs, got {len(json_files)}"

    def test_has_prompt_templates(self):
        from causal_copilot.agent import _CONTEXT_DIR

        for name in [
            "algo_select_prompt.txt",
            "algo_rerank_prompt.txt",
            "hyperparameter_select_prompt.txt",
        ]:
            assert (_CONTEXT_DIR / name).exists(), f"Missing prompt: {name}"

    def test_registered_algos_have_context(self):
        """Every algorithm in REGISTRY must have a profile + HP spec."""
        from causal_copilot.agent import _CONTEXT_DIR
        from causal_copilot.algorithms.registry import REGISTRY

        for name in REGISTRY:
            profile = _CONTEXT_DIR / "algos" / f"{name}.txt"
            hp = _CONTEXT_DIR / "hyperparameters" / f"{name}.json"
            assert profile.exists(), f"Missing profile for {name}"
            assert hp.exists(), f"Missing HP spec for {name}"


# ---------------------------------------------------------------------------
# Task 13: AgentLLM
# ---------------------------------------------------------------------------


@needs_openai
class TestAgentLLM:
    def test_import(self):
        from causal_copilot.agent.llm import AgentLLM

        assert callable(AgentLLM)

    def test_providers_dict(self):
        from causal_copilot.agent.llm import PROVIDERS

        assert "openai" in PROVIDERS
        assert "openrouter" in PROVIDERS
        assert "ollama" in PROVIDERS
        assert "lmstudio" in PROVIDERS
        for name, preset in PROVIDERS.items():
            assert "base_url" in preset
            assert "default_model" in preset

    def test_init_with_provider(self):
        from causal_copilot.agent.llm import AgentLLM

        llm = AgentLLM(provider="ollama")
        assert llm.model == "llama3.2"
        assert "11434" in llm._base_url

    def test_init_custom_base_url(self):
        from causal_copilot.agent.llm import AgentLLM

        llm = AgentLLM(provider="openai", base_url="http://custom:8080/v1")
        assert llm._base_url == "http://custom:8080/v1"

    def test_init_custom_model(self):
        from causal_copilot.agent.llm import AgentLLM

        llm = AgentLLM(provider="openai", model="gpt-4o")
        assert llm.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Mock LLM for selector/pipeline tests
# ---------------------------------------------------------------------------


def _mock_llm_complete(prompt, system="", json_mode=False):
    """Mock LLM that returns PC for selection, alpha for HP tuning."""
    if json_mode:
        # HP tuning prompts start with "You are tuning hyperparameters"
        if prompt.startswith("You are tuning hyperparameters"):
            return {"alpha": 0.05}
        return {"algorithm": "PC", "reasoning": "Mock selection"}
    return "Mock interpretation of causal results."


# ---------------------------------------------------------------------------
# Task 14: Algorithm selector
# ---------------------------------------------------------------------------


@needs_openai
class TestAlgorithmSelector:
    def test_import(self):
        from causal_copilot.agent.selector import select_algorithm

        assert callable(select_algorithm)

    def test_build_catalog(self):
        from causal_copilot.agent.selector import _build_algorithm_catalog

        catalog = _build_algorithm_catalog()
        assert len(catalog) >= 5, f"Expected >=5 algos in catalog, got {len(catalog)}"
        assert "PC" in catalog
        assert "profile" in catalog["PC"]
        assert "hyperparameters" in catalog["PC"]

    def test_select_algorithm_returns_registered(self):
        from causal_copilot.agent.llm import AgentLLM
        from causal_copilot.agent.selector import select_algorithm
        from causal_copilot.algorithms.registry import REGISTRY

        mock_llm = MagicMock()
        mock_llm.complete = _mock_llm_complete

        props = {
            "n_samples": 500,
            "n_features": 5,
            "likely_linear": True,
            "likely_gaussian": True,
            "is_time_series": False,
            "has_missing": False,
        }
        decision = select_algorithm(mock_llm, props, query="What causes Y?")
        assert decision.algorithm in REGISTRY
        assert decision.source == "llm"
        assert decision.reasoning  # non-empty

    def test_select_algorithm_fallback_on_bad_response(self):
        """LLM returning non-REGISTRY algo triggers rule-based fallback."""
        from causal_copilot.agent.llm import AgentLLM
        from causal_copilot.agent.selector import select_algorithm
        from causal_copilot.algorithms.registry import REGISTRY

        def bad_complete(prompt, system="", json_mode=False):
            return {"algorithm": "FakeAlgo", "reasoning": "bad"}

        mock_llm = MagicMock()
        mock_llm.complete = bad_complete

        props = {
            "n_samples": 500,
            "n_features": 5,
            "likely_linear": True,
            "likely_gaussian": True,
            "is_time_series": False,
            "has_missing": False,
        }
        decision = select_algorithm(mock_llm, props)
        assert decision.algorithm in REGISTRY
        assert decision.source == "rule_fallback"

    def test_select_algorithm_fallback_on_exception(self):
        """LLM exception triggers rule-based fallback."""
        from causal_copilot.agent.llm import AgentLLM
        from causal_copilot.agent.selector import select_algorithm
        from causal_copilot.algorithms.registry import REGISTRY

        def raising_complete(prompt, system="", json_mode=False):
            raise ConnectionError("API down")

        mock_llm = MagicMock()
        mock_llm.complete = raising_complete

        props = {
            "n_samples": 500,
            "n_features": 5,
            "likely_linear": True,
            "likely_gaussian": True,
            "is_time_series": False,
            "has_missing": False,
        }
        decision = select_algorithm(mock_llm, props)
        assert decision.algorithm in REGISTRY
        assert decision.source == "rule_fallback"

    def test_agent_decision_dataclass(self):
        from causal_copilot.agent.selector import AgentDecision

        d = AgentDecision(algorithm="PC", hyperparams={}, reasoning="test", source="llm")
        assert d.algorithm == "PC"
        assert d.source == "llm"


@needs_openai
class TestHyperparameterTuner:
    def test_import(self):
        from causal_copilot.agent.selector import tune_hyperparameters

        assert callable(tune_hyperparameters)

    def test_tune_returns_dict(self):
        from causal_copilot.agent.llm import AgentLLM
        from causal_copilot.agent.selector import tune_hyperparameters

        mock_llm = MagicMock()
        mock_llm.complete = _mock_llm_complete

        props = {
            "n_samples": 500,
            "n_features": 5,
            "likely_linear": True,
            "likely_gaussian": True,
            "is_time_series": False,
        }
        hp = tune_hyperparameters(mock_llm, "PC", props)
        assert isinstance(hp, dict)

    def test_tune_returns_empty_on_exception(self):
        from causal_copilot.agent.llm import AgentLLM
        from causal_copilot.agent.selector import tune_hyperparameters

        def raising_complete(prompt, system="", json_mode=False):
            raise ConnectionError("API down")

        mock_llm = MagicMock()
        mock_llm.complete = raising_complete

        props = {
            "n_samples": 500,
            "n_features": 5,
            "likely_linear": True,
            "likely_gaussian": True,
            "is_time_series": False,
        }
        hp = tune_hyperparameters(mock_llm, "PC", props)
        assert hp == {}


# ---------------------------------------------------------------------------
# Task 15: AgentCopilot orchestrator
# ---------------------------------------------------------------------------


@needs_openai
class TestAgentCopilot:
    def test_import(self):
        from causal_copilot.agent import AgentCopilot

        assert callable(AgentCopilot)

    def test_analyze_with_mock_llm(self):
        """AgentCopilot.analyze() should complete with mocked LLM + mocked algo."""
        from causal_copilot.agent.pipeline import AgentCopilot
        from causal_copilot.agent.selector import AgentDecision

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        y = 0.8 * x + rng.normal(size=n) * 0.3
        df = pd.DataFrame({"X": x, "Y": y})

        mock_decision = AgentDecision(
            algorithm="PC", hyperparams={}, reasoning="Mock", source="llm"
        )

        with (
            patch(
                "causal_copilot.agent.selector.select_algorithm",
                return_value=mock_decision,
            ),
            patch(
                "causal_copilot.agent.selector.tune_hyperparameters",
                return_value={},
            ),
            patch.object(
                AgentCopilot, "_interpret_result", return_value=None,
            ),
        ):
            agent = AgentCopilot.__new__(AgentCopilot)
            agent.llm = MagicMock()
            agent.llm.model = "mock-model"
            agent._copilot = __import__("causal_copilot").CausalCopilot()
            result = agent.analyze(df, query="What causes Y?")

        assert result.status == "ok"
        assert result.adjacency_matrix is not None
        assert result.provenance is not None
        assert result.provenance.planner == "llm"
        assert result.provenance.planner_model == "mock-model"

    def test_analyze_sets_reasoning(self):
        """Algorithm selection reasoning should be stored in result."""
        from causal_copilot.agent.pipeline import AgentCopilot
        from causal_copilot.agent.selector import AgentDecision

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"A": rng.normal(size=100), "B": rng.normal(size=100)})

        mock_decision = AgentDecision(
            algorithm="PC",
            hyperparams={},
            reasoning="PC is best for linear Gaussian data",
            source="llm",
        )
        with (
            patch(
                "causal_copilot.agent.selector.select_algorithm",
                return_value=mock_decision,
            ),
            patch(
                "causal_copilot.agent.selector.tune_hyperparameters",
                return_value={},
            ),
            patch.object(
                AgentCopilot, "_interpret_result", return_value=None,
            ),
        ):
            agent = AgentCopilot.__new__(AgentCopilot)
            agent.llm = MagicMock()
            agent.llm.model = "mock-model"
            agent._copilot = __import__("causal_copilot").CausalCopilot()
            result = agent.analyze(df)

        assert "PC is best" in result.algorithm_selection_reason
