"""Causal-Copilot Agent — LLM-driven autonomous causal analysis.

The agent surface wraps the core engine with LLM intelligence:
- Algorithm selection via LLM (from curated context profiles)
- Hyperparameter tuning via LLM
- Natural language result interpretation

Usage:
    from causal_copilot.agent import AgentCopilot

    agent = AgentCopilot(provider="openai")
    result = agent.analyze(df, query="What causes Y?")

Requires: pip install causal-copilot[agent]
"""

from pathlib import Path

_CONTEXT_DIR = Path(__file__).parent / "context"

from causal_copilot.agent.pipeline import AgentCopilot  # noqa: E402

__all__ = ["AgentCopilot", "_CONTEXT_DIR"]
