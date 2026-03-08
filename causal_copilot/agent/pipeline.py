"""AgentCopilot — LLM-driven autonomous causal analysis.

Orchestrates the full agent pipeline:
1. Detect data properties (reuses core/planner.py)
2. LLM selects algorithm (from registered algorithms with context profiles)
3. LLM tunes hyperparameters (from HP specs)
4. Execute algorithm (delegates to CausalCopilot — subprocess isolated)
5. LLM interprets result (optional natural language summary)

The agent ADDS intelligence on top of the core — it never bypasses it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from causal_copilot import CausalCopilot
from causal_copilot.core.planner import detect_data_properties
from causal_copilot.core.result import CausalResult


class AgentCopilot:
    """LLM-driven causal analysis agent.

    Usage:
        agent = AgentCopilot(provider="openai")
        result = agent.analyze(df, query="What causes Y?")

    Provider presets: "openai", "openrouter", "ollama", "lmstudio"
    Or pass base_url for any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        from causal_copilot.agent.llm import AgentLLM

        self.llm = AgentLLM(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self._copilot = CausalCopilot(planner="rule")

    def analyze(
        self,
        data: str | Path | pd.DataFrame,
        *,
        query: str = "",
        timeout: int = 300,
        seed: int = 42,
    ) -> CausalResult:
        """Run LLM-driven causal analysis.

        Args:
            data: CSV file path or DataFrame.
            query: User's causal question (helps LLM select algorithm).
            timeout: Max seconds for algorithm execution.
            seed: Random seed for reproducibility.

        Returns:
            CausalResult with graph, provenance (planner="llm"), and summary.
        """
        from causal_copilot.agent.selector import select_algorithm, tune_hyperparameters

        # Load data if path
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data

        # Detect data properties (reuse core logic)
        numeric_df = df.select_dtypes(include=[np.number])
        props = detect_data_properties(numeric_df)

        # Step 1: LLM selects algorithm (returns AgentDecision, handles errors)
        agent_decision = select_algorithm(self.llm, props, query=query)

        # Step 2: LLM tunes hyperparameters (merges into agent_decision.hyperparams)
        tuned_hp = tune_hyperparameters(self.llm, agent_decision.algorithm, props)
        effective_hp = {**agent_decision.hyperparams, **tuned_hp}

        # Step 3: Execute via core — pass algorithm_params so HPs reach execution
        result = self._copilot.analyze(
            data=df,
            algorithm=agent_decision.algorithm,
            algorithm_params=effective_hp,
            timeout=timeout,
            seed=seed,
        )

        # Thread LLM reasoning into provenance + algorithm_selection_reason
        if result.provenance is not None:
            from dataclasses import replace

            result.provenance = replace(
                result.provenance,
                planner="llm" if agent_decision.source == "llm" else "rule",
                planner_model=self.llm.model if agent_decision.source == "llm" else None,
            )
        result.algorithm_selection_reason = agent_decision.reasoning

        # Step 4: LLM interprets result (stored in discovery_metadata, not summary)
        self._interpret_result(result, query)

        return result

    def _interpret_result(self, result: CausalResult, query: str) -> None:
        """Add LLM-generated interpretation to result (stored in discovery_metadata)."""
        if result.status != "ok" or result.adjacency_matrix is None:
            return

        # Build edge list for LLM
        edges = []
        names = result.node_names or []
        adj = result.adjacency_matrix
        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i, j] == 1 and i < len(names) and j < len(names):
                    edges.append(f"{names[j]} -> {names[i]}")
                elif adj[i, j] == 2 and i < len(names) and j < len(names):
                    edges.append(f"{names[j]} -- {names[i]}")

        edge_text = "\n".join(edges) if edges else "No edges discovered."

        prompt = f"""A causal discovery algorithm found these relationships:
{edge_text}

User question: {query or 'What are the causal relationships?'}

Provide a brief (2-3 sentence) scientific interpretation."""

        try:
            interpretation = self.llm.complete(prompt)
            # Store separately, don't mutate core summary
            result.discovery_metadata["agent_interpretation"] = interpretation
        except Exception:
            pass  # Keep result clean if LLM fails
