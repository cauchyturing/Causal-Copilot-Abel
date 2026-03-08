# Agent Surface (Experimental)

This directory contains the LLM-driven autonomous pipeline — the "agent" surface
of Causal-Copilot. It is **not** included in the pip-installable package.

## Relationship to core

- `agent/` imports FROM `causal_copilot` (the stable core)
- `causal_copilot/` NEVER imports from `agent/`
- The core works offline, deterministically, without any LLM
- The agent adds LLM-driven algorithm selection, hyperparameter tuning, and postprocessing

## Status

Experimental. Used for paper evaluation (arxiv:2504.13263).
The legacy `causal_discovery/` tree is used by the agent for its 39-algorithm wrappers.
