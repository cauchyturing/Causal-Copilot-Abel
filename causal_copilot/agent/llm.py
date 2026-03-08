"""Minimal LLM client — OpenAI-compatible API for all providers.

Uses the OpenAI Python SDK which speaks to any OpenAI-compatible endpoint:
openai.com, OpenRouter, Ollama (/v1), LM Studio (/v1).

One class, one file, no Pydantic dependency.
"""

from __future__ import annotations

import json
import os
import re

PROVIDERS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "key_env": "OPENROUTER_API_KEY",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "key_env": None,
    },
    "lmstudio": {
        "base_url": "http://localhost:1234/v1",
        "default_model": "default",
        "key_env": None,
    },
}

# Providers that support response_format={"type":"json_object"}
_JSON_MODE_PROVIDERS = {"openai", "openrouter", "ollama"}


class AgentLLM:
    """Thin OpenAI-compatible LLM client with provider presets.

    All providers use the same OpenAI SDK — Ollama and LM Studio expose
    an OpenAI-compatible /v1 endpoint, so no special client needed.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        preset = PROVIDERS.get(provider, PROVIDERS["openai"])
        self.provider = provider
        self.model = model or os.getenv("LLM_MODEL") or preset["default_model"]
        self._base_url = base_url or os.getenv("LLM_BASE_URL") or preset["base_url"]

        key_env = preset.get("key_env")
        resolved_key = api_key or (os.getenv(key_env) if key_env else None) or "not-needed"

        # Lazy import — openai is an optional dep ([agent] extra)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "Agent mode requires openai. Install with: pip install causal-copilot[agent]"
            )
        self._client = OpenAI(api_key=resolved_key, base_url=self._base_url)

    def complete(
        self,
        prompt: str,
        system: str = "You are a causal inference expert.",
        json_mode: bool = False,
    ) -> str | dict:
        """Single-turn completion. Returns str or parsed dict (if json_mode).

        JSON mode is provider-aware. LM Studio doesn't support
        response_format, so we fall back to prompt-based JSON extraction.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
        }

        use_json_format = json_mode and self.provider in _JSON_MODE_PROVIDERS
        if use_json_format:
            kwargs["response_format"] = {"type": "json_object"}
            if "json" not in system.lower() and "json" not in prompt.lower():
                messages[0]["content"] += " Respond in JSON format."
        elif json_mode:
            # Provider doesn't support response_format — ask via prompt
            messages[0]["content"] += " You MUST respond with valid JSON only."

        response = self._client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content

        if json_mode:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                if match:
                    return json.loads(match.group(1))
                raise
        return content
