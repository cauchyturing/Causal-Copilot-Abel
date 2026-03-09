"""Smoke tests that verify the package works after pip install.

These simulate what a user experiences after: pip install causal-copilot[agent]
They do NOT require LLM access — they verify structure, imports, and offline functionality.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


class TestWheelSmoke:
    def test_import_core(self):
        from causal_copilot import CausalCopilot, __version__

        assert callable(CausalCopilot)
        assert __version__

    def test_import_agent(self):
        try:
            from causal_copilot.agent import _CONTEXT_DIR, AgentCopilot

            assert callable(AgentCopilot)
            assert _CONTEXT_DIR.is_dir()
        except ImportError:
            pytest.skip("openai not installed")

    def test_context_files_accessible(self):
        try:
            from causal_copilot.agent import _CONTEXT_DIR
        except ImportError:
            pytest.skip("agent not installed")

        from causal_copilot.algorithms.registry import REGISTRY

        for name in REGISTRY:
            assert (_CONTEXT_DIR / "algos" / f"{name}.txt").exists(), f"Missing profile: {name}.txt"
            assert (_CONTEXT_DIR / "hyperparameters" / f"{name}.json").exists(), f"Missing HP spec: {name}.json"

    def test_cli_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "causal_copilot.cli", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "causal-copilot" in result.stdout

    def test_cli_doctor(self):
        result = subprocess.run(
            [sys.executable, "-m", "causal_copilot.cli", "doctor"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "numpy" in result.stdout

    def test_core_analyze_offline(self):
        from causal_copilot import CausalCopilot

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"X": rng.normal(size=100), "Y": rng.normal(size=100)})
        result = CausalCopilot().analyze(df, seed=42)
        assert result.status == "ok"
        assert result.adjacency_matrix is not None
