"""CALM algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class CALMBackend(Backend):
    """CALM (Causal discovery with Acyclicity and Latent Masking)
    using causal-learn. Supports GPU via PyTorch."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causallearn.search.ScoreBased.CALM import calm

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        X = data.values

        # Auto-detect device
        device = self._params.get("device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        calm_params = {
            "lambda1": self._params.get("lambda1", 0.005),
            "alpha": self._params.get("alpha", 0.01),
            "tau": self._params.get("tau", 0.5),
            "rho_init": self._params.get("rho_init", 1e-5),
            "rho_mult": self._params.get("rho_mult", 3),
            "htol": self._params.get("htol", 1e-8),
            "subproblem_iter": self._params.get("subproblem_iter", 40000),
            "standardize": self._params.get("standardize", True),
            "device": device,
        }

        record = calm(X, **calm_params)

        # Extract DAG from GeneralGraph
        G = record["G"]
        d = len(G.get_nodes())
        adj_matrix = np.zeros((d, d))
        for i in range(d):
            for j in range(d):
                if G.graph[i, j] == 1 and G.graph[j, i] == -1:
                    # Directed edge j -> i
                    adj_matrix[i, j] = 1

        info = {
            "B_weighted": record["B_weighted"],
        }

        return adj_matrix, info, G
