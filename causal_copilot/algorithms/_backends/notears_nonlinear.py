"""NOTEARS Nonlinear (MLP/Sobolev) backend — imports gcastle directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class NOTEARSNonlinearBackend(Backend):
    """NOTEARS-Nonlinear using gcastle's NotearsNonlinear."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from castle.algorithms import NotearsNonlinear

        # Remove domain_index if present
        if isinstance(data, pd.DataFrame) and "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        if isinstance(data, pd.DataFrame):
            data_values = data.values
        else:
            data_values = np.array(data)

        # Auto-detect device
        device_type = self._params.get("device_type", "auto")
        if device_type == "auto":
            try:
                import torch

                device_type = "gpu" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device_type = "cpu"

        model = NotearsNonlinear(
            lambda1=self._params.get("lambda1", 0.01),
            lambda2=self._params.get("lambda2", 0.01),
            max_iter=self._params.get("max_iter", 100),
            h_tol=self._params.get("h_tol", 1e-8),
            rho_max=self._params.get("rho_max", 1e16),
            w_threshold=self._params.get("w_threshold", 0.3),
            hidden_layers=self._params.get("hidden_layers", (10, 1)),
            bias=self._params.get("bias", True),
            model_type=self._params.get("model_type", "mlp"),
            device_type=device_type,
            device_ids=self._params.get("device_ids", 0),
            expansions=self._params.get("expansions", 10),
        )

        model.learn(data_values)

        # gcastle: causal_matrix[i,j] != 0 => i -> j  =>  transpose
        adj_matrix = model.causal_matrix.T

        info = {
            "weight_causal_matrix": model.weight_causal_matrix,
        }

        return adj_matrix, info, model
