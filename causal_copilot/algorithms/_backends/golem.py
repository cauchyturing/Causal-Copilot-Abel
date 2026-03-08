"""GOLEM algorithm backend — imports gcastle directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class GOLEMBackend(Backend):
    """GOLEM (A larger class of identifiable DAG models) using gcastle."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from castle.algorithms import GOLEM as CastleGOLEM

        # Remove domain_index if present
        if isinstance(data, pd.DataFrame) and "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data_values = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data_values = np.array(data)

        # Auto-detect device
        device_type = self._params.get("device_type", "auto")
        if device_type == "auto":
            try:
                import torch

                device_type = "gpu" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device_type = "cpu"

        model = CastleGOLEM(
            lambda_1=self._params.get("lambda_1", 0.01),
            lambda_2=self._params.get("lambda_2", 5.0),
            equal_variances=self._params.get("equal_variances", True),
            learning_rate=self._params.get("learning_rate", 1e-3),
            num_iter=self._params.get("num_iter", 10000),
            checkpoint_iter=self._params.get("checkpoint_iter", 5000),
            seed=self._params.get("seed", 1),
            graph_thres=self._params.get("graph_thres", 0.3),
            device_type=device_type,
            device_ids=self._params.get("device_ids", 0),
        )

        model.learn(data_values)

        # gcastle: causal_matrix[i,j] != 0 => i -> j
        # Our convention: mat[i,j] = 1 => j -> i  =>  transpose
        adj_matrix = model.causal_matrix.T

        info = {
            "node_names": node_names,
        }

        return adj_matrix, info, model
