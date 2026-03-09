"""XGES algorithm backend — imports xges directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class XGESBackend(Backend):
    """XGES (fast GES variant) using the xges pip package."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        import xges

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        data_values = data.values

        model = xges.XGES(alpha=self._params.get("alpha", 2.0))
        result = model.fit(data_values)

        # result.to_adjacency_matrix() returns PDAG encoding:
        # adj[i,j]=1 and adj[j,i]=0 => i -> j   (directed)
        # adj[i,j]=1 and adj[j,i]=1 => i -- j   (undirected)
        adj_pdag = result.to_adjacency_matrix()

        adj_matrix = np.zeros_like(adj_pdag, dtype=int)
        indices = np.where(adj_pdag == 1)
        for i, j in zip(indices[0], indices[1], strict=False):
            if adj_pdag[j, i] == 1:
                # Undirected: store once (lower triangle)
                if i > j:
                    adj_matrix[i, j] = 2
            else:
                # Directed i -> j => our convention: mat[j, i] = 1
                adj_matrix[j, i] = 1

        info: dict[str, Any] = {}

        return adj_matrix, info, result
