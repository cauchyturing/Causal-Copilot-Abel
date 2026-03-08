"""Parallel PC algorithm backend — imports gcastle directly."""

import os
from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_castle_cpdag


class PCParallelBackend(Backend):
    """Parallel PC algorithm using gcastle's PC (variant='parallel')."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from castle.algorithms import PC as CastlePC

        # Remove domain_index if present
        if isinstance(data, pd.DataFrame) and "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        if isinstance(data, pd.DataFrame):
            node_names = list(data.columns)
            data_values = data.values
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]
            data_values = np.array(data)

        model = CastlePC(
            variant="parallel",
            alpha=self._params.get("alpha", 0.05),
            ci_test=self._params.get("indep_test", "fisherz"),
            priori_knowledge=self._params.get("background_knowledge", None),
        )

        cores = self._params.get("cores", 8)
        cores = min(cores, 16, os.cpu_count() or 8)

        model.learn(
            data_values,
            columns=node_names,
            p_cores=cores,
            s=self._params.get("memory_efficient", False),
            batch=self._params.get("batch", None),
        )

        adj_matrix = model.causal_matrix
        if isinstance(adj_matrix, pd.DataFrame):
            adj_matrix = adj_matrix.values

        # gcastle PC: causal_matrix[i,j] = 1 => i -> j  =>  transpose first
        adj_matrix = convert_castle_cpdag(adj_matrix.T)

        info = {
            "nodes": node_names,
            "cores": cores,
        }

        return adj_matrix, info, model
