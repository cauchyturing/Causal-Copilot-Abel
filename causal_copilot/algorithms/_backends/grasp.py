"""GRaSP algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_cpdag


class GRaSPBackend(Backend):
    """GRaSP (Greedy Relaxation of Sparsest Permutation) using causal-learn."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causallearn.search.PermutationBased.GRaSP import grasp as cl_grasp

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        cg = cl_grasp(
            data_values,
            score_func=self._params.get("score_func", "local_score_BIC_from_cov"),
            depth=self._params.get("depth", 4),
            parameters=self._params.get("parameters", None),
            verbose=self._params.get("verbose", False),
            node_names=node_names,
        )

        adj_matrix = convert_causallearn_cpdag(cg.graph)

        info = {
            "score_func": self._params.get("score_func", "local_score_BIC_from_cov"),
            "depth": self._params.get("depth", 4),
        }

        return adj_matrix, info, cg
