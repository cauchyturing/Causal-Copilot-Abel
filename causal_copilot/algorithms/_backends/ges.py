"""GES algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd
from causallearn.search.ScoreBased.GES import ges as cl_ges

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_cpdag


class GESBackend(Backend):
    """GES algorithm using causal-learn."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        maxP = self._params.get("maxP", 4)
        if maxP is not None and maxP < 0:
            maxP = None

        record = cl_ges(
            data_values,
            score_func=self._params.get("score_func", "local_score_BIC"),
            maxP=maxP,
            parameters=self._params.get("parameters", None),
            node_names=node_names,
        )

        adj_matrix = convert_causallearn_cpdag(record["G"].graph)

        info = {
            "score": record["score"],
            "update1": record["update1"],
            "update2": record["update2"],
        }

        return adj_matrix, info, record
