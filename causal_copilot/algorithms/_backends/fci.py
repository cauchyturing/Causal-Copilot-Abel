"""FCI algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_pag


class FCIBackend(Backend):
    """FCI algorithm using causal-learn (CPU path only).

    Outputs a PAG (Partial Ancestral Graph) rather than a CPDAG,
    so uses the PAG-specific adjacency conversion.
    """

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causallearn.search.ConstraintBased.FCI import fci as cl_fci

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        # Normalise indep_test: strip _cpu/_gpu suffixes
        indep_test = self._params.get("indep_test", "fisherz")
        indep_test = indep_test.replace("_cpu", "").replace("_gpu", "")

        graph, edges = cl_fci(
            data_values,
            alpha=self._params.get("alpha", 0.05),
            indep_test=indep_test,
            depth=self._params.get("depth", 4),
            max_path_length=self._params.get("max_path_length", -1),
            verbose=self._params.get("verbose", False),
            background_knowledge=None,
            show_progress=self._params.get("show_progress", False),
            node_names=node_names,
        )

        adj_matrix = convert_causallearn_pag(graph.graph)

        info = {
            "edges": edges,
            "graph": graph,
        }

        return adj_matrix, info, (graph, edges)
