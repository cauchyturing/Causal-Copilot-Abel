"""CDNOD algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_cpdag


class CDNODBackend(Backend):
    """CDNOD (Constraint-based causal Discovery from Nonstationary/heterogeneous Data)
    using causal-learn (CPU path only).

    Requires a ``domain_index`` column in data. If absent, defaults to all-ones.
    The domain_index column is appended as the ``c_indx`` argument to the
    upstream cdnod() call, and the resulting extra row/col is stripped from the
    output adjacency matrix.
    """

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causallearn.search.ConstraintBased.CDNOD import cdnod as cl_cdnod

        # Extract or create domain_index
        if "domain_index" in data.columns:
            c_indx = data["domain_index"].values.reshape(-1, 1)
            data = data.drop(columns=["domain_index"])
        else:
            c_indx = np.ones((data.shape[0], 1))

        node_names = list(data.columns)
        data_values = data.values

        # Normalise indep_test: strip _cpu/_gpu suffixes
        indep_test = self._params.get("indep_test", "fisherz")
        indep_test = indep_test.replace("_cpu", "").replace("_gpu", "")

        cg = cl_cdnod(
            data_values,
            c_indx,
            alpha=self._params.get("alpha", 0.05),
            indep_test=indep_test,
            stable=self._params.get("stable", True),
            uc_rule=self._params.get("uc_rule", 0),
            uc_priority=self._params.get("uc_priority", 2),
            depth=self._params.get("depth", 5),
            mvcdnod=self._params.get("mvcdnod", False),
            correction_name=self._params.get("correction_name", "MV_Crtn_Fisher_Z"),
            background_knowledge=None,
            verbose=self._params.get("verbose", False),
            show_progress=self._params.get("show_progress", False),
            node_names=node_names,
        )

        full_adj = convert_causallearn_cpdag(cg.G.graph)
        # Remove the domain_index row/col appended by cdnod
        adj_matrix = full_adj[:-1, :-1]

        info = {
            "PC_elapsed": cg.PC_elapsed if hasattr(cg, "PC_elapsed") else None,
        }

        return adj_matrix, info, cg
