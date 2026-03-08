"""PC algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc as cl_pc

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_cpdag


class PCBackend(Backend):
    """PC algorithm using causal-learn (CPU path only)."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values

        # Normalise indep_test: strip _cpu/_gpu suffixes for the vendored CPU path
        indep_test = self._params.get("indep_test", "fisherz")
        indep_test = indep_test.replace("_cpu", "").replace("_gpu", "")

        cg = cl_pc(
            data_values,
            alpha=self._params.get("alpha", 0.05),
            indep_test=indep_test,
            depth=self._params.get("depth", 4),
            stable=self._params.get("stable", True),
            uc_rule=self._params.get("uc_rule", 0),
            uc_priority=self._params.get("uc_priority", -1),
            mvpc=self._params.get("mvpc", False),
            correction_name=self._params.get("correction_name", "MV_Crtn_Fisher_Z"),
            background_knowledge=None,
            verbose=self._params.get("verbose", False),
            show_progress=self._params.get("show_progress", False),
            node_names=node_names,
        )

        adj_matrix = convert_causallearn_cpdag(cg.G.graph)

        info = {
            "sepset": cg.sepset if hasattr(cg, "sepset") else None,
            "definite_UC": cg.definite_UC if hasattr(cg, "definite_UC") else [],
            "definite_non_UC": cg.definite_non_UC if hasattr(cg, "definite_non_UC") else [],
        }

        return adj_matrix, info, cg
