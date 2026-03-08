"""VARLiNGAM algorithm backend — imports causal-learn directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class VARLiNGAMBackend(Backend):
    """VARLiNGAM (Vector Auto-Regressive LiNGAM) using causal-learn (CPU path only).

    Returns a summary adjacency matrix (any-lag effects collapsed) and includes
    the full lag_matrix in info.
    """

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causallearn.search.FCMBased.lingam import VARLiNGAM as CLVARLiNGAM

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        data_values = data.values

        model = CLVARLiNGAM(
            lags=self._params.get("lags", 10),
            criterion=self._params.get("criterion", "bic"),
            prune=self._params.get("prune", True),
            ar_coefs=self._params.get("ar_coefs", None),
            lingam_model=self._params.get("lingam_model", None),
            random_state=self._params.get("random_state", None),
        )
        model.fit(data_values)

        lag_matrix = model._adjacency_matrices
        summary_matrix = np.any(lag_matrix, axis=0).astype(int)

        info = {
            "lag": model._lags,
            "residuals": model._residuals,
            "causal_order": model.causal_order_,
            "lag_matrix": lag_matrix,
        }

        return summary_matrix, info, model
