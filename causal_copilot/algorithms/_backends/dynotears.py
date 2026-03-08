"""DYNOTEARS algorithm backend — imports causalnex directly."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class DYNOTEARSBackend(Backend):
    """DYNOTEARS (Dynamic NOTEARS) using causalnex.

    Returns a summary adjacency matrix (any-lag effects collapsed) and includes
    the full lag_matrix in info.
    """

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causalnex.structure.dynotears import from_pandas_dynamic

        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        max_lag = self._params.get("p", 10)

        # Run DYNOTEARS
        sm = from_pandas_dynamic(
            data,
            p=max_lag,
            lambda_w=self._params.get("lambda_w", 0.01),
            lambda_a=self._params.get("lambda_a", 0.01),
            max_iter=self._params.get("max_iter", 100),
            h_tol=self._params.get("h_tol", 1e-8),
            w_threshold=self._params.get("w_threshold", 0.05),
        )

        # Convert structure model edges to graph dict, then to adjacency matrices
        graph_dict: dict[str, list[tuple[str, int]]] = {name: [] for name in node_names}

        # Build mapping from temporal node names to original names
        tname_to_name: dict[str, str] = {}
        count_lag = 0
        idx_name = 0
        for tname in sm.nodes:
            tname_to_name[tname] = node_names[idx_name]
            if count_lag == max_lag:
                idx_name += 1
                count_lag = -1
            count_lag += 1

        for c, e in sm.edges:
            tc = int(c.partition("lag")[2])
            te = int(e.partition("lag")[2])
            t = tc - te
            entry = (tname_to_name[c], -t)
            if entry not in graph_dict[tname_to_name[e]]:
                graph_dict[tname_to_name[e]].append(entry)

        # Build lag and summary matrices
        n = len(node_names)
        node_to_idx = {name: idx for idx, name in enumerate(node_names)}
        lagged_adj = np.zeros((max_lag + 1, n, n))

        for target, causes in graph_dict.items():
            target_idx = node_to_idx[target]
            for cause_name, lag in causes:
                cause_idx = node_to_idx[cause_name]
                lag_index = -lag
                if 0 <= lag_index <= max_lag:
                    lagged_adj[lag_index, target_idx, cause_idx] = 1

        summary_matrix = np.any(lagged_adj, axis=0).astype(int)

        info = {
            "lag_matrix": lagged_adj,
            "lag": max_lag,
            "nodes": node_names,
        }

        return summary_matrix, info, sm
