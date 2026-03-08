"""Hybrid algorithm backend — two-stage: constraint-based + functional model."""

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend
from causal_copilot.algorithms._backends._utils import convert_causallearn_cpdag


class HybridBackend(Backend):
    """Hybrid two-stage causal discovery.

    Stage 1: Constraint-based method (PC or GES via causal-learn) to obtain CPDAG.
    Stage 2: Functional model-based method (ANM or PNL via causal-learn) to orient
             undirected edges, followed by Meek's rules.
    """

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        # Remove domain_index if present
        if "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        node_names = list(data.columns)
        data_values = data.values
        n = len(node_names)

        first_stage = self._params.get("first_stage_algo", "pc")
        second_stage = self._params.get("second_stage_method", "pnl")
        alpha = self._params.get("alpha", 0.05)
        m_max = self._params.get("m_max", 3)

        # Stage 1: get CPDAG
        cg_stage1 = self._run_first_stage(data_values, first_stage, alpha, node_names)

        # If no second stage, just convert and return
        if second_stage is None:
            adj_matrix = convert_causallearn_cpdag(cg_stage1.G.graph)
            info = {"initial_cpdag": None, "adj_matrix_final": adj_matrix}
            return adj_matrix, info, cg_stage1

        # Stage 2: orient undirected edges using functional tests
        self._orient_undirected_edges(
            cg_stage1, data_values, second_stage, alpha, m_max, n
        )

        adj_matrix = convert_causallearn_cpdag(cg_stage1.G.graph)
        info = {"initial_cpdag": None, "adj_matrix_final": adj_matrix}
        return adj_matrix, info, cg_stage1

    @staticmethod
    def _run_first_stage(data_values, algo_name, alpha, node_names):
        """Run PC or GES to get initial CPDAG."""
        if algo_name.lower() == "pc":
            from causallearn.search.ConstraintBased.PC import pc as cl_pc

            cg = cl_pc(
                data_values,
                alpha=alpha,
                indep_test="fisherz",
                stable=True,
                node_names=node_names,
                verbose=False,
                show_progress=False,
            )
            return cg
        elif algo_name.lower() == "ges":
            from causallearn.graph.GraphClass import CausalGraph
            from causallearn.search.ScoreBased.GES import ges as cl_ges

            record = cl_ges(data_values, node_names=node_names)
            cg = CausalGraph(len(node_names), node_names)
            cg.G = record["G"]
            return cg
        else:
            raise ValueError(f"Unknown first stage algorithm: {algo_name}")

    @staticmethod
    def _orient_undirected_edges(cg, data_values, method, alpha, m_max, n):
        """Attempt to orient each undirected edge using ANM/PNL + Meek rules."""
        from causallearn.utils.PCUtils.Meek import meek

        if method.lower() == "anm":
            from causallearn.search.FCMBased.ANM.ANM import ANM as FunctionalModel
        elif method.lower() == "pnl":
            from causallearn.search.FCMBased.PNL.PNL import PNL as FunctionalModel
        else:
            raise ValueError(f"Unknown second stage method: {method}")

        cg.to_nx_skeleton()

        has_change = True
        while has_change:
            has_change = False
            edges_to_check = []
            for i in range(n):
                for j in range(i + 1, n):
                    if cg.G.is_undirected_from_to(cg.G.nodes[i], cg.G.nodes[j]):
                        edges_to_check.append((i, j))

            for i, j in edges_to_check:
                pi, qi = _get_confounders(cg, i, j)
                oriented = False
                for m in range(m_max + 1):
                    if oriented:
                        break
                    for cand_conf in combinations(qi, m):
                        full_conf = list(set(pi).union(set(cand_conf)))

                        model = FunctionalModel()
                        data_i = data_values[:, i].reshape(-1, 1)
                        data_j = data_values[:, j].reshape(-1, 1)
                        conf_data = data_values[:, full_conf] if full_conf else None
                        pval_for, pval_back = model.cause_or_effect(
                            data_i, data_j, conf_data
                        )

                        if pval_for > alpha and pval_for > pval_back:
                            _orient_edge(cg, i, j)  # i -> j
                            meek(cg)
                            oriented = True
                            has_change = True
                            break
                        elif pval_back > alpha and pval_back > pval_for:
                            _orient_edge(cg, j, i)  # j -> i
                            meek(cg)
                            oriented = True
                            has_change = True
                            break


def _get_confounders(cg, i, j):
    """Get confirmed (pi) and potential (qi) confounders for edge i--j."""
    neigh_i = cg.neighbors(i)
    pi = []
    qi = []
    for nn in neigh_i:
        if nn == j:
            continue
        if cg.G.is_directed_from_to(
            cg.G.nodes[nn], cg.G.nodes[i]
        ) and cg.G.is_directed_from_to(cg.G.nodes[nn], cg.G.nodes[j]):
            pi.append(nn)
        else:
            is_collider = cg.G.is_directed_from_to(
                cg.G.nodes[i], cg.G.nodes[nn]
            ) and cg.G.is_directed_from_to(cg.G.nodes[j], cg.G.nodes[nn])
            if not is_collider:
                qi.append(nn)
    return pi, qi


def _orient_edge(cg, source, target):
    """Remove undirected edge and add directed source -> target."""
    edge = cg.G.get_edge(cg.G.nodes[source], cg.G.nodes[target])
    if edge is not None:
        cg.G.remove_edge(edge)
    cg.G.add_directed_edge(cg.G.nodes[source], cg.G.nodes[target])
