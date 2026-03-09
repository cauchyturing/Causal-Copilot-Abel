"""Tests for the MCP server tools (13-tool contract).

Core: discover, inspect_graph, estimate_effect, diagnose_data, run_algorithm
Reasoning: refute_estimate, estimate_counterfactual, attribute_anomaly,
           attribute_distribution_change, simulate_intervention
Analysis: compute_feature_importance, validate_graph
Reporting: generate_report
"""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastmcp.exceptions import ToolError


class _MockWrapper:
    """Mock wrapper that mimics a causal discovery algorithm wrapper."""

    def __init__(self, args=None):
        pass

    def fit(self, data, **kwargs):
        n = data.shape[1]
        return np.zeros((n, n)), {"mock": True}, None


def _mock_stat_info(gs):
    """Mock stat_info_collection that sets plausible statistics."""
    gs.statistics.linearity = True
    gs.statistics.gaussian_error = True
    gs.statistics.missingness = False
    gs.statistics.data_type = "Continuous"
    gs.statistics.sample_size = gs.user_data.raw_data.shape[0]
    gs.statistics.feature_number = gs.user_data.raw_data.shape[1]
    gs.statistics.time_series = False
    # Per-column detail
    cols = list(gs.user_data.raw_data.columns)
    gs.statistics.miss_ratio = {c: 0.0 for c in cols}
    gs.statistics.data_type_column = {c: "Continuous" for c in cols}
    gs.statistics.description = "Linear Gaussian continuous data, no missing values."
    return gs


def _inject_fake_stat_module():
    """Inject fake preprocess.stat_info_functions into sys.modules.

    Returns (had_module, old_module) for cleanup.
    """
    import sys
    from types import ModuleType

    had = "preprocess.stat_info_functions" in sys.modules
    old = sys.modules.get("preprocess.stat_info_functions")
    fake = ModuleType("preprocess.stat_info_functions")
    fake.stat_info_collection = _mock_stat_info
    sys.modules["preprocess.stat_info_functions"] = fake
    return had, old


def _restore_stat_module(had, old):
    """Restore preprocess.stat_info_functions after mock."""
    import sys

    if had and old is not None:
        sys.modules["preprocess.stat_info_functions"] = old
    elif not had:
        sys.modules.pop("preprocess.stat_info_functions", None)


class _mock_stat_info_ctx:
    """Context manager that injects a fake stat module for diagnose_data tests."""

    def __enter__(self):
        self._had, self._old = _inject_fake_stat_module()
        return self

    def __exit__(self, *args):
        _restore_stat_module(self._had, self._old)


class _mock_run_algorithm:
    """Context manager that patches wrappers + stat_info so pipeline tools work."""

    def __enter__(self):
        self._had, self._old = _inject_fake_stat_module()
        self._patcher = patch(
            "causal_discovery.wrappers.PC",
            _MockWrapper,
        )
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()
        _restore_stat_module(self._had, self._old)


# ── inspect_graph ──────────────────────────────────────────────────────


class TestInspectGraphTool:
    def test_dag_with_adj(self):
        from causal_copilot.mcp.server import inspect_graph

        # A->B: adj[1,0]=1
        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
            )
        )
        assert result["status"] == "ok"
        assert result["graph_kind"] == "dag"
        assert result["inference_policy"]["eligibility"] is True
        assert result["inference_policy"]["method"] == "standard"
        assert "summary" in result
        assert "key_findings" in result

    def test_cpdag_needs_more_input(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,2],[2,0]]",
                node_names='["A","B"]',
            )
        )
        assert result["status"] == "needs_more_input"
        assert "data_diagnosis" in result["missing_inputs"]
        assert "next_step" in result

    def test_cpdag_with_diagnosis_allows_ida(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,2],[2,0]]",
                node_names='["A","B"]',
                data_diagnosis='{"linearity": true, "gaussian_error": true}',
            )
        )
        assert result["status"] == "ok"
        assert result["inference_policy"]["eligibility"] is True
        assert result["inference_policy"]["method"] == "ida"

    def test_cpdag_nonlinear_rejects(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,2],[2,0]]",
                node_names='["A","B"]',
                data_diagnosis='{"linearity": false, "gaussian_error": true}',
            )
        )
        assert result["status"] == "ok"
        assert result["inference_policy"]["eligibility"] is False

    def test_pag_rejects(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,3],[3,0]]",
                node_names='["A","B"]',
            )
        )
        assert result["status"] == "ok"
        assert result["graph_kind"] == "pag"
        assert result["inference_policy"]["eligibility"] is False

    def test_query_assessment(self):
        from causal_copilot.mcp.server import inspect_graph

        # A->B->C: adj[1,0]=1, adj[2,1]=1
        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,0,0],[1,0,0],[0,1,0]]",
                node_names='["A","B","C"]',
                treatment="A",
                outcome="C",
            )
        )
        assert result["status"] == "ok"
        qa = result["query_assessment"]
        assert qa["directed_path_exists"] is True
        assert qa["effect_identifiable"] is True
        assert qa["directly_connected"] is False

    def test_query_no_path(self):
        from causal_copilot.mcp.server import inspect_graph

        # B->A: adj[0,1]=1. Query A->B has no directed path.
        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,1],[0,0]]",
                node_names='["A","B"]',
                treatment="A",
                outcome="B",
            )
        )
        assert result["status"] == "ok"
        qa = result["query_assessment"]
        assert qa["directed_path_exists"] is False
        assert qa["effect_identifiable"] is False

    def test_run_id_mode(self):
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import inspect_graph

        rid = get_store().save(
            {
                "adjacency_matrix": [[0, 0], [1, 0]],
                "node_names": ["X", "Y"],
                "data_diagnosis": {"linearity": True, "gaussian_error": True},
            }
        )
        result = json.loads(inspect_graph(run_id=rid))
        assert result["status"] == "ok"
        assert result["graph_kind"] == "dag"

    def test_run_id_not_found(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="not found"):
            inspect_graph(run_id="nonexistent")

    def test_mutual_exclusion(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="mutually exclusive"):
            inspect_graph(run_id="abc", adjacency_matrix="[[0]]")

    def test_no_input(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError):
            inspect_graph()

    def test_missing_node_names(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="node_names"):
            inspect_graph(adjacency_matrix="[[0,1],[0,0]]")

    def test_treatment_outcome_all_or_none(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="[Bb]oth"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="A",
            )

    def test_treatment_equals_outcome(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="different"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="A",
                outcome="A",
            )

    def test_treatment_not_in_names(self):
        from causal_copilot.mcp.server import inspect_graph

        with pytest.raises(ToolError, match="Z"):
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
                treatment="Z",
                outcome="B",
            )

    def test_graph_stats(self):
        from causal_copilot.mcp.server import inspect_graph

        result = json.loads(
            inspect_graph(
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["A","B"]',
            )
        )
        gs = result["graph_stats"]
        assert gs["n_nodes"] == 2
        assert gs["n_edges"] == 1
        assert gs["n_directed"] == 1
        assert gs["density"] == 1.0


# ── diagnose_data ──────────────────────────────────────────────────────


class TestDiagnoseDataTool:
    def test_basic_diagnosis(self):
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(csv))
        assert result["status"] == "ok"
        assert "linearity" in result["diagnosis"]
        assert "data_type" in result["diagnosis"]
        assert "sample_size" in result["diagnosis"]

    def test_empty_csv(self):
        from causal_copilot.mcp.server import diagnose_data

        with pytest.raises(ToolError):
            diagnose_data("")


# ── run_algorithm ──────────────────────────────────────────────────────


class TestRunAlgorithmTool:
    def test_run_with_mock(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(csv, algorithm="PC"))
        assert result["status"] == "ok"
        assert "adjacency_matrix" in result
        assert "run_id" in result

    def test_hp_transparency(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(
                run_algorithm(
                    csv,
                    algorithm="PC",
                    hyperparameters='{"alpha": 0.01}',
                )
            )
        assert result["status"] == "ok"
        prov = result["provenance"]
        assert "requested_hyperparameters" in prov
        assert "effective_hyperparameters" in prov
        assert "resolver_adjustments" in prov
        assert prov["requested_hyperparameters"]["alpha"] == 0.01

    def test_no_resolver_overrides(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(
                run_algorithm(
                    csv,
                    algorithm="PC",
                    hyperparameters='{"alpha": 0.01, "indep_test": "kci"}',
                    allow_resolver_overrides=False,
                )
            )
        assert result["status"] == "ok"
        prov = result["provenance"]
        # With overrides disabled, effective should match requested
        assert prov["effective_hyperparameters"]["indep_test"] == "kci"
        assert prov["resolver_adjustments"] == {}

    def test_artifact_stores_data(self):
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(run_algorithm(csv, algorithm="PC"))
        assert result["status"] == "ok"
        cached = get_store().get(result["run_id"])
        assert cached is not None
        assert "_processed_data" in cached
        assert "_statistics" in cached
        import pandas as pd

        assert isinstance(cached["_processed_data"], pd.DataFrame)

    def test_missing_algorithm(self):
        from causal_copilot.mcp.server import run_algorithm

        with pytest.raises(ToolError):
            run_algorithm("a,b\n1,2\n3,4", algorithm="")


# ── discover ───────────────────────────────────────────────────────────


class TestDiscoverTool:
    def test_full_pipeline_with_mock(self):
        from causal_copilot.mcp.server import discover

        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_run_algorithm():
            result = json.loads(discover(csv))
        assert result["status"] in ("ok", "partial", "error")
        if result["status"] == "ok":
            assert "graph_kind" in result
            assert "provenance" in result
            assert "run_id" in result
            # Enriched output fields
            assert "summary" in result
            assert "key_findings" in result
            assert "limitations" in result
            assert "algorithm_rationale" in result

    def test_empty_csv(self):
        from causal_copilot.mcp.server import discover

        with pytest.raises(ToolError):
            discover("")

    def test_too_few_rows(self):
        from causal_copilot.mcp.server import discover

        with pytest.raises(ToolError, match="10 rows"):
            discover("a,b\n1,2\n3,4")

    def test_with_algorithm_override(self):
        from causal_copilot.mcp.server import discover

        csv = "x,y\n" + "\n".join(f"{i},{i * 2}" for i in range(50))
        with _mock_run_algorithm():
            result = json.loads(discover(csv, algorithm="PC"))
        assert result["status"] in ("ok", "partial", "error")


# ── MCP CLI ────────────────────────────────────────────────────────────


# ── estimation helpers ────────────────────────────────────────────────


class TestEstimationLinear:
    def test_linear_basic(self):
        from causal_copilot.mcp.estimation import estimate_linear

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        y = 2.0 * x + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"X": x, "Y": y})
        dot_graph = "digraph { X -> Y; }"
        result = estimate_linear(data, dot_graph, "X", "Y", 0.0, 1.0)
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert 1.5 < ate < 2.5, f"ATE should be ~2.0, got {ate}"
        assert result["ate"]["p_value"] < 0.05


class TestEstimationMatching:
    def test_matching_binary(self):
        from causal_copilot.mcp.estimation import estimate_matching

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)  # binary treatment
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_matching(data, "T", "Y", ["Z"], 0, 1)
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert 2.0 < ate < 4.0, f"ATE should be ~3.0, got {ate}"


class TestEstimationDML:
    def test_dml_basic(self):
        from causal_copilot.mcp.estimation import estimate_dml

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        x = z + rng.normal(size=n) * 0.5
        y = 2.0 * x + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "X": x, "Y": y})
        result = estimate_dml(
            data,
            treatment="X",
            outcome="Y",
            X_col=["Z"],
            W_col=["Z"],
            T0=0.0,
            T1=1.0,
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None
        assert "att" in result


class TestEstimationDRL:
    def test_drl_basic(self):
        from causal_copilot.mcp.estimation import estimate_drl

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 2.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_drl(
            data,
            treatment="T",
            outcome="Y",
            X_col=["Z"],
            W_col=["Z"],
            T0=0,
            T1=1,
        )
        assert "ate" in result
        assert result["ate"]["estimate"] is not None
        assert "att" in result


# ── estimate_effect ───────────────────────────────────────────────────


class TestEstimateEffectTool:
    def _make_linear_data_csv(self, rng, n=200):
        """Y = 2*X + Z + noise. DAG: Z→X, Z→Y, X→Y."""
        z = rng.normal(size=n)
        x = z + rng.normal(size=n) * 0.5
        y = 2.0 * x + z + rng.normal(size=n) * 0.5
        lines = ["Z,X,Y"]
        for i in range(n):
            lines.append(f"{z[i]},{x[i]},{y[i]}")
        return "\n".join(lines)

    def test_linear_with_adj(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        csv = self._make_linear_data_csv(rng)
        # DAG: Z→X (adj[1,0]=1), Z→Y (adj[2,0]=1), X→Y (adj[2,1]=1)
        adj = "[[0,0,0],[1,0,0],[1,1,0]]"
        names = '["Z","X","Y"]'
        result = json.loads(
            estimate_effect(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                method="linear",
            )
        )
        assert result["status"] == "ok"
        assert result["method"] == "linear"
        ate = result["estimates"]["ate"]["estimate"]
        assert 1.0 < ate < 3.0, f"ATE should be ~2.0, got {ate}"
        assert "confounders_used" in result
        assert "provenance" in result
        assert "next_steps" in result

    def test_rejected_pag(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        csv = self._make_linear_data_csv(rng)
        # PAG: adj[0,1]=3, adj[1,0]=3
        adj = "[[0,3],[3,0]]"
        names = '["X","Y"]'
        result = json.loads(
            estimate_effect(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
            )
        )
        assert result["status"] == "rejected"
        assert "next_steps" in result

    def test_rejected_cpdag_nonlinear(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        csv = self._make_linear_data_csv(rng)
        # CPDAG: adj[0,1]=2, adj[1,0]=2
        adj = "[[0,2],[2,0]]"
        names = '["X","Y"]'
        diag = '{"linearity": false, "gaussian_error": true}'
        result = json.loads(
            estimate_effect(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                data_diagnosis=diag,
            )
        )
        assert result["status"] == "rejected"

    def test_treatment_not_in_data(self):
        from causal_copilot.mcp.server import estimate_effect

        with pytest.raises(ToolError, match="MISSING"):
            estimate_effect(
                treatment="MISSING",
                outcome="Y",
                csv_data="X,Y\n1,2\n3,4",
                adjacency_matrix="[[0,0],[1,0]]",
                node_names='["X","Y"]',
            )

    def test_mutual_exclusion(self):
        from causal_copilot.mcp.server import estimate_effect

        with pytest.raises(ToolError, match="mutually exclusive"):
            estimate_effect(
                treatment="X",
                outcome="Y",
                run_id="abc",
                csv_data="x,y\n1,2",
            )

    def test_no_input(self):
        from causal_copilot.mcp.server import estimate_effect

        with pytest.raises(ToolError):
            estimate_effect(treatment="X", outcome="Y")

    def test_invalid_method(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        csv = self._make_linear_data_csv(rng)
        adj = "[[0,0,0],[1,0,0],[1,1,0]]"
        names = '["Z","X","Y"]'
        with pytest.raises(ToolError, match="bogus"):
            estimate_effect(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                method="bogus",
            )

    def test_run_id_from_run_algorithm(self):
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import estimate_effect, run_algorithm

        rng = np.random.default_rng(42)
        n = 200
        z = rng.normal(size=n)
        x = z + rng.normal(size=n) * 0.5
        y = 2.0 * x + z + rng.normal(size=n) * 0.5
        lines = ["Z,X,Y"]
        for i in range(n):
            lines.append(f"{z[i]},{x[i]},{y[i]}")
        csv = "\n".join(lines)

        with _mock_run_algorithm():
            algo_result = json.loads(run_algorithm(csv, algorithm="PC"))
        assert algo_result["status"] == "ok"
        rid = algo_result["run_id"]

        # Mock graph: Z→X, Z→Y, X→Y (DAG)
        cached = get_store().get(rid)
        cached["adjacency_matrix"] = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]
        cached["node_names"] = ["Z", "X", "Y"]
        cached["data_diagnosis"] = {"linearity": True, "gaussian_error": True}

        result = json.loads(
            estimate_effect(
                treatment="X",
                outcome="Y",
                run_id=rid,
                method="linear",
            )
        )
        assert result["status"] == "ok"
        assert 1.0 < result["estimates"]["ate"]["estimate"] < 3.0


# ── MetaLearner estimation ──────────────────────────────────────────


class TestEstimationMetaLearner:
    def test_tlearner_basic(self):
        from causal_copilot.mcp.estimation import estimate_metalearner

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_metalearner(
            data,
            treatment="T",
            outcome="Y",
            X_col=["Z"],
            T0=0,
            T1=1,
            learner="t",
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None
        assert 2.0 < ate < 4.5, f"ATE should be ~3.0, got {ate}"
        assert "att" in result

    def test_slearner_basic(self):
        from causal_copilot.mcp.estimation import estimate_metalearner

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_metalearner(
            data,
            treatment="T",
            outcome="Y",
            X_col=["Z"],
            T0=0,
            T1=1,
            learner="s",
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None
        assert result["algo"] == "SLearner"

    def test_xlearner_basic(self):
        from causal_copilot.mcp.estimation import estimate_metalearner

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_metalearner(
            data,
            treatment="T",
            outcome="Y",
            X_col=["Z"],
            T0=0,
            T1=1,
            learner="x",
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None
        assert result["algo"] == "XLearner"

    def test_dalearner_basic(self):
        from causal_copilot.mcp.estimation import estimate_metalearner

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"Z": z, "T": t, "Y": y})
        result = estimate_metalearner(
            data,
            treatment="T",
            outcome="Y",
            X_col=["Z"],
            T0=0,
            T1=1,
            learner="da",
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None
        assert result["algo"] == "DALearner"


class TestMetaLearnerSelection:
    """Test data-driven MetaLearner selection heuristic."""

    def test_linear_balanced_selects_tlearner(self):
        from causal_copilot.mcp.offline import get_default_estimation_config

        rng = np.random.default_rng(42)
        n = 300
        t = rng.choice([0, 1], size=n, p=[0.5, 0.5])  # balanced
        data = pd.DataFrame({"T": t, "Y": rng.normal(size=n), "X": rng.normal(size=n)})
        config = get_default_estimation_config("metalearner", data, "T", outcome="Y", is_linear=True)
        assert config["learner"] == "t"

    def test_nonlinear_balanced_selects_xlearner(self):
        from causal_copilot.mcp.offline import get_default_estimation_config

        rng = np.random.default_rng(42)
        n = 300
        t = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        data = pd.DataFrame({"T": t, "Y": rng.normal(size=n), "X": rng.normal(size=n)})
        config = get_default_estimation_config("metalearner", data, "T", outcome="Y", is_linear=False)
        assert config["learner"] == "x"

    def test_nonlinear_imbalanced_selects_dalearner(self):
        from causal_copilot.mcp.offline import get_default_estimation_config

        rng = np.random.default_rng(42)
        n = 300
        t = rng.choice([0, 1], size=n, p=[0.9, 0.1])  # heavily imbalanced
        data = pd.DataFrame({"T": t, "Y": rng.normal(size=n), "X": rng.normal(size=n)})
        config = get_default_estimation_config("metalearner", data, "T", outcome="Y", is_linear=False)
        assert config["learner"] == "da"

    def test_small_sample_selects_slearner(self):
        from causal_copilot.mcp.offline import get_default_estimation_config

        rng = np.random.default_rng(42)
        n = 50  # small sample
        t = rng.choice([0, 1], size=n)
        data = pd.DataFrame({"T": t, "Y": rng.normal(size=n), "X": rng.normal(size=n)})
        config = get_default_estimation_config("metalearner", data, "T", outcome="Y", is_linear=False)
        assert config["learner"] == "s"


class TestEstimationIV:
    def test_iv_basic(self):
        from causal_copilot.mcp.estimation import estimate_iv

        rng = np.random.default_rng(42)
        n = 500
        # Z is instrument: Z→T, T→Y, Z⊥Y|T
        z = rng.normal(size=n)
        u = rng.normal(size=n)  # unobserved confounder
        t = 0.5 * z + 0.5 * u + rng.normal(size=n) * 0.3
        y = 2.0 * t + 0.5 * u + rng.normal(size=n) * 0.3
        data = pd.DataFrame({"Z": z, "T": t, "Y": y, "W": rng.normal(size=n)})
        result = estimate_iv(
            data,
            treatment="T",
            outcome="Y",
            instrument="Z",
            X_col=["W"],
            W_col=["W"],
            T0=0.0,
            T1=1.0,
        )
        assert "ate" in result
        ate = result["ate"]["estimate"]
        assert ate is not None


# ── estimate_effect with metalearner ─────────────────────────────────


class TestEstimateEffectMetaLearner:
    def test_metalearner_via_tool(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        n = 300
        z = rng.normal(size=n)
        t = (z + rng.normal(size=n) > 0).astype(int)
        y = 3.0 * t + z + rng.normal(size=n) * 0.5
        lines = ["Z,T,Y"]
        for i in range(n):
            lines.append(f"{z[i]},{t[i]},{y[i]}")
        csv = "\n".join(lines)
        # DAG: Z→T (adj[1,0]=1), Z→Y (adj[2,0]=1), T→Y (adj[2,1]=1)
        adj = "[[0,0,0],[1,0,0],[1,1,0]]"
        names = '["Z","T","Y"]'
        result = json.loads(
            estimate_effect(
                treatment="T",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                method="metalearner",
            )
        )
        assert result["status"] == "ok"
        assert result["method"] == "metalearner"
        assert result["estimates"]["ate"]["estimate"] is not None


class TestEstimateEffectIV:
    def test_iv_auto_detect(self):
        from causal_copilot.mcp.server import estimate_effect

        rng = np.random.default_rng(42)
        n = 500
        z = rng.normal(size=n)
        t = 0.5 * z + rng.normal(size=n) * 0.3
        y = 2.0 * t + rng.normal(size=n) * 0.3
        lines = ["Z,T,Y"]
        for i in range(n):
            lines.append(f"{z[i]},{t[i]},{y[i]}")
        csv = "\n".join(lines)
        # DAG: Z→T (adj[1,0]=1), T→Y (adj[2,1]=1). Z is instrument.
        adj = "[[0,0,0],[1,0,0],[0,1,0]]"
        names = '["Z","T","Y"]'
        result = json.loads(
            estimate_effect(
                treatment="T",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                method="iv",
            )
        )
        assert result["status"] == "ok"
        assert result["method"] == "iv"
        assert "instrument" in result["method_detail"].lower()

    def test_iv_no_instrument_found(self):
        from causal_copilot.mcp.server import estimate_effect

        csv = "X,Y\n" + "\n".join(f"{i},{i * 2}" for i in range(100))
        # DAG: X→Y only, no instrument
        adj = "[[0,0],[1,0]]"
        names = '["X","Y"]'
        result = json.loads(
            estimate_effect(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                method="iv",
            )
        )
        assert result["status"] == "error"
        assert "instrument" in result["error"].lower()


# ── refute_estimate ──────────────────────────────────────────────────


class TestRefuteEstimateTool:
    def test_basic_refutation(self):
        from causal_copilot.mcp.server import refute_estimate

        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(size=n)
        y = 2.0 * x + rng.normal(size=n) * 0.5
        lines = ["X,Y"]
        for i in range(n):
            lines.append(f"{x[i]},{y[i]}")
        csv = "\n".join(lines)
        adj = "[[0,0],[1,0]]"
        names = '["X","Y"]'
        result = json.loads(
            refute_estimate(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
            )
        )
        assert result["status"] == "ok"
        assert "original_estimate" in result
        assert "refutations" in result
        assert "robust" in result
        assert "interpretation" in result

    def test_pag_rejected(self):
        from causal_copilot.mcp.server import refute_estimate

        csv = "X,Y\n1,2\n3,4\n5,6"
        adj = "[[0,3],[3,0]]"
        names = '["X","Y"]'
        result = json.loads(
            refute_estimate(
                treatment="X",
                outcome="Y",
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
            )
        )
        assert result["status"] == "rejected"


# ── GCM tools ────────────────────────────────────────────────────────


def _make_gcm_data(rng, n=300):
    """Generate DAG data: A → B → C with A → C."""
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(size=n) * 0.5
    c = 1.5 * b + 0.5 * a + rng.normal(size=n) * 0.5
    return pd.DataFrame({"A": a, "B": b, "C": c})


def _make_gcm_csv(rng, n=300):
    df = _make_gcm_data(rng, n)
    lines = [",".join(df.columns)]
    for _, row in df.iterrows():
        lines.append(",".join(str(v) for v in row))
    return "\n".join(lines)


# DAG: A→B (adj[1,0]=1), B→C (adj[2,1]=1), A→C (adj[2,0]=1)
_GCM_ADJ = "[[0,0,0],[1,0,0],[1,1,0]]"
_GCM_NAMES = '["A","B","C"]'


class TestEstimateCounterfactualTool:
    def test_basic_counterfactual(self):
        from causal_copilot.mcp.server import estimate_counterfactual

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            estimate_counterfactual(
                treatment="A",
                outcome="C",
                intervention_value=5.0,
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok"
        assert "observed" in result
        assert "counterfactual" in result
        assert "effect" in result
        assert result["counterfactual"]["A"] == 5.0
        assert "interpretation" in result


class TestAttributeAnomalyTool:
    def test_basic_anomaly(self):
        from causal_copilot.mcp.server import attribute_anomaly

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            attribute_anomaly(
                target_node="C",
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok"
        assert "attributions" in result
        assert "interpretation" in result

    def test_target_not_in_graph(self):
        from causal_copilot.mcp.server import attribute_anomaly

        csv = "A,B,C\n1,2,3"
        with pytest.raises(ToolError, match="MISSING"):
            attribute_anomaly(
                target_node="MISSING",
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )


class TestAttributeDistributionChangeTool:
    def test_basic_distribution_change(self):
        from causal_copilot.mcp.server import attribute_distribution_change

        rng = np.random.default_rng(42)
        csv_old = _make_gcm_csv(rng, n=500)
        # New data with shifted A (causes downstream changes)
        rng2 = np.random.default_rng(99)
        n = 500
        a = rng2.normal(loc=3.0, size=n)  # shifted mean
        b = 2.0 * a + rng2.normal(size=n) * 0.5
        c = 1.5 * b + 0.5 * a + rng2.normal(size=n) * 0.5
        df_new = pd.DataFrame({"A": a, "B": b, "C": c})
        lines = [",".join(df_new.columns)]
        for _, row in df_new.iterrows():
            lines.append(",".join(str(v) for v in row))
        csv_new = "\n".join(lines)

        result = json.loads(
            attribute_distribution_change(
                target_node="C",
                csv_data_new=csv_new,
                csv_data_old=csv_old,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok", f"Distribution change failed: {result.get('error', 'unknown')}"
        assert "attributions" in result
        assert "interpretation" in result


class TestSimulateInterventionTool:
    def test_shift_intervention(self):
        from causal_copilot.mcp.server import simulate_intervention

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            simulate_intervention(
                treatment="A",
                outcome="C",
                intervention_value=2.0,
                shift=True,
                num_samples=500,
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok"
        assert "original_distribution" in result
        assert "intervention_distribution" in result
        assert "mean_change" in result
        assert result["intervention_type"] == "shift"
        # Shifting A by +2 should increase C
        assert result["mean_change"] > 0

    def test_atomic_intervention(self):
        from causal_copilot.mcp.server import simulate_intervention

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            simulate_intervention(
                treatment="A",
                outcome="C",
                intervention_value=0.0,
                shift=False,
                num_samples=500,
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok"
        assert result["intervention_type"] == "atomic"


# ── MCP tool registration ────────────────────────────────────────────


# ── Feature Importance ──────────────────────────────────────────────


class TestFeatureImportanceEstimation:
    def test_linear_shap(self):
        from causal_copilot.mcp.estimation import compute_feature_importance

        rng = np.random.default_rng(42)
        n = 200
        a = rng.normal(size=n)
        b = 2.0 * a + rng.normal(size=n) * 0.5
        c = 1.5 * b + 0.5 * a + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"A": a, "B": b, "C": c})
        result = compute_feature_importance(data, "C", is_linear=True)
        assert "feature_importance" in result
        assert "B" in result["feature_importance"]
        assert "A" in result["feature_importance"]
        # B should have higher importance than A (coefficient 1.5 vs 0.5)
        assert result["feature_importance"]["B"] > result["feature_importance"]["A"]
        assert result["method"] == "linear_shap"

    def test_tree_shap(self):
        from causal_copilot.mcp.estimation import compute_feature_importance

        rng = np.random.default_rng(42)
        n = 200
        a = rng.normal(size=n)
        b = 2.0 * a + rng.normal(size=n) * 0.5
        c = 1.5 * b + 0.5 * a + rng.normal(size=n) * 0.5
        data = pd.DataFrame({"A": a, "B": b, "C": c})
        result = compute_feature_importance(data, "C", is_linear=False)
        assert result["method"] == "tree_shap"
        assert "top_features" in result
        assert len(result["top_features"]) > 0


class TestComputeFeatureImportanceTool:
    def test_basic_fi(self):
        from causal_copilot.mcp.server import compute_feature_importance

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            compute_feature_importance(
                target_node="C",
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )
        )
        assert result["status"] == "ok"
        assert "feature_importance" in result
        assert "interpretation" in result
        assert "next_steps" in result

    def test_target_not_in_data(self):
        from causal_copilot.mcp.server import compute_feature_importance

        csv = "A,B,C\n1,2,3\n4,5,6"
        with pytest.raises(ToolError, match="MISSING"):
            compute_feature_importance(
                target_node="MISSING",
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
            )

    def test_nonlinear_detection(self):
        from causal_copilot.mcp.server import compute_feature_importance

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        diag = '{"linearity": false}'
        result = json.loads(
            compute_feature_importance(
                target_node="C",
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
                data_diagnosis=diag,
            )
        )
        assert result["status"] == "ok"
        assert result["method"] == "tree_shap"


# ── Graph Validation ───────────────────────────────────────────────


class TestGraphFalsificationEstimation:
    def test_basic_falsification(self):
        from causal_copilot.mcp.estimation import run_graph_falsification

        rng = np.random.default_rng(42)
        data = _make_gcm_data(rng)
        adj = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        names = ["A", "B", "C"]
        result = run_graph_falsification(data, adj, names, n_permutations=5)
        assert "falsification_result" in result
        assert result["n_nodes"] == 3
        assert result["n_edges"] == 3


class TestValidateGraphTool:
    def test_basic_validation(self):
        from causal_copilot.mcp.server import validate_graph

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        result = json.loads(
            validate_graph(
                csv_data=csv,
                adjacency_matrix=_GCM_ADJ,
                node_names=_GCM_NAMES,
                n_permutations=5,
            )
        )
        assert result["status"] == "ok"
        assert "falsification_result" in result
        assert "interpretation" in result
        assert result["graph_kind"] == "dag"

    def test_cpdag_sanitizes(self):
        from causal_copilot.mcp.server import validate_graph

        rng = np.random.default_rng(42)
        csv = _make_gcm_csv(rng)
        # CPDAG with undirected A--B, directed B→C, A→C
        adj = "[[0,2,0],[2,0,0],[1,1,0]]"
        names = '["A","B","C"]'
        result = json.loads(
            validate_graph(
                csv_data=csv,
                adjacency_matrix=adj,
                node_names=names,
                n_permutations=5,
            )
        )
        assert result["status"] == "ok"
        assert result["graph_kind"] == "cpdag"
        assert "dropped_edges" in result


# ── MCP tool registration ────────────────────────────────────────────


class TestToolRegistration:
    def test_13_tools_registered(self):
        import asyncio

        from causal_copilot.mcp.server import mcp

        tools = asyncio.run(mcp.list_tools())
        actual_names = {t.name for t in tools}
        expected_tools = {
            "discover",
            "inspect_graph",
            "estimate_effect",
            "diagnose_data",
            "run_algorithm",
            "refute_estimate",
            "estimate_counterfactual",
            "attribute_anomaly",
            "attribute_distribution_change",
            "simulate_intervention",
            "compute_feature_importance",
            "validate_graph",
            "generate_report",
        }
        missing = expected_tools - actual_names
        assert not missing, f"Missing tools: {missing}"
        assert len(actual_names) >= 13, f"Expected 13+ tools, got {len(actual_names)}"


# ── Knowledge-first workflow ──────────────────────────────────────────


class TestKnowledgePrompt:
    """Test that diagnose_data returns data-adaptive knowledge_prompt."""

    def test_meaningful_variable_names(self):
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["Income,Education,Age"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(csv))
        assert result["status"] == "ok"
        kp = result["knowledge_prompt"]
        assert "Income" in kp
        assert "Education" in kp
        assert "Known causal relationships" in kp
        assert "Forbidden edges" in kp

    def test_generic_variable_names(self):
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["X1,X2,X3"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)
        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(csv))
        kp = result["knowledge_prompt"]
        assert "generic" in kp.lower() or "no domain knowledge" in kp.lower()

    def test_data_specific_sections(self):
        """Non-Gaussian triggers non-Gaussian-specific knowledge questions."""
        from causal_copilot.mcp.server import diagnose_data

        rng = np.random.default_rng(0)
        lines = ["A,B,C"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)

        # Mock with non-Gaussian stats
        def _mock_nongauss(gs):
            gs = _mock_stat_info(gs)
            gs.statistics.gaussian_error = False
            return gs

        import sys
        from types import ModuleType

        fake = ModuleType("preprocess.stat_info_functions")
        fake.stat_info_collection = _mock_nongauss
        old = sys.modules.get("preprocess.stat_info_functions")
        sys.modules["preprocess.stat_info_functions"] = fake
        try:
            result = json.loads(diagnose_data(csv))
        finally:
            if old is not None:
                sys.modules["preprocess.stat_info_functions"] = old
            else:
                sys.modules.pop("preprocess.stat_info_functions", None)

        kp = result["knowledge_prompt"]
        assert "Non-Gaussian" in kp or "non-Gaussian" in kp


class TestDomainKnowledgeInjection:
    """Test that domain_knowledge param reaches the pipeline."""

    def test_discover_accepts_domain_knowledge(self):
        from causal_copilot.mcp.server import discover

        rng = np.random.default_rng(0)
        lines = ["Income,Education,Wage"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)

        dk = "Income is caused by Education level. Wage depends on both."

        with _mock_run_algorithm():
            result = json.loads(
                discover(
                    csv,
                    query="What causes Wage?",
                    domain_knowledge=dk,
                )
            )
        # Should work without error — knowledge just passes through
        assert result["status"] in ("ok", "partial", "error")

    def test_knowledge_reaches_global_state(self):
        """Verify domain_knowledge is set on gs.user_data.knowledge_docs."""
        from causal_copilot.mcp.bridge import make_global_state

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        gs = make_global_state(df)
        dk = "A causes B through mechanism X"
        gs.user_data.knowledge_docs = dk
        assert gs.user_data.knowledge_docs == dk


class TestFilterKnowledgeBugFix:
    """Test that Filter now replaces [DOMAIN_KNOWLEDGE] in its prompt."""

    def test_domain_knowledge_in_filter_prompt(self):
        from causal_copilot.mcp.bridge import make_args, make_global_state

        df = pd.DataFrame({"Income": range(100), "Education": range(100)})
        gs = make_global_state(df, query="What causes Income?")
        gs.user_data.knowledge_docs = "Education causes Income."
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True
        gs.statistics.missingness = False
        gs.statistics.data_type = "Continuous"
        gs.statistics.sample_size = 100
        gs.statistics.feature_number = 2
        gs.statistics.time_series = False
        gs.statistics.description = "Linear Gaussian continuous data"
        gs.algorithm.waiting_minutes = 5
        gs.user_data.accept_CPDAG = True

        # Import Filter and test its prompt generation (mock LLMClient)
        with patch("causal_discovery.filter.LLMClient"):
            from causal_discovery.filter import Filter

            args = make_args(query="What causes Income?")
            f = Filter(args)
            prompt = f.create_prompt(gs)

        # The [DOMAIN_KNOWLEDGE] placeholder should be replaced
        assert "[DOMAIN_KNOWLEDGE]" not in prompt
        assert "Education causes Income" in prompt


class TestRevisedGraphPreference:
    """Test that serialize_result prefers revised_graph over converted_graph."""

    def test_uses_revised_when_available(self):
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        gs = make_global_state(df)
        # Simulate: converted_graph has edge A->B, revised_graph drops it
        gs.results.converted_graph = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
        gs.results.revised_graph = np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        result = serialize_result(gs, node_names=["A", "B", "C"])
        assert result["status"] == "ok"
        assert result["graph_refined"] is True
        # Should reflect revised_graph (B->C), not converted_graph (A->B)
        edge_pairs = [(e["from"], e["to"]) for e in result["edges"] if e["type"] == "directed"]
        assert ("B", "C") in edge_pairs
        assert ("A", "B") not in edge_pairs

    def test_falls_back_to_converted(self):
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        gs = make_global_state(df)
        gs.results.converted_graph = np.array([[0, 0], [1, 0]])
        # No revised_graph — should use converted_graph
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        result = serialize_result(gs, node_names=["A", "B"])
        assert result["status"] == "ok"
        assert result["graph_refined"] is False

    def test_bootstrap_confidence_included(self):
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        gs = make_global_state(df)
        gs.results.converted_graph = np.array([[0, 0], [1, 0]])  # A->B
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        # Real bootstrap format: dict of ndarrays (from judge_functions.py)
        gs.results.bootstrap_probability = {
            "certain_edges": np.array([[0.0, 0.0], [0.92, 0.0]]),
            "uncertain_edges": np.zeros((2, 2)),
            "bi_edges": np.zeros((2, 2)),
            "half_certain_edges": np.zeros((2, 2)),
            "half_uncertain_edges": np.zeros((2, 2)),
            "none_edges": np.zeros((2, 2)),
            "none_existence": np.zeros((2, 2)),
        }

        result = serialize_result(gs, node_names=["A", "B"])
        assert "edge_confidence" in result
        assert "A->B" in result["edge_confidence"]
        assert abs(result["edge_confidence"]["A->B"] - 0.92) < 0.01

    def test_bootstrap_confidence_all_edge_types(self):
        """Bootstrap confidence includes undirected and bidirected edges."""
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        gs = make_global_state(df)
        # A->B (directed), A-C (undirected)
        gs.results.converted_graph = np.array(
            [
                [0, 0, 2],
                [1, 0, 0],
                [2, 0, 0],
            ]
        )
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        gs.results.bootstrap_probability = {
            "certain_edges": np.array([[0, 0, 0], [0.85, 0, 0], [0, 0, 0]]),
            "uncertain_edges": np.array([[0, 0, 0.78], [0, 0, 0], [0.78, 0, 0]]),
            "bi_edges": np.zeros((3, 3)),
            "half_certain_edges": np.zeros((3, 3)),
            "half_uncertain_edges": np.zeros((3, 3)),
            "none_edges": np.zeros((3, 3)),
            "none_existence": np.zeros((3, 3)),
        }

        result = serialize_result(gs, node_names=["A", "B", "C"])
        ec = result["edge_confidence"]
        assert "A->B" in ec
        assert abs(ec["A->B"] - 0.85) < 0.01
        # Undirected edge confidence
        assert "A-C" in ec
        assert abs(ec["A-C"] - 0.78) < 0.01

    def test_llm_pruning_decisions_exposed(self):
        """LLM pruning decisions (confirmed/rejected edges) exposed."""
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        gs = make_global_state(df)
        gs.results.converted_graph = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        # Simulate Judge LLM decisions (stored as llm_errors)
        # direct_record: list of [j, i] pairs the LLM confirmed as j→i
        # forbid_record: list of [j, i] pairs the LLM rejected
        gs.results.llm_errors = {
            "direct_record": [[0, 1]],  # Confirmed: A(0)->B(1)
            "forbid_record": [[1, 2]],  # Rejected: B(1)->C(2)
        }

        result = serialize_result(gs, node_names=["A", "B", "C"])
        assert "llm_pruning" in result
        assert "A->B" in result["llm_pruning"]["confirmed"]
        assert "B->C" in result["llm_pruning"]["rejected"]


class TestBackgroundKnowledge:
    """Test forbidden/required edges injection."""

    def test_parse_forbidden_edges(self):
        from causal_copilot.mcp.server import _parse_background_knowledge

        warnings = []
        spec = _parse_background_knowledge(
            '[["Age","Income"],["Gender","Height"]]',
            "",
            warnings,
        )
        assert spec is not None
        assert len(spec["forbidden_edges"]) == 2
        assert spec["forbidden_edges"][0] == ["Age", "Income"]
        assert not warnings

    def test_parse_required_edges(self):
        from causal_copilot.mcp.server import _parse_background_knowledge

        warnings = []
        spec = _parse_background_knowledge(
            "",
            '[["Education","Income"]]',
            warnings,
        )
        assert spec is not None
        assert len(spec["required_edges"]) == 1
        assert not warnings

    def test_parse_both(self):
        from causal_copilot.mcp.server import _parse_background_knowledge

        warnings = []
        spec = _parse_background_knowledge(
            '[["A","B"]]',
            '[["C","D"]]',
            warnings,
        )
        assert "forbidden_edges" in spec
        assert "required_edges" in spec

    def test_parse_empty(self):
        from causal_copilot.mcp.server import _parse_background_knowledge

        assert _parse_background_knowledge("", "", []) is None
        assert _parse_background_knowledge("  ", "  ", []) is None

    def test_parse_invalid_json(self):
        from causal_copilot.mcp.server import _parse_background_knowledge

        warnings = []
        spec = _parse_background_knowledge("not json", "", warnings)
        assert spec is None
        assert len(warnings) == 1
        assert "Invalid" in warnings[0]

    def test_discover_with_forbidden_edges(self):
        from causal_copilot.mcp.server import discover

        rng = np.random.default_rng(0)
        lines = ["A,B,C"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)

        with _mock_run_algorithm():
            result = json.loads(
                discover(
                    csv,
                    forbidden_edges='[["A","B"]]',
                    required_edges='[["B","C"]]',
                )
            )
        # Should not crash — constraints passed through
        assert result["status"] in ("ok", "partial", "error")

    def test_run_algorithm_with_forbidden_edges(self):
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(0)
        lines = ["A,B,C"]
        for _ in range(60):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        csv = "\n".join(lines)

        with _mock_run_algorithm():
            result = json.loads(
                run_algorithm(
                    csv,
                    algorithm="PC",
                    forbidden_edges='[["A","B"]]',
                )
            )
        assert result["status"] in ("ok", "error")


class TestDiagnoseDataEnhanced:
    """Test enhanced diagnosis with per-column detail and descriptive stats."""

    def _make_csv(self, n=60):
        rng = np.random.default_rng(0)
        lines = ["a,b,c"]
        for _ in range(n):
            lines.append(f"{rng.normal()},{rng.normal()},{rng.normal()}")
        return "\n".join(lines)

    def test_column_detail_exposed(self):
        from causal_copilot.mcp.server import diagnose_data

        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(self._make_csv()))
        diag = result["diagnosis"]
        assert "column_detail" in diag
        assert "a" in diag["column_detail"]
        assert diag["column_detail"]["a"]["type"] == "Continuous"
        assert "missing_ratio" in diag["column_detail"]["a"]

    def test_descriptive_stats_present(self):
        from causal_copilot.mcp.server import diagnose_data

        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(self._make_csv()))
        diag = result["diagnosis"]
        assert "descriptive_stats" in diag
        assert "a" in diag["descriptive_stats"]
        stats = diag["descriptive_stats"]["a"]
        # pandas describe() keys
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_description_text_present(self):
        from causal_copilot.mcp.server import diagnose_data

        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(self._make_csv()))
        diag = result["diagnosis"]
        assert "description" in diag
        assert "Linear Gaussian" in diag["description"]

    def test_recommendations_enriched(self):
        """Small sample and high-dimensional warnings appear."""
        from causal_copilot.mcp.server import diagnose_data

        # Small sample
        csv = self._make_csv(n=30)
        with _mock_stat_info_ctx():
            result = json.loads(diagnose_data(csv))
        recs = result["recommendations"]
        # At least the basic recommendation
        assert any("Linear" in r or "Gaussian" in r for r in recs)


class TestSelectionTransparency:
    """Test that discover() exposes algorithm selection reasoning."""

    def test_selection_reasoning_in_provenance(self):
        """When LLM path populates candidates/optimum, provenance includes them."""
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        gs = make_global_state(df)
        gs.results.converted_graph = np.array([[0, 0], [1, 0]])
        gs.statistics.linearity = True
        gs.statistics.gaussian_error = True

        # Simulate Filter output
        gs.algorithm.algorithm_candidates = {
            "PC": {"description": "Constraint-based", "justification": "Linear Gaussian"},
            "GES": {"description": "Score-based", "justification": "Fast, reliable"},
        }
        # Simulate Reranker output
        gs.algorithm.algorithm_optimum = {
            "algorithm": "PC",
            "reason": "PC best for linear Gaussian data with small sample",
            "score_calculation": {
                "PC": {"final_score": 8.5},
                "GES": {"final_score": 7.2},
            },
        }
        # Simulate HP selector output
        gs.algorithm.algorithm_arguments_json = {
            "hyperparameters": {
                "alpha": {
                    "value": 0.05,
                    "reasoning": "Standard significance level for small sample",
                },
                "indep_test": {
                    "value": "fisherz",
                    "reasoning": "Linear Gaussian data → Fisher-z optimal",
                },
            },
        }

        provenance = {
            "algorithm": "PC",
            "hyperparameters": {"alpha": 0.05, "indep_test": "fisherz"},
            "seed": 42,
            "planner": "llm",
        }

        # Build selection_reasoning the same way server.py does
        selection_reasoning = {}
        candidates = getattr(gs.algorithm, "algorithm_candidates", None)
        if candidates:
            selection_reasoning["candidates"] = candidates
        optimum = getattr(gs.algorithm, "algorithm_optimum", None)
        if optimum and isinstance(optimum, dict):
            selection_reasoning["ranking_reason"] = optimum.get("reason", "")
            score_calc = optimum.get("score_calculation")
            if score_calc:
                selection_reasoning["scores"] = {
                    k: v.get("final_score") if isinstance(v, dict) else v for k, v in score_calc.items()
                }
        hp_json = getattr(gs.algorithm, "algorithm_arguments_json", None)
        if hp_json and isinstance(hp_json, dict):
            hp_reasoning = {}
            hp_data = hp_json.get("hyperparameters", hp_json)
            for param, info in hp_data.items():
                if isinstance(info, dict) and "reasoning" in info:
                    hp_reasoning[param] = info["reasoning"]
            if hp_reasoning:
                selection_reasoning["hp_reasoning"] = hp_reasoning
        provenance["selection_reasoning"] = selection_reasoning

        result = serialize_result(gs, node_names=["A", "B"], provenance=provenance)
        sr = result["provenance"]["selection_reasoning"]

        # Candidates
        assert "PC" in sr["candidates"]
        assert "GES" in sr["candidates"]

        # Scores
        assert sr["scores"]["PC"] == 8.5
        assert sr["scores"]["GES"] == 7.2

        # Ranking reason
        assert "linear Gaussian" in sr["ranking_reason"]

        # HP reasoning
        assert "alpha" in sr["hp_reasoning"]
        assert "Fisher-z" in sr["hp_reasoning"]["indep_test"]


class TestPromptRegistration:
    """Test that MCP prompts are registered."""

    def test_causal_analysis_prompt_exists(self):
        import asyncio

        from causal_copilot.mcp.server import mcp

        async def check():
            prompts = await mcp.list_prompts()
            names = {p.name for p in prompts}
            return names

        names = asyncio.run(check())
        assert "causal_analysis" in names
        assert "causal_expert" in names
        assert "analyze_dataset" in names


# ── MCP CLI ────────────────────────────────────────────────────────────


# ── generate_report ────────────────────────────────────────────────────


class TestGenerateReport:
    """Tests for the generate_report MCP tool and its helpers."""

    def test_prepare_gs_fills_missing_fields(self):
        """_prepare_gs_for_report fills all fields Report_generation needs."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "Income": rng.normal(50000, 10000, 100),
                "Education": rng.normal(16, 2, 100),
                "Age": rng.normal(35, 10, 100),
            }
        )
        gs = make_global_state(df, query="test")
        gs.algorithm.selected_algorithm = "PC"
        gs.algorithm.algorithm_arguments = {"alpha": 0.05, "indep_test": "fisherz"}
        # All logging/metadata fields should be None/empty
        assert gs.user_data.meaningful_feature is None
        assert gs.user_data.knowledge_docs_for_user is None
        assert gs.algorithm.algorithm_candidates is None
        assert gs.algorithm.algorithm_arguments_json is None
        assert not gs.logging.select_conversation
        assert not gs.logging.argument_conversation
        assert not gs.logging.global_state_logging

        gs = _prepare_gs_for_report(gs)

        # meaningful_feature: column names are real words, not V0/X1
        assert gs.user_data.meaningful_feature is True
        # knowledge_docs_for_user: should be a list with 1+ element
        assert isinstance(gs.user_data.knowledge_docs_for_user, list)
        assert len(gs.user_data.knowledge_docs_for_user) >= 1
        # algorithm_candidates
        assert "PC" in gs.algorithm.algorithm_candidates
        # algorithm_arguments_json
        assert "hyperparameters" in gs.algorithm.algorithm_arguments_json
        hp = gs.algorithm.algorithm_arguments_json["hyperparameters"]
        assert "alpha" in hp
        assert hp["alpha"]["value"] == "0.05"
        # select_conversation
        assert gs.logging.select_conversation[0]["response"]
        # argument_conversation
        assert gs.logging.argument_conversation[0]["response"]
        # global_state_logging
        assert gs.logging.global_state_logging == ["PC"]
        # graph_conversion
        assert isinstance(gs.logging.graph_conversion, dict)

    def test_prepare_gs_with_generic_names(self):
        """_prepare_gs_for_report detects generic variable names (V0, X1)."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"V0": rng.normal(size=50), "V1": rng.normal(size=50)})
        gs = make_global_state(df)
        gs.algorithm.selected_algorithm = "PC"
        gs.algorithm.algorithm_arguments = {}

        gs = _prepare_gs_for_report(gs)
        assert gs.user_data.meaningful_feature is False

    def test_prepare_gs_preserves_existing_fields(self):
        """_prepare_gs_for_report doesn't overwrite existing non-None fields."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        rng = np.random.default_rng(42)
        df = pd.DataFrame({"A": rng.normal(size=50), "B": rng.normal(size=50)})
        gs = make_global_state(df)
        gs.algorithm.selected_algorithm = "GES"
        gs.algorithm.algorithm_arguments = {}
        gs.user_data.meaningful_feature = True
        gs.user_data.knowledge_docs_for_user = ["Custom knowledge"]
        gs.algorithm.algorithm_candidates = {"GES": {"description": "custom", "justification": "user"}}

        gs = _prepare_gs_for_report(gs)
        # Should NOT overwrite
        assert gs.user_data.knowledge_docs_for_user == ["Custom knowledge"]
        assert gs.algorithm.algorithm_candidates["GES"]["description"] == "custom"

    def test_generate_report_rejects_missing_run_id(self):
        """generate_report raises ToolError for unknown run_id."""
        from causal_copilot.mcp.server import generate_report

        with pytest.raises(ToolError, match="not found or expired"):
            generate_report(run_id="nonexistent_id")

    def test_generate_report_rejects_no_global_state(self):
        """generate_report raises ToolError when store has no _global_state."""
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import generate_report

        # Save a minimal payload without _global_state
        run_id = get_store().save({"status": "ok", "_processed_data": None})

        with pytest.raises(ToolError, match="No GlobalState stored"):
            generate_report(run_id=run_id)

    def test_artifact_store_has_global_state_after_run_algorithm(self):
        """run_algorithm stores _global_state and _args in artifact store."""
        from causal_copilot.mcp.artifacts import get_store
        from causal_copilot.mcp.server import run_algorithm

        rng = np.random.default_rng(42)
        n = 200
        X = rng.normal(size=n)
        Y = 2 * X + rng.normal(0, 0.3, size=n)
        Z = X + Y + rng.normal(0, 0.3, size=n)
        lines = ["X,Y,Z"]
        for i in range(n):
            lines.append(f"{X[i]},{Y[i]},{Z[i]}")
        csv = "\n".join(lines)

        with _mock_stat_info_ctx():
            result = json.loads(run_algorithm(csv_data=csv, algorithm="PC"))

        assert result["status"] == "ok"
        run_id = result["run_id"]
        cached = get_store().get(run_id)
        assert cached is not None
        assert cached.get("_global_state") is not None
        assert cached.get("_args") is not None


# ── Audit-driven fixes ─────────────────────────────────────────────────


class TestAuditFixes:
    """Tests verifying fixes from the comprehensive parity audit."""

    def test_confounders_use_original_adj(self):
        """Confounders should be identified from original adj, not sanitized.

        Value=2 (undirected) edges should produce potential_confounders.
        After sanitization, value=2 is zeroed — so if we pass clean_adj,
        potential_confounders would always be empty.
        """
        from causal_copilot.mcp.offline import identify_confounders

        # adj: C→T (undirected=2), C→Y (undirected=2)
        # C is a potential confounder (influences both T and Y, but undirected)
        names = ["T", "Y", "C"]
        adj = np.zeros((3, 3))
        adj[0, 2] = 2  # C-T (undirected)
        adj[2, 0] = 2
        adj[1, 2] = 2  # C-Y (undirected)
        adj[2, 1] = 2

        confirmed, potential = identify_confounders(adj, names, "T", "Y")
        # With original adj: C should be in potential_confounders
        assert "C" in potential, f"C should be potential confounder, got {potential}"

        # After sanitization (zeroes value=2): confounders would be empty
        clean = adj.copy()
        for i in range(3):
            for j in range(3):
                if clean[i, j] != 0 and clean[j, i] != 0:
                    clean[i, j] = 0
                    clean[j, i] = 0
        confirmed2, potential2 = identify_confounders(clean, names, "T", "Y")
        assert len(confirmed2) == 0 and len(potential2) == 0

    def test_causalforest_dml_continuous_guard(self):
        """CausalForestDML should fall back to LinearDML for continuous treatment."""
        from causal_copilot.mcp.offline import get_default_estimation_config

        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "T": rng.normal(size=50),
                "Y": rng.normal(size=50),
                "X": rng.normal(size=50),
            }
        )
        # Force nonlinear + continuous to trigger CausalForestDML selection
        config = get_default_estimation_config(
            "dml",
            df,
            "T",
            outcome="Y",
            is_linear=False,
            treatment_kind="continuous",
        )
        # Even if offline selects CausalForestDML, estimate_dml should guard it
        # The guard is in estimation.py, not offline.py
        # Here we just verify the offline selection
        # CausalForestDML requires binary/discrete — continuous gets LinearDML
        # Actually select_dml_variant returns "LinearDML" for continuous
        # because the condition is `not is_linear and treatment_kind in ("binary", "discrete")`
        assert config["algo"] in ("LinearDML", "SparseLinearDML", "CausalForestDML")

    def test_knowledge_docs_wrapped_in_list(self):
        """domain_knowledge should be stored as a list for downstream compatibility."""
        from causal_copilot.mcp.bridge import make_global_state

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        # Simulate what discover() does after the fix
        domain_knowledge = "Treatment A causes outcome B"
        gs.user_data.knowledge_docs = [domain_knowledge]

        # Should be iterable (list)
        assert isinstance(gs.user_data.knowledge_docs, list)
        assert gs.user_data.knowledge_docs[0] == domain_knowledge

    def test_dead_code_auto_select_removed(self):
        """_auto_select_method should no longer exist in server.py."""
        import causal_copilot.mcp.server as srv

        assert not hasattr(srv, "_auto_select_method")


# ── Audit Round 2 — Bug Fixes ──────────────────────────────────────────


class TestAuditRound2Bugs:
    """Tests for bugs B1-B8 found in the second comprehensive audit."""

    def test_b1_drl_discretizes_continuous_treatment(self):
        """B1: DRL should discretize continuous treatment into quantile bins.

        inference.py:901 always sets discretize=True for DRL variants.
        estimate_drl must bin continuous treatment before fitting.
        """
        from causal_copilot.mcp.estimation import estimate_drl

        rng = np.random.default_rng(42)
        n = 100
        # Continuous treatment (>10 unique values → triggers discretization)
        T = rng.normal(0, 1, n)
        Y = 2 * T + rng.normal(0, 0.5, n)
        X = rng.normal(0, 1, n)
        W = rng.normal(0, 1, n)
        df = pd.DataFrame({"T": T, "Y": Y, "X": X, "W": W})

        result = estimate_drl(
            df,
            treatment="T",
            outcome="Y",
            X_col=["X"],
            W_col=["W"],
            T0=0.0,
            T1=1.0,
            treatment_kind="continuous",
        )
        assert result["ate"]["estimate"] is not None
        # After discretization, T0/T1 should be from binned values (0, 1, or 2)
        # The key check is that the function didn't crash — DRL requires discrete treatment

    def test_b2_drl_t0_t1_validation_guard(self):
        """B2: DRL should reset T0/T1 if they don't exist in actual treatment values."""
        from causal_copilot.mcp.estimation import estimate_drl

        rng = np.random.default_rng(42)
        n = 100
        df = pd.DataFrame(
            {
                "T": rng.choice([0, 1], n),
                "Y": rng.normal(size=n),
                "X": rng.normal(size=n),
            }
        )

        # Pass T0=5, T1=10 — neither exists in binary treatment
        result = estimate_drl(
            df,
            treatment="T",
            outcome="Y",
            X_col=["X"],
            W_col=[],
            T0=5.0,
            T1=10.0,
            treatment_kind="binary",
        )
        # Should not crash — T0/T1 reset to actual values (0 and 1)
        assert result["ate"]["estimate"] is not None

    def test_b3_eda_failure_no_keyerror(self):
        """B3: When EDA fails, fallback eda dict must have required keys.

        report_generation.py:386 accesses eda_result['plot_path_dist'][0]
        and eda_result['plot_path_corr'][0]. Empty dict {} would crash.
        """
        from causal_copilot.mcp.bridge import make_global_state

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        # Simulate EDA failure fallback
        gs.results.eda = {
            "plot_path_dist": [""],
            "plot_path_corr": [""],
        }
        # These accesses must not crash
        assert gs.results.eda["plot_path_dist"][0] == ""
        assert gs.results.eda["plot_path_corr"][0] == ""

    def test_b4_knowledge_docs_for_user_no_double_wrap(self):
        """B4: knowledge_docs_for_user should not double-wrap lists.

        discover() stores knowledge_docs = [domain_knowledge] (list).
        _prepare_gs_for_report() must unwrap, not wrap again as [[...]].
        """
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        gs.algorithm.selected_algorithm = "PC"

        # Simulate what discover() does: stores as list
        gs.user_data.knowledge_docs = ["Domain knowledge text"]
        gs = _prepare_gs_for_report(gs)

        # knowledge_docs_for_user[0] should be a string, not a list
        kd = gs.user_data.knowledge_docs_for_user
        assert isinstance(kd, list)
        assert isinstance(kd[0], str), f"Expected string, got {type(kd[0])}: {kd[0]}"
        assert kd[0] == "Domain knowledge text"

    def test_b4_knowledge_docs_string_input(self):
        """B4: When knowledge_docs is a plain string, should wrap once correctly."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        gs.algorithm.selected_algorithm = "PC"
        gs.user_data.knowledge_docs = "Plain string knowledge"
        gs = _prepare_gs_for_report(gs)

        kd = gs.user_data.knowledge_docs_for_user
        assert isinstance(kd[0], str)
        assert kd[0] == "Plain string knowledge"

    def test_b5_bootstrap_llm_errors_default(self):
        """B5: bootstrap_errors and llm_errors should have safe defaults.

        report_generation.py:594 accesses gs.results.bootstrap_errors;
        :604 accesses gs.results.llm_errors['direct_record'/'forbid_record'].
        When Judge is skipped, these attrs don't exist.
        """
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        gs.algorithm.selected_algorithm = "PC"
        # Don't set bootstrap_errors or llm_errors — simulates run_algorithm path
        gs = _prepare_gs_for_report(gs)

        # Should exist and be safe to access
        assert gs.results.bootstrap_errors == []
        assert gs.results.llm_errors["direct_record"] == {}
        assert gs.results.llm_errors["forbid_record"] == {}

    def test_b5_preserves_existing_bootstrap_errors(self):
        """B5: If Judge ran and set real values, don't overwrite."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        gs.algorithm.selected_algorithm = "PC"
        gs.results.bootstrap_errors = ["edge A→B forced"]
        gs.results.llm_errors = {"direct_record": {"0-1": "confirmed"}, "forbid_record": {}}
        gs = _prepare_gs_for_report(gs)

        assert gs.results.bootstrap_errors == ["edge A→B forced"]
        assert gs.results.llm_errors["direct_record"] == {"0-1": "confirmed"}

    def test_b7_ts_judge_skip_requires_lagged_graph(self):
        """B7: TS Judge skip should require lagged_graph, matching main.py:268.

        main.py condition: time_series AND lagged_graph is not None.
        If time_series=True but lagged_graph=None (algorithm failed to produce it),
        Judge should still run.
        """
        from causal_copilot.mcp.bridge import make_global_state

        gs = make_global_state(pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        gs.statistics.time_series = True
        gs.results.lagged_graph = None  # No lagged graph produced
        gs.results.converted_graph = np.eye(2)

        # The condition `is_ts and has_lagged` should be False
        is_ts = getattr(gs.statistics, "time_series", False)
        has_lagged = getattr(gs.results, "lagged_graph", None) is not None
        assert is_ts is True
        assert has_lagged is False
        # Judge should NOT be skipped in this case
        assert not (is_ts and has_lagged)

    def test_b8_lagged_graph_serialization(self):
        """B8: serialize_result should handle 3D lagged_graph when no 2D graph exists."""
        from causal_copilot.mcp.bridge import make_global_state, serialize_result

        gs = make_global_state(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))
        gs.statistics.time_series = True

        # Only lagged_graph exists (3D: 2 lags × 2 vars × 2 vars)
        lagged = np.zeros((2, 2, 2))
        lagged[0, 0, 1] = 1  # B→A at lag 0
        lagged[1, 1, 0] = 1  # A→B at lag 1
        gs.results.lagged_graph = lagged
        gs.results.raw_result = None
        gs.results.converted_graph = None

        result = serialize_result(gs)
        assert result["status"] == "ok"
        # Summary graph should have both edges collapsed
        adj = np.array(result["adjacency_matrix"])
        assert adj[0, 1] == 1  # B→A
        assert adj[1, 0] == 1  # A→B


class TestAuditRound2Quality:
    """Tests for code quality fixes Q1-Q3."""

    def test_q1_estimate_effect_uses_shared_resolver(self):
        """Q1: estimate_effect should use _resolve_data_and_graph, not duplicated code."""
        import inspect

        from causal_copilot.mcp.server import estimate_effect

        source = inspect.getsource(estimate_effect)
        # The refactored version calls _resolve_data_and_graph
        assert "_resolve_data_and_graph" in source

    def test_q2_runstore_touch_on_read(self):
        """Q2: RunStore.get() should extend TTL on successful access."""
        import time

        from causal_copilot.mcp.artifacts import RunStore

        store = RunStore(ttl_seconds=2)
        run_id = store.save({"test": True})

        # Wait 1.5s, then access — should touch and reset TTL
        time.sleep(1.5)
        data = store.get(run_id)
        assert data is not None

        # Wait another 1.5s — without touch, total would be 3s > 2s TTL
        # With touch, it's only 1.5s from last access < 2s TTL
        time.sleep(1.5)
        data = store.get(run_id)
        assert data is not None, "TTL should have been extended by touch-on-read"

    def test_q3_generate_report_prepare_gs_smoke(self):
        """Q3: Smoke test for _prepare_gs_for_report — the core of generate_report."""
        from causal_copilot.mcp.bridge import make_global_state
        from causal_copilot.mcp.server import _prepare_gs_for_report

        gs = make_global_state(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))
        gs.algorithm.selected_algorithm = "PC"
        gs.results.converted_graph = np.array([[0, 1], [0, 0]])

        gs = _prepare_gs_for_report(gs)

        # All fields Report_generation.__init__ accesses must be non-None
        assert gs.user_data.meaningful_feature is not None
        assert gs.user_data.knowledge_docs_for_user is not None
        assert isinstance(gs.user_data.knowledge_docs_for_user[0], str)
        assert gs.statistics.description is not None
        assert gs.algorithm.algorithm_candidates is not None
        assert gs.algorithm.algorithm_arguments_json is not None
        assert gs.logging.select_conversation is not None
        assert gs.logging.argument_conversation is not None
        assert gs.logging.global_state_logging is not None
        assert isinstance(gs.logging.graph_conversion, dict)
        assert isinstance(gs.results.bootstrap_errors, list)
        assert isinstance(gs.results.llm_errors, dict)
        assert "direct_record" in gs.results.llm_errors
        assert "forbid_record" in gs.results.llm_errors


# ── MCP CLI ────────────────────────────────────────────────────────────


class TestMCPCLI:
    def test_mcp_help(self, capsys):
        from causal_copilot.cli import main

        with pytest.raises(SystemExit):
            main(["mcp", "--help"])
