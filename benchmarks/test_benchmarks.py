"""Benchmark tests — validate scenarios and evaluation metrics.

Run with: pytest benchmarks/ -v
"""

import numpy as np
import pytest

from benchmarks.evaluate import evaluate_adjacency
from benchmarks.scenarios import ALL_SCENARIOS, Scenario

# ---------------------------------------------------------------------------
# Scenario smoke tests
# ---------------------------------------------------------------------------


class TestScenarios:
    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_scenario_structure(self, name):
        s = ALL_SCENARIOS[name]
        assert isinstance(s, Scenario)
        assert s.data.shape[0] >= 100
        assert s.data.shape[1] >= 2
        assert s.ground_truth.shape == (s.data.shape[1], s.data.shape[1])
        assert list(s.data.columns) == s.columns

    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_scenario_deterministic(self, name):
        """Same scenario generated twice must produce identical data."""
        s1 = ALL_SCENARIOS[name]
        # Re-import to regenerate
        from benchmarks.scenarios import _make_scenario, collider, diamond, fork, linear_chain, sparse_10

        gen_fns = {
            "linear_chain": linear_chain,
            "fork": fork,
            "collider": collider,
            "diamond": diamond,
            "sparse_10": sparse_10,
        }
        s2 = _make_scenario(name, s1.description, gen_fns[name])
        np.testing.assert_array_equal(s1.data.values, s2.data.values)
        np.testing.assert_array_equal(s1.ground_truth, s2.ground_truth)

    @pytest.mark.parametrize("name", list(ALL_SCENARIOS.keys()))
    def test_ground_truth_is_dag(self, name):
        """Ground truth adjacency must have zero diagonal (no self-loops)."""
        gt = ALL_SCENARIOS[name].ground_truth
        assert np.all(np.diag(gt) == 0)

    def test_has_5_scenarios(self):
        assert len(ALL_SCENARIOS) == 5


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_perfect_prediction(self):
        gt = np.array([[0, 1], [0, 0]])
        m = evaluate_adjacency(gt, gt)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.shd == 0

    def test_empty_prediction(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.zeros_like(gt)
        m = evaluate_adjacency(pred, gt)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.false_negatives == 1

    def test_all_edges_predicted(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.ones_like(gt)
        np.fill_diagonal(pred, 0)
        m = evaluate_adjacency(pred, gt)
        assert m.recall == 1.0
        assert m.false_positives == 1  # extra reverse edge

    def test_reversed_edge(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.true_positives == 0
        assert m.false_positives == 1
        assert m.false_negatives == 1
        assert m.shd == 1  # one reversal = one SHD edit

    def test_handles_edge_types(self):
        """Predicted edge type must match ground truth for a TP.

        pred=2 (undirected) vs gt=1 (directed) is a type mismatch — counted as
        both FP (wrong type) and FN (missing correct type), not a TP.
        """
        gt = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        pred = np.array([[0, 2, 0], [0, 0, 3], [0, 0, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.true_positives == 0  # type mismatch → not TP
        assert m.false_positives == 2  # wrong type
        assert m.false_negatives == 2  # missing correct type
        # But skeleton should be perfect (both have edges in same places)
        assert m.skeleton_precision == 1.0
        assert m.skeleton_recall == 1.0

    def test_undirected_pred_vs_directed_gt_not_perfect(self):
        """Regression: undirected prediction must NOT score as perfect on directed GT.

        This was the P0 bug — binarizing with (pred > 0) collapsed edge types,
        making CPDAG outputs appear perfect against DAG ground truth.
        """
        gt = np.array([[0, 1], [0, 0]])   # X→Y (directed)
        pred = np.array([[0, 2], [0, 0]])  # X—Y (undirected)
        m = evaluate_adjacency(pred, gt)
        assert m.precision < 1.0, "Undirected pred should NOT be perfect against directed GT"
        assert m.true_positives == 0
        assert m.shd > 0

    def test_shd_reversal_counts_as_one(self):
        """A single reversed edge is SHD=1 (not 2)."""
        gt = np.array([[0, 1], [0, 0]])
        pred = np.array([[0, 0], [1, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.shd == 1  # 1 reversed edge = 1 SHD edit


# ---------------------------------------------------------------------------
# Expanded metrics (skeleton, orientation, FDR)
# ---------------------------------------------------------------------------


class TestExpandedMetrics:
    def test_skeleton_metrics(self):
        """Skeleton ignores direction — only checks adjacency."""
        pred = np.array([[0, 0], [1, 0]])  # Y->X
        gt = np.array([[0, 1], [0, 0]])    # X->Y
        m = evaluate_adjacency(pred, gt)
        assert m.skeleton_precision == 1.0
        assert m.skeleton_recall == 1.0
        assert m.orientation_accuracy < 1.0

    def test_orientation_accuracy_detects_undirected(self):
        """Undirected prediction vs directed GT should have orientation_accuracy < 1."""
        pred = np.array([[0, 2], [0, 0]])  # X—Y (undirected)
        gt = np.array([[0, 1], [0, 0]])    # X→Y (directed)
        m = evaluate_adjacency(pred, gt)
        assert m.skeleton_precision == 1.0  # skeleton is correct
        assert m.orientation_accuracy == 0.0  # but orientation is wrong

    def test_fdr(self):
        pred = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
        gt = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        m = evaluate_adjacency(pred, gt)
        assert m.fdr == pytest.approx(0.5)

    def test_empty_pred_metrics(self):
        gt = np.array([[0, 1], [0, 0]])
        pred = np.zeros((2, 2))
        m = evaluate_adjacency(pred, gt)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.fdr == 0.0

    def test_metrics_to_dict(self):
        pred = np.array([[0, 1], [0, 0]])
        gt = np.array([[0, 1], [0, 0]])
        m = evaluate_adjacency(pred, gt)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "shd" in d
        assert "skeleton_f1" in d
        assert "orientation_accuracy" in d


# ---------------------------------------------------------------------------
# Standard benchmark datasets
# ---------------------------------------------------------------------------

from benchmarks.datasets import BENCHMARK_DATASETS, BenchmarkDataset


class TestBenchmarkDatasets:
    def test_has_benchmark_datasets(self):
        assert len(BENCHMARK_DATASETS) >= 3

    @pytest.mark.parametrize("name", ["sachs", "asia", "alarm"])
    def test_dataset_structure(self, name):
        ds = BENCHMARK_DATASETS[name]
        assert isinstance(ds, BenchmarkDataset)
        assert ds.data.shape[0] > 0
        assert ds.data.shape[1] > 1
        assert ds.ground_truth.shape[0] == ds.ground_truth.shape[1]
        assert ds.ground_truth.shape[0] == ds.data.shape[1]
        assert len(ds.columns) == ds.data.shape[1]

    @pytest.mark.parametrize("name", ["sachs", "asia", "alarm"])
    def test_dataset_ground_truth_is_dag(self, name):
        ds = BENCHMARK_DATASETS[name]
        gt = ds.ground_truth
        assert np.all(np.diag(gt) == 0)
        n = gt.shape[0]
        power = np.eye(n)
        for _ in range(n):
            power = power @ (gt > 0).astype(float)
        assert np.all(np.diag(power) == 0), f"DAG check failed for {name}"

    def test_sachs_edge_list(self):
        """Guard the Sachs edge list against accidental edits (paper Fig. 3A)."""
        ds = BENCHMARK_DATASETS["sachs"]
        gt = ds.ground_truth
        col = {c: i for i, c in enumerate(ds.columns)}
        n_edges = int((gt > 0).sum())
        assert n_edges == 17, f"Sachs should have 17 edges, got {n_edges}"
        # Key edges that were disputed — assert presence
        assert gt[col["Akt"], col["PIP3"]] == 1, "PIP3→Akt must be present"
        assert gt[col["Akt"], col["PKA"]] == 1, "PKA→Akt must be present"
        assert gt[col["PKA"], col["PKC"]] == 1, "PKC→PKA must be present"
        # bnlearn version has Akt→Erk; original paper does NOT
        assert gt[col["Erk"], col["Akt"]] == 0, "Akt→Erk must be absent (bnlearn variant, not paper)"

    def test_alarm_description_is_honest(self):
        """ALARM dataset must not claim to be the real ALARM network."""
        ds = BENCHMARK_DATASETS["alarm"]
        desc = ds.description.lower()
        assert "synthetic" in desc, f"ALARM description should contain 'synthetic': {ds.description}"

    def test_dataset_deterministic(self):
        ds1 = BENCHMARK_DATASETS["sachs"]
        ds2 = BENCHMARK_DATASETS["sachs"]
        np.testing.assert_array_equal(ds1.data.values, ds2.data.values)


# ---------------------------------------------------------------------------
# Algorithm capability matrix
# ---------------------------------------------------------------------------

from benchmarks.capability import CAPABILITY_MATRIX, AlgorithmCapability


class TestCapabilityMatrix:
    def test_has_all_registered_algorithms(self):
        from causal_copilot.algorithms.registry import REGISTRY
        for name in REGISTRY:
            assert name in CAPABILITY_MATRIX, f"Missing capability entry for {name}"

    def test_capability_fields(self):
        for name, cap in CAPABILITY_MATRIX.items():
            assert isinstance(cap, AlgorithmCapability)
            assert cap.name == name
            assert cap.output_type in ("dag", "cpdag", "pag")
            assert isinstance(cap.handles_latent_confounders, bool)
            assert isinstance(cap.handles_nonlinear, bool)
            assert isinstance(cap.handles_non_gaussian, bool)
            assert isinstance(cap.handles_time_series, bool)
            assert isinstance(cap.assumptions, list)
            assert len(cap.assumptions) >= 1


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

from benchmarks.runner import run_benchmark, BenchmarkResult


@pytest.mark.slow
class TestBenchmarkRunner:
    def test_run_single_benchmark(self):
        """Run PC on linear_chain scenario."""
        result = run_benchmark(algorithm="PC", scenario_name="linear_chain", seed=42)
        assert isinstance(result, BenchmarkResult)
        assert result.algorithm == "PC"
        assert result.scenario == "linear_chain"
        assert result.status == "ok"
        assert result.output_type == "cpdag"
        assert result.metrics is not None
        assert result.provenance is not None
        assert result.provenance.seed == 42

    def test_result_to_dict(self):
        result = run_benchmark(algorithm="PC", scenario_name="linear_chain", seed=42)
        d = result.to_dict()
        assert "algorithm" in d
        assert "metrics" in d
        assert "provenance" in d
        assert "output_type" in d
        assert "status" in d
        assert "adjacency_matrix" in d

    def test_run_benchmark_deterministic(self):
        r1 = run_benchmark(algorithm="PC", scenario_name="linear_chain", seed=42)
        r2 = run_benchmark(algorithm="PC", scenario_name="linear_chain", seed=42)
        assert r1.metrics.shd == r2.metrics.shd
        assert r1.metrics.f1 == r2.metrics.f1


class TestBenchmarkCompatibility:
    def test_time_series_algo_skipped_on_iid(self):
        """PCMCI/Granger should be skipped on IID scenarios."""
        result = run_benchmark(algorithm="PCMCI", scenario_name="linear_chain")
        assert result.status == "skipped"
        assert result.metrics is None
        assert "time-series" in result.error.lower()

    def test_granger_skipped_on_iid(self):
        result = run_benchmark(algorithm="GrangerCausality", scenario_name="linear_chain")
        assert result.status == "skipped"
        assert result.metrics is None

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            run_benchmark(algorithm="PC", scenario_name="nonexistent")
