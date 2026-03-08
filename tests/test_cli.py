"""Tests for the CLI commands."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from causal_copilot.cli import main


class TestVersion:
    def test_version_output(self, capsys):
        main(["version"])
        out = capsys.readouterr().out
        assert "causal-copilot" in out
        assert "0.1" in out


class TestDoctor:
    def test_doctor_runs(self, capsys):
        main(["doctor"])
        out = capsys.readouterr().out
        assert "Causal-Copilot Doctor" in out
        assert "numpy" in out
        assert "Platform" in out


class TestAnalyze:
    def test_analyze_file_not_found(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", "/nonexistent.csv"])
        assert exc_info.value.code == 1

    def test_analyze_csv(self, tmp_path, capsys):
        # Create test CSV
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["analyze", str(csv_path)])
        out = capsys.readouterr().out
        assert "Provenance" in out or "directed" in out

    def test_analyze_json_output(self, tmp_path):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        out_path = tmp_path / "result.json"

        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["analyze", str(csv_path), "-o", str(out_path)])

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["status"] == "ok"

    def test_analyze_with_algorithm_flag(self, tmp_path, capsys):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["analyze", str(csv_path), "-a", "PC"])
        out = capsys.readouterr().out
        assert "PC" in out

    def test_analyze_with_seed(self, tmp_path, capsys):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["analyze", str(csv_path), "-s", "99"])
        out = capsys.readouterr().out
        assert "seed=99" in out

    def test_analyze_bad_planner(self, tmp_path, capsys):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(SystemExit) as exc_info:
            main(["analyze", str(csv_path), "-p", "bogus"])
        assert exc_info.value.code == 1


class TestQuickstart:
    def test_quickstart_runs(self, capsys):
        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["quickstart"])
        out = capsys.readouterr().out
        assert "Quickstart" in out
        assert "X → Y → Z" in out

    def test_quickstart_json_output(self, tmp_path):
        out_path = tmp_path / "quickstart.json"
        with _mock_algorithm(), pytest.raises(SystemExit, match="0"):
            main(["quickstart", "-o", str(out_path)])
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["status"] == "ok"


class TestBenchmarkCLI:
    def test_benchmark_help(self, capsys):
        """Benchmark subcommand should exist."""
        with pytest.raises(SystemExit):
            main(["benchmark", "--help"])
        out = capsys.readouterr().out
        assert "benchmark" in out.lower() or "algorithm" in out.lower()

    @pytest.mark.slow
    def test_benchmark_single(self, tmp_path):
        """Run benchmark on one algo + one scenario."""
        out_file = tmp_path / "result.json"
        main(["benchmark", "--algorithm", "PC", "--scenario", "linear_chain",
              "--output", str(out_file)])
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert len(data) == 1
        assert data[0]["algorithm"] == "PC"
        assert "metrics" in data[0]


class TestNoCommand:
    def test_no_command_exits(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockAlgo:
    def default_params(self):
        return {"mock_param": True}

    def fit(self, data, **kwargs):
        n = data.shape[1]
        return np.zeros((n, n)), {"mock": True}, None


class _mock_algorithm:
    """Patches _load_algorithm in copilot module."""

    def __enter__(self):
        self._patcher = patch(
            "causal_copilot.copilot._load_algorithm",
            return_value=_MockAlgo(),
        )
        self._patcher.start()
        return self

    def __exit__(self, *args):
        self._patcher.stop()
