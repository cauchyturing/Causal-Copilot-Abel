"""Adapters wrapping vendored backends to CausalDiscoveryBase ABC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.core.base import CausalDiscoveryBase


class PCAdapter(CausalDiscoveryBase):
    """Adapter for the PC algorithm."""

    @property
    def name(self) -> str:
        return "PC"

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 0.05, "indep_test": "fisherz", "stable": True, "depth": 4}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.pc import PCBackend

        backend = PCBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GESAdapter(CausalDiscoveryBase):
    """Adapter for the GES algorithm."""

    @property
    def name(self) -> str:
        return "GES"

    def default_params(self) -> dict[str, Any]:
        return {"score_func": "local_score_BIC"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.ges import GESBackend

        backend = GESBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class NOTEARSLinearAdapter(CausalDiscoveryBase):
    """Adapter for the NOTEARSLinear algorithm."""

    @property
    def name(self) -> str:
        return "NOTEARSLinear"

    def default_params(self) -> dict[str, Any]:
        return {"lambda1": 0.1, "max_iter": 100, "h_tol": 1e-8, "w_threshold": 0.3}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.notears_linear import NOTEARSLinearBackend

        backend = NOTEARSLinearBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class DirectLiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the DirectLiNGAM algorithm."""

    @property
    def name(self) -> str:
        return "DirectLiNGAM"

    def default_params(self) -> dict[str, Any]:
        return {"measure": "pwling"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.direct_lingam import DirectLiNGAMBackend

        backend = DirectLiNGAMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class PCMCIAdapter(CausalDiscoveryBase):
    """Adapter for the PCMCI time-series algorithm."""

    @property
    def name(self) -> str:
        return "PCMCI"

    def default_params(self) -> dict[str, Any]:
        return {"tau_min": 0, "tau_max": 2, "pc_alpha": 0.05, "alpha_level": 0.05}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.pcmci import PCMCIBackend

        backend = PCMCIBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class ICALiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the ICALiNGAM algorithm."""

    @property
    def name(self) -> str:
        return "ICALiNGAM"

    def default_params(self) -> dict[str, Any]:
        return {"max_iter": 1000}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.ica_lingam import ICALiNGAMBackend

        backend = ICALiNGAMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GrangerCausalityAdapter(CausalDiscoveryBase):
    """Adapter for the GrangerCausality algorithm."""

    @property
    def name(self) -> str:
        return "GrangerCausality"

    def default_params(self) -> dict[str, Any]:
        return {"p": 10, "alpha": 0.05, "criterion": "ssr_ftest"}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.granger import GrangerCausalityBackend

        backend = GrangerCausalityBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class FCIAdapter(CausalDiscoveryBase):
    """Adapter for the FCI algorithm (handles latent confounders, outputs PAG)."""

    @property
    def name(self) -> str:
        return "FCI"

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 0.05, "indep_test": "fisherz", "depth": 4, "max_path_length": -1}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.fci import FCIBackend

        backend = FCIBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GRaSPAdapter(CausalDiscoveryBase):
    """Adapter for the GRaSP algorithm."""

    @property
    def name(self) -> str:
        return "GRaSP"

    def default_params(self) -> dict[str, Any]:
        return {"score_func": "local_score_BIC_from_cov", "depth": 4}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.grasp import GRaSPBackend

        backend = GRaSPBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class CDNODAdapter(CausalDiscoveryBase):
    """Adapter for the CDNOD algorithm (multi-domain constraint-based)."""

    @property
    def name(self) -> str:
        return "CDNOD"

    def default_params(self) -> dict[str, Any]:
        return {
            "alpha": 0.05,
            "indep_test": "fisherz",
            "stable": True,
            "uc_rule": 0,
            "uc_priority": 2,
            "depth": 5,
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.cdnod import CDNODBackend

        backend = CDNODBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class VARLiNGAMAdapter(CausalDiscoveryBase):
    """Adapter for the VARLiNGAM time-series algorithm."""

    @property
    def name(self) -> str:
        return "VARLiNGAM"

    def default_params(self) -> dict[str, Any]:
        return {"lags": 10, "criterion": "bic", "prune": True}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.var_lingam import VARLiNGAMBackend

        backend = VARLiNGAMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class CALMAdapter(CausalDiscoveryBase):
    """Adapter for the CALM algorithm (continuous optimization with GPU support)."""

    @property
    def name(self) -> str:
        return "CALM"

    def default_params(self) -> dict[str, Any]:
        return {"lambda1": 0.005, "alpha": 0.01, "tau": 0.5}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.calm import CALMBackend

        backend = CALMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class GOLEMAdapter(CausalDiscoveryBase):
    """Adapter for the GOLEM algorithm."""

    @property
    def name(self) -> str:
        return "GOLEM"

    def default_params(self) -> dict[str, Any]:
        return {
            "lambda_1": 0.01,
            "lambda_2": 5.0,
            "equal_variances": True,
            "learning_rate": 1e-3,
            "num_iter": 10000,
            "graph_thres": 0.3,
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.golem import GOLEMBackend

        backend = GOLEMBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class NOTEARSNonlinearAdapter(CausalDiscoveryBase):
    """Adapter for the NOTEARS Nonlinear (MLP) algorithm."""

    @property
    def name(self) -> str:
        return "NOTEARSNonlinear"

    def default_params(self) -> dict[str, Any]:
        return {
            "lambda1": 0.01,
            "lambda2": 0.01,
            "max_iter": 100,
            "h_tol": 1e-8,
            "w_threshold": 0.3,
            "hidden_layers": (10, 1),
            "model_type": "mlp",
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.notears_nonlinear import NOTEARSNonlinearBackend

        backend = NOTEARSNonlinearBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class CORLAdapter(CausalDiscoveryBase):
    """Adapter for the CORL (RL-based causal discovery) algorithm."""

    @property
    def name(self) -> str:
        return "CORL"

    def default_params(self) -> dict[str, Any]:
        return {
            "batch_size": 64,
            "input_dim": 100,
            "embed_dim": 256,
            "encoder_name": "transformer",
            "decoder_name": "lstm",
            "reward_score_type": "BIC",
            "iteration": 500,
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.corl import CORLBackend

        backend = CORLBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class PCParallelAdapter(CausalDiscoveryBase):
    """Adapter for the PC-Parallel algorithm (gcastle)."""

    @property
    def name(self) -> str:
        return "PCParallel"

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 0.05, "indep_test": "fisherz", "cores": 8}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.pc_parallel import PCParallelBackend

        backend = PCParallelBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class XGESAdapter(CausalDiscoveryBase):
    """Adapter for the XGES (fast GES variant) algorithm."""

    @property
    def name(self) -> str:
        return "XGES"

    def default_params(self) -> dict[str, Any]:
        return {"alpha": 2.0}

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.xges import XGESBackend

        backend = XGESBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class DYNOTEARSAdapter(CausalDiscoveryBase):
    """Adapter for the DYNOTEARS time-series algorithm."""

    @property
    def name(self) -> str:
        return "DYNOTEARS"

    def default_params(self) -> dict[str, Any]:
        return {
            "p": 10,
            "lambda_w": 0.01,
            "lambda_a": 0.01,
            "max_iter": 100,
            "h_tol": 1e-8,
            "w_threshold": 0.05,
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.dynotears import DYNOTEARSBackend

        backend = DYNOTEARSBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


class HybridAdapter(CausalDiscoveryBase):
    """Adapter for the Hybrid (constraint + functional) algorithm."""

    @property
    def name(self) -> str:
        return "Hybrid"

    def default_params(self) -> dict[str, Any]:
        return {
            "first_stage_algo": "pc",
            "second_stage_method": "pnl",
            "alpha": 0.05,
            "m_max": 3,
        }

    def fit(self, data: pd.DataFrame | np.ndarray, **kwargs) -> tuple[np.ndarray, dict[str, Any], Any]:
        from causal_copilot.algorithms._backends.hybrid import HybridBackend

        backend = HybridBackend({**self.default_params(), **self._params})
        df = pd.DataFrame(data) if isinstance(data, np.ndarray) else data
        return backend.fit(df)


# Registry for programmatic access
STABLE_ALGORITHMS = {
    "PC": PCAdapter,
    "GES": GESAdapter,
    "NOTEARSLinear": NOTEARSLinearAdapter,
    "DirectLiNGAM": DirectLiNGAMAdapter,
    "PCMCI": PCMCIAdapter,
    "ICALiNGAM": ICALiNGAMAdapter,
    "GrangerCausality": GrangerCausalityAdapter,
    "FCI": FCIAdapter,
    "GRaSP": GRaSPAdapter,
    "CDNOD": CDNODAdapter,
    "VARLiNGAM": VARLiNGAMAdapter,
    "CALM": CALMAdapter,
    "GOLEM": GOLEMAdapter,
    "NOTEARSNonlinear": NOTEARSNonlinearAdapter,
    "CORL": CORLAdapter,
    "PCParallel": PCParallelAdapter,
    "XGES": XGESAdapter,
    "DYNOTEARS": DYNOTEARSAdapter,
    "Hybrid": HybridAdapter,
}
