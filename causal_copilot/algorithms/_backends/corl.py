"""CORL (Causal discovery via Ordering-based Reinforcement Learning) backend."""

from typing import Any

import numpy as np
import pandas as pd

from causal_copilot.algorithms._backends._base import Backend


class CORLBackend(Backend):
    """CORL using gcastle's CORL (torch implementation)."""

    def fit(self, data: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any], Any]:
        from castle.algorithms.gradient.corl.torch import CORL as TrustAI_CORL

        # Remove domain_index if present
        if isinstance(data, pd.DataFrame) and "domain_index" in data.columns:
            data = data.drop(columns=["domain_index"])

        # Auto-detect device
        device_type = self._params.get("device_type", "auto")
        if device_type == "auto":
            try:
                import torch

                device_type = "gpu" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device_type = "cpu"

        # Build params for CORL constructor
        corl_params = {
            "batch_size": self._params.get("batch_size", 64),
            "input_dim": self._params.get("input_dim", 100),
            "embed_dim": self._params.get("embed_dim", 256),
            "normalize": self._params.get("normalize", False),
            "encoder_name": self._params.get("encoder_name", "transformer"),
            "encoder_heads": self._params.get("encoder_heads", 2),
            "encoder_blocks": self._params.get("encoder_blocks", 3),
            "encoder_dropout_rate": self._params.get("encoder_dropout_rate", 0.1),
            "decoder_name": self._params.get("decoder_name", "lstm"),
            "reward_mode": self._params.get("reward_mode", "episodic"),
            "reward_score_type": self._params.get("reward_score_type", "BIC"),
            "reward_regression_type": self._params.get("reward_regression_type", "LR"),
            "reward_gpr_alpha": self._params.get("reward_gpr_alpha", 1.0),
            "iteration": self._params.get("iteration", 500),
            "lambda_iter_num": self._params.get("lambda_iter_num", 500),
            "actor_lr": self._params.get("actor_lr", 1e-4),
            "critic_lr": self._params.get("critic_lr", 1e-3),
            "alpha": self._params.get("alpha", 0.99),
            "init_baseline": self._params.get("init_baseline", -1.0),
            "random_seed": self._params.get("random_seed", 0),
            "device_type": device_type,
            "device_ids": self._params.get("device_ids", 0),
        }

        model = TrustAI_CORL(**corl_params)
        model.learn(data)

        # gcastle: causal_matrix[i,j] != 0 => i -> j  =>  transpose
        causal_matrix = np.array(model.causal_matrix).T

        info = {
            "device_type": device_type,
            "iterations": corl_params["iteration"],
            "reward_mode": corl_params["reward_mode"],
        }

        return causal_matrix, info, model
