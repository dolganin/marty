from __future__ import annotations

from typing import Any

import numpy as np

from .actions import ACTION_MACROS, macro_action
from .agent import Agent


class RandomAgent(Agent):
    """Uniform random policy over the same macro vocabulary as robust PPO."""

    def __init__(self, seed: int):
        self._rng = np.random.default_rng(seed)

    def reset(self, trial_start: bool) -> None:
        del trial_start

    def act(self, obs: np.ndarray) -> int:
        del obs
        return int(ACTION_MACROS[self._rng.integers(0, len(ACTION_MACROS))])

    def observe(self, reward: float, done: bool, info: dict[str, Any]) -> None:
        del reward, done, info

    def diagnostics(self) -> dict[str, float]:
        return {"action_entropy": float(np.log(len(ACTION_MACROS)))}


class PPOAgent(Agent):
    """Frozen feed-forward SB3 PPO exposed through the public Agent contract."""

    def __init__(self, model: Any, *, deterministic: bool = True):
        self._model = model
        self._deterministic = bool(deterministic)
        self._last_obs: np.ndarray | None = None

    def reset(self, trial_start: bool) -> None:
        del trial_start
        self._last_obs = None

    def act(self, obs: np.ndarray) -> int:
        self._last_obs = np.asarray(obs, dtype=np.float32)
        action, _ = self._model.predict(self._last_obs, deterministic=self._deterministic)
        return macro_action(int(action))

    def observe(self, reward: float, done: bool, info: dict[str, Any]) -> None:
        del reward, done, info

    def diagnostics(self) -> dict[str, float]:
        if self._last_obs is None:
            return {}
        try:
            import torch

            obs = torch.as_tensor(self._last_obs, device=self._model.device).float().unsqueeze(0)
            with torch.no_grad():
                distribution = self._model.policy.get_distribution(obs)
                entropy = distribution.entropy().mean()
            return {"action_entropy": float(entropy.detach().cpu())}
        except (AttributeError, RuntimeError, TypeError):
            return {}

