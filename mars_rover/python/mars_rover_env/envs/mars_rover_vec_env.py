from __future__ import annotations

import numpy as np

from _mars_rover_cpp import MarsRoverBatchEnv

from mars_rover_env.config import load_env_config


class MarsRoverVecEnv:
    """Thin batch wrapper. Designed for SB3/CleanRL adapters without per-env Python stepping."""

    def __init__(self, num_envs: int, config_path: str | None = None, rig_path: str | None = None):
        self.core = MarsRoverBatchEnv(int(num_envs), load_env_config(config_path, rig_path))
        self.num_envs = self.core.num_envs
        self.obs_dim = self.core.obs_dim
        self.action_dim = self.core.action_dim
        self.obs = np.zeros((self.num_envs, self.obs_dim), dtype=np.float32)
        self.rewards = np.zeros((self.num_envs,), dtype=np.float32)
        self.terminated = np.zeros((self.num_envs,), dtype=np.uint8)
        self.truncated = np.zeros((self.num_envs,), dtype=np.uint8)
        self.terminated_bool = np.zeros((self.num_envs,), dtype=bool)
        self.truncated_bool = np.zeros((self.num_envs,), dtype=bool)

    def reset(self, seed: int = 0):
        self.core.reset_all(int(seed), self.obs)
        return self.obs

    def reset_at(self, env_id: int, seed: int = 0, trial_start: bool = True):
        self.core.reset_at(int(env_id), int(seed), bool(trial_start), self.obs[int(env_id)])
        return self.obs[int(env_id)]

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int32)
        self.core.step(actions, self.obs, self.rewards, self.terminated, self.truncated)
        np.not_equal(self.terminated, 0, out=self.terminated_bool)
        np.not_equal(self.truncated, 0, out=self.truncated_bool)
        return self.obs, self.rewards, self.terminated_bool, self.truncated_bool, {}

    def step_uint8(self, actions: np.ndarray):
        """Fast path for training loops that can consume uint8 done flags without bool copies."""
        actions = np.asarray(actions, dtype=np.int32)
        self.core.step(actions, self.obs, self.rewards, self.terminated, self.truncated)
        return self.obs, self.rewards, self.terminated, self.truncated, {}

    def debug_info(self, env_id: int) -> dict:
        return dict(self.core.debug_info(int(env_id)))
