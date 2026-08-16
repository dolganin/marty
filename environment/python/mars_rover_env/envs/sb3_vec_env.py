from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv


class MarsRoverSb3VecEnv(VecEnv):
    """SB3 adapter over one native batched C++ environment call per step."""

    render_mode = None

    def __init__(
        self,
        num_envs: int,
        action_macros: list[int],
        *,
        config_path: str | None = None,
        biome_split: int = 1,
        seed: int = 0,
        fixed_biome_id: int | None = None,
    ):
                                                                                  
                                                                                   
                       
        self.native = MarsRoverVecEnv(
            num_envs,
            config_path=config_path,
            biome_split=biome_split,
            fixed_biome_id=fixed_biome_id,
        )
        self.action_macros = np.asarray(action_macros, dtype=np.int32)
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.native.obs_dim,),
            dtype=np.float32,
        )
        super().__init__(
            num_envs,
            observation_space,
            gym.spaces.Discrete(len(self.action_macros)),
        )
        self._pending_actions: np.ndarray | None = None
        self._base_seed = int(seed)
        self._rng = np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        seed = (
            int(self._seeds[0])
            if self._seeds and self._seeds[0] is not None
            else self._base_seed
        )
        observations = self.native.reset(seed)
        self.reset_infos = [{"trial_start": True} for _ in range(self.num_envs)]
        self._reset_seeds()
        self._reset_options()
        return observations.copy()

    def step_async(self, actions: np.ndarray) -> None:
        action_indices = np.asarray(actions, dtype=np.int64).reshape(self.num_envs)
        if np.any(action_indices < 0) or np.any(action_indices >= len(self.action_macros)):
            raise ValueError("Macro action index is out of bounds")
        self._pending_actions = np.take(self.action_macros, action_indices)

    def step_wait(self):
        if self._pending_actions is None:
            raise RuntimeError("step_wait called before step_async")
        observations, rewards, terminated, truncated, _ = self.native.step(
            self._pending_actions
        )
        dones = np.logical_or(terminated, truncated)
        infos = [{} for _ in range(self.num_envs)]
        for env_id in np.flatnonzero(dones):
            terminal = observations[env_id].copy()
            trial_start = bool(self.native.next_trial_start[env_id])
            reset_seed = int(
                self._rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
            )
            self.native.reset_at(env_id, reset_seed, trial_start=trial_start)
            infos[env_id] = {
                "terminal_observation": terminal,
                "TimeLimit.truncated": bool(truncated[env_id] and not terminated[env_id]),
                "trial_start": trial_start,
            }
            self.reset_infos[env_id] = {"trial_start": trial_start}
        self._pending_actions = None
        return observations.copy(), rewards.copy(), dones.copy(), infos

    def close(self) -> None:
        self._pending_actions = None

    def get_attr(self, attr_name: str, indices=None) -> list:
        if attr_name == "render_mode":
            return [None for _ in self._get_indices(indices)]
        return [getattr(self.native, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        for _ in self._get_indices(indices):
            setattr(self.native, attr_name, value)

    def env_method(self, method_name: str, *args, indices=None, **kwargs) -> list:
        method = getattr(self.native, method_name)
        return [method(*args, **kwargs) for _ in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
