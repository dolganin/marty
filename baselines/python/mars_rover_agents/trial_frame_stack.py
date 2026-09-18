"""Frame stacking whose memory survives the episode boundary inside a trial.

SB3's VecFrameStack zeroes an environment's stack on every `done`. That is the
right default, and it is also exactly what makes the note's adaptation question
unanswerable: `episodes_per_trial` lives run on ONE fixed chain
(EnvConfig.chain_biomes keeps trial_mechanic_seed_ until trial_start), and the
whole point of those repeated lives is that something learned on life 1 can be
used on life 2. If the stack resets at every death, a frozen deterministic policy
replays an identical trajectory on an identical chain, and "within-trial
adaptation" is identically zero by construction rather than by measurement.

This wrapper resets the stack only when the environment reports `trial_start`,
so the last frames before a crash remain visible on the next life - a real, if
narrow, channel through which the rover can carry "this chain kills me here" into
the next attempt WITHOUT any gradient step and without a recurrent policy.

`terminal_observation` is stacked with the same layout as ordinary observations,
because PPO reads it to bootstrap truncated episodes and would otherwise be fed a
single unstacked frame into a stacked-shape network.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper


class TrialFrameStack(VecEnvWrapper):
    def __init__(self, venv: VecEnv, n_stack: int) -> None:
        if n_stack < 1:
            raise ValueError("n_stack must be >= 1")
        observation_space = venv.observation_space
        if not isinstance(observation_space, spaces.Box) or len(observation_space.shape) != 1:
            raise TypeError("TrialFrameStack expects a 1-D Box observation space")
        self.n_stack = int(n_stack)
        self.obs_dim = int(observation_space.shape[0])
        stacked_space = spaces.Box(
            low=np.repeat(observation_space.low, self.n_stack),
            high=np.repeat(observation_space.high, self.n_stack),
            dtype=observation_space.dtype,
        )
        super().__init__(venv, observation_space=stacked_space)
        self._stack = np.zeros((venv.num_envs, self.obs_dim * self.n_stack), dtype=np.float32)

    def _push(self, obs: np.ndarray) -> np.ndarray:
        if self.n_stack > 1:
            self._stack[:, : -self.obs_dim] = self._stack[:, self.obs_dim :]
        self._stack[:, -self.obs_dim :] = obs
        return self._stack.copy()

    def reset(self) -> np.ndarray:
        self._stack[:] = 0.0
        return self._push(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
                                                                             
                                                                                   
        for env_id, info in enumerate(infos):
            if "terminal_observation" in info:
                terminal = self._stack[env_id].copy()
                if self.n_stack > 1:
                    terminal[: -self.obs_dim] = terminal[self.obs_dim :]
                terminal[-self.obs_dim :] = info["terminal_observation"]
                info["terminal_observation"] = terminal
        for env_id, info in enumerate(infos):
            if info.get("trial_start", False):
                self._stack[env_id] = 0.0
        return self._push(obs), rewards, dones, infos
