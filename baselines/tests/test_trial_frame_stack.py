from __future__ import annotations

import numpy as np
import pytest

from mars_rover_agents.expert_dataset import stack_observations
from mars_rover_agents.trial_frame_stack import TrialFrameStack

gym = pytest.importorskip("gymnasium")
pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import VecEnv              


class _ScriptedVecEnv(VecEnv):
    """Two envs emitting a counter, with scripted done/trial_start boundaries."""

    render_mode = None

    def __init__(self, boundaries: dict[int, tuple[bool, bool]]):
                                                                              
        super().__init__(
            2,
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            gym.spaces.Discrete(3),
        )
        self.boundaries = boundaries
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros((2, 2), dtype=np.float32)

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        self.t += 1
        obs = np.full((2, 2), float(self.t), dtype=np.float32)
        dones = np.zeros(2, dtype=bool)
        infos = [{}, {}]
        if self.t in self.boundaries:
            done, trial_start = self.boundaries[self.t]
            dones[0] = done
            if done:
                infos[0] = {
                    "terminal_observation": np.full(2, -float(self.t), dtype=np.float32),
                    "trial_start": trial_start,
                }
        return obs, np.zeros(2, dtype=np.float32), dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [None] * self.num_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs


def test_stack_layout_is_oldest_first_and_zero_padded():
    history = [np.array([1.0, 1.0], np.float32), np.array([2.0, 2.0], np.float32)]
    stacked = stack_observations(history, stack=3, obs_dim=2)
    assert stacked.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]


def test_stack_of_one_returns_the_latest_frame():
    history = [np.array([1.0, 1.0], np.float32), np.array([2.0, 2.0], np.float32)]
    assert stack_observations(history, stack=1, obs_dim=2).tolist() == [2.0, 2.0]


def test_memory_survives_a_death_but_not_a_new_trial():
                                                                            
    env = TrialFrameStack(_ScriptedVecEnv({2: (True, False), 4: (True, True)}), n_stack=3)
    env.reset()
    env.step(np.zeros(2))       
    obs, _, _, infos = env.step(np.zeros(2))                                     
                                                                                    
    assert obs[0].tolist() == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
                                                                            
    assert infos[0]["terminal_observation"].tolist() == [0.0, 0.0, 1.0, 1.0, -2.0, -2.0]

    env.step(np.zeros(2))       
    obs, _, _, _ = env.step(np.zeros(2))                       
                                                                                   
    assert obs[0].tolist() == [0.0, 0.0, 0.0, 0.0, 4.0, 4.0]


def test_untouched_env_keeps_stacking_normally():
    env = TrialFrameStack(_ScriptedVecEnv({2: (True, True)}), n_stack=3)
    env.reset()
    for _ in range(3):
        obs, _, _, _ = env.step(np.zeros(2))
    assert obs[1].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]


def test_observation_space_is_widened_by_the_stack():
    env = TrialFrameStack(_ScriptedVecEnv({}), n_stack=4)
    assert env.observation_space.shape == (8,)
