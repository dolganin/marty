from __future__ import annotations

import numpy as np

import mars_rover_env                                                         
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def test_idling_behind_a_deployed_panel_only_burns_the_clock() -> None:
    """v13 has no stuck loss: standing still simply wastes the shared timer."""
    config = EnvConfig()
    config.fixed_biome_id = 0
    config.termination.max_steps = 500
    config.termination.stuck_steps = 30
    config.reward.stuck_penalty = 300.0
    batch = MarsRoverBatchEnv(1, config)
    observations = np.zeros((1, batch.obs_dim), dtype=np.float32)
    rewards = np.zeros(1, dtype=np.float32)
    terminated = np.zeros(1, dtype=np.uint8)
    truncated = np.zeros(1, dtype=np.uint8)
    actions = np.zeros(1, dtype=np.int32)
    batch.reset_at(0, 17, True, observations[0])

    start = dict(batch.debug_info(0))
    actions[0] = 1 << 10  # deploy the solar panel
    batch.step(actions, observations, rewards, terminated, truncated)
    actions[0] = 0
    for _ in range(250):
        batch.step(actions, observations, rewards, terminated, truncated)
        assert not terminated[0], "an idle rover is never killed off in v13"

    debug = batch.debug_info(0)
    assert debug["x"] < 1.5
    assert debug["trial_steps_used"] > start["trial_steps_used"]
    # Only the millimetres of settling creep earn anything at all.
    assert rewards[0] < 0.01

