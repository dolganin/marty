from __future__ import annotations

import numpy as np

import mars_rover_env                                                         
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def test_deployed_panel_cannot_exempt_idle_policy_from_stuck_loss() -> None:
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

                                                                            
    actions[0] = 1 << 10
    batch.step(actions, observations, rewards, terminated, truncated)
    actions[0] = 0
    for _ in range(250):
        batch.step(actions, observations, rewards, terminated, truncated)
        if terminated[0]:
            break

    assert terminated[0]
    assert not truncated[0]
    assert rewards[0] <= -299.0
    assert batch.debug_info(0)["x"] < 1.5

