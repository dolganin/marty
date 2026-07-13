import numpy as np

import mars_rover_env  # noqa: F401 - registers MinGW runtime DLLs on Windows
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def test_lidar_spends_energy_and_reveals_forward_observation() -> None:
    config = EnvConfig()
    config.biome_split = 2
    config.physics.initial_energy = 50.0
    config.physics.energy_capacity = 50.0
    config.physics.lidar_energy_cost = 2.0
    batch = MarsRoverBatchEnv(1, config)
    obs = np.zeros((1, batch.obs_dim), dtype=np.float32)
    rewards = np.zeros(1, dtype=np.float32)
    terminated = np.zeros(1, dtype=np.uint8)
    truncated = np.zeros(1, dtype=np.uint8)
    actions = np.zeros(1, dtype=np.int32)
    batch.reset_at(0, 3, True, obs[0])

    # Forward height and slope fields are unavailable without an active scan.
    terrain_begin = 8 + 8 * 3
    terrain_end = terrain_begin + 24 * 2
    assert np.all(obs[0, terrain_begin:terrain_end] == 0.0)
    energy_before = batch.debug_info(0)["energy"]

    actions[0] = 1 << 11
    batch.step(actions, obs, rewards, terminated, truncated)
    debug = batch.debug_info(0)
    assert debug["lidar_active"]
    assert debug["lidar_range"] > 0.0
    assert debug["lidar_cooldown"] > 0.0
    assert debug["energy"] <= energy_before - 1.99
    assert np.any(np.abs(obs[0, terrain_begin:terrain_end]) > 1.0e-5)
