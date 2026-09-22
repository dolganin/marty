import unittest

import numpy as np

from _mars_rover_cpp import MarsRoverBatchEnv, biome_catalog
from mars_rover_env.config import load_env_config


class CryogenicHeadwindTest(unittest.TestCase):
    def test_heater_is_required_to_keep_engine_warm(self) -> None:
        item = next(entry for entry in biome_catalog() if entry["id"] == "cryogenic_headwind_run")
        self.assertEqual(int(item["split"]), 2)
        configs = []
        for _ in range(2):
            config = load_env_config()
            config.fixed_biome_id = int(item["index"])
            config.chain_biomes = False
            config.termination.max_steps = 4_000
            config.termination.stuck_steps = 100_000
            config.termination.flip_angle = 100.0
            config.termination.fatal_fall_y = -100_000.0
            config.termination.min_energy = -100_000.0
            configs.append(config)
        env = MarsRoverBatchEnv(configs)
        observations = np.zeros((2, env.obs_dim), dtype=np.float32)
        rewards = np.zeros(2, dtype=np.float32)
        terminated = np.zeros(2, dtype=np.uint8)
        truncated = np.zeros(2, dtype=np.uint8)
        env.reset_all(8423432780, observations)
        actions = np.asarray([1, 1 | (1 << 12)], dtype=np.int32)
        for _ in range(2_400):
            env.step(actions, observations, rewards, terminated, truncated)
        cold = env.debug_info(0)
        heated = env.debug_info(1)
        self.assertLess(float(cold["engine_temperature"]), -38.0)
        self.assertFalse(bool(cold["engine_running"]))
        self.assertGreater(float(heated["engine_temperature"]), -25.0)
        self.assertTrue(bool(heated["heater_active"]))
        self.assertGreater(float(heated["x"]), float(cold["x"]) + 5.0)


if __name__ == "__main__":
    unittest.main()
