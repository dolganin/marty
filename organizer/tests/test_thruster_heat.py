import unittest

import numpy as np

from _mars_rover_cpp import MarsRoverBatchEnv
from mars_rover_env.config import load_env_config


class ThrusterHeatTest(unittest.TestCase):
    @staticmethod
    def make_config():
        config = load_env_config()
        config.fixed_biome_id = 0
        config.chain_biomes = False
        config.termination.stuck_steps = 100_000
        config.termination.flip_angle = 100.0
        config.termination.fatal_fall_y = -100_000.0
        config.termination.min_energy = -100_000.0
        return config

    def test_thruster_overheats_and_stops_after_about_ten_seconds(self) -> None:
        config = self.make_config()
        env = MarsRoverBatchEnv(1, config)
        observations = np.zeros((1, env.obs_dim), dtype=np.float32)
        rewards = np.zeros(1, dtype=np.float32)
        terminated = np.zeros(1, dtype=np.uint8)
        truncated = np.zeros(1, dtype=np.uint8)
        env.reset_all(2026, observations)

        actions = np.asarray([1 << 23], dtype=np.int32)
        failure_step = None
        for step in range(round(15.0 / config.physics.dt)):
            env.step(actions, observations, rewards, terminated, truncated)
            debug = env.debug_info(0)
            if debug["engine_overheated"]:
                failure_step = step + 1
                self.assertEqual(float(debug["thruster_thrust"]), 0.0)
                break

        self.assertIsNotNone(failure_step)
        failure_time = failure_step * config.physics.dt
        self.assertGreaterEqual(failure_time, 8.0)
        self.assertLessEqual(failure_time, 12.0)

    def test_heater_can_run_while_solar_panel_is_charging(self) -> None:
        config = self.make_config()
        env = MarsRoverBatchEnv(1, config)
        observations = np.zeros((1, env.obs_dim), dtype=np.float32)
        rewards = np.zeros(1, dtype=np.float32)
        terminated = np.zeros(1, dtype=np.uint8)
        truncated = np.zeros(1, dtype=np.uint8)
        env.reset_all(2027, observations)

        actions = np.asarray([(1 << 10) | (1 << 12)], dtype=np.int32)
        env.step(actions, observations, rewards, terminated, truncated)
        actions[0] = 1 << 12
        for _ in range(round(1.5 / config.physics.dt)):
            env.step(actions, observations, rewards, terminated, truncated)

        debug = env.debug_info(0)
        self.assertTrue(bool(debug["charging_active"]))
        self.assertTrue(bool(debug["heater_active"]))
        self.assertGreater(float(debug["energy_gain_rate"]), 0.0)
        self.assertGreaterEqual(
            float(debug["energy_cost_rate"]), config.physics.engine_heater_energy_rate
        )


if __name__ == "__main__":
    unittest.main()
