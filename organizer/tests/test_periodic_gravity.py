import unittest

import numpy as np

from _mars_rover_cpp import MarsRoverBatchEnv, biome_catalog
from mars_rover_env.config import load_env_config


class PeriodicGravityTest(unittest.TestCase):
    def test_all_waveforms_have_visible_amplitude(self) -> None:
        item = next(entry for entry in biome_catalog() if entry["id"] == "periodic_gravity_relay")
        self.assertEqual(int(item["split"]), 2)
        self.assertGreater(float(item["parameters"]["gravity_wave_amplitude"]), 0.7)

        config = load_env_config()
        config.fixed_biome_id = int(item["index"])
        config.chain_biomes = False
        config.termination.max_steps = 2_000
        config.termination.stuck_steps = 100_000
        config.termination.flip_angle = 100.0
        config.termination.fatal_fall_y = -100_000.0
        config.termination.min_energy = -100_000.0
        config.termination.trial_time_limit = 60.0
        env = MarsRoverBatchEnv(1, config)
        observations = np.zeros((1, env.obs_dim), dtype=np.float32)
        rewards = np.zeros(1, dtype=np.float32)
        terminated = np.zeros(1, dtype=np.uint8)
        truncated = np.zeros(1, dtype=np.uint8)
        actions = np.zeros(1, dtype=np.int32)
        env.reset_at(0, 2026, True, observations[0])

        samples = {index: [] for index in range(4)}
        gravities = []
        for _ in range(1_600):
            env.step(actions, observations, rewards, terminated, truncated)
            debug = env.debug_info(0)
            waveform = int(debug["gravity_waveform"])
            if waveform in samples:
                samples[waveform].append(float(debug["gravity_wave_value"]))
            gravities.append(float(debug["gravity"]))

        self.assertTrue(all(len(values) > 100 for values in samples.values()))
        for values in samples.values():
            self.assertLess(min(values), -0.9)
            self.assertGreater(max(values), 0.9)
        self.assertLessEqual(len({round(value, 3) for value in samples[2]}), 2)
        self.assertGreater(max(gravities) - min(gravities), 5.0)


if __name__ == "__main__":
    unittest.main()
