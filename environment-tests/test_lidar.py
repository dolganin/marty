import numpy as np

import mars_rover_env  # noqa: F401  (ensures the native module is importable)
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


# Observation layout (environment/cpp/include/mars/observation.hpp):
#   8 body | 8*3 wheels | 24*2 terrain | 8*3 biome | 18 drivetrain and sensors
#   | 4 imu | 3 body contacts | 8*2 wheel encoders | 15 organs and contacts
BODY_DIM = 8
WHEEL_DIM = 8 * 3
TERRAIN_SAMPLES = 24
TERRAIN_BEGIN = BODY_DIM + WHEEL_DIM
TERRAIN_END = TERRAIN_BEGIN + TERRAIN_SAMPLES * 2
BIOME_BEGIN = TERRAIN_END
MISC_BEGIN = BIOME_BEGIN + 8 * 3
ENGINE_TEMPERATURE_INDEX = MISC_BEGIN + 7
IMU_BEGIN = MISC_BEGIN + 18
BODY_CONTACT_BEGIN = IMU_BEGIN + 4
ENCODER_BEGIN = BODY_CONTACT_BEGIN + 3
SUSPENSION_BEGIN = ENCODER_BEGIN + 8
OBSERVATION_DIM = 160


def _batch(config: EnvConfig, seed: int):
    batch = MarsRoverBatchEnv(1, config)
    obs = np.zeros((1, batch.obs_dim), dtype=np.float32)
    rewards = np.zeros(1, dtype=np.float32)
    terminated = np.zeros(1, dtype=np.uint8)
    truncated = np.zeros(1, dtype=np.uint8)
    actions = np.zeros(1, dtype=np.int32)
    batch.reset_at(0, seed, True, obs[0])

    def step(action: int) -> dict:
        actions[0] = action
        batch.step(actions, obs, rewards, terminated, truncated)
        return dict(batch.debug_info(0))

    return batch, obs, step


def _far_channels(obs: np.ndarray) -> np.ndarray:
    """Terrain and slope samples that only a lidar scan can reveal."""
    half = TERRAIN_SAMPLES // 2
    heights = obs[TERRAIN_BEGIN + half : TERRAIN_BEGIN + TERRAIN_SAMPLES]
    slopes = obs[TERRAIN_BEGIN + TERRAIN_SAMPLES + half : TERRAIN_END]
    return np.concatenate((heights, slopes))


def test_lidar_spends_energy_and_reveals_the_far_terrain() -> None:
    config = EnvConfig()
    config.fixed_biome_id = 0
    config.physics.initial_energy = 50.0
    config.physics.energy_capacity = 50.0
    config.physics.lidar_energy_cost = 0.65
    batch, obs, step = _batch(config, seed=3)

    # v13 gives the rover short-range passive sight; only the far half of the
    # scan needs the lidar.
    assert np.all(_far_channels(obs[0]) == 0.0)
    energy_before = batch.debug_info(0)["energy"]

    debug = step(1 << 11)
    assert debug["lidar_active"]
    assert debug["lidar_range"] > 0.0
    assert debug["lidar_cooldown"] > 0.0
    assert debug["energy"] <= energy_before - 0.64
    assert np.any(np.abs(_far_channels(obs[0])) > 1.0e-5)


def test_proprioception_and_environment_sensors_are_observable() -> None:
    config = EnvConfig()
    config.fixed_biome_id = 0
    batch, obs, step = _batch(config, seed=11)

    assert batch.obs_dim == OBSERVATION_DIM
    debug = batch.debug_info(0)
    assert np.isclose(obs[0, ENGINE_TEMPERATURE_INDEX], debug["engine_temperature"] / 120.0)

    step((1 << 3) | (1 << 6))
    for _ in range(60):
        debug = step(1)

    assert np.isfinite(obs[0, IMU_BEGIN : IMU_BEGIN + 4]).all()
    assert np.all(np.abs(obs[0, IMU_BEGIN : IMU_BEGIN + 4]) <= 4.0)
    contacts = obs[0, BODY_CONTACT_BEGIN : BODY_CONTACT_BEGIN + 3]
    assert np.all(np.isin(contacts, (0.0, 1.0)))
    encoders = obs[0, ENCODER_BEGIN : ENCODER_BEGIN + 8]
    assert np.all(np.abs(encoders) <= 4.0)
    suspension = obs[0, SUSPENSION_BEGIN : SUSPENSION_BEGIN + 8]
    assert np.all((suspension >= 0.0) & (suspension <= 1.0))


def test_anchor_courses_receive_reproducible_rare_terrain_surprises() -> None:
    config = EnvConfig()
    config.biome_split = 3
    config.chain_biomes = True
    config.terrain_surprise_probability = 1.0
    first = MarsRoverBatchEnv(1, config)
    second = MarsRoverBatchEnv(1, config)
    obs_a = np.zeros((1, first.obs_dim), dtype=np.float32)
    obs_b = np.zeros((1, second.obs_dim), dtype=np.float32)

    # The surprise probability is scaled by the course difficulty ramp, so the
    # opening zone stays mostly calm even at probability 1.0: the property to
    # check is that surprises do occur and repeat for the same seed.
    modes = []
    for seed in range(40):
        first.reset_at(0, seed, True, obs_a[0])
        second.reset_at(0, seed, True, obs_b[0])
        assert np.array_equal(obs_a, obs_b)
        modes.append(first.debug_info(0)["terrain_surprise_mode"])
    assert set(modes) <= {0, 1, 2, 3, 4}
    assert any(mode > 0 for mode in modes), "no zone ever received a surprise"
