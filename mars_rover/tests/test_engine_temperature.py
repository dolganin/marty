import numpy as np

import mars_rover_env  # Registers the MinGW runtime DLL directory on Windows.
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def _batch(config: EnvConfig, seed: int):
    # Engine regression tests target built-in thermal behavior, not an LLM bank.
    config.biome_split = 2
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


def test_overheat_shutdown_and_cooldown_restart() -> None:
    config = EnvConfig()
    config.physics.initial_engine_temperature = 20.0
    config.physics.overheat_temperature = 25.0
    config.physics.overheat_restart_temperature = 22.0
    config.physics.engine_heat_rate = 80.0
    config.physics.engine_cooling_rate = 0.02
    _, obs, step = _batch(config, seed=0)  # Ice provides a cold cooling medium.

    for _ in range(300):
        debug = step(1 | 8)  # Full throttle with clutch disengaged.
        if debug["engine_overheated"]:
            break
    else:
        raise AssertionError("engine did not overheat")

    assert not debug["engine_running"]
    assert debug["engine_temperature"] >= config.physics.overheat_temperature

    for _ in range(300):
        debug = step(0)
        if not debug["engine_overheated"]:
            break
    else:
        raise AssertionError("engine did not cool to its restart threshold")

    assert not debug["engine_overheated"]
    assert not debug["engine_running"]
    debug = step(512)  # Ignition is independent from the clutch pedal.
    assert debug["engine_running"]
    assert debug["engine_temperature"] <= config.physics.overheat_restart_temperature
    assert np.isfinite(obs).all()


def test_cold_engine_cannot_start_in_a_colder_martian_biome() -> None:
    config = EnvConfig()
    config.physics.initial_engine_temperature = -30.0
    config.physics.cold_start_temperature = -18.0
    config.physics.engine_cooling_rate = 0.2
    batch, _, step = _batch(config, seed=3)

    debug = dict(batch.debug_info(0))
    assert debug["engine_cold_locked"]
    assert not debug["engine_running"]

    for _ in range(300):
        debug = step(512)

    assert debug["engine_cold_locked"]
    assert not debug["engine_running"]
    assert debug["engine_temperature"] < config.physics.initial_engine_temperature


def test_high_rpm_warms_engine_faster_than_idle() -> None:
    config = EnvConfig()
    config.physics.engine_cooling_rate = 0.0
    config.physics.engine_heat_rate = 10.0
    _, _, idle_step = _batch(config, seed=3)
    _, _, rev_step = _batch(config, seed=3)

    for _ in range(600):
        idle = idle_step(0)
        revving = rev_step(1 | 8)

    assert revving["engine_rpm"] > idle["engine_rpm"]
    assert revving["engine_temperature"] > idle["engine_temperature"] + 5.0
