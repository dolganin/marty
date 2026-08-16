import numpy as np

import mars_rover_env                                                         
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def _solar_env(seed: int, biome_id: int):
    config = EnvConfig()
                                                                               
    config.biome_split = 2
    config.fixed_biome_id = biome_id
    config.physics.initial_energy = 10.0
    config.physics.energy_capacity = 100.0
    config.physics.panel_deploy_time = 0.5
    config.physics.panel_retract_time = 0.4
    config.termination.max_steps = 2000
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
        assert not terminated[0]
        assert not truncated[0]
        return dict(batch.debug_info(0))

    return batch, obs, step


def _deploy(step) -> dict:
    step(1024)
    for _ in range(300):
        debug = step(0)
        if debug["charging_active"]:
            return debug
    raise AssertionError("solar panel did not deploy")


def test_solar_cycle_locks_rover_and_charges_fastest_on_ice() -> None:
                                                                   
    ice_batch, ice_obs, ice_step = _solar_env(seed=0, biome_id=2)
    sand_batch, _, sand_step = _solar_env(seed=5, biome_id=1)
    assert ice_batch.debug_info(0)["mechanic"] == "Ice"
    assert sand_batch.debug_info(0)["mechanic"] == "Sand"

    ice_start = dict(ice_batch.debug_info(0))
    ice = _deploy(ice_step)
    sand = _deploy(sand_step)
    assert not ice["engine_running"]
    assert ice["solar_panel_deployment"] >= 0.999
    assert ice["solar_charge_rate"] > sand["solar_charge_rate"] * 4.0
    assert abs(ice["x"] - ice_start["x"]) < 0.05
    assert ice["engine_temperature"] < ice_start["engine_temperature"]

    energy_before = ice["energy"]
    for _ in range(120):
        ice = ice_step(0)
    assert ice["energy"] > energy_before + ice["solar_charge_rate"] * 1.9
    assert np.isfinite(ice_obs).all()

                                                                            
    ice_step(1024)
    stow_x = ice["x"]
    for _ in range(120):
        ice = ice_step(1)
        if ice["solar_panel_deployment"] <= 0.001:
            break
    assert not ice["charging_active"]
    assert abs(ice["x"] - stow_x) < 0.05
    assert not ice["engine_running"]
    ice = ice_step(512)
    assert ice["engine_running"]
