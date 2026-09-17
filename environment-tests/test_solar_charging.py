import numpy as np

import mars_rover_env                                                         
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv


def _biome_index(biome_id: str) -> int:
    from _mars_rover_cpp import biome_catalog

    return next(int(item["index"]) for item in biome_catalog() if item["id"] == biome_id)


def _solar_env(seed: int, biome_id: int):
    config = EnvConfig()
    # Builtin split: fixed_biome_id then means the catalog index itself.
    config.biome_split = 0 if biome_id >= 0 else 1
    config.chain_biomes = biome_id < 0
    config.fixed_biome_id = biome_id
    config.physics.initial_energy = 10.0
    config.physics.energy_capacity = 100.0
    config.physics.panel_deploy_time = 0.5
    config.physics.panel_retract_time = 0.4
    # max_steps is the guard for clockless configs; this scenario needs room.
    config.termination.max_steps = 20000
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
                                                                   
    ice_batch, ice_obs, ice_step = _solar_env(seed=0, biome_id=_biome_index("ice"))
    assert ice_batch.debug_info(0)["mechanic"] == "Ice"

    ice_start = dict(ice_batch.debug_info(0))
    ice = _deploy(ice_step)
    # v13 gates charging on standing still, not on killing the engine.
    assert ice["solar_panel_stationary"]
    assert ice["solar_panel_deployment"] >= 0.999
    assert ice["solar_charge_rate"] > 0.0
    assert abs(ice["x"] - ice_start["x"]) < 0.05
    assert ice["engine_temperature"] < ice_start["engine_temperature"]

    energy_before = ice["energy"]
    for _ in range(120):
        ice = ice_step(0)
    # 120 steps is two seconds of charging; the panel also ramps in.
    assert ice["energy"] > energy_before + ice["solar_charge_rate"] * 1.5
    assert np.isfinite(ice_obs).all()

                                                                            
    ice_step(1024)
    stow_x = ice["x"]
    for _ in range(120):
        ice = ice_step(1)
        if ice["solar_panel_deployment"] <= 0.001:
            break
    assert not ice["charging_active"]
    # The rover creeps only slightly while the panel is still folding away.
    assert abs(ice["x"] - stow_x) < 0.25
    ice = ice_step(512 | 8)
    assert ice["engine_running"]


def test_charge_rate_is_a_property_of_the_world(tmp_path=None) -> None:
    """The charge rate comes from the layer stack, so it differs per world."""
    rates = set()
    for seed in range(8):
        _, _, step = _solar_env(seed=seed, biome_id=-1)
        step(1024)
        for _ in range(400):
            debug = step(0)
            if debug["charging_active"]:
                break
        assert debug["charging_active"]
        rates.add(round(debug["solar_charge_rate"], 3))
    assert len(rates) > 1
