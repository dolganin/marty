import yaml

from mars_rover_env import MarsRoverEnv


GAS = 1
CLUTCH = 1 << 3
SHIFT_UP = 1 << 6
SHIFT_DOWN = 1 << 7
IGNITION = 1 << 9


def _flat_config(tmp_path) -> str:
    """A flat, single-biome course: gearing is the only thing under test."""
    path = tmp_path / "flat.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "env": {
                    "chain_biomes": False,
                    "fixed_biome_id": 0,
                    "biome_split": 0,
                },
                "terrain": {
                    "amplitude": 0.0,
                    "roughness": 0.0,
                    "crater_count": 0,
                    "step_count": 0,
                },
                "physics": {
                    "initial_energy": 20000.0,
                    "energy_capacity": 20000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _start_engine(env) -> None:
    """v13 starts with the engine off; the key needs neutral or the clutch."""
    for _ in range(10):
        if env.debug_info()["engine_running"]:
            return
        env.step(IGNITION | CLUTCH)
    raise AssertionError("the engine never started")


def _running_env(tmp_path) -> MarsRoverEnv:
    env = MarsRoverEnv(config_path=_flat_config(tmp_path))
    env.reset(seed=3)
    _start_engine(env)
    return env


def test_upshift_needs_the_clutch_and_lowers_the_projected_rpm(tmp_path) -> None:
    env = _running_env(tmp_path)
    for _ in range(900):
        env.step(GAS)

    before = env.debug_info()
    assert before["gear"] == 1
    assert before["engine_rpm"] > before["next_gear_rpm"]

    env.step(CLUTCH | SHIFT_UP)
    assert env.debug_info()["gear"] == 2
    # Let the clutch close again: during the cut the road speed dips and the
    # projected rpm is not comparable.
    for _ in range(60):
        env.step(GAS)
    after = env.debug_info()
    # Compare rpm per unit of road speed: the rover keeps accelerating between
    # the two samples, so the raw numbers are not taken at the same speed.
    before_rate = before["next_gear_rpm"] / max(0.1, before["speed_kmh"])
    after_rate = after["next_gear_rpm"] / max(0.1, after["speed_kmh"])
    assert after_rate < before_rate
    env.close()


def test_the_shift_window_tracks_road_speed(tmp_path) -> None:
    env = _running_env(tmp_path)
    standing = env.debug_info()
    assert not standing["upshift_speed_ok"]
    assert standing["next_gear_rpm"] <= standing["minimum_shift_up_rpm"]

    for _ in range(900):
        env.step(GAS)
    rolling = env.debug_info()
    assert rolling["next_gear_rpm"] > standing["next_gear_rpm"]
    assert rolling["upshift_speed_ok"]
    env.close()


def test_clutch_separates_engine_rpm_from_vehicle_speed(tmp_path) -> None:
    env = _running_env(tmp_path)

    # Clutch pedal down: the engine revs freely and the rover stays put.
    for _ in range(120):
        env.step(GAS | CLUTCH)
    disengaged = env.debug_info()
    assert disengaged["clutch_engagement"] < 0.01
    assert disengaged["engine_rpm"] > 2500.0
    assert disengaged["speed_kmh"] < 1.0

    # Pedal released: the same throttle now reaches the wheels.
    for _ in range(120):
        env.step(GAS)
    engaged = env.debug_info()
    assert engaged["clutch_engagement"] > 0.99
    assert engaged["speed_kmh"] > disengaged["speed_kmh"] + 1.0
    env.close()


def test_taller_base_speed_reaches_a_dynamic_first_gear_pace(tmp_path) -> None:
    env = _running_env(tmp_path)
    for _ in range(900):
        env.step(GAS)
    assert env.debug_info()["speed_kmh"] > 8.0
    env.close()


def test_a_tall_gear_lugs_the_engine_at_walking_speed(tmp_path) -> None:
    env = _running_env(tmp_path)
    for _ in range(900):
        env.step(GAS)

    # Climb as high as the road speed allows, then crawl. On flat ground with
    # slipping wheels that ceiling is third.
    for _ in range(3):
        env.step(CLUTCH | SHIFT_UP)
        for _ in range(30):
            env.step(GAS)
    top_gear = env.debug_info()["gear"]
    assert top_gear >= 3

    for _ in range(240):
        env.step(0)
    lugging = env.debug_info()
    assert lugging["gear"] == top_gear
    assert lugging["should_shift_down"] or lugging["engine_stalled"]
    env.close()


def test_downshift_returns_a_usable_gear(tmp_path) -> None:
    env = _running_env(tmp_path)
    for _ in range(900):
        env.step(GAS)
    env.step(CLUTCH | SHIFT_UP)
    assert env.debug_info()["gear"] == 2

    # A downshift is refused while the lower gear would overspeed the engine,
    # so coast down first.
    for _ in range(240):
        env.step(0)
    env.step(CLUTCH | SHIFT_DOWN)
    assert env.debug_info()["gear"] == 1
    env.close()
