from pathlib import Path

from mars_rover_env import MarsRoverEnv


def _play_config() -> str:
    return str(
        Path(__file__).parents[1]
        / "python"
        / "mars_rover_env"
        / "configs"
        / "play.yaml"
    )


def test_upshift_uses_projected_next_gear_rpm() -> None:
    env = MarsRoverEnv(config_path=_play_config(), biome_split=2, fixed_biome_id=0)
    env.reset(seed=3)

    # Select first, then allow the synchronizer cooldown to expire at rest.
    env.step(8 | 64)
    for _ in range(46):
        env.step(0)

    # First -> second is rejected when the projected RPM would lug the engine.
    env.step(8 | 64)
    debug = env.debug_info()
    assert debug["gear"] == 1
    assert not debug["upshift_speed_ok"]
    assert debug["next_gear_rpm"] < debug["minimum_shift_up_rpm"]

    # Under full throttle the recommendation rises into the useful power band.
    for _ in range(3000):
        env.step(1)
        debug = env.debug_info()
        if debug["upshift_recommended"] and debug["shift_cooldown"] <= 0.0:
            break
    else:
        raise AssertionError("first-to-second recommendation was never reached")

    assert debug["engine_rpm"] >= debug["shift_up_rpm"]
    assert debug["next_gear_rpm"] >= 3000.0
    env.step(8 | 64)
    assert env.debug_info()["gear"] == 2


def test_clutch_separates_engine_rpm_from_vehicle_speed() -> None:
    env = MarsRoverEnv(config_path=_play_config(), biome_split=2, fixed_biome_id=0)
    env.reset(seed=3)
    env.step(8 | 64)

    # With the pedal held, the engine can rev while almost no torque reaches
    # the wheels. Engagement decays over a short, finite mechanical interval.
    for _ in range(120):
        env.step(1 | 8)
    disengaged = env.debug_info()
    assert disengaged["clutch_engagement"] < 0.01
    assert disengaged["engine_rpm"] > 3000.0
    assert disengaged["speed_kmh"] < 1.0

    # Releasing the pedal reconnects the drivetrain and launches the rover.
    for _ in range(120):
        env.step(1)
    engaged = env.debug_info()
    assert engaged["clutch_engagement"] > 0.99
    assert engaged["speed_kmh"] > disengaged["speed_kmh"] + 1.0


def test_eight_speed_box_lugs_and_stalls_in_fourth_at_walking_speed() -> None:
    env = MarsRoverEnv(config_path=_play_config(), biome_split=2, fixed_biome_id=0)
    env.reset(seed=3)
    env.step(8 | 64)
    multipliers = []

    for gear in range(1, 4):
        for _ in range(6000):
            _, _, terminated, truncated, _ = env.step(1)
            debug = env.debug_info()
            assert not terminated and not truncated
            if debug["upshift_recommended"] and debug["shift_cooldown"] <= 0.0:
                break
        else:
            raise AssertionError(f"gear {gear} never reached its shift point")
        assert debug["gear"] == gear
        assert debug["gear_count"] == 8
        multipliers.append(debug["gear_energy_multiplier"])
        env.step(8 | 64)
        assert env.debug_info()["gear"] == gear + 1
        for _ in range(3):
            env.step(1)

    multipliers.append(env.debug_info()["gear_energy_multiplier"])
    assert multipliers == sorted(multipliers)

    for _ in range(600):
        env.step(2)
        if env.debug_info()["speed_kmh"] <= 5.0:
            break
    for _ in range(60):
        env.step(1)
    debug = env.debug_info()
    assert debug["gear"] == 4
    assert debug["speed_kmh"] < 5.0
    assert debug["should_shift_down"]
    assert not debug["engine_running"]
    env.step(512)
    assert not env.debug_info()["engine_running"]
    env.step(0)
    env.step(512 | 8)
    assert env.debug_info()["engine_running"]
