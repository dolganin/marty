from __future__ import annotations

import numpy as np
import pytest

import _mars_rover_cpp as native
from mars_rover_env import MarsRoverEnv


GAS = 1
CLUTCH = 1 << 3
TILT_LEFT = 1 << 4
TILT_RIGHT = 1 << 5
SHIFT_UP = 1 << 6
IGNITION = 1 << 9
LIDAR = 1 << 11

# The ledge field is only carved where the course difficulty ramp has risen
# (env.cpp, Env::build_zones), which on the default 20 km course sits far beyond
# anything a scripted drive reaches. A short course brings it into range.
SHORT_COURSE = "terrain:\n  sample_count: 2000\n  dx: 0.25\n"


def _ledge_index() -> int:
    return next(
        int(item["index"])
        for item in native.biome_catalog()
        if item["id"] == "commitment_ledge_field"
    )


def _short_course_config(tmp_path, extra: str = "") -> str:
    config = tmp_path / "ledge_course.yaml"
    config.write_text(SHORT_COURSE + extra, encoding="utf-8")
    return str(config)


def _drive_to_first_landing(
    *, correct_pitch: bool, config_path: str, steps: int = 9000
) -> tuple[dict, np.ndarray]:
    """Drive the ledge field until the rover completes one ballistic landing."""
    env = MarsRoverEnv(
        config_path=config_path, fixed_biome_id=_ledge_index(), biome_split=0
    )
    obs, _ = env.reset(seed=7, options={"trial_start": True})
    airborne_obs = None
    result: dict = {}
    for _ in range(steps):
        before = env.debug_info()
        if not before["engine_running"]:
            # v13 does not start the engine for you.
            action = IGNITION
        elif before["upshift_recommended"] and before["can_shift_up"]:
            action = CLUTCH | SHIFT_UP
        else:
            action = GAS
            if before["airborne"]:
                # Pitch control in flight decides whether the rover lands flat.
                action |= TILT_RIGHT if correct_pitch else TILT_LEFT
        if before["airborne"] and airborne_obs is None:
            action |= LIDAR
        obs, _, terminated, truncated, _ = env.step(action)
        after = env.debug_info()
        if after["airborne"] and airborne_obs is None:
            airborne_obs = obs.copy()
        if after["landing_event"]:
            result = after
            break
        if terminated or truncated:
            env.reset()
    env.close()
    assert result, "the rover never completed a ballistic landing"
    assert airborne_obs is not None
    return result, airborne_obs


def test_commitment_ledge_has_a_real_ballistic_phase(tmp_path) -> None:
    landing, _ = _drive_to_first_landing(
        correct_pitch=True, config_path=_short_course_config(tmp_path)
    )
    # A carved ledge gap produces a genuine flight phase ending in a measured
    # impact, not a step the rover simply drives over.
    assert landing["impact_speed"] > 0.0
    assert landing["landing_angle"] >= 0.0


def test_scanners_are_silent_while_airborne(tmp_path) -> None:
    """env.cpp gates the lidar on ground contact: no scan while in flight."""
    env = MarsRoverEnv(
        config_path=_short_course_config(tmp_path),
        fixed_biome_id=_ledge_index(),
        biome_split=0,
    )
    env.reset(seed=7, options={"trial_start": True})
    checked = 0
    for _ in range(9000):
        before = env.debug_info()
        if not before["engine_running"]:
            action = IGNITION
        elif before["upshift_recommended"] and before["can_shift_up"]:
            action = CLUTCH | SHIFT_UP
        else:
            action = GAS
        # Ask for a scan on every step, including in flight.
        env.step(action | LIDAR)
        after = env.debug_info()
        if after["airborne"]:
            assert not after["lidar_active"], "the lidar must not scan in flight"
            checked += 1
            if checked > 20:
                break
    env.close()
    assert checked > 0, "the rover never left the ground"


def test_landing_reports_the_body_angle_without_teleporting_it(tmp_path) -> None:
    landing, _ = _drive_to_first_landing(
        correct_pitch=False, config_path=_short_course_config(tmp_path)
    )
    assert landing["landing_angle"] == pytest.approx(abs(landing["angle"]), abs=0.05)


def test_landing_settles_on_the_terrain_instead_of_tunnelling(tmp_path) -> None:
    config_path = _short_course_config(tmp_path)
    env = MarsRoverEnv(
        config_path=config_path, fixed_biome_id=_ledge_index(), biome_split=0
    )
    env.reset(seed=7, options={"trial_start": True})
    landed = False
    debug: dict = {}
    for _ in range(9000):
        before = env.debug_info()
        if not before["engine_running"]:
            action = IGNITION
        elif before["upshift_recommended"] and before["can_shift_up"]:
            action = CLUTCH | SHIFT_UP
        else:
            action = GAS
        env.step(action)
        debug = env.debug_info()
        if debug["landing_event"]:
            landed = True
            break
    assert landed

    minimum_y = float(debug["y"])
    for _ in range(600):
        env.step(0)
        debug = env.debug_info()
        minimum_y = min(minimum_y, float(debug["y"]))
        if debug["fatal_error"]:
            # Falling into the next gap is a legitimate outcome; the check here
            # is only that the rover never sank through solid terrain.
            break
    env.close()

    assert minimum_y > -2.0


def test_excessive_impact_speed_is_fatal(tmp_path) -> None:
    config_path = _short_course_config(
        tmp_path, "physics:\n  safe_landing_speed: 0.05\n  safe_landing_angle: 3.2\n"
    )
    landing, _ = _drive_to_first_landing(correct_pitch=True, config_path=config_path)
    assert landing["impact_speed"] > 0.05
    assert landing["landing_angle"] < 3.2
    assert landing["landing_fatal"]
    assert landing["fatal_error"]


def test_scanner_channels_are_raw_geometry_not_mechanic_ids() -> None:
    scanner_start = 8 + 8 * 3 + 24 * 2
    scanner_end = scanner_start + 8 * 3
    observations = []
    for biome_index in (0, 1):
        env = MarsRoverEnv(fixed_biome_id=biome_index, biome_split=0)
        env.reset(seed=3, options={"trial_start": True})
        obs, *_ = env.step(LIDAR)
        observations.append(obs[scanner_start:scanner_end])
        env.close()
    assert np.allclose(observations[0], observations[1], atol=1.0e-4)
