from __future__ import annotations

import numpy as np

import _mars_rover_cpp as native
from mars_rover_env import MarsRoverEnv


def _ledge_index() -> int:
    return next(
        int(item["index"])
        for item in native.biome_catalog()
        if item["id"] == "commitment_ledge_field"
    )


def _drive_to_first_landing(
    *, correct_pitch: bool, config_path: str | None = None
) -> tuple[dict, np.ndarray]:
    env = MarsRoverEnv(
        config_path=config_path, fixed_biome_id=_ledge_index(), biome_split=0
    )
    obs, _ = env.reset(seed=7, options={"trial_start": True})
    airborne_obs = None
    result = {}
    for step in range(700):
        before = env.debug_info()
        if step in (5, 80, 170, 280):
            action = (1 << 3) | (1 << 6)                             
        elif correct_pitch and before["airborne"] and before["x"] > 10.0:
            action = 1 | (1 << 5)                                       
        else:
            action = 1
                                                                               
        if before["airborne"] and before["x"] > 10.0 and airborne_obs is None:
            action |= 1 << 11
        obs, _, terminated, truncated, _ = env.step(action)
        after = env.debug_info()
        if after["airborne"] and after["x"] > 10.0 and airborne_obs is None:
            airborne_obs = obs.copy()
        if after["landing_event"] and after["x"] > 10.0:
            result = after
            break
        assert not (terminated or truncated), after
    env.close()
    assert result, "the rover never completed a ballistic landing"
    assert airborne_obs is not None
    return result, airborne_obs


def test_commitment_ledge_has_real_ballistic_phase_and_silent_scanners() -> None:
    landing, airborne_obs = _drive_to_first_landing(correct_pitch=True)
    assert landing["impact_speed"] > 0.5
    assert not landing["landing_fatal"]
    assert not landing["fatal_error"]

    terrain_start = 8 + 8 * 3
    scanner_end = terrain_start + 24 * 2 + 8 * 3
    assert np.allclose(airborne_obs[terrain_start:scanner_end], 0.0)


def test_bad_landing_angle_is_a_fatal_rollover() -> None:
    landing, _ = _drive_to_first_landing(correct_pitch=False)
    assert landing["landing_angle"] > 0.55
    assert landing["landing_fatal"]
    assert landing["fatal_error"]
    assert abs(landing["angle"]) >= 2.2


def test_excessive_impact_speed_is_fatal_even_with_good_angle(tmp_path) -> None:
    config = tmp_path / "strict_landing.yaml"
    config.write_text(
        "physics:\n  safe_landing_speed: 1.0\n  safe_landing_angle: 1.0\n"
        "termination:\n  max_steps: 3000\n  fatal_fall_y: -8.0\n",
        encoding="utf-8",
    )
    landing, _ = _drive_to_first_landing(
        correct_pitch=True, config_path=str(config)
    )
    assert landing["impact_speed"] > 1.0
    assert landing["landing_angle"] < 1.0
    assert landing["landing_fatal"]
    assert landing["fatal_error"]


def test_scanner_channels_are_raw_geometry_not_mechanic_ids() -> None:
    scanner_start = 8 + 8 * 3 + 24 * 2
    scanner_end = scanner_start + 8 * 3
    observations = []
    for biome_index in (0, 1):                                               
        env = MarsRoverEnv(fixed_biome_id=biome_index, biome_split=0)
        env.reset(seed=3, options={"trial_start": True})
        obs, *_ = env.step(1 << 11)
        observations.append(obs[scanner_start:scanner_end])
        env.close()
    assert np.allclose(observations[0], observations[1], atol=1.0e-4)


def test_trial_keeps_the_same_hidden_mechanic_across_episode_resets() -> None:
    """An RL² trial must be repeated experience of one world, not four iid worlds.

    The mechanic name is deliberately available only through ``debug_info`` here;
    it is not part of the public observation.  A different episode seed is still
    allowed to change the concrete terrain draw, but it must not replace the
    biome or its sampled physical parameters while ``trial_start`` is false.
    """
    env = MarsRoverEnv(biome_split=MarsRoverEnv.BIOME_MODE_TRAIN)
    env.reset(seed=314, options={"trial_start": True})
    first = env.debug_info()
    env.reset(seed=2718, options={"trial_start": False})
    later = env.debug_info()
    env.close()

    assert later["episode_in_trial"] == 1
    for field in ("mechanic", "friction_mul", "sink_rate", "viscosity", "gravity_mul"):
        assert later[field] == first[field]
