from __future__ import annotations

import math

import pytest

from mars_rover_env import MarsRoverEnv
from mars_rover_env.config import load_env_config
from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv

DT = 1.0 / 60.0
IGNITION = 1 << 9


@pytest.fixture()
def budget_config(tmp_path):
    import yaml

    path = tmp_path / "budget.yaml"
    payload = {
        "env": {"episodes_per_trial": 0, "biome_split": 1, "chain_biomes": True},
        "physics": {"dt": DT},
        "termination": {"max_steps": 0, "trial_time_limit": 2.0, "stuck_steps": 100000},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_config_carries_the_trial_budget(budget_config):
    config = load_env_config(budget_config)
    assert config.termination.trial_time_limit == pytest.approx(2.0)
    assert config.termination.max_steps == 0
    assert config.episodes_per_trial == 0


def test_budget_is_measured_in_whole_steps(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=1, options={"trial_start": True})
    debug = env.debug_info()
    assert debug["trial_step_budget"] == int(round(2.0 / DT))
    assert debug["trial_steps_used"] == 0
    assert debug["trial_time_left"] == pytest.approx(2.0, abs=1e-3)
    env.close()


def test_budget_truncates_the_trial_and_not_the_attempt(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=2, options={"trial_start": True})
    budget = env.debug_info()["trial_step_budget"]
    truncated = False
    for _ in range(budget + 10):
        _, _, terminated, truncated, _ = env.step(IGNITION)
        if terminated or truncated:
            break
    assert truncated
    assert env.debug_info()["trial_exhausted"]
    env.close()


def test_clock_survives_a_death_inside_the_trial(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=3, options={"trial_start": True})
    for _ in range(120):
        env.step(IGNITION)
    spent = env.debug_info()["trial_steps_used"]
    assert spent == 120

    env.reset(seed=4, options={"trial_start": False})
    assert env.debug_info()["trial_steps_used"] == spent

    env.reset(seed=5, options={"trial_start": True})
    assert env.debug_info()["trial_steps_used"] == 0
    env.close()


def test_exhausted_budget_requests_a_fresh_course(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=6, options={"trial_start": True})
    budget = env.debug_info()["trial_step_budget"]
    for _ in range(budget):
        _, _, terminated, truncated, _ = env.step(IGNITION)
        if terminated or truncated:
            break
    _, info = env.reset()
    assert info["trial_start"]
    assert env.debug_info()["trial_steps_used"] == 0
    env.close()


def test_attempts_are_unlimited_until_the_clock_runs_out(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=7, options={"trial_start": True})
    budget = env.debug_info()["trial_step_budget"]
    attempts = 0
    spent = 0
    while spent < budget and attempts < 50:
        for _ in range(30):
            env.step(IGNITION)
            spent += 1
            if spent >= budget:
                break
        spent_reported = env.debug_info()["trial_steps_used"]
        assert spent_reported == spent
        if spent < budget:
            _, info = env.reset()
            assert not info["trial_start"]
            attempts += 1
    assert attempts >= 3
    env.close()


def test_vector_wrapper_tracks_the_same_boundary(budget_config):
    vec = MarsRoverVecEnv(2, config_path=str(budget_config))
    vec.reset(seed=11)
    assert not vec.next_trial_start.any()
    budget = vec.core.trial_step_budget(0)
    actions = [IGNITION, IGNITION]
    for _ in range(budget + 5):
        _, _, terminated, truncated, _ = vec.step(actions)
        if terminated.any() or truncated.any():
            break
    assert vec.next_trial_start.any()
    assert vec.core.trial_steps_used(0) >= budget or vec.core.trial_steps_used(1) >= budget


def test_disabled_budget_keeps_the_old_episode_protocol(tmp_path):
    import yaml

    path = tmp_path / "classic.yaml"
    path.write_text(
        yaml.safe_dump({
            "env": {"episodes_per_trial": 2, "biome_split": 1},
            "termination": {"max_steps": 40, "trial_time_limit": 0.0},
        }),
        encoding="utf-8",
    )
    env = MarsRoverEnv(config_path=str(path))
    env.reset(seed=21, options={"trial_start": True})
    assert env.debug_info()["trial_step_budget"] == 0
    assert env.debug_info()["trial_time_left"] == pytest.approx(-1.0)
    truncated = False
    for _ in range(60):
        _, _, terminated, truncated, _ = env.step(IGNITION)
        if terminated or truncated:
            break
    assert truncated
    _, info = env.reset()
    assert not info["trial_start"]
    _, info = env.reset()
    assert info["trial_start"]
    env.close()


def test_budget_in_seconds_matches_wall_clock_of_the_simulation(budget_config):
    env = MarsRoverEnv(config_path=str(budget_config))
    env.reset(seed=31, options={"trial_start": True})
    budget = env.debug_info()["trial_step_budget"]
    assert math.isclose(budget * DT, 2.0, rel_tol=1e-3)
    env.close()
