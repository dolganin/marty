import numpy as np
import pytest

from mars_rover_env import MarsRoverEnv, MarsRoverVecEnv
from mars_rover_env.config import load_env_config


def test_single_env_reset_step_and_render():
    env = MarsRoverEnv(render_mode="rgb_array", render_width=80, render_height=45)
    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert info["trial_start"] is True
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.dtype == np.float32
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert not truncated
    assert info == {}
    assert env.render().shape == (45, 80, 3)


def test_native_observation_uses_policy_friendly_scales():
    config = load_env_config()
    env = MarsRoverEnv(fixed_biome_id=0)
    obs, _ = env.reset(seed=42)
    # x is reported in units of the 1000 m reference distance (env.cpp, finish_scale).
    assert obs[0] == pytest.approx(1.0 / 1000.0)
    # energy is reported as a fraction of the battery capacity.
    assert obs[6] == pytest.approx(config.physics.initial_energy / config.physics.energy_capacity)
    assert 0.0 < obs[6] < 1.0
    obs, *_ = env.step(4095)
    assert np.all(np.isfinite(obs))
    assert np.max(np.abs(obs)) < 20.0
    env.close()


def test_vector_env():
    env = MarsRoverVecEnv(4)
    assert env.max_steps == load_env_config().termination.max_steps
    assert env.reset(seed=7).shape == (4, env.obs_dim)
    result = env.step(np.zeros(4, dtype=np.int32))
    assert result[0].shape == (4, env.obs_dim)


def test_vector_env_tracks_trial_boundaries(tmp_path):
    # v13 ends a trial on the shared clock, not after a fixed number of episodes.
    config = tmp_path / "short_trial.yaml"
    config.write_text("termination:\n  trial_time_limit: 1.0\n", encoding="utf-8")
    env = MarsRoverVecEnv(2, config_path=str(config))
    env.reset(seed=7)
    assert not env.next_trial_start.any()
    actions = np.zeros(2, dtype=np.int32)
    steps_per_trial = int(round(1.0 / load_env_config().physics.dt))
    for _ in range(steps_per_trial + 1):
        env.step(actions)
    assert env.next_trial_start.all()


def test_sb3_adapter_steps_native_batch_without_training(tmp_path):
    pytest.importorskip("stable_baselines3")
    from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

    env = MarsRoverSb3VecEnv(8, [0, 1, 1 << 9], seed=11)
    obs = env.reset()
    assert obs.shape == (8, env.native.obs_dim)
    obs, rewards, dones, infos = env.step(np.ones(8, dtype=np.int64))
    assert obs.shape == (8, env.native.obs_dim)
    assert rewards.shape == dones.shape == (8,)
    assert len(infos) == 8
    env.close()

    config = tmp_path / "short_trial.yaml"
    config.write_text("termination:\n  trial_time_limit: 0.1\n", encoding="utf-8")
    env = MarsRoverSb3VecEnv(2, [0], config_path=str(config), seed=11)
    env.reset()
    steps_per_trial = int(round(0.1 / load_env_config().physics.dt))
    boundaries = []
    for step in range(3 * steps_per_trial):
        _, _, dones, infos = env.step(np.zeros(2, dtype=np.int64))
        if dones.any():
            assert dones.all(), "the shared clock ends the trial in every env at once"
            boundaries.append((step, [info["trial_start"] for info in infos]))
    # The clock runs out once per trial_time_limit and opens a new trial each time.
    assert len(boundaries) == 3, boundaries
    assert all(all(flags) for _, flags in boundaries)
    assert [step for step, _ in boundaries] == [
        steps_per_trial * repeat - 1 for repeat in (1, 2, 3)
    ]
    env.close()
