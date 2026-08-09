import numpy as np
import pytest

from mars_rover_env import MarsRoverEnv, MarsRoverVecEnv


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
    assert not truncated  # max_steps=0 disables the episode time limit.
    assert info == {}
    assert env.render().shape == (45, 80, 3)


def test_native_observation_uses_policy_friendly_scales():
    env = MarsRoverEnv(fixed_biome_id=0)
    obs, _ = env.reset(seed=42)
    assert obs[0] == pytest.approx(0.01)  # spawn x / finish_x
    assert obs[6] == pytest.approx(1.0)  # energy / capacity
    obs, *_ = env.step(4095)
    assert obs[-3] == pytest.approx(1.0)  # 12-bit action mask
    assert np.max(np.abs(obs)) < 20.0
    env.close()


def test_vector_env():
    env = MarsRoverVecEnv(4)
    assert env.max_steps == 3000
    assert env.reset(seed=7).shape == (4, env.obs_dim)
    result = env.step(np.zeros(4, dtype=np.int32))
    assert result[0].shape == (4, env.obs_dim)


def test_vector_env_tracks_trial_boundaries():
    env = MarsRoverVecEnv(2)
    env.reset(seed=7)
    assert not env.next_trial_start.any()
    for env_id in range(2):
        for episode in range(1, env.episodes_per_trial):
            env.reset_at(env_id, seed=7 + episode, trial_start=False)
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

    config = tmp_path / "short.yaml"
    config.write_text(
        "env:\n  episodes_per_trial: 2\ntermination:\n  max_steps: 1\n",
        encoding="utf-8",
    )
    env = MarsRoverSb3VecEnv(2, [0], config_path=str(config), seed=11)
    env.reset()
    _, _, dones, infos = env.step(np.zeros(2, dtype=np.int64))
    assert dones.all()
    assert not any(info["trial_start"] for info in infos)
    _, _, dones, infos = env.step(np.zeros(2, dtype=np.int64))
    assert dones.all()
    assert all(info["trial_start"] for info in infos)
    env.close()
