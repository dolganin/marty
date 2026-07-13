import numpy as np

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


def test_vector_env():
    env = MarsRoverVecEnv(4)
    assert env.reset(seed=7).shape == (4, env.obs_dim)
    result = env.step(np.zeros(4, dtype=np.int32))
    assert result[0].shape == (4, env.obs_dim)
