from __future__ import annotations

import numpy as np
import torch

from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv
from mars_rover_agents.rl2_agent import RL2TransformerAgent
from mars_rover_agents.rl2_model import RL2TransformerActorCritic, rl2_features
from mars_rover_agents.rl2_training import collect_trial_batch, episode_objective_weights


def _small_model() -> RL2TransformerActorCritic:
    torch.manual_seed(4)
    return RL2TransformerActorCritic(
        124, model_dim=32, heads=4, layers=2, ff_dim=64, memory_len=8
    )


def test_chunk_and_online_transformer_are_causally_equivalent() -> None:
    model = _small_model().eval()
    features = torch.randn(2, 13, model.feature_dim)
    chunk_state = model.initial_state(2, torch.device("cpu"))
    chunk_logits, chunk_values, _ = model.forward_chunk(features, chunk_state)
    step_state = model.initial_state(2, torch.device("cpu"))
    logits = []
    values = []
    for step in range(features.shape[1]):
        step_logits, step_values, step_state = model.forward_step(features[:, step], step_state)
        logits.append(step_logits)
        values.append(step_values)
    assert torch.allclose(chunk_logits, torch.stack(logits, dim=1), atol=2.0e-5)
    assert torch.allclose(chunk_values, torch.stack(values, dim=1), atol=2.0e-5)


def test_agent_keeps_memory_at_episode_boundary_and_resets_at_trial() -> None:
    model = _small_model().eval()
    agent = RL2TransformerAgent(model, device=torch.device("cpu"))
    observation = np.zeros(124, dtype=np.float32)
    agent.reset(True)
    agent.act(observation)
    agent.observe(1.0, True, {"episode_index": 0})
    valid_before = int(agent.state.valid.sum())
    agent.reset(False)
    assert int(agent.state.valid.sum()) == valid_before
    assert agent.previous_reward == 1.0
    assert agent.previous_done is True
    agent.reset(True)
    assert int(agent.state.valid.sum()) == 0
    assert agent.previous_action == -1


def test_explicit_rl2_feedback_features() -> None:
    features = rl2_features(
        np.zeros((1, 124), dtype=np.float32),
        np.asarray([3]),
        np.asarray([2.5]),
        np.asarray([True]),
        device=torch.device("cpu"),
    )
    assert features.shape == (1, 138)
    assert features[0, 124 + 3] == 1.0
    assert torch.isclose(features[0, -2], torch.tensor(0.25))
    assert features[0, -1] == 1.0


def test_trial_gae_bootstraps_across_episode_boundaries(tmp_path) -> None:
    config = tmp_path / "short.yaml"
    config.write_text(
        "env:\n  episodes_per_trial: 4\ntermination:\n  max_steps: 2\n  stuck_steps: 180\n",
        encoding="utf-8",
    )
    env = MarsRoverVecEnv(2, config_path=str(config), biome_split=1)
    model = _small_model()
    batch = collect_trial_batch(
        env,
        model,
        device=torch.device("cpu"),
        seed=71,
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert batch.lengths.tolist() == [8, 8]
    assert batch.episode_ends[:, 0].nonzero().flatten().tolist() == [1, 3, 5, 7]
    assert torch.isclose(batch.returns[0, 0], batch.rewards[:, 0].sum(), atol=1.0e-5)
    weights = episode_objective_weights(batch, 4.0)
    assert weights[:, 0].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
