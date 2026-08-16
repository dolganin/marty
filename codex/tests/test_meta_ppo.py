import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codex" / "src"))

from meta_ppo import Agent, RunningMeanStd, summary


def test_trial_start_clears_recurrent_memory():
    torch.manual_seed(0)
    agent = Agent(obs_dim=7, action_dim=4, hidden_size=16)
    obs = torch.randn(2, 7)
    action = torch.zeros(2, dtype=torch.long)
    scalar = torch.zeros(2)
    old_hidden = torch.randn(2, 16)
    start = torch.ones(2)
    logits_a, values_a, hidden_a = agent.step(
        obs, action, scalar, scalar, scalar, start, old_hidden
    )
    logits_b, values_b, hidden_b = agent.step(
        obs, action, scalar, scalar, scalar, start, torch.zeros_like(old_hidden)
    )
    assert torch.equal(logits_a, logits_b)
    assert torch.equal(values_a, values_b)
    assert torch.equal(hidden_a, hidden_b)


def test_episode_boundary_does_not_clear_memory():
    torch.manual_seed(1)
    agent = Agent(obs_dim=7, action_dim=4, hidden_size=16)
    obs = torch.randn(2, 7)
    action = torch.zeros(2, dtype=torch.long)
    scalar = torch.zeros(2)
    previous_done = torch.ones(2)
    start = torch.zeros(2)
    old_hidden = torch.randn(2, 16)
    _, _, hidden_a = agent.step(
        obs, action, scalar, previous_done, scalar, start, old_hidden
    )
    _, _, hidden_b = agent.step(
        obs, action, scalar, previous_done, scalar, start, torch.zeros_like(old_hidden)
    )
    assert not torch.allclose(hidden_a, hidden_b)


def test_running_statistics_round_trip():
    rms = RunningMeanStd(3)
    rms.update(np.asarray([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
    restored = RunningMeanStd(3)
    restored.load_state_dict(rms.state_dict())
    probe = np.asarray([[2, 3, 4]], dtype=np.float32)
    np.testing.assert_array_equal(rms.normalize(probe), restored.normalize(probe))


def test_trial_summary_measures_adaptation_after_respawn():
    attempts = [
        {"env": 0.0, "trial": 0.0, "attempt": 1.0, "distance_m": 10.0},
        {"env": 0.0, "trial": 0.0, "attempt": 2.0, "distance_m": 25.0},
    ]
    trials = [{"return": 1.0, "best_distance_m": 25.0, "total_distance_m": 35.0,
               "attempts": 2.0, "success": 0.0}]
    result = summary(trials, attempts)
    assert result["adaptation_distance_delta_mean"] == 15.0
    assert result["best_distance_m_mean"] == 25.0
