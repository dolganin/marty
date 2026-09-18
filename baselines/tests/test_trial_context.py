from __future__ import annotations

import numpy as np
import pytest

from mars_rover_agents.trial_context import (
    EPISODE_SUMMARY_DIM,
    OBS_ENERGY,
    OBS_VX,
    OBS_X,
    TrialContextTracker,
    trial_context_dim,
)


OBS_DIM = 124


def _observation(overrides: dict[int, float] | None = None) -> np.ndarray:
    obs = np.zeros((1, OBS_DIM), dtype=np.float32)
    for index, value in (overrides or {}).items():
        obs[0, index] = value
    return obs


def test_context_is_empty_before_any_episode_finishes():
    tracker = TrialContextTracker(1, history=2, max_steps=100)
    for _ in range(10):
        tracker.observe(_observation({OBS_VX: 0.5}), np.array([1.0]), np.array([False]))
    features = tracker.features()
    assert features.shape == (1, trial_context_dim(2))
    assert np.all(features == 0.0), "unfinished episodes must not leak into the summary"


def test_finished_episode_becomes_the_most_recent_summary():
    tracker = TrialContextTracker(1, history=2, max_steps=100)
    for _ in range(9):
        tracker.observe(_observation({OBS_VX: 0.5}), np.array([1.0]), np.array([False]))
    tracker.observe(_observation({OBS_VX: 0.5, OBS_X: 0.42}), np.array([1.0]), np.array([True]))

    features = tracker.features()[0]
    summary = features[:EPISODE_SUMMARY_DIM]
    validity = features[EPISODE_SUMMARY_DIM]
    assert validity == pytest.approx(1.0)
    assert summary[0] == pytest.approx(0.5, abs=1e-6), "mean forward velocity"
    assert summary[11] == pytest.approx(0.42, abs=1e-6), "final progress, not mean progress"
    assert summary[13] == pytest.approx(10 / 100, abs=1e-6), "episode length fraction"
                                      
    assert features[EPISODE_SUMMARY_DIM + 1 :][EPISODE_SUMMARY_DIM] == pytest.approx(0.0)


def test_older_summary_shifts_back_and_statistics_do_not_bleed_across_episodes():
    tracker = TrialContextTracker(1, history=2, max_steps=100)
    for _ in range(4):
        tracker.observe(_observation({OBS_VX: 1.0}), np.array([0.0]), np.array([False]))
    tracker.observe(_observation({OBS_VX: 1.0}), np.array([0.0]), np.array([True]))
    for _ in range(4):
        tracker.observe(_observation({OBS_VX: 3.0}), np.array([0.0]), np.array([False]))
    tracker.observe(_observation({OBS_VX: 3.0}), np.array([0.0]), np.array([True]))

    features = tracker.features()[0]
    newest = features[:EPISODE_SUMMARY_DIM]
    older = features[EPISODE_SUMMARY_DIM + 1 : EPISODE_SUMMARY_DIM + 1 + EPISODE_SUMMARY_DIM]
    assert newest[0] == pytest.approx(3.0, abs=1e-6), "slot 0 holds the most recent episode"
    assert older[0] == pytest.approx(1.0, abs=1e-6), "the earlier episode shifts back"


def test_energy_consumption_is_a_difference_not_a_mean():
    tracker = TrialContextTracker(1, history=1, max_steps=100)
    tracker.observe(_observation({OBS_ENERGY: 1.0}), np.array([0.0]), np.array([False]))
    tracker.observe(_observation({OBS_ENERGY: 0.4}), np.array([0.0]), np.array([True]))
    summary = tracker.features()[0][:EPISODE_SUMMARY_DIM]
    assert summary[12] == pytest.approx(0.6, abs=1e-6)


def test_reset_trial_clears_history():
    tracker = TrialContextTracker(2, history=2, max_steps=100)
    tracker.observe(
        np.zeros((2, OBS_DIM), dtype=np.float32), np.array([1.0, 1.0]), np.array([True, True])
    )
    assert tracker.features().any()
    tracker.reset_trial(np.array([True, False]))
    features = tracker.features()
    assert not features[0].any(), "reset environment forgets the trial"
    assert features[1].any(), "untouched environment keeps its trial context"


def test_inactive_environments_are_not_accumulated():
    tracker = TrialContextTracker(2, history=1, max_steps=100)
    observations = np.zeros((2, OBS_DIM), dtype=np.float32)
    observations[:, OBS_VX] = 2.0
    tracker.observe(
        observations,
        np.array([0.0, 0.0]),
        np.array([True, True]),
        active=np.array([True, False]),
    )
    features = tracker.features()
    assert features[0, EPISODE_SUMMARY_DIM] == pytest.approx(1.0)
    assert features[1, EPISODE_SUMMARY_DIM] == pytest.approx(0.0), "finished slot must stay idle"
