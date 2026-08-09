"""Explicit compressed record of earlier episodes in the current trial.

Test 1 showed the Transformer-XL memory channel is effectively dead: swapping in a
state accumulated on a completely different mechanic changed 0.9% of actions, and the
influence vanished entirely past ~128 steps. Carrying mechanic identity across an
800-step episode boundary through attention alone is not happening.

Rather than hope a bigger memory window fixes it, this module hands the policy the
same information directly: a fixed-width summary of each finished episode of the
trial, fed as ordinary input features on every step. Identifying the mechanic then
needs a linear read of a present input instead of attention over a long-evicted past.

Everything summarised here is derived from what the agent already receives
(observation, reward, done). No biome id, no privileged simulator state — the summary
is a convenience, not extra information, so held-out evaluation stays honest.
"""

from __future__ import annotations

import numpy as np


# Observation layout (mars/env.cpp build_observation), 124 floats:
#   0 x/finish, 1 y/10, 2 vx/20, 3 vy/20, 4 angle, 5 angular_velocity/10,
#   6 energy/capacity, 7 trial_start, 8..31 wheels (8 x contact/slip/normal_force),
#   32..79 lidar terrain, 80..103 lidar biome samples, 104..123 drivetrain/thermal/lidar.
OBS_X = 0
OBS_VX = 2
OBS_VY = 3
OBS_ANGLE = 4
OBS_ANGULAR_VELOCITY = 5
OBS_ENERGY = 6
WHEEL_BASE = 8
WHEEL_COUNT = 8
OBS_ENGINE_RPM = 107
OBS_ENGINE_TEMPERATURE = 111
OBS_AMBIENT_TEMPERATURE = 112

EPISODE_SUMMARY_DIM = 16


class TrialContextTracker:
    """Accumulates per-episode statistics and exposes the last `history` summaries.

    Batched over environments so training collection and single-agent evaluation can
    share one implementation; an agent is simply the `num_envs == 1` case.
    """

    def __init__(self, num_envs: int, *, history: int = 2, max_steps: int = 800) -> None:
        if num_envs < 1 or history < 1 or max_steps < 1:
            raise ValueError("num_envs, history and max_steps must be positive")
        self.num_envs = int(num_envs)
        self.history = int(history)
        self.max_steps = float(max_steps)
        self.feature_dim = self.history * (EPISODE_SUMMARY_DIM + 1)
        self._summaries = np.zeros(
            (self.num_envs, self.history, EPISODE_SUMMARY_DIM), dtype=np.float32
        )
        self._valid = np.zeros((self.num_envs, self.history), dtype=np.float32)
        self._reset_accumulators(np.ones(self.num_envs, dtype=bool))

    def _reset_accumulators(self, mask: np.ndarray) -> None:
        if not hasattr(self, "_sums"):
            self._sums = np.zeros((self.num_envs, EPISODE_SUMMARY_DIM), dtype=np.float64)
            self._steps = np.zeros(self.num_envs, dtype=np.float64)
            self._start_energy = np.zeros(self.num_envs, dtype=np.float64)
            self._started = np.zeros(self.num_envs, dtype=bool)
            self._max_speed = np.zeros(self.num_envs, dtype=np.float64)
            return
        self._sums[mask] = 0.0
        self._steps[mask] = 0.0
        self._started[mask] = False
        self._max_speed[mask] = 0.0

    def reset_trial(self, mask: np.ndarray) -> None:
        """Forget everything for the selected environments (trial boundary)."""
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not mask.any():
            return
        self._summaries[mask] = 0.0
        self._valid[mask] = 0.0
        self._reset_accumulators(mask)

    def observe(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        active: np.ndarray | None = None,
    ) -> None:
        """Fold one environment step into the running episode statistics."""
        obs = np.asarray(observations, dtype=np.float32).reshape(self.num_envs, -1)
        reward = np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
        done = np.asarray(dones, dtype=bool).reshape(self.num_envs)
        live = (
            np.ones(self.num_envs, dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool).reshape(self.num_envs)
        )

        energy = obs[:, OBS_ENERGY].astype(np.float64)
        fresh = live & ~self._started
        if fresh.any():
            self._start_energy[fresh] = energy[fresh]
            self._started[fresh] = True

        contact = obs[:, WHEEL_BASE : WHEEL_BASE + WHEEL_COUNT * 3 : 3].astype(np.float64)
        slip = obs[:, WHEEL_BASE + 1 : WHEEL_BASE + WHEEL_COUNT * 3 : 3].astype(np.float64)
        normal = obs[:, WHEEL_BASE + 2 : WHEEL_BASE + WHEEL_COUNT * 3 : 3].astype(np.float64)
        contact_fraction = contact.mean(axis=1)
        # Slip only means something on a wheel that is touching ground; averaging over
        # airborne wheels would dilute exactly the signal that identifies traction.
        grounded = contact.sum(axis=1)
        mean_slip = np.where(grounded > 0.0, (slip * contact).sum(axis=1) / np.maximum(grounded, 1.0), 0.0)

        velocity_x = obs[:, OBS_VX].astype(np.float64)
        step_values = np.stack(
            [
                velocity_x,
                np.abs(obs[:, OBS_VY].astype(np.float64)),
                np.abs(obs[:, OBS_ANGULAR_VELOCITY].astype(np.float64)),
                np.abs(obs[:, OBS_ANGLE].astype(np.float64)),
                mean_slip,
                contact_fraction,
                normal.mean(axis=1),
                obs[:, OBS_ENGINE_RPM].astype(np.float64),
                obs[:, OBS_ENGINE_TEMPERATURE].astype(np.float64),
                obs[:, OBS_AMBIENT_TEMPERATURE].astype(np.float64),
                reward,
                obs[:, OBS_X].astype(np.float64),
                energy,
                np.zeros(self.num_envs),
                np.zeros(self.num_envs),
                np.zeros(self.num_envs),
            ],
            axis=1,
        )
        self._sums[live] += step_values[live]
        self._steps[live] += 1.0
        self._max_speed[live] = np.maximum(self._max_speed[live], velocity_x[live])

        finished = live & done
        if finished.any():
            self._close_episodes(finished, obs, energy)

    def _close_episodes(
        self, finished: np.ndarray, obs: np.ndarray, energy: np.ndarray
    ) -> None:
        steps = np.maximum(self._steps[finished], 1.0)
        means = self._sums[finished] / steps[:, None]
        summary = means.astype(np.float32)
        # Replace the placeholder slots with terminal facts that a mean would destroy.
        summary[:, 11] = obs[finished, OBS_X]  # final progress, not average progress
        summary[:, 12] = (self._start_energy[finished] - energy[finished]).astype(np.float32)
        summary[:, 13] = (steps / self.max_steps).astype(np.float32)
        summary[:, 14] = self._max_speed[finished].astype(np.float32)
        summary[:, 15] = np.abs(obs[finished, OBS_ANGLE]).astype(np.float32)
        rows = np.flatnonzero(finished)
        self._summaries[rows] = np.roll(self._summaries[rows], shift=1, axis=1)
        self._summaries[rows, 0] = summary
        self._valid[rows] = np.roll(self._valid[rows], shift=1, axis=1)
        self._valid[rows, 0] = 1.0
        self._reset_accumulators(finished)

    def features(self) -> np.ndarray:
        """Return [num_envs, history * (EPISODE_SUMMARY_DIM + 1)] context features."""
        flat = self._summaries.reshape(self.num_envs, self.history, EPISODE_SUMMARY_DIM)
        with_validity = np.concatenate((flat, self._valid[:, :, None]), axis=2)
        return np.ascontiguousarray(
            with_validity.reshape(self.num_envs, self.feature_dim), dtype=np.float32
        )


def trial_context_dim(history: int = 2) -> int:
    return int(history) * (EPISODE_SUMMARY_DIM + 1)
