"""Collect a scripted-expert dataset for behaviour cloning, with frame stacking.

Why this exists: AGENT_TASK_ADAPTATION.md's decisive negative fact is that a 7x
larger training budget moved nothing, and that every candidate covers the same
distance on the same trial seed regardless of algorithm. That is not a
meta-learning result, it is a bootstrapping result - nothing in the action
distribution reaches the first positive reward, because the drivetrain only
delivers torque after ignition -> powered upshift -> throttle, an edge-triggered
sequence (physics.cpp:212) that random exploration effectively never emits while
a stuck/flip termination is charging a large constant penalty either way.

scripted_driver.scripted_action already clears that barrier from five reactive
rules. So the cheapest way past the wall is not a better meta-algorithm: it is to
hand the learner a competent locomotion prior and let RL spend its budget on the
part that is actually about the biome chain.

Two details make the dataset useful rather than merely imitable:

* Frame stacking. The observation deliberately hides the mechanic (see the notes
  in env.cpp:build_observation), and the terrain/lidar channels are zero unless a
  scan is active - which costs energy and is on cooldown most of the time. The
  ONLY channel that carries mechanic identity is how the rover's own dynamics
  respond over several steps (slip vs throttle, normal force, acceleration). A
  memoryless MLP cannot compute that difference; a stack of K frames can. The
  stack layout here matches SB3's VecFrameStack (oldest first, newest last, zero
  padding at episode start) so a cloned policy transfers into PPO unchanged.
* Epsilon perturbation. Pure expert rollouts only cover the expert's own state
  distribution, so the clone falls off it on the first mistake. A fraction of
  uniformly random macros is injected into the ROLLOUT while the label stays the
  expert's action for the visited state - a single-pass DAgger-style widening
  that costs nothing extra to collect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import ACTION_MACROS
from .scripted_driver import scripted_action

                                                                              
                                                                          
                                                               
_MACRO_INDEX = {int(mask): index for index, mask in enumerate(ACTION_MACROS)}


def expert_macro_index(debug: dict) -> int:
    mask = int(scripted_action(debug))
    try:
        return _MACRO_INDEX[mask]
    except KeyError as exc:                                                     
        raise RuntimeError(
            f"scripted driver emitted mask {mask}, which is not in ACTION_MACROS"
        ) from exc


def stack_observations(history: list[np.ndarray], stack: int, obs_dim: int) -> np.ndarray:
    """Oldest-first concatenation, zero-padded at the head - VecFrameStack layout."""
    if stack <= 1:
        return history[-1]
    out = np.zeros(obs_dim * stack, dtype=np.float32)
    recent = history[-stack:]
    offset = (stack - len(recent)) * obs_dim
    out[offset:] = np.concatenate(recent)
    return out


@dataclass
class ExpertDataset:
    observations: np.ndarray                        
    actions: np.ndarray                      
                                                                                
                                                                                  
                                                                                
                                                               
    returns_to_go: np.ndarray
    episode_distances: np.ndarray
    episode_returns: np.ndarray

    def __len__(self) -> int:
        return int(self.observations.shape[0])


def collect_expert_dataset(
    *,
    config_path: str,
    episodes: int,
    stack: int,
    max_steps: int,
    seed_begin: int,
    epsilon: float,
    gamma: float = 0.995,
    biome_split: int = 1,
    progress_every: int = 25,
) -> ExpertDataset:
    from mars_rover_env import MarsRoverEnv

    rng = np.random.default_rng(seed_begin)
    env = MarsRoverEnv(config_path=config_path, biome_split=biome_split)
    obs_dim = int(env.observation_space.shape[0])

    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    all_returns_to_go: list[np.ndarray] = []
    distances: list[float] = []
    returns: list[float] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed_begin + episode, options={"trial_start": True})
        history: list[np.ndarray] = [obs.astype(np.float32)]
        start_x = env.debug_info()["x"]
        episode_return = 0.0
        episode_rewards: list[float] = []
        for _ in range(max_steps):
            label = expert_macro_index(env.debug_info())
            all_obs.append(stack_observations(history, stack, obs_dim))
            all_actions.append(label)
            taken = (
                int(rng.integers(0, len(ACTION_MACROS)))
                if epsilon > 0.0 and rng.random() < epsilon
                else label
            )
            obs, reward, terminated, truncated, _ = env.step(int(ACTION_MACROS[taken]))
            episode_return += float(reward)
            episode_rewards.append(float(reward))
            history.append(obs.astype(np.float32))
            if len(history) > stack:
                history.pop(0)
            if terminated or truncated:
                break
        rtg = np.zeros(len(episode_rewards), dtype=np.float32)
        running = 0.0
        for index in range(len(episode_rewards) - 1, -1, -1):
            running = episode_rewards[index] + gamma * running
            rtg[index] = running
        all_returns_to_go.append(rtg)
        distances.append(float(env.debug_info()["x"] - start_x))
        returns.append(episode_return)
        if progress_every and (episode + 1) % progress_every == 0:
            print(
                f"  expert episodes {episode + 1}/{episodes} "
                f"transitions={len(all_obs)} mean_distance={np.mean(distances):.1f}",
                flush=True,
            )
    env.close()
    return ExpertDataset(
        observations=np.asarray(all_obs, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        returns_to_go=np.concatenate(all_returns_to_go) if all_returns_to_go else np.zeros(0, np.float32),
        episode_distances=np.asarray(distances, dtype=np.float32),
        episode_returns=np.asarray(returns, dtype=np.float32),
    )
