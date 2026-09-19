from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from _mars_rover_cpp import MarsRoverBatchEnv, biome_catalog
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import load_env_config


EPISODE_SECONDS = 300.0
SEEDS = (104729, 130363, 155921)
CHAIN_GROUPS = 5


def _test_biomes() -> list[dict]:
    biomes = [dict(item) for item in biome_catalog() if int(item["split"]) == 2]
    if len(biomes) != 10:
        raise RuntimeError(f"organizer build must contain exactly 10 test biomes, got {len(biomes)}")
    return biomes


def _config(*, fixed_biome_id: int | None):
    config = load_env_config()
    config.biome_split = 2
    config.fixed_biome_id = -1 if fixed_biome_id is None else fixed_biome_id
    config.chain_biomes = fixed_biome_id is None
    config.force_endgame_difficulty = True
    config.termination.trial_time_limit = EPISODE_SECONDS
    config.termination.max_steps = round(EPISODE_SECONDS / config.physics.dt)
    return config


def _suite():
    configs = []
    seeds = []
    labels = []
    for biome in _test_biomes():
        for seed in SEEDS:
            configs.append(_config(fixed_biome_id=int(biome["index"])))
            seeds.append(seed)
            labels.append(f"biome:{biome['id']}")
    for group in range(CHAIN_GROUPS):
        for seed in SEEDS:
            configs.append(_config(fixed_biome_id=None))
            seeds.append(seed + 1_000_003 * (group + 1))
            labels.append(f"chain:{group + 1}")
    return configs, seeds, labels


def evaluate(model_path: Path, output_path: Path, device: str) -> dict:
    from stable_baselines3 import PPO

    configs, seeds, labels = _suite()
    batch = MarsRoverBatchEnv(configs)
    count = batch.num_envs
    observations = np.zeros((count, batch.obs_dim), dtype=np.float32)
    rewards = np.zeros(count, dtype=np.float32)
    terminated = np.zeros(count, dtype=np.uint8)
    truncated = np.zeros(count, dtype=np.uint8)
    actions = np.zeros(count, dtype=np.int32)
    for index, seed in enumerate(seeds):
        batch.reset_at(index, seed, True, observations[index])
    starts = np.asarray([batch.debug_info(i)["x"] for i in range(count)], dtype=np.float64)
    model = PPO.load(str(model_path), device=device)
    action_count = int(model.action_space.n)
    if action_count < 1 or action_count > len(ACTION_MACROS):
        raise RuntimeError(
            f"model has {action_count} actions, organizer contract supports 1..{len(ACTION_MACROS)}"
        )
    macro_table = np.asarray(ACTION_MACROS[:action_count], dtype=np.int32)

    done = np.zeros(count, dtype=bool)
    final_x = starts.copy()
    started = time.perf_counter()
    max_steps = max(config.termination.max_steps for config in configs)
    for _ in range(max_steps):
        action_indices, _ = model.predict(observations, deterministic=True)
        actions[:] = macro_table[np.asarray(action_indices, dtype=np.int64).reshape(count)]
        actions[done] = 0
        batch.step(actions, observations, rewards, terminated, truncated)
        just_done = np.logical_and(~done, np.logical_or(terminated != 0, truncated != 0))
        for index in np.flatnonzero(just_done):
            final_x[index] = float(batch.debug_info(int(index))["x"])
        done |= just_done
        if done.all():
            break
    for index in np.flatnonzero(~done):
        final_x[index] = float(batch.debug_info(int(index))["x"])
    wall_seconds = time.perf_counter() - started
    distances = final_x - starts
    rows = [
        {"scenario": labels[i], "seed": seeds[i], "distance": float(distances[i])}
        for i in range(count)
    ]
    result = {
        "score": float(np.mean(distances)),
        "episodes": count,
        "episode_seconds": EPISODE_SECONDS,
        "simulated_seconds": count * EPISODE_SECONDS,
        "wall_seconds": wall_seconds,
        "realtime_speedup": count * EPISODE_SECONDS / max(wall_seconds, 1e-9),
        "biome_mean": float(np.mean(distances[:30])),
        "chain_mean": float(np.mean(distances[30:])),
        "results": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, default=Path("score.json"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate(args.model, args.output, args.device)
    print(json.dumps({key: result[key] for key in (
        "score", "biome_mean", "chain_mean", "episodes", "wall_seconds", "realtime_speedup"
    )}, indent=2))


if __name__ == "__main__":
    main()
