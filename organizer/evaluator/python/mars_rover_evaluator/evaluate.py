from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from _mars_rover_cpp import MarsRoverBatchEnv, biome_catalog
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import load_env_config


EPISODE_SECONDS = 300.0
BASE_SEED_COUNT = 16
REPLICAS_PER_SCENARIO = 3
CHAIN_GROUPS = 7
BIOME_SCENARIOS = BASE_SEED_COUNT - CHAIN_GROUPS
BIOME_RUNS = BIOME_SCENARIOS * REPLICAS_PER_SCENARIO
CHAIN_ONLY_BIOMES = ("gale_dark_flats", "battery_bay_tycho")
TEST_BIOME_COUNT = BIOME_SCENARIOS + len(CHAIN_ONLY_BIOMES)


def _test_biomes() -> list[dict]:
    biomes = [dict(item) for item in biome_catalog() if int(item["split"]) == 2]
    if len(biomes) != TEST_BIOME_COUNT:
        raise RuntimeError(
            f"organizer build must contain exactly {TEST_BIOME_COUNT} test biomes, got {len(biomes)}"
        )
    scenarios = [item for item in biomes if item["id"] not in CHAIN_ONLY_BIOMES]
    if len(scenarios) != BIOME_SCENARIOS:
        present = {item["id"] for item in biomes}
        missing = sorted(set(CHAIN_ONLY_BIOMES) - present)
        raise RuntimeError(f"chain-only biomes are not in the bank: {', '.join(missing)}")
    return scenarios


def _config(*, fixed_biome_id: int | None):
    config = load_env_config()
    config.biome_split = 2
    config.fixed_biome_id = -1 if fixed_biome_id is None else fixed_biome_id
    config.chain_biomes = fixed_biome_id is None
    config.force_endgame_difficulty = True
    config.termination.trial_time_limit = EPISODE_SECONDS
    config.termination.max_steps = round(EPISODE_SECONDS / config.physics.dt)
    return config


def _seed_file() -> Path:
    configured = os.environ.get("MARS_ROVER_TEST_SEEDS_FILE")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / ".env"


def _base_seeds() -> tuple[int, ...]:
    path = _seed_file()
    if not path.is_file():
        raise RuntimeError(f"test seed file does not exist: {path}")
    seed_list = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        if key.strip() == "MARS_ROVER_TEST_SEEDS" and separator:
            seed_list = value.strip()
            break
    if seed_list is None:
        raise RuntimeError(f"{path} must define MARS_ROVER_TEST_SEEDS")
    values = []
    for value in seed_list.split(","):
        try:
            seed = int(value.strip(), 0)
        except ValueError as error:
            raise RuntimeError(f"invalid test seed in {path}: {value}") from error
        if seed < 0 or seed > (1 << 64) - REPLICAS_PER_SCENARIO:
            raise RuntimeError(f"test seed is out of range in {path}: {seed}")
        values.append(seed)
    if len(values) != BASE_SEED_COUNT:
        raise RuntimeError(
            f"{path} must contain exactly {BASE_SEED_COUNT} test seeds, got {len(values)}"
        )
    return tuple(values)


def _suite():
    configs = []
    seeds = []
    labels = []
    scenarios = [
        (f"biome:{biome['id']}", int(biome["index"]))
        for biome in _test_biomes()
    ]
    scenarios.extend((f"chain:{group + 1}", None) for group in range(CHAIN_GROUPS))
    base_seeds = _base_seeds()
    if len(scenarios) != len(base_seeds):
        raise RuntimeError("test scenario and seed counts differ")
    for (label, biome_id), base_seed in zip(scenarios, base_seeds, strict=True):
        for offset in range(REPLICAS_PER_SCENARIO):
            configs.append(_config(fixed_biome_id=biome_id))
            seeds.append(base_seed + offset)
            labels.append(label)
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
    model = PPO.load(str(model_path), device=device)
    action_count = int(model.action_space.n)
    if action_count != len(ACTION_MACROS):
        raise RuntimeError(
            f"model has {action_count} actions, organizer contract requires {len(ACTION_MACROS)}"
        )
    macro_table = np.asarray(ACTION_MACROS, dtype=np.int32)

    done = np.zeros(count, dtype=bool)
    best_distances = np.zeros(count, dtype=np.float64)
    started = time.perf_counter()
    max_steps = max(config.termination.max_steps for config in configs)
    for _ in range(max_steps):
        action_indices, _ = model.predict(observations, deterministic=True)
        actions[:] = macro_table[np.asarray(action_indices, dtype=np.int64).reshape(count)]
        actions[done] = 0
        batch.step(actions, observations, rewards, terminated, truncated)
        just_done = np.logical_and(~done, np.logical_or(terminated != 0, truncated != 0))
        for index in np.flatnonzero(just_done):
            best_distances[index] = float(batch.debug_info(int(index))["best_distance_m"])
        done |= just_done
        if done.all():
            break
    for index in np.flatnonzero(~done):
        best_distances[index] = float(batch.debug_info(int(index))["best_distance_m"])
    wall_seconds = time.perf_counter() - started
    if not np.isfinite(best_distances).all():
        raise RuntimeError("evaluation produced a non-finite distance")
    distances = np.maximum(best_distances, 0.0)
    rows = [{
        "scenario": labels[i],
        "seed": seeds[i],
        "base_seed": seeds[i] - i % REPLICAS_PER_SCENARIO,
        "replica": i % REPLICAS_PER_SCENARIO,
        "distance": float(distances[i]),
    } for i in range(count)]
    result = {
        "score": float(np.median(distances)),
        "episodes": count,
        "episode_seconds": EPISODE_SECONDS,
        "simulated_seconds": count * EPISODE_SECONDS,
        "wall_seconds": wall_seconds,
        "realtime_speedup": count * EPISODE_SECONDS / max(wall_seconds, 1e-9),
        "biome_mean": float(np.mean(distances[:BIOME_RUNS])),
        "chain_mean": float(np.mean(distances[BIOME_RUNS:])),
        "results": rows,
    }
    summary_keys = ("score", "biome_mean", "chain_mean", "wall_seconds", "realtime_speedup")
    if not all(np.isfinite(result[key]) for key in summary_keys):
        raise RuntimeError("evaluation produced a non-finite summary metric")
    for key in ("score", "biome_mean", "chain_mean"):
        if result[key] < 0.0:
            raise RuntimeError(f"evaluation produced an out-of-range {key}: {result[key]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
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
