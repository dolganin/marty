"""Why is episode 1 of a trial so much better than episodes 2-4?

The distilled control drives ~3x better in episode 1 (raw return 1.89 vs 0.44/0.66/0.61)
even though its memory is provably inert, so "the agent adapts badly" cannot be the
explanation. Two candidates remain, and they need opposite fixes:

  (course)  the per-episode terrain seed makes later courses harder, so the metric is
            measuring the course lottery rather than the policy;
  (state)   entering an episode with accumulated recurrent state / non-reset RL2 inputs
            (previous action, previous reward, trial_start and episode_in_trial flags)
            degrades the policy, regardless of which course it faces.

This probe separates them by running the SAME course twice: once entered cold as a trial
start, once entered warm as a later episode of a trial. Any gap is attributable to entry
state alone, because the terrain seed is held identical.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mars_rover_env import MarsRoverEnv

from .actions import macro_action
from .rl2_agent import RL2TransformerAgent


def _run_episode(
    agent: RL2TransformerAgent,
    env: MarsRoverEnv,
    *,
    seed: int,
    trial_start: bool,
    max_steps: int,
) -> dict[str, float]:
    obs, _ = env.reset(seed=seed, options={"trial_start": trial_start})
    agent.reset(trial_start=trial_start)
    total = 0.0
    steps = 0
    for _ in range(max_steps):
        action = int(agent.act(obs))
        obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        agent.observe(float(reward), done, {})
        total += float(reward)
        steps += 1
        if done:
            break
    debug = env.debug_info()
    return {
        "raw_return": total,
        "steps": steps,
        "distance": float(debug.get("x", 0.0)) - 1.0,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import _mars_rover_cpp as native

    device = torch.device(args.device)
    catalog = [dict(item) for item in native.biome_catalog()]
    split_code = {"train": 1, "test": 2}[args.split]
    biomes = [item for item in catalog if int(item["split"]) == split_code][: args.biomes]

    loaded = RL2TransformerAgent.load(args.model, device=device, max_steps=args.max_steps)
    cold: list[float] = []
    warm: list[float] = []
    rows: list[dict[str, Any]] = []

    for trial in range(args.trials):
        trial_seed = args.seed + trial
        # The course episode `probe_episode` would normally face inside a trial.
        course_seed = trial_seed + args.probe_episode * 1_000_003
        for item in biomes:
            index = int(item["index"])

            # COLD: identical course, entered as a fresh trial start.
            env = MarsRoverEnv(
                config_path=args.config,
                biome_split=MarsRoverEnv.BIOME_MODE_ALL,
                fixed_biome_id=index,
            )
            agent = RL2TransformerAgent(
                loaded.model, device=device, deterministic=True, max_steps=args.max_steps
            )
            cold_result = _run_episode(
                agent, env, seed=course_seed, trial_start=True, max_steps=args.max_steps
            )
            env.close()

            # WARM: same course, but reached after playing the earlier episodes of the trial.
            env = MarsRoverEnv(
                config_path=args.config,
                biome_split=MarsRoverEnv.BIOME_MODE_ALL,
                fixed_biome_id=index,
            )
            agent = RL2TransformerAgent(
                loaded.model, device=device, deterministic=True, max_steps=args.max_steps
            )
            for episode in range(args.probe_episode):
                _run_episode(
                    agent,
                    env,
                    seed=trial_seed + episode * 1_000_003,
                    trial_start=episode == 0,
                    max_steps=args.max_steps,
                )
            warm_result = _run_episode(
                agent, env, seed=course_seed, trial_start=False, max_steps=args.max_steps
            )
            env.close()

            cold.append(cold_result["raw_return"])
            warm.append(warm_result["raw_return"])
            rows.append(
                {
                    "seed": trial_seed,
                    "biome": str(item["id"]),
                    "cold_return": cold_result["raw_return"],
                    "warm_return": warm_result["raw_return"],
                    "cold_distance": cold_result["distance"],
                    "warm_distance": warm_result["distance"],
                    "cold_steps": cold_result["steps"],
                    "warm_steps": warm_result["steps"],
                }
            )
        print(
            f"seed={trial_seed} cold_mean={np.mean(cold):.4f} warm_mean={np.mean(warm):.4f}",
            flush=True,
        )

    cold_array = np.asarray(cold)
    warm_array = np.asarray(warm)
    difference = warm_array - cold_array
    # Paired over (seed, biome): both conditions saw the identical course.
    standard_error = float(difference.std(ddof=1) / np.sqrt(len(difference))) if len(difference) > 1 else 0.0
    return {
        "model": str(args.model),
        "probe_episode": args.probe_episode,
        "pairs": len(rows),
        "cold_mean_return": float(cold_array.mean()),
        "warm_mean_return": float(warm_array.mean()),
        "warm_minus_cold": float(difference.mean()),
        "standard_error": standard_error,
        "z": float(difference.mean() / standard_error) if standard_error > 0 else 0.0,
        "rows": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cold vs warm entry on an identical course")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--biomes", type=int, default=10)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=90_000)
    parser.add_argument("--probe-episode", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"\ncold={results['cold_mean_return']:.4f} warm={results['warm_mean_return']:.4f} "
        f"warm-cold={results['warm_minus_cold']:+.4f} (se={results['standard_error']:.4f}, "
        f"z={results['z']:+.2f}) over {results['pairs']} paired courses"
    )


if __name__ == "__main__":
    main()
