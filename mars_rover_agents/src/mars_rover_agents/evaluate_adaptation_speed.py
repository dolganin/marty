"""Validation protocol: which checkpoint adapts FASTEST to a brand-new chain?

Generates a pile of fresh procedurally-chained trials (see EnvConfig.chain_biomes)
and, for each candidate checkpoint, measures how many attempts (episodes) and how
many environment steps of on-trial fine-tuning it needs before it clears a target
distance on that specific trial - then averages over trials. This is deliberately
NOT a return curve: it is "time/attempts to adapt", which is the quantity the
Reptile baseline (train_reptile_ppo.py) is actually optimised for, and the one a
memoryless robust-PPO checkpoint (no fine-tuning ability built in, but still
capable of gradient updates) can be compared against fairly on the same axis.

Per trial, per candidate:
  1. Load the candidate's weights fresh (every trial starts from the same point).
  2. attempt 1: roll one deterministic episode on the trial, record distance.
  3. If short of the target: fine-tune for --finetune-steps on that SAME trial
     (the chain does not change until the vec-env's own episodes_per_trial count
     rolls over - see MarsRoverSb3VecEnv.step_wait), then try again.
  4. Stop at --max-attempts or once the target is cleared.

Report per candidate: mean attempts-to-target, mean env-steps-to-target, success
rate, averaged over all trials. The candidate with the lowest attempts/steps at a
comparable success rate is the fastest-adapting one.

Usage:
    python -m mars_rover_agents.evaluate_adaptation_speed \\
        --candidate reptile=runs/reptile/final/model.zip \\
        --candidate robust=runs/robust/model.zip \\
        --trials 30 --target-distance 150 --max-attempts 6
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure adaptation speed on fresh chained trials")
    parser.add_argument(
        "--candidate", action="append", required=True, metavar="NAME=PATH",
        help="repeatable; a checkpoint to score, e.g. --candidate reptile=runs/reptile/final/model.zip",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed-begin", type=int, default=50_000)
    parser.add_argument("--target-distance", type=float, default=150.0)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--max-episode-steps", type=int, default=1500)
    parser.add_argument("--finetune-steps", type=int, default=4096)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def _parse_candidates(specs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--candidate must be NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        out[name] = Path(path)
    return out


def _score_trial(model_path: Path, config_path: Path, trial_seed: int, *,
                  target_distance: float, max_attempts: int, max_episode_steps: int,
                  finetune_steps: int) -> dict[str, Any]:
    """One trial: repeated (deterministic attempt, fine-tune) rounds on ONE fixed chain."""
    from stable_baselines3 import PPO

    from mars_rover_env import MarsRoverEnv

    # A single-env vec keeps the fine-tuning env pinned to one chain (see
    # MarsRoverSb3VecEnv.step_wait / trial_start), and the model is loaded
    # against it up front so a later .learn() call never has to change the
    # env's n_envs (PPO.set_env requires an exact match).
    vec_env = MarsRoverSb3VecEnv(
        1, list(ACTION_MACROS), config_path=str(config_path), biome_split=1, seed=trial_seed,
    )
    model = PPO.load(str(model_path), env=vec_env, device="cpu")
    env = MarsRoverEnv(config_path=str(config_path), biome_split=1)
    total_steps = 0
    best_distance = 0.0
    success = False
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        obs, _ = env.reset(seed=trial_seed, options={"trial_start": attempt == 1})
        start_x = env.debug_info()["x"]
        for step in range(max_episode_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            total_steps += 1
            if terminated or truncated:
                break
        distance = env.debug_info()["x"] - start_x
        best_distance = max(best_distance, distance)
        attempts_used = attempt
        if distance >= target_distance:
            success = True
            break
        if attempt == max_attempts:
            break
        model.learn(total_timesteps=finetune_steps, reset_num_timesteps=False, progress_bar=False)
        total_steps += finetune_steps
    env.close()
    vec_env.close()
    return {
        "trial_seed": trial_seed,
        "success": success,
        "attempts_used": attempts_used,
        "env_steps_used": total_steps,
        "best_distance": best_distance,
    }


def main() -> None:
    args = build_parser().parse_args()

    candidates = _parse_candidates(args.candidate)
    results: dict[str, list[dict[str, Any]]] = {}

    for name, path in candidates.items():
        print(f"=== {name} ({path}) ===", flush=True)
        trial_rows = []
        wall_start = time.perf_counter()
        for trial_index in range(args.trials):
            trial_seed = args.seed_begin + trial_index
            row = _score_trial(
                path, args.config, trial_seed,
                target_distance=args.target_distance, max_attempts=args.max_attempts,
                max_episode_steps=args.max_episode_steps, finetune_steps=args.finetune_steps,
            )
            trial_rows.append(row)
            print(f"  trial {trial_index + 1}/{args.trials} seed={trial_seed} "
                  f"success={row['success']} attempts={row['attempts_used']} "
                  f"steps={row['env_steps_used']} best_distance={row['best_distance']:.1f}", flush=True)
        wall_seconds = time.perf_counter() - wall_start
        results[name] = trial_rows

        success_rate = float(np.mean([r["success"] for r in trial_rows]))
        mean_attempts = float(np.mean([r["attempts_used"] for r in trial_rows]))
        mean_steps = float(np.mean([r["env_steps_used"] for r in trial_rows]))
        mean_distance = float(np.mean([r["best_distance"] for r in trial_rows]))
        print(f"  -- {name}: success_rate={success_rate:.2f} mean_attempts={mean_attempts:.2f} "
              f"mean_env_steps={mean_steps:.0f} mean_best_distance={mean_distance:.1f} "
              f"wall_seconds={wall_seconds:.0f}\n")

    summary = {
        name: {
            "success_rate": float(np.mean([r["success"] for r in rows])),
            "mean_attempts_to_target": float(np.mean([r["attempts_used"] for r in rows])),
            "mean_env_steps_to_target": float(np.mean([r["env_steps_used"] for r in rows])),
            "mean_best_distance": float(np.mean([r["best_distance"] for r in rows])),
        }
        for name, rows in results.items()
    }
    ranked = sorted(
        summary.items(),
        key=lambda item: (-item[1]["success_rate"], item[1]["mean_attempts_to_target"]),
    )
    print("RANKED (highest success rate, then fewest attempts, wins):")
    for name, stats in ranked:
        print(f"  {name}: {stats}")

    payload = {
        "target_distance": args.target_distance,
        "max_attempts": args.max_attempts,
        "trials": args.trials,
        "summary": summary,
        "per_trial": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name=f"adaptation-speed-eval_{args.trials}trials",
            params={"target_distance": args.target_distance, "max_attempts": args.max_attempts,
                    "trials": args.trials, "candidates": list(candidates)},
            tags={"algorithm": "adaptation-speed-eval"},
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
        for name, stats in summary.items():
            tracker.log_metrics({f"{name}.{key}": value for key, value in stats.items()}, step=0)
        if args.output:
            tracker.log_artifact(args.output, artifact_path="adaptation_speed")
        tracker.close()


if __name__ == "__main__":
    main()
