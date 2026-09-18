"""Why did each episode end? Distance alone cannot tell a skill ceiling from a clock.

The adaptation metrics in this repo are all distance-shaped, and distance is only
a skill measurement while episodes end for skill reasons. Measured on the cloned
policy, 14 of 20 evaluation episodes ran the full 3000-step budget without dying:
at that point "mean distance" is just "mean speed x 50 s", and two policies that
differ in how well they read a mechanic can score the same because both ran out
of clock rather than out of ability.

This script reports the termination cause, the achieved average speed and the top
gear reached, per episode and aggregated, so any distance number produced
elsewhere can be read with the right meaning attached:

  CLOCK   - truncated at max_steps; distance is speed-limited, not skill-limited
  BATTERY - energy hit the termination floor
  CRASH   - flip or fatal fall
  STUCK   - the no-progress watchdog fired
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mars_rover_env.config import DEFAULT_ENV_CONFIG

from .actions import ACTION_MACROS
from .expert_dataset import stack_observations
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun

CAUSES = ("CLOCK", "BATTERY", "CRASH", "STUCK")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Termination-cause breakdown for a checkpoint")
    parser.add_argument("--candidate", action="append", required=True, metavar="NAME=PATH:STACK",
                        help="repeatable, e.g. --candidate k8=runs/x/model.zip:8")
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed-begin", type=int, default=900_000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def _classify(debug: dict, truncated: bool, terminated: bool, flip_angle: float = 2.2) -> str:
    if truncated and not terminated:
        return "CLOCK"
    if debug["energy"] <= 0.0:
        return "BATTERY"
    if abs(debug["angle"]) > flip_angle or debug["fatal_error"]:
        return "CRASH"
    return "STUCK"


def diagnose(model_path: Path, stack: int, *, config_path: Path, trials: int,
             seed_begin: int, max_steps: int) -> dict:
    from stable_baselines3 import PPO

    from mars_rover_env import MarsRoverEnv

    model = PPO.load(str(model_path), device="cpu")
    env = MarsRoverEnv(config_path=str(config_path), biome_split=1)
    obs_dim = int(env.observation_space.shape[0])
    rows = []
    for trial in range(trials):
        obs, _ = env.reset(seed=seed_begin + trial, options={"trial_start": True})
        history = [obs.astype(np.float32)]
        start_x = env.debug_info()["x"]
        top_gear = 0
        terminated = truncated = False
        for step in range(max_steps):
            action, _ = model.predict(
                stack_observations(history, stack, obs_dim), deterministic=True
            )
            obs, _, terminated, truncated, _ = env.step(int(ACTION_MACROS[int(action)]))
            history.append(obs.astype(np.float32))
            if len(history) > stack:
                history.pop(0)
            gear = str(env.debug_info()["gear"])
            if gear.isdigit():
                top_gear = max(top_gear, int(gear))
            if terminated or truncated:
                break
        debug = env.debug_info()
        steps = step + 1
        distance = float(debug["x"] - start_x)
        rows.append({
            "trial_seed": seed_begin + trial,
            "cause": _classify(debug, truncated, terminated),
            "distance": distance,
            "steps": steps,
            "top_gear": top_gear,
            "mean_speed": distance / (steps * 0.0166667),
        })
    env.close()

    summary = {
        "mean_distance": float(np.mean([r["distance"] for r in rows])),
        "mean_speed": float(np.mean([r["mean_speed"] for r in rows])),
        "mean_top_gear": float(np.mean([r["top_gear"] for r in rows])),
    }
    for cause in CAUSES:
        subset = [r for r in rows if r["cause"] == cause]
        summary[f"frac_{cause.lower()}"] = len(subset) / max(1, len(rows))
        summary[f"mean_distance_{cause.lower()}"] = (
            float(np.mean([r["distance"] for r in subset])) if subset else 0.0
        )
    return {"summary": summary, "per_trial": rows}


def main() -> None:
    args = build_parser().parse_args()
    candidates = {}
    for spec in args.candidate:
        name, rest = spec.split("=", 1)
        path, _, stack = rest.rpartition(":")
        candidates[name] = (Path(path), int(stack))

    payload = {}
    for name, (path, stack) in candidates.items():
        print(f"=== {name} ({path}, stack={stack}) ===", flush=True)
        result = diagnose(path, stack, config_path=args.config, trials=args.trials,
                          seed_begin=args.seed_begin, max_steps=args.max_steps)
        for row in result["per_trial"]:
            print(f"  seed={row['trial_seed']} {row['cause']:8s} dist={row['distance']:7.1f} "
                  f"steps={row['steps']:5d} top_gear={row['top_gear']} "
                  f"v={row['mean_speed']:.2f} m/s", flush=True)
        print("  " + " ".join(f"{k}={v:.3f}" for k, v in result["summary"].items()), flush=True)
        payload[name] = result

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name=f"ceiling-diagnosis_{'_'.join(candidates)}",
            params={"trials": args.trials, "max_steps": args.max_steps,
                    "candidates": {k: str(v[0]) for k, v in candidates.items()}},
            tags={"algorithm": "ceiling-diagnosis"},
            tracking_uri=args.tracking_uri, experiment_name=args.experiment,
        )
        for name, result in payload.items():
            tracker.log_metrics({f"{name}.{k}": v for k, v in result["summary"].items()}, step=0)
        if args.output:
            tracker.log_artifact(args.output, artifact_path="ceiling")
        tracker.close()


if __name__ == "__main__":
    main()
