"""The floor baseline: a five-rule scripted driver (see scripted_driver.py), no
learning at all. Any RL/meta-RL candidate that cannot beat this on the same
protocol as evaluate_adaptation_speed.py is not "bad at adapting" - it has not
even matched a driver that does not look at the biome at all.

Also records a GIF per trial (rendered with the HUD via debug_rgb_array) so a
human can actually see where the rover gets stuck, and logs everything -
metrics + GIFs - to MLflow as artifacts.

Usage:
    python -m mars_rover_agents.run_scripted_baseline \\
        --trials 16 --target-distance 60 --record-first 4
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from mars_rover_env.config import DEFAULT_ENV_CONFIG

from .scripted_driver import scripted_action
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scripted no-learning floor baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed-begin", type=int, default=80_000)
    parser.add_argument("--target-distance", type=float, default=60.0)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--record-first", type=int, default=4, help="GIFs for the first N trials")
    parser.add_argument("--record-every-n-steps", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("scripted_baseline_results.json"))
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def _save_gif(frames: list, path: Path, fps: int) -> None:
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    duration_ms = max(20, int(1000 / max(1, fps)))
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=True
    )


def _run_trial(config_path: Path, seed: int, max_steps: int, *, record: bool, record_every: int):
    from mars_rover_env import MarsRoverEnv

    render_mode = "debug_rgb_array" if record else None
    env = MarsRoverEnv(config_path=str(config_path), biome_split=1, render_mode=render_mode)
    env.reset(seed=seed, options={"trial_start": True})
    start_x = env.debug_info()["x"]
    energy_zero_step = None
    frames = []
    step = 0
    final = env.debug_info()
    for step in range(1, max_steps + 1):
        debug = env.debug_info()
        action = scripted_action(debug)
        _, _, terminated, truncated, _ = env.step(action)
        if record and (step % record_every == 0):
            frames.append(env.render())
        if terminated or truncated:
            final = env.debug_info()
            if final.get("energy", 1.0) <= 0.0:
                energy_zero_step = step
            break
    end_x = env.debug_info()["x"]
    env.close()
    return {
        "seed": seed,
        "distance": end_x - start_x,
        "steps": step,
        "energy_zero_step": energy_zero_step,
        "flipped": abs(final.get("angle", 0.0)) > 1.0,
        "final_gear": final.get("gear"),
        "mechanic_at_end": final.get("mechanic"),
    }, frames


def main() -> None:
    args = build_parser().parse_args()
    tracker = None
    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name="scripted-baseline-floor",
            params={
                "algorithm": "scripted-driver-no-learning",
                "trials": args.trials,
                "target_distance": args.target_distance,
                "max_steps": args.max_steps,
            },
            tags={"algorithm": "scripted-baseline"},
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="mars_scripted_gifs_") as gif_dir:
            gif_dir_path = Path(gif_dir)
            for i in range(args.trials):
                seed = args.seed_begin + i
                record = i < args.record_first
                row, frames = _run_trial(
                    args.config, seed, args.max_steps,
                    record=record, record_every=args.record_every_n_steps,
                )
                results.append(row)
                print(
                    f"trial {i + 1}/{args.trials} seed={seed} distance={row['distance']:.1f} "
                    f"steps={row['steps']} energy_zero_step={row['energy_zero_step']} "
                    f"flipped={row['flipped']} final_gear={row['final_gear']} "
                    f"mechanic_at_end={row['mechanic_at_end']}",
                    flush=True,
                )
                if record and frames:
                    gif_path = gif_dir_path / f"trial_{i:02d}_seed{seed}.gif"
                    _save_gif(frames, gif_path, fps=60 // args.record_every_n_steps)
                    if tracker is not None:
                        tracker.log_artifact(gif_path, artifact_path="scripted_baseline_gifs")
                if tracker is not None:
                    tracker.log_metrics({"trial.distance": row["distance"]}, step=i)

            distances = np.array([row["distance"] for row in results], dtype=float)
            success = distances >= args.target_distance
            energy_deaths = sum(1 for row in results if row["energy_zero_step"] is not None)
            summary = {
                "mean_distance": float(distances.mean()),
                "median_distance": float(np.median(distances)),
                "max_distance": float(distances.max()),
                "success_rate_at_target": float(success.mean()),
                "energy_death_fraction": energy_deaths / len(results),
                "trials": results,
            }
            args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(
                f"\n-- scripted baseline: mean={summary['mean_distance']:.1f} "
                f"median={summary['median_distance']:.1f} max={summary['max_distance']:.1f} "
                f"success_rate={summary['success_rate_at_target']:.2f} "
                f"energy_death_fraction={summary['energy_death_fraction']:.2f}",
                flush=True,
            )
            if tracker is not None:
                tracker.log_metrics(
                    {
                        "summary.mean_distance": summary["mean_distance"],
                        "summary.median_distance": summary["median_distance"],
                        "summary.max_distance": summary["max_distance"],
                        "summary.success_rate": summary["success_rate_at_target"],
                        "summary.energy_death_fraction": summary["energy_death_fraction"],
                    },
                    step=0,
                )
                tracker.log_artifact(args.output, artifact_path="results")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
