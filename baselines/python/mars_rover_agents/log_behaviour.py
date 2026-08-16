"""What does the policy actually DO at inference? Render it, trace it, log it.

Aggregate distance numbers say a policy scores 60 m and say nothing about how.
This produces, per episode, the three artefacts that answer that question, and
uploads all of them to MLflow:

* behaviour_<i>.gif - the rendered episode (debug overlay on), so the run can be
  watched rather than inferred from scalars.
* behaviour_<i>.png - per-step traces of position, speed, gear, energy, body
  angle and the chosen macro, on a shared time axis, with the termination cause
  in the title.
* behaviour.jsonl / summary.json - the same trace as data, plus two breakdowns
  that scalar metrics hide: how often each of the 15 public macros is chosen,
  and where the time and the deaths go per biome mechanic.

Episodes are picked to be representative rather than flattering: the trials are
sorted by distance and the run keeps the best, the median and the worst, so a
policy that looks fine on average but dies stupidly on a third of the chains
shows both faces.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from mars_rover_env.config import DEFAULT_ENV_CONFIG

from .actions import ACTION_MACROS
from .diagnose_ceiling import _classify
from .expert_dataset import stack_observations
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun

                                                                                
                                        
MACRO_NAMES = (
    "idle", "gas", "brake", "gas+reverse", "clutch+shiftup", "gas+shiftup",
    "gas+tiltleft+shiftup", "gas+tiltright+shiftup", "clutch+shiftdown", "ignition",
    "gas+tiltleft", "gas+tiltright", "lidar", "toggle_drive", "toggle_charge",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render and trace a checkpoint's inference behaviour")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--seed-begin", type=int, default=900_000)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--render-every", type=int, default=4,
                        help="keep every Nth frame in the GIF (60 fps native is unwatchable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def roll_episode(model, env, *, seed: int, stack: int, max_steps: int,
                 render_every: int, record: bool) -> dict[str, Any]:
    obs, _ = env.reset(seed=seed, options={"trial_start": True})
    obs_dim = int(env.observation_space.shape[0])
    history = [obs.astype(np.float32)]
    start_x = float(env.debug_info()["x"])
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    terminated = truncated = False
    total_reward = 0.0
    for step in range(max_steps):
        action, _ = model.predict(stack_observations(history, stack, obs_dim), deterministic=True)
        action = int(action)
        debug = env.debug_info()
        trace.append({
            "step": step,
            "x": float(debug["x"]),
            "speed": float(debug["speed"]),
            "gear": str(debug["gear"]),
            "rpm": float(debug["engine_rpm"]),
            "energy": float(debug["energy"]),
            "angle": float(debug["angle"]),
            "airborne": bool(debug["airborne"]),
            "mechanic": str(debug["mechanic"]),
            "action": action,
            "action_name": MACRO_NAMES[action],
        })
        if record and step % render_every == 0:
            rendered = env.render()
            if rendered is not None:
                frames.append(rendered)
        obs, reward, terminated, truncated, _ = env.step(int(ACTION_MACROS[action]))
        total_reward += float(reward)
        history.append(obs.astype(np.float32))
        if len(history) > stack:
            history.pop(0)
        if terminated or truncated:
            break
    debug = env.debug_info()
    return {
        "seed": seed,
        "distance": float(debug["x"]) - start_x,
        "steps": len(trace),
        "reward": total_reward,
                                                                              
                                                                           
        "cause": _classify(debug, truncated, terminated)
        if (terminated or truncated)
        else "BUDGET",
        "trace": trace,
        "frames": frames,
    }


def _write_gif(frames: list[np.ndarray], path: Path) -> bool:
    if not frames:
        return False
    from PIL import Image

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:],
                   duration=67, loop=0, optimize=False)
    return True


def _plot_trace(episode: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = episode["trace"]
    steps = [row["step"] for row in trace]
    figure, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    axes[0].plot(steps, [row["x"] for row in trace], color="tab:blue")
    axes[0].set_ylabel("x, m")
    axes[1].plot(steps, [row["speed"] for row in trace], color="tab:green")
    axes[1].set_ylabel("speed, m/s")
    gears = [int(row["gear"]) if row["gear"].isdigit() else 0 for row in trace]
    axes[2].step(steps, gears, color="tab:orange", where="post")
    axes[2].set_ylabel("gear (0 = N/R)")
    axes[3].plot(steps, [row["energy"] for row in trace], color="tab:red", label="energy")
    axes[3].plot(steps, [row["angle"] for row in trace], color="tab:purple", label="body angle")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_ylabel("energy / angle")
    axes[4].scatter(steps, [row["action"] for row in trace], s=2, color="tab:gray")
    axes[4].set_yticks(range(len(MACRO_NAMES)))
    axes[4].set_yticklabels(MACRO_NAMES, fontsize=6)
    axes[4].set_ylabel("chosen macro")
    axes[4].set_xlabel("step")
    figure.suptitle(
        f"seed {episode['seed']} - {episode['cause']} after {episode['steps']} steps, "
        f"{episode['distance']:.1f} m, return {episode['reward']:.1f}"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    from stable_baselines3 import PPO

    from mars_rover_env import MarsRoverEnv

    stack = max(1, int(args.frame_stack))
    args.output.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(args.model), device="cpu")
    env = MarsRoverEnv(config_path=str(args.config), biome_split=1, render_mode="debug_rgb_array")

                                                                              
                                                                                
    episodes = [
        roll_episode(model, env, seed=args.seed_begin + index, stack=stack,
                     max_steps=args.max_steps, render_every=args.render_every, record=False)
        for index in range(args.trials)
    ]
    for episode in episodes:
        print(f"  seed={episode['seed']} {episode['cause']:8s} dist={episode['distance']:7.1f} "
              f"steps={episode['steps']:5d} return={episode['reward']:8.1f}", flush=True)

    ranked = sorted(episodes, key=lambda item: item["distance"])
    picks = {"worst": ranked[0], "median": ranked[len(ranked) // 2], "best": ranked[-1]}

    action_counter: Counter[str] = Counter()
    mechanic_steps: Counter[str] = Counter()
    mechanic_deaths: Counter[str] = Counter()
    for episode in episodes:
        for row in episode["trace"]:
            action_counter[row["action_name"]] += 1
            mechanic_steps[row["mechanic"]] += 1
        if episode["cause"] in {"CRASH", "STUCK"} and episode["trace"]:
            mechanic_deaths[episode["trace"][-1]["mechanic"]] += 1

    total_steps = max(1, sum(action_counter.values()))
    summary: dict[str, Any] = {
        "model": str(args.model),
        "frame_stack": stack,
        "trials": args.trials,
        "mean_distance": float(np.mean([e["distance"] for e in episodes])),
        "mean_steps": float(np.mean([e["steps"] for e in episodes])),
        "cause_counts": dict(Counter(e["cause"] for e in episodes)),
        "action_fractions": {
            name: count / total_steps for name, count in action_counter.most_common()
        },
        "mechanic_step_fractions": {
            name: count / total_steps for name, count in mechanic_steps.most_common()
        },
        "deaths_by_mechanic": dict(mechanic_deaths),
        "episodes": [
            {key: value for key, value in episode.items() if key not in {"trace", "frames"}}
            for episode in episodes
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output / "behaviour.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            for row in episode["trace"]:
                handle.write(json.dumps({"seed": episode["seed"], **row}) + "\n")

    print("\naction usage (fraction of all steps):")
    for name, fraction in list(summary["action_fractions"].items())[:8]:
        print(f"  {name:24s} {fraction:6.3f}")
    print("deaths by mechanic:", dict(mechanic_deaths))

    rendered: dict[str, Path] = {}
    for label, episode in picks.items():
        replay = roll_episode(model, env, seed=episode["seed"], stack=stack,
                              max_steps=args.max_steps, render_every=args.render_every,
                              record=True)
        gif_path = args.output / f"behaviour_{label}.gif"
        if _write_gif(replay["frames"], gif_path):
            rendered[label] = gif_path
        _plot_trace(replay, args.output / f"behaviour_{label}.png")
        print(f"  rendered {label}: seed={episode['seed']} {episode['cause']} "
              f"{episode['distance']:.1f} m -> {gif_path.name}", flush=True)
    env.close()

    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name=args.run_name or f"behaviour_{args.model.parent.name}",
            params={"model": str(args.model), "frame_stack": stack, "trials": args.trials,
                    "seed_begin": args.seed_begin},
            tags={"algorithm": "behaviour-log"},
            tracking_uri=args.tracking_uri, experiment_name=args.experiment,
        )
        tracker.log_metrics({
            "behaviour.mean_distance": summary["mean_distance"],
            "behaviour.mean_steps": summary["mean_steps"],
            **{f"behaviour.action.{name}": value
               for name, value in summary["action_fractions"].items()},
            **{f"behaviour.cause.{name}": float(count) / args.trials
               for name, count in summary["cause_counts"].items()},
        }, step=0)
        tracker.log_artifact(args.output, artifact_path="behaviour")
        for label, path in rendered.items():
            if not tracker.artifact_exists(f"behaviour/{path.name}"):
                print(f"warning: {label} GIF did not land on the tracking server", flush=True)
        tracker.close()


if __name__ == "__main__":
    main()
