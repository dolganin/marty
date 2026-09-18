"""Expert-bootstrapped, history-conditioned PPO for procedurally chained courses.

The method, and why it follows from AGENT_TASK_ADAPTATION.md rather than from the
meta-RL literature:

1. The note's decisive negative result is that 15M -> 100M steps changed nothing
   and that all four candidates covered the same distance on the same trial seed.
   A budget that buys no improvement and an outcome determined by the seed rather
   than by the weights both say the same thing: no candidate ever left the
   degenerate region of policy space. Measured here directly - holding throttle
   from spawn moves the rover 0.4 m in 601 steps, because the drivetrain delivers
   no torque until an edge-triggered ignition -> powered-upshift -> throttle
   sequence has been emitted (physics.cpp:212), while stuck termination charges a
   large constant penalty regardless of what the policy did. Every gradient early
   in training points away from the actions that would eventually pay.
   => Stage 1 removes the barrier by cloning scripted_driver, which clears it from
   five reactive rules and covers ~40 m where the trained candidates covered ~4.

2. What is left after locomotion is the actual task: the chain hides which
   mechanic the rover is standing on. The observation withholds MechanicType,
   biome params and ambient temperature by design, and the terrain/lidar channels
   are ZERO unless a scan is active - which costs energy and spends most of the
   episode on cooldown. The mechanic is therefore observable only in how the
   rover's own dynamics answer its own commands over several steps: slip against
   throttle, normal force, achieved acceleration. A memoryless MLP is
   architecturally unable to compute that; it can only react to an instantaneous
   slice. This is the concrete reason a memoryless policy plateaus here, and it is
   a representation problem, not a meta-learning one.
   => Stage 2 conditions the policy on a stack of K recent frames, so mechanic
   inference becomes a function the network can actually represent, and finetunes
   with PPO on the full chained distribution.

The two stages are deliberately separable so the experiment answers a question
instead of just producing a number: running this script with --frame-stack 1 and
--frame-stack K gives two policies with identical bootstrapping, identical budget
and identical evaluation, differing only in whether the architecture can see
history. Both are logged to MLflow as separate runs under the same experiment.

Evaluation deliberately reports distance-based metrics rather than the note's
"attempts to 150 m": at the achievable speeds a 3000-step episode cannot reach
150 m at all, so that target was unreachable by construction and its 0% success
rate carried no information about any policy. Reported instead: mean/median
distance on fresh chains, success rate at a reachable target, and the within-trial
delta across the episodes of a trial - which is the adaptation quantity a
history-conditioned policy is supposed to move without any gradient step.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS
from .expert_dataset import collect_expert_dataset, stack_observations
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun, training_provenance
from .trial_frame_stack import TrialFrameStack

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--frame-stack", type=int, default=8,
                        help="1 = memoryless control arm; >1 = history-conditioned")
    parser.add_argument("--expert-episodes", type=int, default=300)
    parser.add_argument("--expert-epsilon", type=float, default=0.1)
    parser.add_argument("--expert-max-steps", type=int, default=3000)
    parser.add_argument("--bc-epochs", type=int, default=12)
    parser.add_argument("--bc-batch-size", type=int, default=1024)
    parser.add_argument("--bc-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--ppo-timesteps", type=int, default=3_000_000)
    parser.add_argument("--ppo-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--eval-trials", type=int, default=24)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    parser.add_argument("--eval-target", type=float, default=60.0)
    parser.add_argument("--eval-seed-begin", type=int, default=900_000)
    parser.add_argument("--seed", type=int, default=7001)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "artifacts" / "runs")
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def evaluate_policy(
    model,
    *,
    config_path: Path,
    stack: int,
    trials: int,
    max_steps: int,
    target: float,
    seed_begin: int,
    episodes_per_trial: int = 4,
) -> dict[str, float]:
    """Deterministic rollouts on fresh chains, reported per episode-in-trial.

    Every trial replays the SAME chain for `episodes_per_trial` lives (the C++
    side keeps trial_mechanic_seed_ fixed while trial_start is False), so the
    difference between the first life and the later ones on one chain is exactly
    the within-trial adaptation a history-conditioned policy is supposed to buy -
    measured with frozen weights, so it cannot be confused with fine-tuning.
    """
    from mars_rover_env import MarsRoverEnv

    env = MarsRoverEnv(config_path=str(config_path), biome_split=1)
    obs_dim = int(env.observation_space.shape[0])
    per_life: list[list[float]] = [[] for _ in range(episodes_per_trial)]
    all_distances: list[float] = []
    successes: list[bool] = []
    for trial in range(trials):
        best = 0.0
        history: list[np.ndarray] = []
        for life in range(episodes_per_trial):
            obs, _ = env.reset(seed=seed_begin + trial, options={"trial_start": life == 0})
            if life == 0:
                                                                                 
                                                                              
                                  
                history = []
            history.append(obs.astype(np.float32))
            if len(history) > stack:
                history.pop(0)
            start_x = env.debug_info()["x"]
            for _ in range(max_steps):
                action, _ = model.predict(
                    stack_observations(history, stack, obs_dim), deterministic=True
                )
                obs, _, terminated, truncated, _ = env.step(int(ACTION_MACROS[int(action)]))
                history.append(obs.astype(np.float32))
                if len(history) > stack:
                    history.pop(0)
                if terminated or truncated:
                    break
            distance = float(env.debug_info()["x"] - start_x)
            per_life[life].append(distance)
            all_distances.append(distance)
            best = max(best, distance)
        successes.append(best >= target)
    env.close()

    metrics = {
        "eval.mean_distance": float(np.mean(all_distances)),
        "eval.median_distance": float(np.median(all_distances)),
        "eval.max_distance": float(np.max(all_distances)),
        "eval.success_rate": float(np.mean(successes)),
    }
    for life, values in enumerate(per_life):
        metrics[f"eval.life{life + 1}_distance"] = float(np.mean(values))
    metrics["eval.within_trial_delta"] = (
        metrics[f"eval.life{episodes_per_trial}_distance"] - metrics["eval.life1_distance"]
    )
    return metrics


def behaviour_clone(model, dataset, *, epochs: int, batch_size: int, learning_rate: float,
                    tracker: MLflowRun | None, device: str) -> dict[str, float]:
    """Supervised warm start of BOTH heads: policy by cross-entropy, value by MSE."""
    import torch

    observations = torch.as_tensor(dataset.observations, device=device)
    actions = torch.as_tensor(dataset.actions, device=device)
    returns = torch.as_tensor(dataset.returns_to_go, device=device)
    optimiser = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    count = observations.shape[0]
    generator = torch.Generator(device="cpu").manual_seed(0)
    stats: dict[str, float] = {}
    for epoch in range(epochs):
        order = torch.randperm(count, generator=generator).to(device)
        losses, accuracies, value_losses = [], [], []
        for start in range(0, count, batch_size):
            index = order[start : start + batch_size]
            batch_obs, batch_actions = observations[index], actions[index]
            values, log_prob, entropy = model.policy.evaluate_actions(batch_obs, batch_actions)
            policy_loss = -log_prob.mean()
            value_loss = torch.nn.functional.mse_loss(values.flatten(), returns[index])
            loss = policy_loss + 0.5 * value_loss - 0.001 * entropy.mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5)
            optimiser.step()
            with torch.no_grad():
                predicted = model.policy.get_distribution(batch_obs).distribution.probs.argmax(-1)
                accuracies.append(float((predicted == batch_actions).float().mean()))
            losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
        stats = {
            "bc.policy_loss": float(np.mean(losses)),
            "bc.value_loss": float(np.mean(value_losses)),
            "bc.accuracy": float(np.mean(accuracies)),
        }
        print(f"  bc epoch {epoch + 1}/{epochs} " + " ".join(f"{k}={v:.4f}" for k, v in stats.items()),
              flush=True)
        if tracker is not None:
            tracker.log_metrics(stats, step=epoch)
    return stats


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    stack = max(1, int(args.frame_stack))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"adaptive-bc-ppo_k{stack}_s{args.seed}_{stamp}"
    run_dir = (args.output_root / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, Any] = {
        "algorithm": "expert_bootstrapped_history_conditioned_ppo_v1",
        "frame_stack": stack,
        "expert_episodes": args.expert_episodes,
        "expert_epsilon": args.expert_epsilon,
        "bc_epochs": args.bc_epochs,
        "bc_learning_rate": args.bc_learning_rate,
        "ppo_timesteps": args.ppo_timesteps,
        "ppo_learning_rate": args.ppo_learning_rate,
        "num_envs": args.num_envs,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "ent_coef": args.ent_coef,
        "eval_target": args.eval_target,
        "eval_trials": args.eval_trials,
        "seed": args.seed,
        "config": str(args.config),
        **training_provenance(workspace=PROJECT_ROOT.parent, source_root=PROJECT_ROOT / "python"),
    }
    tracker = None if args.no_mlflow else MLflowRun(
        run_name=run_name,
        params=params,
        tags={"algorithm": "adaptive-bc-ppo", "frame_stack": stack,
              "arm": "history" if stack > 1 else "memoryless"},
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment,
    )

    try:
        print(f"[1/3] collecting scripted-expert dataset ({args.expert_episodes} chains, stack={stack})",
              flush=True)
        dataset = collect_expert_dataset(
            config_path=str(args.config),
            episodes=args.expert_episodes,
            stack=stack,
            max_steps=args.expert_max_steps,
            seed_begin=args.seed,
            epsilon=args.expert_epsilon,
        )
        expert_stats = {
            "expert.transitions": float(len(dataset)),
            "expert.mean_distance": float(dataset.episode_distances.mean()),
            "expert.median_distance": float(np.median(dataset.episode_distances)),
            "expert.max_distance": float(dataset.episode_distances.max()),
            "expert.mean_return": float(dataset.episode_returns.mean()),
        }
        print("  " + " ".join(f"{k}={v:.2f}" for k, v in expert_stats.items()), flush=True)
        if tracker is not None:
            tracker.log_metrics(expert_stats, step=0)

        base_env = MarsRoverSb3VecEnv(
            args.num_envs, list(ACTION_MACROS), config_path=str(args.config),
            biome_split=1, seed=args.seed,
        )
        env = VecMonitor(TrialFrameStack(base_env, n_stack=stack) if stack > 1 else base_env)
        model = PPO(
            "MlpPolicy", env, device=device, seed=args.seed,
            n_steps=args.n_steps, batch_size=args.batch_size,
            gamma=0.995, gae_lambda=0.95, ent_coef=args.ent_coef,
            learning_rate=args.ppo_learning_rate,
            policy_kwargs={"net_arch": [512, 512, 256]}, verbose=0,
        )
        assert model.observation_space.shape == (dataset.observations.shape[1],), (
            f"frame-stacked env obs {model.observation_space.shape} does not match "
            f"expert dataset {dataset.observations.shape[1]}"
        )

        print(f"[2/3] behaviour cloning ({len(dataset)} transitions)", flush=True)
        bc_stats = behaviour_clone(
            model, dataset, epochs=args.bc_epochs, batch_size=args.bc_batch_size,
            learning_rate=args.bc_learning_rate, tracker=tracker, device=device,
        )
        model.save(str(run_dir / "bc_model"))
        bc_metrics = evaluate_policy(
            model, config_path=args.config, stack=stack, trials=args.eval_trials,
            max_steps=args.eval_max_steps, target=args.eval_target,
            seed_begin=args.eval_seed_begin,
        )
        print("  after BC: " + " ".join(f"{k}={v:.2f}" for k, v in bc_metrics.items()), flush=True)
        if tracker is not None:
            tracker.log_metrics({f"bc_{k}": v for k, v in bc_metrics.items()}, step=0)
            tracker.log_metrics(bc_metrics, step=0)

        print(f"[3/3] PPO finetuning for {args.ppo_timesteps} steps", flush=True)
        done_steps = 0
        chunk = max(args.eval_every, args.num_envs * args.n_steps)
        while done_steps < args.ppo_timesteps:
            budget = min(chunk, args.ppo_timesteps - done_steps)
            model.learn(total_timesteps=budget, reset_num_timesteps=False, progress_bar=False)
            done_steps += budget
            metrics = evaluate_policy(
                model, config_path=args.config, stack=stack, trials=args.eval_trials,
                max_steps=args.eval_max_steps, target=args.eval_target,
                seed_begin=args.eval_seed_begin,
            )
            rollout_return = float(np.mean([info["r"] for info in model.ep_info_buffer]))\
                if model.ep_info_buffer else float("nan")
            print(f"  steps={done_steps} rollout_return={rollout_return:.1f} "
                  + " ".join(f"{k}={v:.2f}" for k, v in metrics.items()), flush=True)
            model.save(str(run_dir / f"checkpoint_{done_steps:09d}"))
            if tracker is not None:
                tracker.log_metrics({**metrics, "train.rollout_return": rollout_return},
                                    step=done_steps)

        model.save(str(run_dir / "model"))
        final_metrics = evaluate_policy(
            model, config_path=args.config, stack=stack, trials=args.eval_trials * 2,
            max_steps=args.eval_max_steps, target=args.eval_target,
            seed_begin=args.eval_seed_begin + 100_000,
        )
        summary = {
            "params": params, "expert": expert_stats, "bc": bc_stats,
            "bc_eval": bc_metrics, "final_eval": final_metrics,
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("FINAL: " + " ".join(f"{k}={v:.2f}" for k, v in final_metrics.items()), flush=True)
        if tracker is not None:
            tracker.log_metrics({f"final_{k}": v for k, v in final_metrics.items()},
                                step=args.ppo_timesteps)
            tracker.log_artifact(run_dir / "summary.json", artifact_path="summary")
            tracker.log_artifact(run_dir / "model.zip", artifact_path="model")
            tracker.set_tags({"status": "complete"})
        env.close()
    except BaseException:
        if tracker is not None:
            tracker.set_tags({"status": "failed"})
            tracker.close("FAILED")
            tracker = None
        raise
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
