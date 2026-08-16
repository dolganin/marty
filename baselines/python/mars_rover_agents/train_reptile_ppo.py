"""Reptile meta-training over PPO: a cheap alternative to RL2 for this benchmark.

Why Reptile and not RL2/MAML here: RL2 has to learn to adapt FROM SCRATCH inside
its recurrent state, on a task distribution that just became "a new procedurally
assembled chain of zones every trial" (see EnvConfig.chain_biomes) - that is a much
harder credit-assignment problem than the fixed-bank setting it was built for, and
the earlier gen7 measurements already show it barely learns to move (auc ~ -0.005)
without heavy scaffolding (distillation + KL anchor). MAML needs second-order
gradients through a PPO inner loop, which is expensive and unstable in practice.

Reptile sidesteps both: it never differentiates through the inner loop. Each outer
iteration is ordinary first-order PPO training for a short burst on a freshly
sampled chain (a "task"), starting from the current meta-weights; the meta-weights
then take a small step toward wherever that burst ended up:

    meta_theta <- meta_theta + reptile_lr * (theta_after_burst - meta_theta)

Over many bursts on many different chains this converges to an initialisation that
is a few gradient steps from good on ANY chain from the distribution - i.e. an
agent that adapts fast, without ever needing a differentiable adaptation procedure
or a recurrent memory. It is a plain, well-understood algorithm bolted onto the
PPO/vec-env machinery this repo already has (train_robust_ppo.py,
train_single_biome_ppo.py); nothing new to debug at the RL-algorithm level.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS, macro_action
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun, training_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reptile meta-training: many short PPO bursts on fresh chained "
        "courses, each nudging a shared meta-initialisation toward fast adaptation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--init-model", type=Path, help="warm-start the meta-init from a robust checkpoint")
    parser.add_argument("--outer-iterations", type=int, default=200)
    parser.add_argument("--inner-timesteps", type=int, default=8192, help="PPO burst length per task")
    parser.add_argument("--num-envs", type=int, default=32, help="parallel chains per burst")
    parser.add_argument("--reptile-lr", type=float, default=0.2, help="meta step size (epsilon)")
    parser.add_argument("--reptile-lr-final", type=float, default=0.02, help="linearly annealed to this")
    parser.add_argument("--inner-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=1500)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run-name")
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def _reptile_step(
    meta_state: dict[str, torch.Tensor], adapted_state: dict[str, torch.Tensor], lr: float
) -> dict[str, torch.Tensor]:
    return {
        key: meta_state[key] + lr * (adapted_state[key].to(meta_state[key].device) - meta_state[key])
        for key in meta_state
    }


def _quick_eval(model, config_path: Path, seed: int, episodes: int, max_steps: int) -> float:
    """Mean distance covered per life on a handful of fresh chains, no training."""
    from mars_rover_env import MarsRoverEnv

    distances = []
    env = MarsRoverEnv(config_path=str(config_path), biome_split=1)
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode, options={"trial_start": True})
        start_x = env.debug_info()["x"]
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(macro_action(int(action)))
            if terminated or truncated:
                break
        distances.append(env.debug_info()["x"] - start_x)
    env.close()
    return float(np.mean(distances))


def main() -> None:
    args = build_parser().parse_args()
    from stable_baselines3 import PPO

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    bootstrap_env = MarsRoverSb3VecEnv(
        args.num_envs, list(ACTION_MACROS), config_path=str(args.config),
        biome_split=1, seed=args.seed,
    )
    model = PPO(
        "MlpPolicy", bootstrap_env, device=device, seed=args.seed,
        n_steps=256, batch_size=2048, gamma=0.995, gae_lambda=0.95,
        ent_coef=0.01, learning_rate=args.inner_learning_rate,
        policy_kwargs={"net_arch": [256, 256]}, verbose=0,
    )
    if args.init_model is not None:
                                                                              
                                                                                
                                                                            
                                                                                
                                                                             
                                                                          
        source = PPO.load(str(args.init_model), device=device)
        source_state = source.policy.state_dict()
        target_state = model.policy.state_dict()
        transplanted, skipped = 0, 0
        for key, value in source_state.items():
            if key in target_state and target_state[key].shape == value.shape:
                target_state[key] = value
                transplanted += 1
            else:
                skipped += 1
        model.policy.load_state_dict(target_state)
        print(f"warm start from {args.init_model}: transplanted {transplanted} tensors, "
              f"reinitialised {skipped} (action head / shape mismatch)")
    bootstrap_env.close()
    meta_state = _clone_state(model.policy.state_dict())

    run_name = args.run_name or f"reptile-ppo_s{args.seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    tracker = None
    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name=run_name,
            params={
                "algorithm": "reptile-ppo",
                "outer_iterations": args.outer_iterations,
                "inner_timesteps": args.inner_timesteps,
                "num_envs": args.num_envs,
                "reptile_lr": args.reptile_lr,
                "reptile_lr_final": args.reptile_lr_final,
                "inner_learning_rate": args.inner_learning_rate,
                "seed": args.seed,
                "init_model": str(args.init_model) if args.init_model else None,
                **training_provenance(workspace=PROJECT_ROOT.parent, source_root=PROJECT_ROOT / "python"),
            },
            tags={"algorithm": "reptile-ppo"},
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for outer_iter in range(args.outer_iterations):
            task_seed = int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            progress = outer_iter / max(1, args.outer_iterations - 1)
            reptile_lr = args.reptile_lr + (args.reptile_lr_final - args.reptile_lr) * progress

            task_env = MarsRoverSb3VecEnv(
                args.num_envs, list(ACTION_MACROS), config_path=str(args.config),
                biome_split=1, seed=task_seed,
            )
            model.set_env(task_env)
            model.policy.load_state_dict(meta_state)
                                                                                 
                                                                      
            model.policy.optimizer = type(model.policy.optimizer)(
                model.policy.parameters(), lr=args.inner_learning_rate
            )
            model.learn(total_timesteps=args.inner_timesteps, reset_num_timesteps=True, progress_bar=False)
            adapted_state = _clone_state(model.policy.state_dict())
            meta_state = _reptile_step(meta_state, adapted_state, reptile_lr)
            task_env.close()

            if tracker is not None:
                tracker.log_metrics(
                    {"reptile.lr": reptile_lr, "reptile.task_seed": task_seed},
                    step=outer_iter,
                )

            if (outer_iter + 1) % args.checkpoint_every == 0 or outer_iter + 1 == args.outer_iterations:
                model.policy.load_state_dict(meta_state)
                checkpoint_path = args.output / f"meta_iter_{outer_iter + 1:05d}"
                model.save(str(checkpoint_path / "model"))
                eval_distance = _quick_eval(
                    model, args.config, seed=10_000_000 + outer_iter,
                    episodes=args.eval_episodes, max_steps=args.eval_max_steps,
                )
                print(f"[{outer_iter + 1}/{args.outer_iterations}] "
                      f"reptile_lr={reptile_lr:.3f} eval_mean_distance={eval_distance:.2f}", flush=True)
                if tracker is not None:
                    tracker.log_metrics({"eval.mean_distance": eval_distance}, step=outer_iter)

        model.policy.load_state_dict(meta_state)
        final_path = args.output / "final"
        final_path.mkdir(parents=True, exist_ok=True)
        model.save(str(final_path / "model"))
        artifact = {
            "artifact_type": "reptile_ppo_meta_init",
            "outer_iterations": args.outer_iterations,
            "inner_timesteps": args.inner_timesteps,
            "num_envs": args.num_envs,
            "reptile_lr_schedule": [args.reptile_lr, args.reptile_lr_final],
            "seed": args.seed,
            "init_model": str(args.init_model) if args.init_model else None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (final_path / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        if tracker is not None:
            tracker.log_artifact(final_path, artifact_path="final")
        print(f"reptile meta-init -> {final_path}")
    finally:
        if tracker is not None:
            tracker.close()


if __name__ == "__main__":
    main()
