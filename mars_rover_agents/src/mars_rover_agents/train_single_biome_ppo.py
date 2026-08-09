"""Layer-1 control: PPO trained on ONE mechanic, then evaluated on held-out biomes.

This is the "pure memorisation" baseline. An agent that only ever sees a single mechanic
can perfect that mechanic without ever learning to identify which world it is in. Running
it on the held-out bank prices that shortcut: if it transfers, the benchmark can be beaten
by memorising rather than adapting, and the whole track is compromised. It should fail.

Deliberately NOT wired into the difficulty-gate / manifest machinery: this artifact is a
diagnostic control, never a reference, so it must not be able to influence normalisation
bands or gate decisions.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PPO specialised on a single mechanic")
    parser.add_argument("--biome-index", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--timesteps", type=int, default=1_024_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument(
        "--init-model",
        type=Path,
        help=(
            "Warm-start from an existing PPO. PPO cannot learn a single hard mechanic from "
            "scratch (the mixed bank acts as a curriculum), so specialising an agent that "
            "already drives is the only way to measure the cost of specialisation itself."
        ),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3.0e-4,
        help="Fine-tuning a competent policy needs a far smaller step than training one.",
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from stable_baselines3 import PPO

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    import _mars_rover_cpp as native

    catalog = {int(item["index"]): dict(item) for item in native.biome_catalog()}
    if args.biome_index not in catalog:
        raise SystemExit(f"unknown biome index {args.biome_index}")
    biome = catalog[args.biome_index]

    env = MarsRoverSb3VecEnv(
        args.num_envs,
        list(ACTION_MACROS),
        config_path=str(args.config),
        biome_split=0,
        fixed_biome_id=args.biome_index,
        seed=args.seed,
    )
    if args.init_model is not None:
        model = PPO.load(
            str(args.init_model),
            env=env,
            device="cuda:0",
            learning_rate=args.learning_rate,
            ent_coef=args.entropy_coefficient,
        )
        model.set_random_seed(args.seed)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cuda:0",
            seed=args.seed,
            n_steps=256,
            batch_size=2048,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.01,
            learning_rate=3.0e-4,
            verbose=0,
        )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output / "model"))
    (args.output / "artifact.json").write_text(
        json.dumps(
            {
                "artifact_type": "single_biome_ppo_control",
                "purpose": "prices the memorisation shortcut; never a reference",
                "biome_index": args.biome_index,
                "biome_id": str(biome["id"]),
                "timesteps": args.timesteps,
                "seed": args.seed,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"single-biome PPO on {biome['id']} -> {args.output}")


if __name__ == "__main__":
    main()
