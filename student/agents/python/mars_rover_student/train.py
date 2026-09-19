from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--biome-index", type=int)
    args = parser.parse_args()
    env = MarsRoverSb3VecEnv(
        args.num_envs,
        list(ACTION_MACROS),
        biome_split=1,
        seed=args.seed,
        fixed_biome_id=args.biome_index,
    )
    model = PPO("MlpPolicy", env, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))


if __name__ == "__main__":
    main()
