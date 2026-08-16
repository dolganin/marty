from __future__ import annotations

import argparse
import os
import time

import numpy as np

from mars_rover_env import MarsRoverVecEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vectorized Mars rover simulation.")
    parser.add_argument("--envs", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    parser.add_argument("--steps", type=int, default=1_000_000, help="Target env-steps per batch size.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup vector steps per batch size.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bool-flags", action="store_true", help="Use Gym-style bool flags instead of uint8 fast path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'default')}")
    print("envs,vector_steps,env_steps_per_sec,seconds,flag_mode")
    for num_envs in args.envs:
        env = MarsRoverVecEnv(num_envs)
        env.reset(args.seed)
        actions = np.ones(num_envs, dtype=np.int32)
        step_fn = env.step if args.bool_flags else env.step_uint8
        for _ in range(args.warmup):
            step_fn(actions)
        loops = max(1, args.steps // num_envs)
        start = time.perf_counter()
        for _ in range(loops):
            step_fn(actions)
        elapsed = time.perf_counter() - start
        env_steps = num_envs * loops
        mode = "bool" if args.bool_flags else "uint8"
        print(f"{num_envs},{loops},{env_steps / elapsed:.0f},{elapsed:.4f},{mode}")


if __name__ == "__main__":
    main()
