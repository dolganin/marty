"""Algorithm Distillation: can adaptation be learned purely in-context?

RL2 adapts through a recurrent state that is carried across episodes. AD asks a different
question: if a model is trained to imitate a LEARNING PROCESS — sequences that run from a
weak policy to a strong one on the same task — does it then improve within its own context
window on a task it has never seen, with frozen weights and no recurrent state?

Dataset construction here uses the robust-PPO training checkpoints as the learning curve
(250k -> 500k -> 750k -> 1M -> final, whose held-out return climbs -0.62 -> 20.85). For each
biome we concatenate one episode from each checkpoint in improving order, so the context a
model reads is literally "an agent getting better at this world".

Deliberate limitation, stated up front: 5 checkpoints is a coarse sampling of a learning
curve, and the source policies are all descendants of one run. AD in the literature uses many
independent learning histories. This is therefore a feasibility probe, not a faithful AD
reproduction — read a positive result as "in-context improvement is reachable here" and a
negative one as "not reachable at this budget", never as a refutation of AD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mars_rover_env import MarsRoverEnv

from .actions import ACTION_MACROS


def collect_learning_history(
    checkpoints: list[Path],
    *,
    biome_index: int,
    seeds: list[int],
    max_steps: int,
    config_path: str | None,
) -> list[dict]:
    """One improving sequence per seed: weakest policy first, strongest last."""
    from stable_baselines3 import PPO

    policies = [PPO.load(str(path), device="cuda:0") for path in checkpoints]
    action_macros = np.asarray(ACTION_MACROS, dtype=np.int32)
    rows: list[dict] = []
    for seed in seeds:
        observations: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        for stage, policy in enumerate(policies):
            env = MarsRoverEnv(
                config_path=config_path,
                biome_split=MarsRoverEnv.BIOME_MODE_ALL,
                fixed_biome_id=biome_index,
            )
            obs, _ = env.reset(seed=seed + stage * 7919, options={"trial_start": stage == 0})
            for _ in range(max_steps):
                action_index, _ = policy.predict(obs, deterministic=False)
                action_index = int(action_index)
                observations.append(np.asarray(obs, dtype=np.float32))
                actions.append(action_index)
                obs, reward, terminated, truncated, _ = env.step(
                    int(action_macros[action_index])
                )
                rewards.append(float(reward))
                if terminated or truncated:
                    break
            env.close()
        rows.append(
            {
                "biome_index": biome_index,
                "seed": seed,
                "observations": np.stack(observations, axis=0),
                "actions": np.asarray(actions, dtype=np.int64),
                "rewards": np.asarray(rewards, dtype=np.float32),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect learning histories for AD")
    parser.add_argument("--run-dir", type=Path, required=True, help="robust-PPO run directory")
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed-begin", type=int, default=80_000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import _mars_rover_cpp as native

    checkpoints = sorted(
        (args.run_dir / "checkpoints").glob("robust_ppo_*_steps.zip"),
        key=lambda path: int(path.stem.split("_")[2]),
    )
    final = args.run_dir / "model.zip"
    if final.is_file():
        checkpoints.append(final)
    if len(checkpoints) < 2:
        raise SystemExit("need at least two checkpoints to form a learning curve")

    train_biomes = [
        int(item["index"]) for item in native.biome_catalog() if int(item["split"]) == 1
    ]
    seeds = list(range(args.seed_begin, args.seed_begin + args.seeds))
    args.output.mkdir(parents=True, exist_ok=True)

    total = 0
    for biome_index in train_biomes:
        rows = collect_learning_history(
            checkpoints,
            biome_index=biome_index,
            seeds=seeds,
            max_steps=args.max_steps,
            config_path=args.config,
        )
        for row in rows:
            path = args.output / f"biome{biome_index:03d}_seed{row['seed']}.npz"
            np.savez_compressed(
                path,
                observations=row["observations"],
                actions=row["actions"],
                rewards=row["rewards"],
            )
            total += 1
        print(f"biome {biome_index}: {len(rows)} histories", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoints": [str(path) for path in checkpoints],
                "train_biomes": train_biomes,
                "seeds": seeds,
                "max_steps_per_stage": args.max_steps,
                "histories": total,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {total} learning histories to {args.output}")


if __name__ == "__main__":
    main()
