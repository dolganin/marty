from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mars_rover_env.actions import ACTION_MACROS


OBSERVATION_DIM = 160


def validate_policy(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"policy artifact is missing or empty: {path}")
    model = PPO.load(path, device="cpu")
    if model.observation_space.shape != (OBSERVATION_DIM,):
        raise RuntimeError(
            f"expected observation shape ({OBSERVATION_DIM},), got {model.observation_space.shape}"
        )
    if getattr(model.action_space, "n", None) != len(ACTION_MACROS):
        raise RuntimeError(
            f"expected {len(ACTION_MACROS)} discrete actions, got {model.action_space}"
        )
    observations = np.zeros((2, OBSERVATION_DIM), dtype=np.float32)
    actions, _ = model.predict(observations, deterministic=True)
    actions = np.asarray(actions)
    if actions.shape != (2,) or not np.isfinite(actions).all():
        raise RuntimeError(f"policy returned invalid actions with shape {actions.shape}")
    if np.any(actions < 0) or np.any(actions >= len(ACTION_MACROS)):
        raise RuntimeError(f"policy returned actions outside [0, {len(ACTION_MACROS) - 1}]")
    print(f"policy artifact is valid: {path} ({path.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("/output/policy.zip"))
    args = parser.parse_args()
    validate_policy(args.path)


if __name__ == "__main__":
    main()
