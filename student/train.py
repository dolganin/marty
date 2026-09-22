from __future__ import annotations

import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv


DEFAULT_OUTPUT = Path("/output/policy.zip")


def training_settings() -> dict[str, int | float | str]:
    smoke = os.environ.get("MARS_ROVER_SMOKE_TEST") == "1"
    return {
        "total_timesteps": int(
            os.environ.get("MARS_ROVER_SMOKE_FRAMES", "512")
            if smoke
            else os.environ.get("TOTAL_FRAMES", "1000000000")
        ),
        "num_envs": 4 if smoke else 64,
        "n_steps": 64 if smoke else 128,
        "batch_size": 64 if smoke else 2048,
        "seed": int(os.environ.get("ARENA_SEED", "2026")),
        "device": "cpu" if smoke else os.environ.get("MARS_ROVER_DEVICE", "auto"),
        "checkpoint_steps": 256 if smoke else 100000,
        "train_seconds": 120.0 if smoke else float(os.environ.get("MARS_ROVER_TRAIN_SECONDS", "3540")),
    }


def atomic_save(model: PPO, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.zip")
    model.save(temporary)
    temporary.replace(output)


class ArtifactCheckpoint(BaseCallback):
    def __init__(self, output: Path, every_timesteps: int, max_seconds: float):
        super().__init__()
        self.output = output
        self.every_timesteps = every_timesteps
        self.max_seconds = max_seconds
        self.last_saved = 0
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_saved >= self.every_timesteps:
            atomic_save(self.model, self.output)
            self.last_saved = self.num_timesteps
        return time.monotonic() - self.started < self.max_seconds


def main() -> None:
    settings = training_settings()
    output = Path(os.environ.get("MARS_ROVER_OUTPUT_PATH", str(DEFAULT_OUTPUT)))
    env = MarsRoverSb3VecEnv(
        int(settings["num_envs"]),
        list(ACTION_MACROS),
        biome_split=1,
        seed=int(settings["seed"]),
    )
    try:
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=int(settings["n_steps"]),
            batch_size=int(settings["batch_size"]),
            n_epochs=1,
            learning_rate=3e-4,
            seed=int(settings["seed"]),
            device=str(settings["device"]),
            policy_kwargs={"net_arch": [128, 128]},
            verbose=1,
        )
        callback = ArtifactCheckpoint(
            output,
            int(settings["checkpoint_steps"]),
            float(settings["train_seconds"]),
        )
        atomic_save(model, output)
        try:
            model.learn(total_timesteps=int(settings["total_timesteps"]), callback=callback)
        finally:
            atomic_save(model, output)
    finally:
        env.close()


if __name__ == "__main__":
    main()
