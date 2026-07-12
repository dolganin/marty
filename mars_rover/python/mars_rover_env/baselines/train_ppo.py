from __future__ import annotations

from mars_rover_env import MarsRoverEnv


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError("Install stable-baselines3 to run this baseline") from exc

    env = MarsRoverEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)


if __name__ == "__main__":
    main()
