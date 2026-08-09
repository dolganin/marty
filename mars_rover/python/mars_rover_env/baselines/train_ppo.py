from __future__ import annotations

from mars_rover_env.tools.train_robust_ppo import main as train_robust_ppo


def main() -> None:
    # Keep the historical entry point, but use the same frozen-bank, batched,
    # CUDA-only implementation as the versioned reference workflow.
    train_robust_ppo()


if __name__ == "__main__":
    main()
