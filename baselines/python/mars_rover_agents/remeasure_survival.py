from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mars_rover_env.bank import DEFAULT_MANIFEST
from mars_rover_env.config import DEFAULT_ENV_CONFIG

from .agents import PPOAgent
from .harness import EvaluationSpec, evaluate
from .rl2_agent import RL2TransformerAgent
from .rl2_model import RL2TransformerActorCritic


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-measure a frozen robust PPO or RL2 checkpoint on survival curves."
    )
    parser.add_argument("kind", choices=("robust", "rl2"))
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--split", choices=("train", "test", "anchor"), default="train")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=90_000)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("survival re-measurement requires an available CUDA device")
    device = torch.device(args.device)
    if args.kind == "robust":
        from stable_baselines3 import PPO

        model = PPO.load(args.checkpoint, device=device)
        factory = lambda _seed: PPOAgent(model, deterministic=True)
    else:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = RL2TransformerActorCritic(**payload["model_config"]).to(device)
        model.load_compatible_state(payload["model_state"])
        model.eval()
        factory = lambda _seed: RL2TransformerAgent(
            model, device=device, deterministic=True, max_steps=args.max_steps
        )

    payload = evaluate(
        factory,
        EvaluationSpec(
            split=args.split,
            seeds=tuple(range(args.seed_start, args.seed_start + args.trials)),
            episodes_per_trial=args.episodes,
            max_steps=args.max_steps,
            config_path=str(args.config),
            render_trials=0,
        ),
        manifest_path=args.manifest,
        output_dir=args.output,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
