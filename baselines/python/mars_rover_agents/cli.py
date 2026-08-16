from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import PPOAgent, RandomAgent
from .harness import EvaluationSpec, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Common public-Agent Mars Rover evaluation")
    parser.add_argument("--agent", choices=("random", "ppo", "rl2"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--split", choices=("train", "test", "anchor"), required=True)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes-per-trial", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-trials", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if args.agent == "random":
        factory = lambda seed: RandomAgent(seed)
    elif args.agent == "rl2":
        if not args.model:
            raise SystemExit("--model is required for RL2")
        from .rl2_agent import RL2TransformerAgent

                                                                                
                                                                                
                                                          
        rl2_agent = RL2TransformerAgent.load(args.model, device=args.device)
        factory = lambda _seed: RL2TransformerAgent(
            rl2_agent.model, device=rl2_agent.device, deterministic=True
        )
    else:
        if not args.model:
            raise SystemExit("--model is required for PPO")
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        factory = lambda _seed: PPOAgent(model)
    kwargs = {}
    if args.manifest is not None:
        kwargs["manifest_path"] = args.manifest
    payload = evaluate(
        factory,
        EvaluationSpec(
            split=args.split,
            seeds=tuple(range(args.seed, args.seed + args.trials)),
            episodes_per_trial=args.episodes_per_trial,
            max_steps=args.max_steps,
            config_path=args.config,
            render_trials=args.render_trials,
        ),
        output_dir=args.output,
        **kwargs,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

