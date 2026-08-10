"""Does a biome require a DIFFERENT policy, or only a harder version of the same one?

Four generations of bank edits attacked perception — fingerprint separation, a crippled
lidar, darkness, a starved battery, a withheld thermometer — and the memoryless reference
never moved (59.4 -> 56.3 -> 60.1 -> 60.1 m). The reason showed up directly: that policy
emits 93-98% identical actions in all ten held-out biomes. It never identifies the world
because identifying it changes nothing about what to do.

So the property a meta-RL bank actually needs is not "the mechanic is hard to see" but
"the mechanic changes which behaviour wins". This measures exactly that, cheaply and
without training anything: run a fixed repertoire of hand-written strategies on every
biome and look at which one wins where.

  - If one strategy wins everywhere, no memory can pay: a reflex that plays that strategy
    is optimal, and knowing the biome is worthless.
  - If different strategies win in different biomes, then identifying the biome is worth
    real reward, and an agent that infers it from experience can beat one that cannot.

Report:
  winner_entropy   - spread of winning strategies across biomes (0 = one winner everywhere)
  regret           - what the best SINGLE fixed strategy loses against per-biome winners.
                     This is the ceiling on what adaptation can ever be worth on this bank.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.actions import ACTION_MACROS


# Deliberately coarse and mutually exclusive driving styles. They are not meant to be good;
# they are meant to be DIFFERENT, so that "which one wins" is informative about the biome.
def _strategies() -> dict[str, callable]:
    GAS = 1
    BRAKE = 1 << 1
    REVERSE_ASSIST = 1 | (1 << 2)
    CLUTCH_UP = (1 << 3) | (1 << 6)
    CLUTCH_DOWN = (1 << 3) | (1 << 7)
    IGNITION = 1 << 9
    TILT_L = 1 | (1 << 4)
    TILT_R = 1 | (1 << 5)
    CHARGE = 1 << 8
    LIDAR = 1 << 11

    def ignite_first(step: int, inner):
        return IGNITION if step < 12 else inner(step)

    return {
        "full_throttle": lambda s: ignite_first(s, lambda t: GAS),
        "feathered": lambda s: ignite_first(s, lambda t: GAS if t % 3 else 0),
        "low_gear_crawl": lambda s: ignite_first(
            s, lambda t: CLUTCH_DOWN if t % 40 == 0 else GAS
        ),
        "high_gear_cruise": lambda s: ignite_first(
            s, lambda t: CLUTCH_UP if t % 40 == 0 else GAS
        ),
        "brake_and_creep": lambda s: ignite_first(
            s, lambda t: BRAKE if t % 8 < 2 else GAS
        ),
        "tilt_stabilised": lambda s: ignite_first(
            s, lambda t: TILT_L if (t // 30) % 2 else TILT_R
        ),
        "charge_then_run": lambda s: ignite_first(
            s, lambda t: CHARGE if t < 400 else GAS
        ),
        "scan_and_go": lambda s: ignite_first(
            s, lambda t: LIDAR if t % 120 == 0 else GAS
        ),
    }


def score_strategy(policy, biome_index: int, seeds: list[int], max_steps: int, config_path):
    returns = []
    for seed in seeds:
        env = MarsRoverEnv(
            config_path=config_path,
            biome_split=MarsRoverEnv.BIOME_MODE_ALL,
            fixed_biome_id=biome_index,
        )
        env.reset(seed=seed, options={"trial_start": True})
        total = 0.0
        for step in range(max_steps):
            _, reward, terminated, truncated, _ = env.step(int(policy(step)))
            total += float(reward)
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
    return float(np.mean(returns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure whether biomes demand different behaviour")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed-begin", type=int, default=90_000)
    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--config")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import _mars_rover_cpp as native

    split_code = {"train": 1, "test": 2}[args.split]
    biomes = [
        (int(item["index"]), str(item["id"]))
        for item in native.biome_catalog()
        if int(item["split"]) == split_code
    ]
    seeds = list(range(args.seed_begin, args.seed_begin + args.seeds))
    strategies = _strategies()

    table: dict[str, dict[str, float]] = {}
    for index, name in biomes:
        row = {
            label: score_strategy(policy, index, seeds, args.max_steps, args.config)
            for label, policy in strategies.items()
        }
        table[name] = row
        best = max(row, key=row.get)
        print(f"  {name:<26} best={best:<18} {row[best]:8.2f}", flush=True)

    winners = Counter(max(row, key=row.get) for row in table.values())
    total = sum(winners.values())
    entropy = -sum(
        (count / total) * math.log(count / total) for count in winners.values()
    )
    # What does the best single fixed strategy give up against per-biome winners?
    per_biome_best = {name: max(row.values()) for name, row in table.items()}
    fixed_scores = {
        label: sum(table[name][label] for name in table) / len(table)
        for label in strategies
    }
    best_fixed_label = max(fixed_scores, key=fixed_scores.get)
    oracle_mean = sum(per_biome_best.values()) / len(per_biome_best)
    regret = oracle_mean - fixed_scores[best_fixed_label]

    print()
    print(f"winning strategies: {dict(winners)}")
    print(f"winner entropy    : {entropy:.3f} (0 = one strategy wins everywhere)")
    print(f"best fixed policy : {best_fixed_label} (mean {fixed_scores[best_fixed_label]:.2f})")
    print(f"per-biome best    : mean {oracle_mean:.2f}")
    print(f"ADAPTATION HEADROOM (regret of the best fixed policy): {regret:.2f}")
    print()
    print(
        "A bank where this headroom is small cannot reward memory: the best reflex already"
        "\nmatches the best per-biome behaviour, so identifying the biome buys nothing."
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "split": args.split,
                    "table": table,
                    "winners": dict(winners),
                    "winner_entropy": entropy,
                    "best_fixed_policy": best_fixed_label,
                    "best_fixed_mean": fixed_scores[best_fixed_label],
                    "per_biome_best_mean": oracle_mean,
                    "adaptation_headroom": regret,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
