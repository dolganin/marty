"""Gate on how deep into a chained, procedurally-assembled course an agent gets.

The bank is no longer one fixed biome per trial (see EnvConfig.chain_biomes):
a trial is a random concatenation of zones, refreshed with fresh LLM biomes
before each run (mars-rover-bootstrap-run), so the specific course a trial
presents is unique to that run. "Adaptation" here is not a return curve on a
fixed task - it is whether memory lets the rover push FARTHER into an
unfamiliar chain than a memoryless policy, given the same number of lives.

`episodes_per_trial` already IS the life count: every episode in a trial
reuses the same trial_mechanic_seed_ (see Env::reset), so all N episodes of a
trial are N attempts at the identical chained course, each restarting from the
spawn point. harness.py already records `distance` per episode (end_x -
start_x, i.e. how far that one life reached). The metric this module adds is
the one harness.py does not compute: the BEST life per trial, i.e.
max(distance over the trial's episodes) - "how deep the agent got with its
lives", not the average life.

Usage:
    python -m mars_rover_agents.depth_gate <adaptive_run> <robust_run> --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _split_dir(run_dir: Path, split: str) -> Path:
    if (run_dir / "episodes.jsonl").is_file():
        return run_dir
    direct = run_dir / split
    finalized = run_dir / "final" / split
    if finalized.is_dir():
        return finalized
    if direct.is_dir():
        return direct
    if run_dir.name == split and run_dir.is_dir():
        return run_dir
    raise FileNotFoundError(f"no {split!r} evaluation under {run_dir}")


def _load_trial_max_depth(path: Path) -> tuple[list[tuple[str, int]], np.ndarray]:
    """Best-life depth per (biome_id, seed) trial, keyed identically across runs."""
    rows = [
        json.loads(line)
        for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        key = (str(row["biome_id"]), int(row["seed"]))
        grouped.setdefault(key, []).append(float(row["distance"]))
    keys = sorted(grouped)
    if not keys:
        raise ValueError(f"no episodes in {path}")
    depth = np.asarray([max(grouped[key]) for key in keys], dtype=np.float64)
    return keys, depth


def _bootstrap_mean(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def verify_depth_separation(
    adaptive_run: Path,
    robust_run: Path,
    *,
    split: str = "test",
    min_depth_gap: float = 5.0,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    adaptive_keys, adaptive_depth = _load_trial_max_depth(_split_dir(Path(adaptive_run), split))
    robust_keys, robust_depth = _load_trial_max_depth(_split_dir(Path(robust_run), split))
    if adaptive_keys != robust_keys:
        raise ValueError("adaptive and robust evaluations must use identical biome/seed trials")

    rng = np.random.default_rng(seed)
    adaptive_ci = _bootstrap_mean(adaptive_depth, samples=bootstrap_samples, rng=rng)
    robust_ci = _bootstrap_mean(robust_depth, samples=bootstrap_samples, rng=rng)
    gap = adaptive_depth - robust_depth
    gap_ci = _bootstrap_mean(gap, samples=bootstrap_samples, rng=rng)

    checks = {
        "adaptive_deeper_on_average": bool(gap_ci[0] > 0.0),
        "gap_clears_floor": bool(gap_ci[1] > min_depth_gap),
        "gap_not_explained_by_noise": bool(gap_ci[1] > 0.0),
    }
    return {
        "schema_version": 1,
        "metric": "best_life_depth_in_chained_course",
        "split": split,
        "num_trials": int(adaptive_depth.size),
        "adaptive_mean_depth_ci": adaptive_ci,
        "robust_mean_depth_ci": robust_ci,
        "adaptive_minus_robust_depth_ci": gap_ci,
        "min_depth_gap": min_depth_gap,
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require the memory-carrying agent to reach farther into a chained, "
        "procedurally-assembled course than the memoryless reference, given the same "
        "number of lives (episodes_per_trial)."
    )
    parser.add_argument("adaptive_run", type=Path)
    parser.add_argument("robust_run", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--min-depth-gap", type=float, default=5.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = verify_depth_separation(
        args.adaptive_run,
        args.robust_run,
        split=args.split,
        min_depth_gap=args.min_depth_gap,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
