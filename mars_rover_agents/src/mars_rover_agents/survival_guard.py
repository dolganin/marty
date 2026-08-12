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


def _load_trials(path: Path) -> tuple[list[tuple[str, int]], np.ndarray, np.ndarray]:
    rows = [
        json.loads(line)
        for line in (path / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[tuple[str, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["biome_id"]), int(row["seed"])), {})[
            int(row["episode_index"])
        ] = row
    keys = sorted(grouped)
    if not keys:
        raise ValueError(f"no episodes in {path}")
    episode_count = max(max(grouped[key]) for key in keys) + 1
    survival = np.asarray(
        [
            [1.0 - float(bool(grouped[key][episode]["fatal_error"])) for episode in range(episode_count)]
            for key in keys
        ],
        dtype=np.float64,
    )
    distance = np.asarray(
        [
            [float(grouped[key][episode]["distance"]) for episode in range(episode_count)]
            for key in keys
        ],
        dtype=np.float64,
    )
    return keys, survival, distance


def _bootstrap_mean(
    values: np.ndarray, *, samples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def verify_survival_separation(
    adaptive_run: Path,
    robust_run: Path,
    *,
    split: str = "test",
    late_survival_floor: float = 0.95,
    min_delta_gap: float = 0.10,
    robust_flat_tolerance: float = 0.05,
    bootstrap_samples: int = 20_000,
    seed: int = 0,
) -> dict[str, Any]:
    adaptive_keys, adaptive, adaptive_distance = _load_trials(
        _split_dir(Path(adaptive_run), split)
    )
    robust_keys, robust, _ = _load_trials(_split_dir(Path(robust_run), split))
    if adaptive_keys != robust_keys:
        raise ValueError("adaptive and robust evaluations must use identical biome/seed trials")
    if adaptive.shape != robust.shape or adaptive.shape[1] < 2:
        raise ValueError("adaptive and robust survival curves must have the same >=2 episode shape")

    half = max(1, adaptive.shape[1] // 2)
    adaptive_delta = adaptive[:, -half:].mean(axis=1) - adaptive[:, :half].mean(axis=1)
    robust_delta = robust[:, -half:].mean(axis=1) - robust[:, :half].mean(axis=1)
    delta_gap = adaptive_delta - robust_delta
    rng = np.random.default_rng(seed)
    adaptive_ci = _bootstrap_mean(adaptive_delta, samples=bootstrap_samples, rng=rng)
    robust_ci = _bootstrap_mean(robust_delta, samples=bootstrap_samples, rng=rng)
    gap_ci = _bootstrap_mean(delta_gap, samples=bootstrap_samples, rng=rng)

    adaptive_curve = adaptive.mean(axis=0)
    robust_curve = robust.mean(axis=0)
    late_survival = float(adaptive[:, -half:].mean())
    # Fatal episodes contribute zero useful distance. This remains defined even
    # when every first probe crashes, which is a valid (if expensive) learning curve.
    distance_curve = np.where(adaptive > 0.5, adaptive_distance, 0.0).mean(axis=0)
    distance_growth = float(distance_curve[-1] - distance_curve[0])

    checks = {
        "episode_one_is_risky": bool(adaptive_curve[0] < 1.0),
        "late_survival": bool(late_survival >= late_survival_floor),
        "adaptive_curve_rises": bool(adaptive_ci[1] > 0.0),
        "robust_curve_is_flat": bool(
            abs(robust_ci[0]) <= robust_flat_tolerance
            and robust_ci[1] <= 0.0 <= robust_ci[2]
        ),
        "adaptive_beats_robust_shape": bool(gap_ci[1] > min_delta_gap),
        "surviving_distance_grows": bool(distance_growth > 0.0),
    }
    return {
        "schema_version": 1,
        "metric": "survival_after_fatal_error",
        "split": split,
        "num_trials": int(adaptive.shape[0]),
        "episodes_per_trial": int(adaptive.shape[1]),
        "adaptive_survival_curve": adaptive_curve.tolist(),
        "robust_survival_curve": robust_curve.tolist(),
        "adaptive_delta_ci": adaptive_ci,
        "robust_delta_ci": robust_ci,
        "adaptive_minus_robust_delta_ci": gap_ci,
        "late_survival": late_survival,
        "late_survival_floor": late_survival_floor,
        "min_delta_gap": min_delta_gap,
        "robust_flat_tolerance": robust_flat_tolerance,
        "surviving_distance_curve": distance_curve.tolist(),
        "surviving_distance_growth": distance_growth,
        "checks": checks,
        "overall_pass": bool(all(checks.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require rising adaptive survival and a flat robust baseline on paired trials."
    )
    parser.add_argument("adaptive_run", type=Path)
    parser.add_argument("robust_run", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--late-survival-floor", type=float, default=0.95)
    parser.add_argument("--min-delta-gap", type=float, default=0.10)
    parser.add_argument("--robust-flat-tolerance", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = verify_survival_separation(
        args.adaptive_run,
        args.robust_run,
        split=args.split,
        late_survival_floor=args.late_survival_floor,
        min_delta_gap=args.min_delta_gap,
        robust_flat_tolerance=args.robust_flat_tolerance,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
