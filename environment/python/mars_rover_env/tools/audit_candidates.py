from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.candidates import DEFAULT_CATALOG, candidate_config, candidates, load_catalog


def _debug_bounded(debug: dict) -> bool:
    values = (debug.get("energy", 0.0), debug.get("latent_traction", 1.0),
              debug.get("latent_viscosity", 0.0), debug.get("gravity_multiplier", 1.0),
              debug.get("latent_charge_reserve", 0.0), debug.get("latent_suspension", 1.0))
    return all(math.isfinite(float(value)) for value in values) and (
        0.0 <= float(debug.get("energy", 0.0)) <= float(debug.get("energy_capacity", 1.0)) + 1e-4
        and 0.05 <= float(debug.get("latent_traction", 1.0)) <= 2.5
        and 0.0 <= float(debug.get("latent_viscosity", 0.0)) <= 12.0
        and 0.25 <= float(debug.get("gravity_multiplier", 1.0)) <= 1.6
        and 0.0 <= float(debug.get("latent_charge_reserve", 0.0)) <= 1.0
        and 0.65 <= float(debug.get("latent_suspension", 1.0)) <= 1.25
    )


def _action(step: int) -> int:
    action = 1
    if step == 0:
        action |= 512
    if step in (180, 360):
        action |= 8 | 64
    if step % 211 == 40:
        action |= 2048
    if step % 307 in range(80, 105):
        action |= 16384
    if step % 401 in range(120, 155):
        action |= 8192
    return action


def _run_one(name: str, seed: int, steps: int, catalog_path: Path) -> dict:
    item = candidates(catalog_path)[name]
    env = MarsRoverEnv(config_override=candidate_config(item))
    obs, _ = env.reset(seed=seed)
    observations = [obs.copy()]
    rewards: list[float] = []
    finite = bool(np.isfinite(obs).all())
    initial_debug = env.debug_info()
    bounded = _debug_bounded(initial_debug)
    start_x = float(initial_debug["x"])
    for step in range(steps):
        obs, reward, terminated, truncated, _ = env.step(_action(step))
        finite &= bool(np.isfinite(obs).all()) and math.isfinite(reward)
        rewards.append(reward)
        if step % 60 == 59:
            observations.append(obs.copy())
            bounded &= _debug_bounded(env.debug_info())
        if terminated or truncated:
            break
    debug = env.debug_info()
    sampled = np.concatenate(observations).astype(np.float64)
    return {
        "name": name, "seed": seed, "finite": finite, "bounded": bounded,
        "steps": len(rewards),
        "distance": float(debug["x"]) - start_x, "reward": float(sum(rewards)),
        "fingerprint": [float(np.mean(sampled)), float(np.std(sampled)),
                        float(np.min(sampled)), float(np.max(sampled)),
                        float(debug.get("latent_traction", 0.0)),
                        float(debug.get("latent_viscosity", 0.0)),
                        float(debug.get("gravity", 0.0)),
                        float(debug.get("generated_pit_count", 0.0))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit provisional candidate worlds")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw_catalog = load_catalog(args.catalog)
    items = candidates(args.catalog)
    errors: list[str] = []
    expected = int(raw_catalog["expected_candidates"])
    if len(items) != expected:
        errors.append(f"expected {expected} candidates, got {len(items)}")
    for item in items.values():
        if len(item.seeds) < 3 or len(set(item.seeds)) != len(item.seeds):
            errors.append(f"{item.name}: needs at least three unique seeds")
        try:
            candidate_config(item)
        except Exception as exc:
            errors.append(f"{item.name}: {exc}")

    jobs = [(name, seed) for name, item in items.items() for seed in item.seeds]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        first = list(executor.map(lambda job: _run_one(*job, args.steps, args.catalog), jobs))
        second = list(executor.map(lambda job: _run_one(*job, args.steps, args.catalog), jobs))

    deterministic = all(
        a["steps"] == b["steps"] and a["finite"] == b["finite"] and
        np.allclose(a["fingerprint"], b["fingerprint"], rtol=0.0, atol=1e-7) and
        math.isclose(a["distance"], b["distance"], rel_tol=0.0, abs_tol=1e-7) and
        math.isclose(a["reward"], b["reward"], rel_tol=0.0, abs_tol=1e-7)
        for a, b in zip(first, second)
    )
    if not deterministic:
        errors.append("determinism check failed")
    if not all(row["finite"] for row in first):
        errors.append("non-finite physics output")
    if not all(row["bounded"] for row in first):
        errors.append("physics latent/state bounds failed")

    means = {name: np.mean([row["fingerprint"] for row in first if row["name"] == name], axis=0)
             for name in items}
    matrix = np.stack(list(means.values()))
    scale = np.maximum(np.ptp(matrix, axis=0), 1e-6)
    near_duplicates = []
    names = list(means)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            distance = float(np.linalg.norm((means[left] - means[right]) / scale) / math.sqrt(scale.size))
            if distance < 0.12:
                near_duplicates.append({"left": left, "right": right, "distance": distance})

    result = {
        "catalog_version": raw_catalog["catalog_version"],
        "environment_version": raw_catalog["environment_version"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps_per_probe": args.steps, "seed_runs": len(first),
        "deterministic": deterministic, "finite": all(row["finite"] for row in first),
        "bounded": all(row["bounded"] for row in first),
        "near_duplicates": near_duplicates, "errors": errors,
        "runs": [{key: value for key, value in row.items() if key != "fingerprint"} for row in first],
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
