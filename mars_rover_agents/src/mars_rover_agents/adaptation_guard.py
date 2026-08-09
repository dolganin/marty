from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


NOISE_FLOOR = 0.03  # normalized adaptation_delta of the collapsed/degenerate fixed point
# observed on 2026-08-03 in runs s2031 (from-scratch) and late-training s2041
# (post-forgetting): AUC ~0.013-0.014, delta ~+0.031-0.033. A "real" delta must
# clear this by a margin, not just be positive.


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _trial_curves(episodes: list[dict[str, Any]]) -> np.ndarray:
    """Return shape (num_trials, episodes_per_trial) of normalized_return, ordered by episode_index."""
    trials: dict[tuple[str, int], dict[int, float]] = {}
    for row in episodes:
        if row["normalized_return"] is None:
            raise ValueError("episode row has no normalized_return; anchor references not frozen")
        key = (row["biome_id"], row["seed"])
        trials.setdefault(key, {})[int(row["episode_index"])] = float(row["normalized_return"])
    episode_count = max(max(d) for d in trials.values()) + 1
    curves = np.array(
        [[trials[key][i] for i in range(episode_count)] for key in sorted(trials)],
        dtype=np.float64,
    )
    return curves


def _bootstrap_ci(values: np.ndarray, *, samples: int, rng: np.random.Generator, alpha: float = 0.05) -> tuple[float, float, float]:
    n = values.shape[0]
    means = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def verify_split(
    final_dir: Path,
    *,
    split: str,
    auc_floor: float,
    noise_floor: float,
    min_win_fraction: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    split_dir = final_dir / split
    summary = json.loads((split_dir / "summary.json").read_text(encoding="utf-8"))
    episodes = _load_jsonl(split_dir / "episodes.jsonl")
    curves = _trial_curves(episodes)
    episode_count = curves.shape[1]
    if episode_count < 2:
        raise ValueError("need at least two episodes per trial to measure adaptation")
    half = episode_count // 2
    early = curves[:, :half].mean(axis=1)
    late = curves[:, episode_count - half:].mean(axis=1)
    per_trial_delta = late - early

    rng = np.random.default_rng(seed)
    mean_delta, ci_lo, ci_hi = _bootstrap_ci(per_trial_delta, samples=bootstrap_samples, rng=rng)
    win_fraction = float((per_trial_delta > 0).mean())

    mean_curve = curves.mean(axis=0)
    monotonic_gap = float(mean_curve[-1] - mean_curve[0])
    non_decreasing = bool(np.all(np.diff(mean_curve) >= -1e-9))

    trial_auc = float(summary["trial_auc"])
    auc_pass = trial_auc >= auc_floor
    # Strict separation from the known degenerate fixed point: the whole CI must
    # clear it, not just the point estimate (rules out delta > 0 that is really noise).
    delta_pass = ci_lo > noise_floor
    shape_pass = non_decreasing and win_fraction >= min_win_fraction

    return {
        "split": split,
        "trial_auc": trial_auc,
        "auc_floor": auc_floor,
        "auc_pass": auc_pass,
        "num_trials": int(curves.shape[0]),
        "mean_delta": mean_delta,
        "delta_ci_low": ci_lo,
        "delta_ci_high": ci_hi,
        "noise_floor": noise_floor,
        "delta_pass": delta_pass,
        "win_fraction": win_fraction,
        "min_win_fraction": min_win_fraction,
        "mean_episode_curve": mean_curve.tolist(),
        "curve_non_decreasing": non_decreasing,
        "shape_pass": shape_pass,
        "overall_pass": bool(auc_pass and delta_pass and shape_pass),
    }


def verify_run(
    run_dir: Path,
    *,
    auc_floor: float = 0.25,
    noise_floor: float = NOISE_FLOOR,
    min_win_fraction: float = 0.7,
    bootstrap_samples: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    final_dir = run_dir / "final"
    if not final_dir.is_dir():
        raise FileNotFoundError(f"no finalization output at {final_dir}; run mars-agent-finalize-rl2 first")
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run_record.get("status") != "complete":
        raise RuntimeError("training run is not complete")

    results = {
        split: verify_split(
            final_dir,
            split=split,
            auc_floor=auc_floor,
            noise_floor=noise_floor,
            min_win_fraction=min_win_fraction,
            bootstrap_samples=bootstrap_samples,
            seed=seed + hash(split) % 1000,
        )
        for split in ("test", "anchor")
    }

    # No-forgetting check across the training trajectory: every logged evaluation
    # that beat the previous best selection score must not have done so by
    # dropping below the AUC floor (this mirrors the guard already enforced live
    # in train_rl2.consider_evaluation, re-derived here for an auditable record).
    eval_dir = run_dir / "evaluations"
    trajectory = []
    if eval_dir.is_dir():
        for step_dir in sorted(eval_dir.iterdir(), key=lambda p: int(p.name.split("_")[-1])):
            summary_path = step_dir / "summary.json"
            if not summary_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            trajectory.append(
                {
                    "transitions": int(step_dir.name.split("_")[-1]),
                    "trial_auc": summary.get("trial_auc"),
                    "adaptation_delta": summary.get("adaptation_delta"),
                }
            )
    selected_step = run_record.get("selected_at_transitions")
    selected_point = next((p for p in trajectory if p["transitions"] == selected_step), None)
    no_forgetting_pass = (
        selected_point is not None
        and selected_point["trial_auc"] is not None
        and selected_point["trial_auc"] >= auc_floor
    )

    overall_pass = (
        results["test"]["overall_pass"]
        and no_forgetting_pass
    )

    return {
        "run_dir": str(run_dir),
        "selected_at_transitions": selected_step,
        "selected_point": selected_point,
        "no_forgetting_pass": no_forgetting_pass,
        "training_trajectory": trajectory,
        "splits": results,
        "overall_pass": bool(overall_pass),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a finalized RL2 artifact clears the AUC floor and shows a real, "
        "non-degenerate adaptation delta (not the ~0.03 collapsed-policy noise floor)."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--auc-floor", type=float, default=0.25)
    parser.add_argument("--noise-floor", type=float, default=NOISE_FLOOR)
    parser.add_argument("--min-win-fraction", type=float, default=0.7)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = verify_run(
        args.run_dir,
        auc_floor=args.auc_floor,
        noise_floor=args.noise_floor,
        min_win_fraction=args.min_win_fraction,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
