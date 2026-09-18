"""PPO curriculum from basic locomotion to the full procedural chain.

The failed baselines begin directly in a 14-zone, 800 m task where a random early
crash costs 300 reward.  This schedule first makes forward motion learnable, then
adds mechanic transitions and finally restores the exact target environment.  It
does not expose biome identity or simulator state and therefore preserves the
public-agent contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank
from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS
from .curriculum import DEFAULT_CURRICULUM, allocate_timesteps, curriculum_payload, write_stage_configs
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun, training_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chain-length curriculum PPO (GPU-only)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--seed", type=int, default=2042)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "artifacts" / "runs")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError as exc:                    
        raise SystemExit("Install with: python -m pip install -e '.[train]'") from exc

    if args.timesteps < len(DEFAULT_CURRICULUM) or args.num_envs < 1:
        raise SystemExit("timesteps must cover every stage and num-envs must be positive")
    manifest = load_manifest(args.manifest)
    bank_version = require_compiled_bank(manifest)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.output_root / f"curriculum-ppo_s{args.seed}_{stamp}").resolve()
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    stage_paths = write_stage_configs(args.config, run_dir / "configs", DEFAULT_CURRICULUM)
    budgets = allocate_timesteps(args.timesteps, DEFAULT_CURRICULUM)
    provenance = training_provenance(workspace=WORKSPACE_ROOT, source_root=PROJECT_ROOT / "python")
    spec = {
        "algorithm": "chain_consequence_curriculum_ppo_v1",
        "hypothesis": "learn locomotion before long-horizon risk avoidance",
        "base_config": str(args.config.resolve()),
        "base_config_sha256": _sha256(args.config),
        "bank_version": bank_version,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_timesteps": args.timesteps,
        "stages": curriculum_payload(DEFAULT_CURRICULUM),
        "stage_budgets": budgets,
        "action_macros": list(ACTION_MACROS),
    }
    (run_dir / "run.json").write_text(json.dumps({**spec, "status": "running"}, indent=2) + "\n")
    tracker = None
    if not args.no_mlflow:
        tracker = MLflowRun(
            run_name=run_dir.name,
            params={**spec, "provenance": provenance},
            tags={"agent": "curriculum_ppo", "architecture": "feedforward_mlp", "bank_version": bank_version},
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
        tracker.log_dict(spec, "provenance/run_spec.json")
        tracker.log_artifact(run_dir / "configs", "curriculum_configs")

    if args.preflight:
        print(json.dumps({"run_dir": str(run_dir), "stage_budgets": budgets, "configs": [str(p) for p in stage_paths]}, indent=2))
        if tracker is not None:
            tracker.set_tags({"status": "preflight"})
            tracker.close("FINISHED")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; refusing CPU fallback")

    model = None
    env = None
    completed = 0
    try:
        for index, (stage, config_path, budget) in enumerate(zip(DEFAULT_CURRICULUM, stage_paths, budgets), start=1):
            if env is not None:
                env.close()
            env = VecMonitor(MarsRoverSb3VecEnv(
                args.num_envs, list(ACTION_MACROS), config_path=str(config_path),
                biome_split=MarsRoverEnv.BIOME_MODE_TRAIN, seed=args.seed + index * 1009,
            ))
            if model is None:
                model = PPO(
                    "MlpPolicy", env, device="cuda:0", seed=args.seed, verbose=1,
                    n_steps=256, batch_size=2048, learning_rate=3e-4,
                    gamma=0.995, gae_lambda=0.95, ent_coef=0.02,
                    policy_kwargs={"net_arch": [256, 256]},
                )
            else:
                model.set_env(env)
            model.learn(total_timesteps=budget, reset_num_timesteps=False, progress_bar=False)
            completed += budget
            checkpoint = run_dir / "checkpoints" / f"stage_{index:02d}_zones_{stage.zone_count}"
            checkpoint.parent.mkdir(exist_ok=True)
            model.save(checkpoint)
            if tracker is not None:
                tracker.log_metrics({
                    "curriculum.stage": index,
                    "curriculum.zone_count": stage.zone_count,
                    "curriculum.crash_penalty_scale": stage.crash_penalty_scale,
                }, completed)
                tracker.log_artifact(checkpoint.with_suffix(".zip"), "checkpoints")

        assert model is not None
        model.save(run_dir / "model")
        archive = run_dir / "model.zip"
        result = {**spec, "status": "complete", "actual_timesteps": int(model.num_timesteps), "model_sha256": _sha256(archive)}
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        if tracker is not None:
            tracker.log_artifact(archive, "model")
            tracker.log_artifact(run_dir / "run.json", "provenance")
            tracker.set_tags({"status": "complete", "model_sha256": result["model_sha256"]})
            tracker.close("FINISHED")
            tracker = None
        print(f"Curriculum PPO complete: {run_dir}")
    except BaseException:
        if tracker is not None:
            tracker.set_tags({"status": "failed"})
            tracker.close("FAILED")
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
