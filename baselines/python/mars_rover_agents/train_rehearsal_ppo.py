"""Consolidate a curriculum policy without catastrophic locomotion forgetting."""

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
from .curriculum import CurriculumStage, write_stage_configs
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
REHEARSAL_STAGES = (
    CurriculumStage(4, 0.25, 1.0, finish_x=190.0, max_steps=1600),
    CurriculumStage(8, 0.25, 1.0, finish_x=390.0, max_steps=2200),
    CurriculumStage(14, 0.50, 1.0),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interleaved 4/8/14-zone PPO consolidation")
    parser.add_argument("--initial-model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--timesteps", type=int, default=3_000_000)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2043)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--output-root", type=Path, default=WORKSPACE_ROOT / "artifacts" / "runs")
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    if not args.initial_model.is_file():
        raise SystemExit(f"initial model does not exist: {args.initial_model}")
    if args.timesteps < args.cycles * len(REHEARSAL_STAGES) or args.cycles < 1:
        raise SystemExit("timesteps must cover every stage of every cycle")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; refusing CPU fallback")

    bank_version = require_compiled_bank(load_manifest(args.manifest))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (args.output_root / f"rehearsal-ppo_s{args.seed}_{stamp}").resolve()
    run_dir.mkdir(parents=True)
    configs = write_stage_configs(args.config, run_dir / "configs", REHEARSAL_STAGES)
    per_cycle = args.timesteps // args.cycles
    unit = per_cycle // 4
    phase_budgets = [unit, unit, per_cycle - 2 * unit]
    spec = {
        "algorithm": "interleaved_rehearsal_curriculum_ppo_v1",
        "initial_model": str(args.initial_model.resolve()),
        "initial_model_sha256": hashlib.sha256(args.initial_model.read_bytes()).hexdigest(),
        "bank_version": bank_version,
        "cycles": args.cycles,
        "phase_order_zones": [4, 8, 14],
        "phase_budgets_per_cycle": phase_budgets,
        "requested_timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
    }
    (run_dir / "run.json").write_text(json.dumps({**spec, "status": "running"}, indent=2) + "\n")
    tracker = None if args.no_mlflow else MLflowRun(
        run_name=run_dir.name, params=spec,
        tags={"agent": "rehearsal_curriculum_ppo", "bank_version": bank_version},
        tracking_uri=args.tracking_uri, experiment_name=args.experiment,
    )
    model = None
    env = None
    completed = 0
    try:
        for cycle in range(args.cycles):
            for stage_index, (stage, config_path, budget) in enumerate(
                zip(REHEARSAL_STAGES, configs, phase_budgets), start=1
            ):
                if env is not None:
                    env.close()
                env = VecMonitor(MarsRoverSb3VecEnv(
                    args.num_envs, list(ACTION_MACROS), config_path=str(config_path),
                    biome_split=MarsRoverEnv.BIOME_MODE_TRAIN,
                    seed=args.seed + cycle * 1009 + stage_index * 97,
                ))
                if model is None:
                    model = PPO.load(args.initial_model, env=env, device="cuda:0")
                    model.learning_rate = args.learning_rate
                    model.lr_schedule = lambda _progress: args.learning_rate
                    for group in model.policy.optimizer.param_groups:
                        group["lr"] = args.learning_rate
                else:
                    model.set_env(env)
                model.learn(total_timesteps=budget, reset_num_timesteps=False, progress_bar=False)
                completed += budget
                if tracker is not None:
                    tracker.log_metrics({
                        "rehearsal.cycle": cycle + 1,
                        "rehearsal.zone_count": stage.zone_count,
                    }, completed)
            checkpoint = run_dir / "checkpoints" / f"cycle_{cycle + 1:02d}"
            checkpoint.parent.mkdir(exist_ok=True)
            model.save(checkpoint)

        assert model is not None
        model.save(run_dir / "model")
        archive = run_dir / "model.zip"
        result = {
            **spec, "status": "complete", "actual_rehearsal_timesteps": completed,
            "model_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
        (run_dir / "run.json").write_text(json.dumps(result, indent=2) + "\n")
        if tracker is not None:
            tracker.log_artifact(archive, "model")
            tracker.log_artifact(run_dir / "run.json", "provenance")
            tracker.set_tags({"status": "complete", "model_sha256": result["model_sha256"]})
            tracker.close("FINISHED")
            tracker = None
        print(f"Rehearsal PPO complete: {run_dir}")
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
