from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    load_manifest,
    require_compiled_bank,
    require_frozen_train_bank,
    write_json,
)
from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv

from .actions import ACTION_MACROS
from .callbacks import MLflowTrainingCallback
from .tracking import (
    DEFAULT_EXPERIMENT,
    DEFAULT_TRACKING_URI,
    MLflowRun,
    training_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
EXTRA_SPACE_RUNS_ROOT = Path("/extra_space/media/mars_rover/runs")
TRAINING_PROTOCOL: dict[str, Any] = {
    "name": "robust-ppo-public-agent-v2",
    "n_steps": 256,
    "batch_size": 2048,
    "learning_rate": 3.0e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "policy_network": [256, 256],
    "memory": "none",
    "frame_stack": 1,
    "domain_randomization": "episode_reset",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_name(seed: int, train_version: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"robust-ppo_s{seed}_{train_version.split(':')[-1][:8]}_{stamp}"


def _latest_checkpoint(run_dir: Path) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for path in (run_dir / "checkpoints").glob("robust_ppo_*_steps.zip"):
        try:
            candidates.append((int(path.stem.rsplit("_", 2)[-2]), path))
        except ValueError:
            continue
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-only robust PPO with public-agent evaluation")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--eval-seeds", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=1200)
    parser.add_argument("--output-root", type=Path, default=EXTRA_SPACE_RUNS_ROOT)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
        from stable_baselines3.common.vec_env import VecMonitor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install with: python -m pip install -e './mars_rover_agents[train]'") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; refusing CPU fallback for every training mode")
    if args.timesteps < 1 or args.num_envs < 1 or args.checkpoint_every < 1 or args.eval_every < 1:
        raise SystemExit("timesteps, num-envs, checkpoint-every, and eval-every must be positive")
    if args.eval_seeds < 1 or args.eval_episodes < 1 or args.eval_max_steps < 1:
        raise SystemExit("evaluation dimensions must be positive")
    if args.preflight and args.resume_dir is not None:
        raise SystemExit("--preflight and --resume-dir are mutually exclusive")

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)

    manifest = load_manifest(args.manifest)
    bank_version = require_compiled_bank(manifest)
    train_version = require_frozen_train_bank(manifest)
    config_hash = _sha256(args.config)
    gate_hash = manifest["difficulty_gates"]["train"].get("config_sha256")
    if gate_hash != config_hash:
        raise SystemExit("training config differs from the frozen train difficulty gate")

    import _mars_rover_cpp as native

    run_name = _run_name(args.seed, train_version)
    run_dir = args.resume_dir.resolve() if args.resume_dir else (args.output_root / run_name).resolve()
    prior_record: dict[str, Any] | None = None
    if args.resume_dir:
        if not (run_dir / "run.json").is_file():
            raise SystemExit(f"resume directory has no run.json: {run_dir}")
        prior_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        run_name = run_dir.name
    elif run_dir.exists() and not args.preflight:
        raise SystemExit(f"run directory already exists: {run_dir}")

    run_spec = {
        "schema_version": 2,
        "agent_interface": "Agent.reset/act/observe-v1",
        "agent": "robust_ppo",
        "bank_version": bank_version,
        "train_version": train_version,
        "environment_version": native.environment_version(),
        "config_path": str(args.config.resolve()),
        "config_sha256": config_hash,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_timesteps": args.timesteps,
        "action_macros": list(ACTION_MACROS),
        "training_protocol": TRAINING_PROTOCOL,
        "eval": {
            "split": "test",
            "seed_begin": 40_000,
            "seeds": args.eval_seeds,
            "episodes_per_trial": args.eval_episodes,
            "max_steps": args.eval_max_steps,
            "interval": args.eval_every,
        },
    }

    env = VecMonitor(
        MarsRoverSb3VecEnv(
            args.num_envs,
            list(ACTION_MACROS),
            config_path=str(args.config),
            biome_split=MarsRoverEnv.BIOME_MODE_TRAIN,
            seed=args.seed,
        )
    )
    if env.venv.native.max_steps <= 0:
        env.close()
        raise SystemExit("training environment must have a finite positive max_steps")

    checkpoint = _latest_checkpoint(run_dir) if args.resume_dir else None
    if args.resume_dir and checkpoint is None:
        env.close()
        raise SystemExit(f"no checkpoint found in {run_dir}")
    if checkpoint is not None:
        model = PPO.load(checkpoint, env=env, device=device)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            seed=args.seed,
            verbose=1,
            n_steps=TRAINING_PROTOCOL["n_steps"],
            batch_size=TRAINING_PROTOCOL["batch_size"],
            learning_rate=TRAINING_PROTOCOL["learning_rate"],
            gamma=TRAINING_PROTOCOL["gamma"],
            gae_lambda=TRAINING_PROTOCOL["gae_lambda"],
            ent_coef=TRAINING_PROTOCOL["ent_coef"],
            clip_range=TRAINING_PROTOCOL["clip_range"],
            max_grad_norm=TRAINING_PROTOCOL["max_grad_norm"],
            policy_kwargs={
                "net_arch": {
                    "pi": TRAINING_PROTOCOL["policy_network"],
                    "vf": TRAINING_PROTOCOL["policy_network"],
                }
            },
            device=device,
        )
    if model.device.type != "cuda":
        env.close()
        raise RuntimeError(f"invalid PPO device: {model.device}")

    observations = env.reset()
    actions, _ = model.predict(observations, deterministic=True)
    next_observations, rewards, _, _ = env.step(actions)
    if not np.isfinite(next_observations).all() or not np.isfinite(rewards).all():
        env.close()
        raise RuntimeError("CUDA preflight produced non-finite rollout data")
    if args.preflight:
        env.close()
        print(
            f"CUDA+env preflight PASS device={model.device} gpu={gpu_name!r} "
            f"envs={args.num_envs} obs={next_observations.shape}"
        )
        return

    run_dir.mkdir(parents=True, exist_ok=prior_record is not None)
    provenance = training_provenance(
        workspace=WORKSPACE_ROOT,
        source_root=PROJECT_ROOT / "src",
        extra={
            "torch.version": torch.__version__,
            "torch.cuda": torch.version.cuda,
            "stable_baselines3.version": stable_baselines3.__version__,
            "gpu.name": gpu_name,
            "gpu.count": torch.cuda.device_count(),
        },
    )
    if prior_record is not None:
        if prior_record.get("spec") != run_spec:
            raise SystemExit("resume configuration differs from the original frozen run")
        run_record = prior_record
        run_record["status"] = "running"
        run_record.pop("interrupted_at", None)
        run_record.setdefault("resume_events", []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(checkpoint),
            "source_sha256": provenance["source.sha256"],
        })
        run_record["resume_checkpoint"] = str(checkpoint)
    else:
        run_record = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "spec": run_spec,
            "provenance": provenance,
            "resume_checkpoint": None,
        }
    write_json(run_dir / "run.json", run_record)

    tracker: MLflowRun | None = None
    try:
        tracker = MLflowRun(
            run_name=run_name,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            params={**run_spec, "provenance": provenance},
            tags={
                "agent": "robust_ppo",
                "architecture": "feedforward_mlp",
                "memory": "none",
                "bank_version": bank_version,
                "train_version": train_version,
                "split.training": "train",
                "split.heldout": "anchor+test",
                "seed": args.seed,
                "gpu": gpu_name,
                "config_sha256": config_hash,
                "source_sha256": provenance["source.sha256"],
                "resumed": bool(checkpoint),
                "resume_checkpoint": checkpoint.name if checkpoint else None,
                "resume_source_sha256": provenance["source.sha256"] if checkpoint else None,
            },
            existing_run_id=(
                str(prior_record["mlflow"]["run_id"])
                if prior_record is not None
                else None
            ),
        )
        run_record["mlflow"] = {
            "tracking_uri": args.tracking_uri,
            "experiment": args.experiment,
            "run_id": tracker.run_id,
        }
        write_json(run_dir / "run.json", run_record)
        tracker.log_dict(run_spec, "provenance/run_spec.json")
        tracker.log_dict(provenance, "provenance/source.json")
        tracker.log_artifact(args.manifest, "provenance")
        tracker.log_artifact(args.config, "provenance")

        mlflow_callback = MLflowTrainingCallback(
            tracker=tracker,
            run_dir=run_dir,
            manifest_path=args.manifest,
            eval_every=args.eval_every,
            eval_seeds=tuple(range(40_000, 40_000 + args.eval_seeds)),
            eval_episodes=args.eval_episodes,
            eval_max_steps=args.eval_max_steps,
            config_path=str(args.config),
        )
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=prior_record is not None)
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.num_envs),
            save_path=str(checkpoint_dir),
            name_prefix="robust_ppo",
            verbose=1,
        )
        remaining = args.timesteps - int(model.num_timesteps)
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=CallbackList([mlflow_callback, checkpoint_callback]),
                reset_num_timesteps=checkpoint is None,
            )
        current_manifest = load_manifest(args.manifest)
        if current_manifest.get("bank_version") != bank_version:
            raise RuntimeError("bank changed during training; refusing to freeze model")
        model.save(run_dir / "model")
        archive = run_dir / "model.zip"
        model_hash = _sha256(archive)
        artifact = {
            "artifact_type": "robust_ppo",
            "agent_interface": run_spec["agent_interface"],
            "bank_version_at_training": bank_version,
            "train_version": train_version,
            "reference_version": "sha256:" + model_hash,
            "model_sha256": model_hash,
            "model": archive.name,
            "timesteps": int(model.num_timesteps),
            "seed": args.seed,
            "num_envs": args.num_envs,
            "accelerator": "cuda",
            "gpu_name": gpu_name,
            "environment_version": native.environment_version(),
            "config_sha256": config_hash,
            "action_macros": list(ACTION_MACROS),
            "training_protocol": TRAINING_PROTOCOL,
            "trained_splits": ["train"],
            "excluded_splits": ["anchor", "test"],
            "mlflow_run_id": tracker.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "artifact.json", artifact)
        current_manifest["reference_version"] = artifact["reference_version"]
        write_json(args.manifest, current_manifest)
        tracker.log_artifact(archive, "model")
        tracker.log_artifact(run_dir / "artifact.json", "model")
        run_record.update(
            {
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "actual_timesteps": int(model.num_timesteps),
                "model_sha256": model_hash,
            }
        )
        write_json(run_dir / "run.json", run_record)
        tracker.log_artifact(run_dir / "run.json", "provenance")
        tracker.set_tags({"status": "complete", "model_sha256": model_hash})
        tracker.close("FINISHED")
        tracker = None
        print(f"Robust PPO complete: {run_dir}")
    except BaseException:
        run_record["status"] = "failed"
        run_record["failed_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "run.json", run_record)
        if tracker is not None:
            tracker.set_tags({"status": "failed"})
            tracker.log_artifact(run_dir / "run.json", "provenance")
            tracker.close("FAILED")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
