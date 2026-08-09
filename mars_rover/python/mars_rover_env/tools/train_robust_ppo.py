from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    load_manifest,
    require_compiled_bank,
    require_frozen_train_bank,
    write_json,
)


TRAINING_PROTOCOL = {
    "name": "robust-ppo-v1",
    "n_steps": 256,
    "batch_size": 2048,
    "learning_rate": 3.0e-4,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "policy_network": [256, 256],
    "deterministic_algorithms": True,
}


def _latest_checkpoint(output: Path) -> Path | None:
    pattern = re.compile(r"robust_ppo_(\d+)_steps\.zip$")
    matches = []
    for path in (output / "checkpoints").glob("robust_ppo_*_steps.zip"):
        match = pattern.search(path.name)
        if match:
            matches.append((int(match.group(1)), path))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and version the frozen robust-PPO reference")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=250_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="construct the exact CUDA PPO and execute one batch step without learn() or writes",
    )
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts" / "robust_ppo")
    args = parser.parse_args()

    try:
        import stable_baselines3
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import VecMonitor
        from mars_rover_env.envs.sb3_vec_env import MarsRoverSb3VecEnv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install with: python -m pip install -e '.[benchmark]'") from exc
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA GPU is required for robust-PPO training; refusing to fall back to CPU"
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    cuda_device = torch.device("cuda")
    print(f"Training accelerator: {torch.cuda.get_device_name(cuda_device)}")

    manifest = load_manifest(args.manifest)
    require_compiled_bank(manifest)
    train_version = require_frozen_train_bank(manifest)
    version_slug = train_version.split(":", 1)[-1][:16]
    output = args.output_root / version_slug
    if not args.preflight:
        output.mkdir(parents=True, exist_ok=True)
        existing_artifacts = [
            path for path in (output / "artifact.json", output / "model.zip") if path.exists()
        ]
        if existing_artifacts:
            raise SystemExit(
                "Frozen robust-PPO output already exists and will not be overwritten: "
                + ", ".join(str(path) for path in existing_artifacts)
            )

    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    if args.timesteps < 1:
        raise SystemExit("--timesteps must be positive")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be positive")
    if args.preflight and args.resume:
        raise SystemExit("--preflight and --resume are mutually exclusive")

    import _mars_rover_cpp as native

    config_path = Path(args.config) if args.config else DEFAULT_ENV_CONFIG
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    gate_config_sha256 = manifest["difficulty_gates"]["train"].get("config_sha256")
    if gate_config_sha256 != config_sha256:
        raise SystemExit(
            "Training env config differs from the frozen train difficulty-gate config"
        )
    run_spec = {
        "schema_version": 1,
        "train_version": train_version,
        "environment_version": native.environment_version(),
        "config_path": str(config_path.resolve()),
        "config_sha256": config_sha256,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "requested_timesteps": args.timesteps,
        "action_macros": ACTION_MACROS,
        "training_protocol": {
            **TRAINING_PROTOCOL,
            "train_difficulty_gate": manifest["difficulty_gates"]["train"],
        },
    }
    run_path = output / "run.json"
    if not args.preflight:
        if run_path.is_file():
            if not args.resume:
                raise SystemExit(
                    f"Interrupted run metadata exists; use --resume after inspection: {run_path}"
                )
            prior = json.loads(run_path.read_text(encoding="utf-8"))
            if prior.get("spec") != run_spec:
                raise SystemExit("Resume configuration does not match the interrupted frozen run")
        elif args.resume:
            raise SystemExit(f"Cannot resume without run metadata: {run_path}")
        else:
            write_json(
                run_path,
                {
                    "status": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "spec": run_spec,
                },
            )

    # Native simulation remains a single batched C++ call; observations/actions
    # cross into PyTorch in arrays, allowing PPO inference and updates on CUDA.
    native_adapter = MarsRoverSb3VecEnv(
        args.num_envs,
        list(ACTION_MACROS),
        config_path=args.config,
        biome_split=MarsRoverEnv.BIOME_MODE_TRAIN,
        seed=args.seed,
    )
    if native_adapter.native.max_steps <= 0:
        native_adapter.close()
        raise SystemExit("Training configuration must set termination.max_steps > 0")
    env = VecMonitor(native_adapter)
    checkpoint = _latest_checkpoint(output) if args.resume else None
    if args.resume:
        if checkpoint is None:
            env.close()
            raise SystemExit("No valid PPO checkpoint exists for the interrupted run")
        print(f"Resuming frozen run from {checkpoint}")
        run_record = json.loads(run_path.read_text(encoding="utf-8"))
        run_record.setdefault("resume_events", []).append(
            {
                "at": datetime.now(timezone.utc).isoformat(),
                "checkpoint": checkpoint.name,
            }
        )
        write_json(run_path, run_record)
        model = PPO.load(checkpoint, env=env, device=cuda_device)
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
            policy_kwargs={
                "net_arch": {
                    "pi": TRAINING_PROTOCOL["policy_network"],
                    "vf": TRAINING_PROTOCOL["policy_network"],
                }
            },
            device=cuda_device,
        )
    if model.device.type != "cuda":
        raise RuntimeError(f"PPO resolved an invalid training device: {model.device}")
    if args.preflight:
        observations = env.reset()
        actions, _ = model.predict(observations, deterministic=True)
        next_observations, rewards, _, _ = env.step(actions)
        env.close()
        if not np.isfinite(next_observations).all() or not np.isfinite(rewards).all():
            raise RuntimeError("CUDA preflight produced non-finite rollout data")
        print(
            f"CUDA preflight PASS: device={model.device} envs={args.num_envs} "
            f"obs={next_observations.shape} max_steps={native_adapter.native.max_steps}"
        )
        return
    model_path = output / "model"
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // args.num_envs),
        save_path=str(checkpoint_dir),
        name_prefix="robust_ppo",
        verbose=1,
    )
    try:
        remaining = args.timesteps - int(model.num_timesteps)
        if remaining > 0:
            model.learn(
                total_timesteps=remaining,
                callback=callback,
                reset_num_timesteps=not args.resume,
            )
        current_manifest = load_manifest(args.manifest)
        if current_manifest.get("train_version") != train_version:
            raise RuntimeError("Train bank changed during robust-PPO training; refusing to freeze")
        model.save(model_path)
    finally:
        env.close()
    archive = model_path.with_suffix(".zip")
    model_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
    reference_version = "sha256:" + model_sha256
    metadata = {
        "artifact_type": "robust_ppo",
        "train_version": train_version,
        "bank_version_at_training": manifest["bank_version"],
        "timesteps": int(model.num_timesteps),
        "requested_timesteps": args.timesteps,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "accelerator": "cuda",
        "gpu_name": torch.cuda.get_device_name(cuda_device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "stable_baselines3_version": stable_baselines3.__version__,
        "training_protocol": run_spec["training_protocol"],
        "config_sha256": config_sha256,
        "resumed": bool(args.resume),
        "resume_checkpoint": checkpoint.name if checkpoint is not None else None,
        "trained_splits": ["train"],
        "excluded_splits": ["anchor", "test"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": model_sha256,
        "reference_version": reference_version,
        "environment_version": native.environment_version(),
        "model": archive.name,
        "action_macros": list(ACTION_MACROS),
    }
    write_json(output / "artifact.json", metadata)
    current_manifest["reference_version"] = reference_version
    write_json(args.manifest, current_manifest)
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    run_record.update(
        {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "model_sha256": model_sha256,
            "actual_timesteps": int(model.num_timesteps),
        }
    )
    write_json(run_path, run_record)
    print(f"Frozen robust-PPO artifact: {output}")


if __name__ == "__main__":
    main()
