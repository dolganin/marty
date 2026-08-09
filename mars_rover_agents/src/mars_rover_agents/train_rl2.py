from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank, write_json
from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv
from mars_rover_env.tools.audit_bank import audit

from .harness import EvaluationSpec, evaluate
from .distillation import distill_from_ppo_teacher
from .rl2_agent import RL2TransformerAgent
from .rl2_model import RL2TransformerActorCritic, rl2_features
from .trial_context import trial_context_dim
from .rl2_training import collect_trial_batch, ppo_update
from .tracking import DEFAULT_EXPERIMENT, DEFAULT_TRACKING_URI, MLflowRun, training_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
# Runs (checkpoints, evaluation GIFs, trajectories) go on the large data disk,
# not the small workspace disk. python/mars_rover_agents/runs is kept as a
# symlink here for anything that still looks up a relative "runs/" path.
EXTRA_SPACE_RUNS_ROOT = Path("/extra_space/media/mars_rover/runs")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_name(seed: int, bank_version: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"rl2-transformer_s{seed}_{bank_version.split(':')[-1][:8]}_{stamp}"


def _latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = list((run_dir / "checkpoints").glob("rl2_*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda path: int(path.stem.split("_")[-1]))


def _log_evaluation(
    *,
    model: RL2TransformerActorCritic,
    device: torch.device,
    tracker: MLflowRun,
    run_dir: Path,
    manifest_path: Path,
    config_path: str,
    transitions: int,
    seeds: tuple[int, ...],
    max_steps: int,
) -> dict[str, Any]:
    eval_dir = run_dir / "evaluations" / f"step_{transitions:09d}"
    payload = evaluate(
        lambda _seed: RL2TransformerAgent(
            model, device=device, deterministic=True, max_steps=max_steps
        ),
        EvaluationSpec(
            split="test",
            seeds=seeds,
            episodes_per_trial=4,
            max_steps=max_steps,
            config_path=config_path,
            render_trials=1,
            render_stride=12,
            render_max_frames=180,
        ),
        manifest_path=manifest_path,
        output_dir=eval_dir,
    )
    summary = payload["summary"]
    metrics: dict[str, float | None] = {
        "heldout.mean_raw_return": summary["mean_raw_return"],
        "heldout.raw_adaptation_delta": summary["raw_adaptation_delta"],
        "heldout.trial_auc": summary["trial_auc"],
        "heldout.adaptation_delta": summary["adaptation_delta"],
        "heldout.success_rate": summary["success_rate"],
        "heldout.flip_rate": summary["flip_rate"],
        "heldout.mean_battery_consumed": summary["mean_battery_consumed"],
        "heldout.mean_distance": summary["mean_distance"],
    }
    for index, value in enumerate(summary["raw_return_by_episode"], start=1):
        metrics[f"heldout.raw_return.episode_{index}"] = value
    for index, value in enumerate(summary["normalized_return_by_episode"] or [], start=1):
        metrics[f"heldout.normalized_return.episode_{index}"] = value
    for index, value in enumerate(summary["action_entropy_by_episode"], start=1):
        metrics[f"heldout.action_entropy.episode_{index}"] = value
    tracker.log_metrics(metrics, transitions)
    local_gif = eval_dir / "behavior.gif"
    remote_gif_path = f"evaluations/step_{transitions:09d}/behavior.gif"
    try:
        tracker.log_artifact(eval_dir, artifact_path=f"evaluations/step_{transitions:09d}")
    finally:
        if local_gif.is_file():
            if tracker.artifact_exists(remote_gif_path):
                local_gif.unlink()
            else:
                print(
                    f"warning: MLflow did not confirm upload of {remote_gif_path}; "
                    f"keeping local copy at {local_gif}"
                )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-only trial-native RL2 Transformer PPO")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--total-transitions", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--memory-len", type=int, default=256)
    parser.add_argument(
        "--context-history",
        type=int,
        default=0,
        help=(
            "Number of finished episodes summarised into explicit input features "
            "(0 keeps the pure attention-only RL2 architecture)."
        ),
    )
    parser.add_argument("--chunk-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--trial-minibatch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--final-episode-weight", type=float, default=1.0)
    parser.add_argument("--selection-min-auc", type=float, default=0.0)
    parser.add_argument(
        "--kl-anchor-coefficient",
        type=float,
        default=0.0,
        help="Penalty weight for KL(current || frozen post-warm-start snapshot); 0 disables the anchor.",
    )
    parser.add_argument(
        "--value-warmup-updates",
        type=int,
        default=0,
        help="Number of initial PPO updates with policy/entropy gradient zeroed (value head only).",
    )
    parser.add_argument(
        "--dynamics-coefficient",
        type=float,
        default=0.0,
        help=(
            "Weight of the self-supervised next-step dynamics prediction loss. Reward alone "
            "does not pressure the representation to identify the mechanic; this does."
        ),
    )
    parser.add_argument(
        "--belief-dim",
        type=int,
        default=0,
        help=(
            "VariBAD: size of the explicit posterior over the mechanic. 0 keeps pure RL2 "
            "(implicit memory only), so the two differ by the belief alone."
        ),
    )
    parser.add_argument("--belief-kl-coefficient", type=float, default=0.0)
    parser.add_argument("--eval-every-updates", type=int, default=5)
    # One evaluation seed makes trial_auc/adaptation_delta dominated by which course
    # the shared per-episode seed happened to draw, not by the policy. Averaging over
    # several trials is what makes the adaptation signal readable at all.
    parser.add_argument("--eval-seeds", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=1200)
    parser.add_argument("--checkpoint-every-updates", type=int, default=2)
    parser.add_argument("--teacher-model", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
    parser.add_argument("--distill-transitions", type=int, default=0)
    parser.add_argument("--distill-chunk-length", type=int, default=128)
    parser.add_argument("--distill-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--distill-temperature", type=float, default=1.0)
    parser.add_argument("--distill-rollout-temperature", type=float, default=1.0)
    parser.add_argument("--output-root", type=Path, default=EXTRA_SPACE_RUNS_ROOT)
    parser.add_argument("--resume-dir", type=Path)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--preflight", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; RL2 training refuses CPU fallback")
    if args.num_envs < 1 or args.total_transitions < 1:
        raise SystemExit("num-envs and total-transitions must be positive")
    if args.chunk_length < 1 or args.memory_len < args.chunk_length:
        raise SystemExit("memory-len must be at least chunk-length")
    if args.trial_minibatch_size < 1 or args.trial_minibatch_size > args.num_envs:
        raise SystemExit("invalid trial minibatch size")
    if (args.teacher_model is None) != (args.distill_transitions == 0):
        raise SystemExit("teacher-model and a positive distill-transitions must be used together")
    if args.teacher_model is not None and args.initialization_checkpoint is not None:
        raise SystemExit("teacher-model and initialization-checkpoint are mutually exclusive")
    if args.distill_chunk_length < 1 or args.distill_learning_rate <= 0:
        raise SystemExit("invalid distillation configuration")
    if not 0.0 < args.distill_temperature <= 1.0:
        raise SystemExit("distill-temperature must be in (0, 1]")
    if args.final_episode_weight < 1.0:
        raise SystemExit("final-episode-weight must be at least one")
    if not 0.0 <= args.selection_min_auc <= 1.0:
        raise SystemExit("selection-min-auc must be in [0, 1]")
    audit(args.manifest, require_reference=True, require_test_gate=True)
    manifest = load_manifest(args.manifest)
    bank_version = require_compiled_bank(manifest)
    if not manifest.get("anchor_references"):
        raise SystemExit("anchor references must be frozen before RL2 training")
    config_hash = _sha256(args.config)
    if config_hash != manifest["difficulty_gates"]["train"].get("config_sha256"):
        raise SystemExit("RL2 config differs from the frozen bank protocol")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    env = MarsRoverVecEnv(
        args.num_envs,
        config_path=str(args.config),
        biome_split=MarsRoverEnv.BIOME_MODE_TRAIN,
    )
    model_config = {
        "observation_dim": env.obs_dim,
        "model_dim": args.model_dim,
        "heads": args.heads,
        "layers": args.layers,
        "ff_dim": args.ff_dim,
        "memory_len": args.memory_len,
        "context_dim": trial_context_dim(args.context_history) if args.context_history else 0,
        "belief_dim": args.belief_dim,
    }
    model = RL2TransformerActorCritic(**model_config).to(device)
    initialization_sha256: str | None = None
    if args.initialization_checkpoint is not None:
        initialization_path = args.initialization_checkpoint.resolve()
        initialization_record_path = initialization_path.parent / "run.json"
        if not initialization_record_path.is_file():
            raise SystemExit("initialization checkpoint has no adjacent run.json")
        initialization_record = json.loads(
            initialization_record_path.read_text(encoding="utf-8")
        )
        initialization_spec = initialization_record.get("spec", {})
        if initialization_spec.get("bank_version") != bank_version:
            raise SystemExit("initialization checkpoint and current bank versions differ")
        if initialization_spec.get("reference_version") != manifest["reference_version"]:
            raise SystemExit("initialization checkpoint and reference versions differ")
        initialization_payload = torch.load(
            initialization_path, map_location=device, weights_only=False
        )
        if initialization_payload.get("model_config") != model_config:
            raise SystemExit("initialization checkpoint model config differs")
        model.load_state_dict(initialization_payload["model_state"])
        initialization_sha256 = _sha256(initialization_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-5)
    one_features = rl2_features(
        env.reset(args.seed),
        np.full(args.num_envs, -1),
        np.zeros(args.num_envs),
        np.ones(args.num_envs, dtype=bool),
        device=device,
        trial_context=(
            np.zeros((args.num_envs, model.context_dim), dtype=np.float32)
            if model.context_dim
            else None
        ),
    )
    with torch.no_grad():
        logits, values, _ = model.forward_step(
            one_features, model.initial_state(args.num_envs, device)
        )
    if logits.device.type != "cuda" or values.device.type != "cuda":
        raise RuntimeError("RL2 preflight tensors are not on CUDA")
    if args.preflight:
        print(
            f"RL2 CUDA preflight PASS gpu={gpu_name!r} envs={args.num_envs} "
            f"features={tuple(one_features.shape)} logits={tuple(logits.shape)}"
        )
        return

    run_name = _run_name(args.seed, bank_version)
    prior_record: dict[str, Any] | None = None
    run_dir = args.resume_dir.resolve() if args.resume_dir else (args.output_root / run_name).resolve()
    if args.resume_dir:
        run_path = run_dir / "run.json"
        if not run_path.is_file():
            raise SystemExit(f"resume run has no metadata: {run_path}")
        prior_record = json.loads(run_path.read_text(encoding="utf-8"))
        run_name = run_dir.name
    elif run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=prior_record is not None)
    (run_dir / "checkpoints").mkdir(exist_ok=prior_record is not None)

    training_protocol = {
        "name": "rl2-transformer-xl-ppo-v1",
        "batch_unit": "complete_trial",
        "episode_boundary_terminal": False,
        "trial_boundary_terminal": True,
        "explicit_inputs": ["observation", "previous_action", "previous_reward", "previous_done"],
        "num_envs": args.num_envs,
        "episodes_per_trial": env.episodes_per_trial,
        "model": model_config,
        "chunk_length": args.chunk_length,
        "epochs": args.epochs,
        "trial_minibatch_size": args.trial_minibatch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "value_coefficient": args.value_coefficient,
        "entropy_coefficient": args.entropy_coefficient,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "final_episode_weight": args.final_episode_weight,
        "kl_anchor_coefficient": args.kl_anchor_coefficient,
        "dynamics_coefficient": args.dynamics_coefficient,
        "belief_dim": args.belief_dim,
        "belief_kl_coefficient": args.belief_kl_coefficient,
        "value_warmup_updates": args.value_warmup_updates,
        "model_selection": {
            "minimum_trial_auc": args.selection_min_auc,
            "score": "adaptation_delta + 0.1 * trial_auc",
            "fallback": "highest_trial_auc_if_no_candidate_passes_floor",
        },
        "warm_start": {
            "method": (
                "public_observation_ppo_distillation"
                if args.teacher_model
                else "rl2_initialization_checkpoint"
                if args.initialization_checkpoint
                else "none"
            ),
            "teacher_model": str(args.teacher_model.resolve()) if args.teacher_model else None,
            "initialization_checkpoint": (
                str(args.initialization_checkpoint.resolve())
                if args.initialization_checkpoint
                else None
            ),
            "initialization_sha256": initialization_sha256,
            "transitions": args.distill_transitions,
            "chunk_length": args.distill_chunk_length,
            "learning_rate": args.distill_learning_rate,
            "teacher_temperature": args.distill_temperature,
            "privileged_inputs": False,
        },
    }
    run_spec = {
        "schema_version": 1,
        "agent": "rl2_transformer",
        "agent_interface": "Agent.reset/act/observe-v1",
        "bank_version": bank_version,
        "train_version": manifest["train_version"],
        "reference_version": manifest["reference_version"],
        "environment_version": __import__("_mars_rover_cpp").environment_version(),
        "config_sha256": config_hash,
        "seed": args.seed,
        "total_transitions": args.total_transitions,
        "training_protocol": training_protocol,
    }
    provenance = training_provenance(
        workspace=WORKSPACE_ROOT,
        source_root=PROJECT_ROOT / "src",
        extra={
            "torch.version": torch.__version__,
            "torch.cuda": torch.version.cuda,
            "gpu.name": gpu_name,
            "gpu.count": torch.cuda.device_count(),
        },
    )
    transitions = 0
    update = 0
    checkpoint = _latest_checkpoint(run_dir) if prior_record is not None else None
    if prior_record is not None:
        if prior_record.get("spec") != run_spec:
            raise SystemExit("resume configuration differs from the original RL2 run")
        if checkpoint is None:
            raise SystemExit("RL2 resume run has no checkpoint")
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_compatible_state(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        transitions = int(state["transitions"])
        update = int(state["update"])
        run_record = prior_record
        run_record["status"] = "running"
        run_record.setdefault("resume_events", []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint.name,
            "source_sha256": provenance["source.sha256"],
        })
    else:
        run_record = {
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "spec": run_spec,
            "provenance": provenance,
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
                "agent": "rl2_transformer",
                "architecture": "transformer_xl",
                "memory": "trial_persistent",
                "batch_unit": "trial",
                "bank_version": bank_version,
                "train_version": manifest["train_version"],
                "reference_version": manifest["reference_version"],
                "split.training": "train",
                "split.heldout": "anchor+test",
                "seed": args.seed,
                "gpu": gpu_name,
                "source_sha256": provenance["source.sha256"],
                "resumed": bool(checkpoint),
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
        if args.initialization_checkpoint is not None and update == 0:
            tracker.log_artifact(args.initialization_checkpoint, "provenance/initialization")
            tracker.set_tags({
                "warm_start": "rl2_initialization_checkpoint",
                "warm_start.initialization_sha256": initialization_sha256,
            })
        if args.teacher_model is not None and update == 0:
            from stable_baselines3 import PPO

            teacher_artifact_path = args.teacher_model.parent / "artifact.json"
            if not teacher_artifact_path.is_file():
                raise SystemExit("teacher model has no adjacent artifact.json")
            teacher_artifact = json.loads(teacher_artifact_path.read_text(encoding="utf-8"))
            if teacher_artifact.get("reference_version") != manifest["reference_version"]:
                raise SystemExit("teacher and current reference versions differ")
            teacher = PPO.load(args.teacher_model, device=device)
            tracker.log_artifact(teacher_artifact_path, "provenance/teacher")
            completed_distillation = distill_from_ppo_teacher(
                model,
                teacher,
                env,
                device=device,
                transitions=args.distill_transitions,
                chunk_length=args.distill_chunk_length,
                learning_rate=args.distill_learning_rate,
                teacher_temperature=args.distill_temperature,
                rollout_temperature=args.distill_rollout_temperature,
                seed=args.seed ^ 0xD157111,
                metric_callback=lambda metrics, step: tracker.log_metrics(metrics, step),
            )
            tracker.set_tags({
                "warm_start": "public_observation_ppo_distillation",
                "warm_start.transitions": completed_distillation,
                "warm_start.teacher_sha256": _sha256(args.teacher_model),
            })
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "teacher_model_sha256": _sha256(args.teacher_model),
                    "distill_transitions": completed_distillation,
                },
                run_dir / "distilled_initialization.pt",
            )
            tracker.log_artifact(run_dir / "distilled_initialization.pt", "checkpoints")
        evaluation_seeds = tuple(range(60_000, 60_000 + args.eval_seeds))
        best_path = run_dir / "best_model_state.pt"
        best_score = float("-inf")
        best_auc = float("-inf")
        best_step = 0
        best_summary: dict[str, Any] | None = None

        def consider_evaluation(payload: dict[str, Any], step: int) -> None:
            nonlocal best_score, best_auc, best_step, best_summary
            summary = payload["summary"]
            auc = float(summary["trial_auc"])
            delta = float(summary["adaptation_delta"])
            passes_floor = auc >= args.selection_min_auc
            score = delta + 0.1 * auc if passes_floor else float("-inf")
            should_replace = score > best_score
            if best_score == float("-inf") and score == float("-inf") and auc > best_auc:
                should_replace = True
            tracker.log_metrics(
                {
                    "heldout.selection_score": score if passes_floor else None,
                    "heldout.selection_passes_auc_floor": float(passes_floor),
                },
                step,
            )
            if not should_replace:
                return
            best_score = score
            best_auc = auc
            best_step = step
            best_summary = dict(summary)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config,
                    "evaluation_step": step,
                    "selection_score": score,
                    "summary": summary,
                },
                best_path,
            )
            tracker.log_artifact(best_path, "checkpoints/best")

        if update == 0:
            initial_evaluation = _log_evaluation(
                model=model,
                device=device,
                tracker=tracker,
                run_dir=run_dir,
                manifest_path=args.manifest,
                config_path=str(args.config),
                transitions=transitions,
                seeds=evaluation_seeds,
                max_steps=args.eval_max_steps,
            )
            consider_evaluation(initial_evaluation, transitions)

        reference_model: RL2TransformerActorCritic | None = None
        if args.kl_anchor_coefficient > 0.0:
            reference_model = RL2TransformerActorCritic(**model_config).to(device)
            reference_model.load_state_dict(model.state_dict())
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
            reference_model.eval()

        rng = np.random.default_rng(args.seed + update * 7919)
        while transitions < args.total_transitions:
            batch = collect_trial_batch(
                env,
                model,
                device=device,
                seed=args.seed + update * 104729,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            metrics = ppo_update(
                model,
                optimizer,
                batch,
                device=device,
                epochs=args.epochs,
                trial_minibatch_size=args.trial_minibatch_size,
                chunk_length=args.chunk_length,
                clip_range=args.clip_range,
                value_coefficient=args.value_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                max_grad_norm=args.max_grad_norm,
                target_kl=args.target_kl,
                final_episode_weight=args.final_episode_weight,
                rng=rng,
                reference_model=reference_model,
                kl_anchor_coefficient=args.kl_anchor_coefficient,
                freeze_policy=update < args.value_warmup_updates,
                dynamics_coefficient=args.dynamics_coefficient,
                observation_dim=env.obs_dim,
                belief_kl_coefficient=args.belief_kl_coefficient,
            )
            transitions += batch.total_steps
            update += 1
            episode_returns = batch.episode_returns.mean(axis=0)
            episode_entropies = batch.episode_entropies.mean(axis=0)
            metrics.update({
                "learning_rate": optimizer.param_groups[0]["lr"],
                "rollout.trial_auc_raw": float(episode_returns.mean()),
                "rollout.adaptation_delta_raw": float(episode_returns[-1] - episode_returns[0]),
                "rollout.mean_trial_length": float(batch.lengths.float().mean()),
                "rollout.transitions": float(batch.total_steps),
                "system.gpu.memory_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
                "system.gpu.memory_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
            })
            for index, value in enumerate(episode_returns, start=1):
                metrics[f"rollout.raw_return.episode_{index}"] = float(value)
            for index, value in enumerate(episode_entropies, start=1):
                metrics[f"rollout.action_entropy.episode_{index}"] = float(value)
            tracker.log_metrics(metrics, transitions)

            checkpoint_due = update % args.checkpoint_every_updates == 0
            evaluation_due = update % args.eval_every_updates == 0
            if checkpoint_due or transitions >= args.total_transitions:
                checkpoint_path = run_dir / "checkpoints" / f"rl2_{transitions}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": model_config,
                    "transitions": transitions,
                    "update": update,
                    "run_spec": run_spec,
                }, checkpoint_path)
                tracker.log_artifact(checkpoint_path, "checkpoints")
            if evaluation_due or transitions >= args.total_transitions:
                heldout_evaluation = _log_evaluation(
                    model=model,
                    device=device,
                    tracker=tracker,
                    run_dir=run_dir,
                    manifest_path=args.manifest,
                    config_path=str(args.config),
                    transitions=transitions,
                    seeds=evaluation_seeds,
                    max_steps=args.eval_max_steps,
                )
                consider_evaluation(heldout_evaluation, transitions)

        if not best_path.is_file() or best_summary is None:
            raise RuntimeError("RL2 model selection produced no checkpoint")
        best_payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_payload["model_state"])
        model_path = run_dir / "model.pt"
        torch.save({
            "model_state": model.state_dict(),
            "model_config": model_config,
            "agent": "rl2_transformer",
            "bank_version": bank_version,
            "transitions": transitions,
            "selected_at_transitions": best_step,
            "selection_score": best_score,
            "selected_heldout_summary": best_summary,
        }, model_path)
        model_hash = _sha256(model_path)
        artifact = {
            "artifact_type": "rl2_transformer",
            "model": model_path.name,
            "model_sha256": model_hash,
            "bank_version_at_training": bank_version,
            "train_version": manifest["train_version"],
            "reference_version": manifest["reference_version"],
            "environment_version": __import__("_mars_rover_cpp").environment_version(),
            "config_sha256": config_hash,
            "transitions": transitions,
            "updates": update,
            "seed": args.seed,
            "accelerator": "cuda",
            "gpu_name": gpu_name,
            "training_protocol": training_protocol,
            "trained_splits": ["train"],
            "excluded_splits": ["anchor", "test"],
            "mlflow_run_id": tracker.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "artifact.json", artifact)
        tracker.log_artifact(model_path, "model")
        tracker.log_artifact(run_dir / "artifact.json", "model")
        run_record.update({
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "actual_transitions": transitions,
            "updates": update,
            "model_sha256": model_hash,
            "selected_at_transitions": best_step,
            "selection_score": best_score,
        })
        write_json(run_dir / "run.json", run_record)
        tracker.log_artifact(run_dir / "run.json", "provenance")
        tracker.set_tags({
            "status": "complete",
            "model_sha256": model_hash,
            "model.selected_at_transitions": best_step,
            "model.selection_score": best_score,
        })
        tracker.close("FINISHED")
        tracker = None
        print(f"RL2 Transformer complete: {run_dir}")
    except BaseException as exc:
        run_record["status"] = "failed"
        run_record["failed_at"] = datetime.now(timezone.utc).isoformat()
        run_record["error_type"] = type(exc).__name__
        run_record["error_message"] = str(exc)
        run_record["traceback"] = traceback.format_exc()
        write_json(run_dir / "run.json", run_record)
        if tracker is not None:
            tracker.set_tags({"status": "failed"})
            tracker.log_dict(
                {
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": run_record["traceback"],
                },
                "diagnostics/failure.json",
            )
            tracker.log_artifact(run_dir / "run.json", "provenance")
            tracker.close("FAILED")
        raise


if __name__ == "__main__":
    main()
