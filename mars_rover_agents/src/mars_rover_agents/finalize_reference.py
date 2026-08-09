from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest
from mars_rover_env.tools.audit_bank import audit
from mars_rover_env.tools.evaluate_biomes import gate_bank

from .agents import PPOAgent, RandomAgent
from .calibration import freeze_anchor_references, renormalize_evaluation
from .harness import EvaluationSpec, evaluate
from .tracking import DEFAULT_TRACKING_URI, artifact_exists


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_payload(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    summary = payload["summary"]
    result = {
        f"{prefix}.mean_raw_return": float(summary["mean_raw_return"]),
        f"{prefix}.raw_adaptation_delta": float(summary["raw_adaptation_delta"]),
        f"{prefix}.success_rate": float(summary["success_rate"]),
        f"{prefix}.flip_rate": float(summary["flip_rate"]),
        f"{prefix}.mean_battery_consumed": float(summary["mean_battery_consumed"]),
        f"{prefix}.mean_distance": float(summary["mean_distance"]),
    }
    for key in ("trial_auc", "adaptation_delta"):
        if summary.get(key) is not None:
            result[f"{prefix}.{key}"] = float(summary[key])
    for index, value in enumerate(summary["raw_return_by_episode"], start=1):
        result[f"{prefix}.raw_return.episode_{index}"] = float(value)
    for index, value in enumerate(summary.get("normalized_return_by_episode") or [], start=1):
        result[f"{prefix}.normalized_return.episode_{index}"] = float(value)
    for biome_id in summary["mechanics"]:
        values = [
            float(row["normalized_return"])
            for row in payload["episodes"]
            if row["biome_id"] == biome_id and row.get("normalized_return") is not None
        ]
        if values:
            result[f"{prefix}.mechanic.{biome_id}.normalized_return"] = float(np.mean(values))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze test/anchor references and finalize MLflow")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config")
    parser.add_argument("--gate-episodes", type=int, default=20)
    parser.add_argument("--gate-max-steps", type=int, default=3000)
    parser.add_argument("--gate-seed", type=int, default=20_000)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.run_dir.resolve()
    artifact_path = run_dir / "artifact.json"
    model_path = run_dir / "model.zip"
    run_path = run_dir / "run.json"
    if not artifact_path.is_file() or not model_path.is_file() or not run_path.is_file():
        raise SystemExit(f"incomplete robust-PPO run: {run_dir}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    run_record = json.loads(run_path.read_text(encoding="utf-8"))
    if artifact.get("model_sha256") != _sha256(model_path):
        raise SystemExit("model hash differs from artifact metadata")
    if artifact.get("accelerator") != "cuda":
        raise SystemExit("refusing to finalize a non-CUDA training artifact")
    manifest = load_manifest(args.manifest)
    if manifest.get("reference_version") != artifact.get("reference_version"):
        raise SystemExit("manifest and trained reference version disagree")

    gate_args = SimpleNamespace(
        manifest=args.manifest,
        config=args.config,
        split="test",
        robust_model=str(model_path),
        episodes=args.gate_episodes,
        max_steps=args.gate_max_steps,
        seed=args.gate_seed,
        tau_low=0.20,
        solve_min=0.05,
        robust_max=0.85,
        min_reference_gap=1.0,
    )
    gate_bank(gate_args)
    audit(args.manifest, require_reference=True, require_test_gate=True)

    try:
        import torch
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("stable-baselines3 and torch are required") from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to evaluate the trained reference")
    model = PPO.load(model_path, device="cuda:0")
    if model.device.type != "cuda":
        raise RuntimeError(f"reference loaded on invalid device: {model.device}")

    final_root = run_dir / "final"
    seeds = tuple(range(50_000, 50_000 + args.eval_trials))
    common = {
        "seeds": seeds,
        "episodes_per_trial": args.eval_episodes,
        "max_steps": args.eval_max_steps,
        "config_path": args.config,
    }
    random_anchor_dir = final_root / "anchor_random"
    robust_anchor_dir = final_root / "anchor_robust"
    test_dir = final_root / "test_robust"
    random_anchor = evaluate(
        lambda seed: RandomAgent(seed),
        EvaluationSpec(split="anchor", render_trials=0, **common),
        manifest_path=args.manifest,
        output_dir=random_anchor_dir,
    )
    robust_anchor = evaluate(
        lambda _seed: PPOAgent(model),
        EvaluationSpec(split="anchor", render_trials=1, render_stride=60, **common),
        manifest_path=args.manifest,
        output_dir=robust_anchor_dir,
    )
    freeze_anchor_references(
        manifest_path=args.manifest,
        random_evaluation=random_anchor_dir,
        robust_evaluation=robust_anchor_dir,
        reference_version=artifact["reference_version"],
    )
    robust_anchor = renormalize_evaluation(robust_anchor_dir, args.manifest)
    test_result = evaluate(
        lambda _seed: PPOAgent(model),
        EvaluationSpec(split="test", render_trials=1, render_stride=60, **common),
        manifest_path=args.manifest,
        output_dir=test_dir,
    )
    audit(args.manifest, require_reference=True, require_test_gate=True)

    mlflow_data = run_record.get("mlflow", {})
    run_id = mlflow_data.get("run_id")
    if not run_id:
        raise SystemExit("training run has no MLflow run_id")
    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(_metric_payload("final.anchor.robust", robust_anchor), step=int(artifact["timesteps"]))
            mlflow.log_metrics(_metric_payload("final.test.robust", test_result), step=int(artifact["timesteps"]))
            mlflow.log_artifacts(str(final_root), artifact_path="final")
            mlflow.log_artifact(str(args.manifest), artifact_path="final/provenance")
            mlflow.set_tags({
                "reference.finalized": "true",
                "reference.version": artifact["reference_version"],
                "test_gate.protocol": "rollout-v2-public-macro-random",
                "anchor.count": str(len(robust_anchor["summary"]["mechanics"])),
                "final.test.trial_auc": str(test_result["summary"]["trial_auc"]),
                "final.test.adaptation_delta": str(test_result["summary"]["adaptation_delta"]),
            })
    finally:
        for local_gif in final_root.rglob("*.gif"):
            remote_path = f"final/{local_gif.relative_to(final_root)}"
            if artifact_exists(args.tracking_uri, str(run_id), remote_path):
                local_gif.unlink()
            else:
                print(
                    f"warning: MLflow did not confirm upload of {remote_path}; "
                    f"keeping local copy at {local_gif}"
                )
    print(json.dumps({
        "run_dir": str(run_dir),
        "mlflow_run_id": run_id,
        "reference_version": artifact["reference_version"],
        "anchor_trial_auc": robust_anchor["summary"]["trial_auc"],
        "test_trial_auc": test_result["summary"]["trial_auc"],
        "test_adaptation_delta": test_result["summary"]["adaptation_delta"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
