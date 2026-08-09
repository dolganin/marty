from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank
from mars_rover_env.config import DEFAULT_ENV_CONFIG

from .harness import EvaluationSpec, evaluate
from .rl2_agent import RL2TransformerAgent
from .rl2_model import RL2TransformerActorCritic
from .tracking import DEFAULT_TRACKING_URI


def _metric_payload(prefix: str, payload: dict[str, Any]) -> dict[str, float]:
    summary = payload["summary"]
    metrics = {
        f"{prefix}.mean_raw_return": float(summary["mean_raw_return"]),
        f"{prefix}.raw_adaptation_delta": float(summary["raw_adaptation_delta"]),
        f"{prefix}.trial_auc": float(summary["trial_auc"]),
        f"{prefix}.adaptation_delta": float(summary["adaptation_delta"]),
        f"{prefix}.success_rate": float(summary["success_rate"]),
        f"{prefix}.flip_rate": float(summary["flip_rate"]),
        f"{prefix}.mean_battery_consumed": float(summary["mean_battery_consumed"]),
        f"{prefix}.mean_distance": float(summary["mean_distance"]),
    }
    for index, value in enumerate(summary["raw_return_by_episode"], start=1):
        metrics[f"{prefix}.raw_return.episode_{index}"] = float(value)
    for index, value in enumerate(summary["normalized_return_by_episode"] or [], start=1):
        metrics[f"{prefix}.normalized_return.episode_{index}"] = float(value)
    for index, value in enumerate(summary["action_entropy_by_episode"], start=1):
        metrics[f"{prefix}.action_entropy.episode_{index}"] = float(value)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full held-out finalization for a trained RL2 model")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_ENV_CONFIG)
    parser.add_argument("--eval-trials", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    parser.add_argument("--seed-start", type=int, default=70_000)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to evaluate the trained RL2 artifact")
    if args.eval_trials < 1 or args.eval_episodes < 1 or args.eval_max_steps < 1:
        raise SystemExit("evaluation dimensions must be positive")

    run_dir = args.run_dir.resolve()
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    artifact = json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))
    if run_record.get("status") != "complete":
        raise SystemExit("RL2 training run must be complete before finalization")
    if artifact.get("accelerator") != "cuda":
        raise SystemExit("refusing to finalize a non-CUDA training artifact")

    manifest = load_manifest(args.manifest)
    bank_version = require_compiled_bank(manifest)
    if artifact.get("bank_version_at_training") != bank_version:
        raise SystemExit("model and current bank versions differ")
    if artifact.get("reference_version") != manifest.get("reference_version"):
        raise SystemExit("model and frozen reference versions differ")

    device = torch.device("cuda:0")
    model_payload = torch.load(run_dir / artifact["model"], map_location=device, weights_only=False)
    model = RL2TransformerActorCritic(**model_payload["model_config"]).to(device)
    model.load_state_dict(model_payload["model_state"])
    model.eval()
    seeds = tuple(range(args.seed_start, args.seed_start + args.eval_trials))
    final_root = run_dir / "final"
    spec = {
        "seeds": list(seeds),
        "episodes_per_trial": args.eval_episodes,
        "max_steps": args.eval_max_steps,
        "bank_version": bank_version,
        "reference_version": manifest["reference_version"],
    }
    (final_root).mkdir(parents=True, exist_ok=True)
    (final_root / "protocol.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    def agent_factory(_seed: int) -> RL2TransformerAgent:
        return RL2TransformerAgent(model, device=device, deterministic=True)

    results: dict[str, dict[str, Any]] = {}
    for split in ("anchor", "test"):
        results[split] = evaluate(
            agent_factory,
            EvaluationSpec(
                split=split,
                seeds=seeds,
                episodes_per_trial=args.eval_episodes,
                max_steps=args.eval_max_steps,
                config_path=str(args.config),
                render_trials=1,
                render_stride=20,
                render_max_frames=240,
            ),
            manifest_path=args.manifest,
            output_dir=final_root / split,
        )

    run_id = run_record.get("mlflow", {}).get("run_id") or artifact.get("mlflow_run_id")
    if not run_id:
        raise SystemExit("training run has no MLflow run_id")
    try:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        with mlflow.start_run(run_id=str(run_id)):
            step = int(artifact["transitions"])
            for split, payload in results.items():
                mlflow.log_metrics(_metric_payload(f"final.{split}", payload), step=step)
            mlflow.log_artifacts(str(final_root), artifact_path="final")
            mlflow.log_artifact(str(args.manifest), artifact_path="final/provenance")
            mlflow.set_tags({
                "rl2.finalized": "true",
                "rl2.final.protocol": json.dumps(spec, separators=(",", ":")),
                "final.test.trial_auc": str(results["test"]["summary"]["trial_auc"]),
                "final.test.adaptation_delta": str(
                    results["test"]["summary"]["adaptation_delta"]
                ),
            })
    finally:
        from .tracking import artifact_exists

        for local_gif in final_root.rglob("*.gif"):
            remote_path = f"final/{local_gif.relative_to(final_root)}"
            if artifact_exists(args.tracking_uri, str(run_id), remote_path):
                local_gif.unlink()
            else:
                print(
                    f"warning: MLflow did not confirm upload of {remote_path}; "
                    f"keeping local copy at {local_gif}"
                )

    print(json.dumps({split: value["summary"] for split, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
