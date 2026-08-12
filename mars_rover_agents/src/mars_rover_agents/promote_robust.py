from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank, write_json

from .tracking import DEFAULT_TRACKING_URI


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote a train-selected robust checkpoint to the frozen reference artifact."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selected-at", type=int, required=True)
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    checkpoint = args.checkpoint.resolve()
    if run_dir not in checkpoint.parents or not checkpoint.is_file():
        raise SystemExit("checkpoint must be an existing file inside run_dir")
    source_artifact = json.loads((run_dir / "artifact.json").read_text(encoding="utf-8"))
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    manifest = load_manifest(args.manifest)
    bank_version = require_compiled_bank(manifest)
    if source_artifact.get("bank_version_at_training") != bank_version:
        raise SystemExit("source robust run belongs to a different bank")
    if source_artifact.get("train_version") != manifest.get("train_version"):
        raise SystemExit("source robust run belongs to a different train split")
    if source_artifact.get("accelerator") != "cuda":
        raise SystemExit("refusing to promote a non-CUDA artifact")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    model_path = args.output_dir / "model.zip"
    shutil.copy2(checkpoint, model_path)
    model_hash = _sha256(model_path)
    artifact = {
        **source_artifact,
        "model": model_path.name,
        "model_sha256": model_hash,
        "reference_version": "sha256:" + model_hash,
        "selected_at_timesteps": int(args.selected_at),
        "selection_split": "train",
        "selection_metric": "survival_auc_then_distance",
        "source_run_dir": str(run_dir),
        "source_checkpoint": checkpoint.name,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output_dir / "artifact.json", artifact)
    manifest["reference_version"] = artifact["reference_version"]
    # The held-out gate must be rerun against this exact frozen reference.
    manifest.setdefault("difficulty_gates", {}).pop("test", None)
    write_json(args.manifest, manifest)

    run_id = run_record.get("mlflow", {}).get("run_id") or source_artifact.get("mlflow_run_id")
    if run_id:
        import mlflow

        mlflow.set_tracking_uri(args.tracking_uri)
        client = mlflow.tracking.MlflowClient()
        client.log_artifact(str(run_id), str(model_path), "model/selected")
        client.log_artifact(str(run_id), str(args.output_dir / "artifact.json"), "model/selected")
        client.set_tag(str(run_id), "selected.reference_version", artifact["reference_version"])
        client.set_tag(str(run_id), "selected.at_timesteps", str(args.selected_at))
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
