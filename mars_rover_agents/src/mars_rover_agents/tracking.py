from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_TRACKING_URI = "http://swagstation.netcraze.pro:4249"
DEFAULT_EXPERIMENT = "Marty/mars_rover_agents"


def artifact_exists(tracking_uri: str, run_id: str, remote_path: str) -> bool:
    """Confirm a logged artifact actually landed in the tracking server's store.

    ``mlflow.log_artifact`` resolves the experiment's ``artifact_location`` on
    the *client's own filesystem* whenever that location is a bare local path
    (no ``mlflow-artifacts:``/``s3:``/... scheme). If the server was started
    without artifact proxying and the client isn't on a filesystem shared with
    the server, the upload silently no-ops onto local disk and the run shows
    an empty artifact tree in the UI. Callers use this to avoid deleting the
    only copy of a local file when that has happened.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    directory = remote_path.rsplit("/", 1)[0] if "/" in remote_path else ""
    try:
        entries = client.list_artifacts(run_id, path=directory or None)
    except Exception:
        return False
    return any(entry.path == remote_path and not entry.is_dir for entry in entries)


def flatten_params(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            result.update(flatten_params(value, name))
        elif isinstance(value, (list, tuple)):
            result[name] = json.dumps(value, separators=(",", ":"))[:6000]
        elif value is not None:
            result[name] = str(value)[:6000]
    return result


def source_sha256(root: str | Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "__pycache__"} for part in path.parts):
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_provenance(workspace: str | Path) -> dict[str, str]:
    workspace = Path(workspace)
    result = {"git.commit": "unknown", "git.dirty": "unknown"}
    try:
        result["git.commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=workspace, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=workspace, text=True, stderr=subprocess.DEVNULL
        )
        result["git.dirty"] = str(bool(status.strip())).lower()
    except (OSError, subprocess.CalledProcessError):
        pass
    return result


class MLflowRun:
    """Small MLflow lifecycle wrapper with explicit experiment and provenance."""

    def __init__(
        self,
        *,
        run_name: str,
        params: dict[str, Any],
        tags: dict[str, Any],
        tracking_uri: str | None = None,
        experiment_name: str = DEFAULT_EXPERIMENT,
        system_metrics: bool = True,
        existing_run_id: str | None = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("install the 'train' extra to enable MLflow") from exc

        self.mlflow = mlflow
        self.tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        if system_metrics:
            mlflow.set_system_metrics_sampling_interval(10)
            mlflow.set_system_metrics_samples_before_logging(1)
        if existing_run_id is None:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = (
                mlflow.create_experiment(experiment_name)
                if experiment is None
                else experiment.experiment_id
            )
            self._run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name,
                log_system_metrics=system_metrics,
            )
            mlflow.log_params(flatten_params(params))
        else:
            self._run = mlflow.start_run(
                run_id=existing_run_id,
                log_system_metrics=system_metrics,
            )
        base_tags = {
            "project": "mars_rover_agents",
            "run_name": run_name,
            "host": platform.node(),
            "resume.existing_run": str(existing_run_id is not None).lower(),
            **{str(key): str(value) for key, value in tags.items() if value is not None},
        }
        mlflow.set_tags(base_tags)

    @property
    def run_id(self) -> str:
        return str(self._run.info.run_id)

    def log_metrics(self, metrics: dict[str, float | int | None], step: int) -> None:
        clean = {
            str(key).replace("/", "."): float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if clean:
            self.mlflow.log_metrics(clean, step=int(step), synchronous=True)

    def set_tags(self, tags: dict[str, Any]) -> None:
        self.mlflow.set_tags({str(k): str(v) for k, v in tags.items() if v is not None})

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        path = Path(path)
        if path.is_dir():
            self.mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        elif path.is_file():
            self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def artifact_exists(self, remote_path: str) -> bool:
        return artifact_exists(self.tracking_uri, self.run_id, remote_path)

    def log_dict(self, payload: dict[str, Any], artifact_file: str) -> None:
        self.mlflow.log_dict(payload, artifact_file)

    def close(self, status: str = "FINISHED") -> None:
        self.mlflow.end_run(status=status)


def training_provenance(
    *, workspace: str | Path, source_root: str | Path, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        **git_provenance(workspace),
        "source.sha256": source_sha256(source_root),
        "python.version": platform.python_version(),
        "platform": platform.platform(),
        **(extra or {}),
    }
