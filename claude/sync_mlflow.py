"""Заливка истории прогона (runs/<name>/log.jsonl + ckpt.pt) в удалённый MLflow.

Нужна, когда обучение уже шло, а трекинг надо перенести на сервер:

    .venv/bin/python claude/sync_mlflow.py --run rl2_v12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

HERE = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, required=True, help="имя каталога в claude/runs")
    p.add_argument("--mlflow-uri", type=str, default="http://swagstation.netcraze.pro:4249")
    p.add_argument("--experiment-id", type=str, default="18")
    p.add_argument("--run-name", type=str, default="")
    args = p.parse_args()

    run_dir = HERE / "runs" / args.run
    rows = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines() if line]
    ckpt_path = run_dir / "ckpt.pt"

    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(experiment_id=args.experiment_id, run_name=args.run_name or args.run):
        import torch

        ckpt = torch.load(ckpt_path, map_location="cpu")
        mlflow.log_params(ckpt.get("args", {}))
        import _mars_rover_cpp as native

        mlflow.set_tags(
            {
                "env_version": str(native.environment_version()),
                "algo": "PPO+GRU (RL^2)",
                "backfilled": "true",
            }
        )
        for row in rows:
            step = int(row.get("step", 0))
            prefix = "" if any(k.startswith("eval_") for k in row) else "train/"
            metrics = {
                (prefix + k): float(v)
                for k, v in row.items()
                if isinstance(v, (int, float)) and k not in ("update", "step")
            }
            for i, v in enumerate(row.get("prog_by_attempt", []) or []):
                if v is not None:
                    metrics[f"train/prog_attempt_{i}"] = float(v)
            if metrics:
                mlflow.log_metrics(metrics, step=step)
        mlflow.log_artifact(str(ckpt_path), artifact_path="weights")
        print("synced", len(rows), "rows ->", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
