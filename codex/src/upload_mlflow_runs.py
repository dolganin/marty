from __future__ import annotations

import argparse

from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-uri", required=True)
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--target-uri", required=True)
    parser.add_argument("--target-experiment-id", required=True)
    parser.add_argument("--name-prefix", default="v12-")
    args = parser.parse_args()

    source = MlflowClient(args.source_uri)
    target = MlflowClient(args.target_uri)
    experiment = source.get_experiment_by_name(args.source_experiment)
    if experiment is None:
        raise SystemExit(f"source experiment not found: {args.source_experiment}")
    if target.get_experiment(args.target_experiment_id) is None:
        raise SystemExit(f"target experiment not found: {args.target_experiment_id}")

    existing = {}
    for run in target.search_runs([args.target_experiment_id], max_results=5000):
        source_id = run.data.tags.get("imported_from_local_run_id")
        if source_id:
            existing[source_id] = run.info.run_id

    runs = source.search_runs(
        [experiment.experiment_id],
        order_by=["attributes.start_time ASC"],
        max_results=5000,
    )
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        if not name.startswith(args.name_prefix):
            continue
        if run.info.run_id in existing:
            print(f"SKIP {name} -> {existing[run.info.run_id]}", flush=True)
            continue

        tags = {
            key: value
            for key, value in run.data.tags.items()
            if not key.startswith("mlflow.")
        }
        tags.update(
            {
                "imported_from_local_run_id": run.info.run_id,
                "imported_from_tracking_uri": args.source_uri,
            }
        )
        created = target.create_run(
            args.target_experiment_id,
            start_time=run.info.start_time,
            tags=tags,
            run_name=name,
        )
        target_id = created.info.run_id
        try:
            params = [Param(key, str(value)) for key, value in run.data.params.items()]
            metrics = []
            for key in run.data.metrics:
                for point in source.get_metric_history(run.info.run_id, key):
                    metrics.append(Metric(key, point.value, point.timestamp, point.step))
            user_tags = [RunTag(key, value) for key, value in tags.items()]
            for index, batch in enumerate(chunks(metrics, 1000)):
                target.log_batch(
                    target_id,
                    metrics=batch,
                    params=params if index == 0 else [],
                    tags=user_tags if index == 0 else [],
                )
            if not metrics:
                target.log_batch(target_id, params=params, tags=user_tags)

            status = run.info.status if run.info.status in {"FINISHED", "FAILED", "KILLED"} else "KILLED"
            target.set_terminated(target_id, status=status)
            print(f"COPIED {name} {run.info.run_id} -> {target_id}", flush=True)
        except BaseException:
            target.set_terminated(target_id, status="FAILED")
            raise


if __name__ == "__main__":
    main()
