from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from mlflow.tracking import MlflowClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    args = parser.parse_args()

    client = MlflowClient(args.tracking_uri)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    aliases = {
        "best_distance_m_mean": "inference.distance_mean_m",
        "best_distance_m_median": "inference.distance_median_m",
        "best_distance_m_p90": "inference.distance_p90_m",
        "best_distance_m_max": "inference.distance_max_m",
        "attempts_mean": "inference.attempts_mean",
        "return_mean": "inference.return_mean",
    }
    timestamp = int(time.time() * 1000)
    for repeat in report["runs"]:
        step = int(repeat["repeat"])
        for source_key, target_key in aliases.items():
            client.log_metric(
                args.run_id,
                target_key,
                float(repeat[source_key]),
                timestamp=timestamp + step,
                step=step,
            )
    client.log_artifacts(args.run_id, str(args.videos), artifact_path="videos")
    client.set_tag(args.run_id, "video_trials", "50000,52000,54000")
    client.set_tag(args.run_id, "video_format", "MP4 H.264, 640x360, 30fps, full 120s")
    print(f"published curves and videos to {args.run_id}")


if __name__ == "__main__":
    main()
