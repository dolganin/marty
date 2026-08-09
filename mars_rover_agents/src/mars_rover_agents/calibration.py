from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank, write_json


def _load_evaluation(path: Path) -> dict[str, Any]:
    summary_path = path / "summary.json"
    episodes_path = path / "episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"evaluation directory is incomplete: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
    if summary.get("split") != "anchor" or not episodes:
        raise ValueError(f"not an anchor evaluation: {path}")
    return {"summary": summary, "episodes": episodes}


def freeze_anchor_references(
    *,
    manifest_path: Path,
    random_evaluation: Path,
    robust_evaluation: Path,
    reference_version: str,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    bank_version = require_compiled_bank(manifest)
    random_payload = _load_evaluation(random_evaluation)
    robust_payload = _load_evaluation(robust_evaluation)
    random_summary = random_payload["summary"]
    robust_summary = robust_payload["summary"]
    for summary in (random_summary, robust_summary):
        if summary.get("bank_version") != bank_version:
            raise ValueError("anchor evaluation belongs to a different bank")
    protocol_fields = ("seeds", "episodes_per_trial", "mechanics")
    if any(random_summary.get(key) != robust_summary.get(key) for key in protocol_fields):
        raise ValueError("random and robust anchor evaluations use different protocols")
    expected_anchors = set(manifest.get("anchors", []))
    actual_anchors = set(random_summary.get("mechanics", []))
    if actual_anchors != expected_anchors:
        raise ValueError(
            "anchor evaluation coverage differs from the manifest: "
            f"missing={sorted(expected_anchors - actual_anchors)} "
            f"extra={sorted(actual_anchors - expected_anchors)}"
        )
    if reference_version != manifest.get("reference_version"):
        raise ValueError("anchor calibration reference does not match the manifest")

    def means(payload: dict[str, Any]) -> dict[str, float]:
        result = {}
        for biome_id in payload["summary"]["mechanics"]:
            values = [
                float(row["raw_return"])
                for row in payload["episodes"]
                if row["biome_id"] == biome_id
            ]
            if not values:
                raise ValueError(f"missing anchor rows for {biome_id}")
            result[biome_id] = float(np.mean(values))
        return result

    lows = means(random_payload)
    highs = means(robust_payload)
    invalid = [biome_id for biome_id in lows if highs[biome_id] <= lows[biome_id] + 1.0e-8]
    if invalid:
        raise ValueError("invalid anchor normalization bands: " + ", ".join(invalid))
    manifest["anchor_references"] = {
        "schema_version": 1,
        "bank_version": bank_version,
        "reference_version": reference_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seeds": random_summary["seeds"],
        "episodes_per_trial": random_summary["episodes_per_trial"],
        "items": {
            biome_id: {"r_random": lows[biome_id], "r_robust": highs[biome_id]}
            for biome_id in random_summary["mechanics"]
        },
    }
    write_json(manifest_path, manifest)
    return manifest["anchor_references"]


def renormalize_evaluation(evaluation_dir: Path, manifest_path: Path) -> dict[str, Any]:
    payload = _load_evaluation(evaluation_dir)
    manifest = load_manifest(manifest_path)
    bounds = {
        biome_id: (float(item["r_random"]), float(item["r_robust"]))
        for biome_id, item in manifest.get("anchor_references", {}).get("items", {}).items()
    }
    mechanics = payload["summary"]["mechanics"]
    missing = [biome_id for biome_id in mechanics if biome_id not in bounds]
    if missing:
        raise ValueError("missing anchor normalization bands: " + ", ".join(missing))
    for row in payload["episodes"]:
        low, high = bounds[row["biome_id"]]
        row["normalized_return"] = (float(row["raw_return"]) - low) / (high - low)
    episode_count = int(payload["summary"]["episodes_per_trial"])
    curve = [
        float(np.mean([
            row["normalized_return"]
            for row in payload["episodes"]
            if int(row["episode_index"]) == index
        ]))
        for index in range(episode_count)
    ]
    payload["summary"]["normalized_return_by_episode"] = curve
    payload["summary"]["trial_auc"] = float(np.mean(curve))
    payload["summary"]["adaptation_delta"] = curve[-1] - curve[0]
    (evaluation_dir / "summary.json").write_text(
        json.dumps(payload["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episodes_path = evaluation_dir / "episodes.jsonl"
    temporary = episodes_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in payload["episodes"]:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(episodes_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze public-agent random/robust anchor bands")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--random-evaluation", type=Path, required=True)
    parser.add_argument("--robust-evaluation", type=Path, required=True)
    parser.add_argument("--reference-version", required=True)
    args = parser.parse_args()
    result = freeze_anchor_references(
        manifest_path=args.manifest,
        random_evaluation=args.random_evaluation,
        robust_evaluation=args.robust_evaluation,
        reference_version=args.reference_version,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
