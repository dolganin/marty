from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mars_rover_env.candidates import DEFAULT_CATALOG, candidates, load_catalog


def _action_tv(left: dict, right: dict) -> float:
    a = np.asarray(left["action_index_counts"], dtype=np.float64)
    b = np.asarray(right["action_index_counts"], dtype=np.float64)
    a /= max(float(a.sum()), 1.0)
    b /= max(float(b.sum()), 1.0)
    return float(0.5 * np.abs(a - b).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Conservative agent-assisted candidate deduplication")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--transfer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance-gap", type=float, default=0.35)
    parser.add_argument("--action-tv", type=float, default=0.10)
    parser.add_argument("--mechanic-overlap", type=float, default=0.67)
    args = parser.parse_args()

    catalogue = load_catalog(args.catalog)
    items = candidates(args.catalog)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    rows = {row["candidate"]: row for row in transfer["candidates"]}
    common = transfer.get("common_evaluation_seeds", [])
    if len(common) < 4:
        raise SystemExit("transfer result must use at least four common evaluation seeds")

    comparisons = []
    confirmed_edges: list[tuple[str, str]] = []
    response_similar_count = 0
    for pair in audit["near_duplicates"]:
        left, right = pair["left"], pair["right"]
        a, b = rows[left], rows[right]
        da = np.asarray(a["distances_m"], dtype=np.float64)
        db = np.asarray(b["distances_m"], dtype=np.float64)
        scale = max(5.0, float(0.5 * (da.mean() + db.mean())))
        distance_gap = float(np.mean(np.abs(da - db)) / scale)
        action_tv = _action_tv(a, b)
        response_similar = distance_gap <= args.distance_gap and action_tv <= args.action_tv
        response_similar_count += int(response_similar)
        left_mechanics = set(items[left].mechanics)
        right_mechanics = set(items[right].mechanics)
        mechanic_overlap = len(left_mechanics & right_mechanics) / len(left_mechanics | right_mechanics)
        left_generation = items[left].generation
        right_generation = items[right].generation
        non_lidar_left = {k: v for k, v in left_generation.items() if "lidar" not in k}
        non_lidar_right = {k: v for k, v in right_generation.items() if "lidar" not in k}
        controlled_lidar_contrast = (
            items[left].mechanics == items[right].mechanics
            and items[left].seeds == items[right].seeds
            and non_lidar_left == non_lidar_right
            and left_generation != right_generation
        )
        confirmed = (
            response_similar
            and mechanic_overlap >= args.mechanic_overlap
            and not controlled_lidar_contrast
        )
        if confirmed:
            classification = "confirmed_redundant"
        elif controlled_lidar_contrast:
            classification = "protected_controlled_contrast"
        elif response_similar:
            classification = "different_mechanics_same_agent_failure_mode"
        else:
            classification = "distinct_agent_response"
        comparisons.append({
            "left": left, "right": right,
            "physics_probe_distance": pair["distance"],
            "agent_distance_gap": distance_gap,
            "agent_action_tv": action_tv,
            "mechanic_overlap": mechanic_overlap,
            "response_similar": response_similar,
            "controlled_lidar_contrast": controlled_lidar_contrast,
            "classification": classification,
            "confirmed_redundant": confirmed,
        })
        if confirmed:
            confirmed_edges.append((left, right))

    def rank(name: str) -> tuple[float, float, int, str]:
        row = rows[name]
        support = 1 if items[name].support == "implemented" else 0
        return (float(row["positive_seed_fraction"]), float(row["mean_distance_m"]), support, name)

    adjacency = {name: set() for name in items}
    for left, right in confirmed_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    kept: set[str] = set()
    deprioritized: dict[str, str] = {}
    for name in sorted(items, key=rank, reverse=True):
        blockers = sorted(adjacency[name] & kept)
        if blockers:
            deprioritized[name] = max(blockers, key=rank)
        else:
            kept.add(name)

    result = {
        "selection_version": "candidate-agent-dedup-v1",
        "catalog_version": catalogue["catalog_version"],
        "method": {
            "status": "provisional_agent_assisted",
            "rule": "physics probe < 0.12, paired distance gap <= threshold, action TV <= threshold",
            "distance_gap_threshold": args.distance_gap,
            "action_tv_threshold": args.action_tv,
            "mechanic_overlap_threshold": args.mechanic_overlap,
            "common_seeds": common,
            "note_ru": "Пометка не удаляет кандидата. При равенстве сохраняется вариант с большей долей успешных seed и средней дистанцией.",
        },
        "candidate_count": len(items),
        "input_similar_pairs": len(comparisons),
        "agent_response_similar_pairs": response_similar_count,
        "confirmed_redundant_pairs": len(confirmed_edges),
        "kept_count": len(kept),
        "deprioritized_count": len(deprioritized),
        "kept": sorted(kept),
        "deprioritized": [
            {"name": name, "representative": representative,
             "reason": "agent_and_physics_response_redundant"}
            for name, representative in sorted(deprioritized.items())
        ],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "candidate_count", "input_similar_pairs", "confirmed_redundant_pairs",
        "agent_response_similar_pairs", "kept_count", "deprioritized_count")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
