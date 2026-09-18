from __future__ import annotations

import json

from mars_rover_agents.depth_gate import verify_depth_separation


def _write_eval(path, *, distances_by_episode: list[float]) -> None:
    split = path / "test"
    split.mkdir(parents=True)
    rows = []
    for seed in range(10):
        for episode, distance in enumerate(distances_by_episode):
            rows.append(
                {
                    "biome_id": "chained_course",
                    "seed": seed,
                    "episode_index": episode,
                    "distance": distance,
                }
            )
    (split / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_depth_gate_passes_when_adaptive_reaches_farther(tmp_path) -> None:
    adaptive = tmp_path / "adaptive"
    robust = tmp_path / "robust"
                                                                              
                                                                               
    _write_eval(robust, distances_by_episode=[20.0, 19.0, 21.0, 20.0])
    _write_eval(adaptive, distances_by_episode=[18.0, 30.0, 45.0, 50.0])
    result = verify_depth_separation(adaptive, robust, bootstrap_samples=500, seed=1)
    assert result["overall_pass"] is True
    assert result["checks"]["adaptive_deeper_on_average"] is True


def test_depth_gate_fails_when_no_agent_gets_farther(tmp_path) -> None:
    adaptive = tmp_path / "adaptive"
    robust = tmp_path / "robust"
    _write_eval(robust, distances_by_episode=[20.0, 19.0, 21.0, 20.0])
    _write_eval(adaptive, distances_by_episode=[20.0, 21.0, 19.0, 20.0])
    result = verify_depth_separation(adaptive, robust, bootstrap_samples=500, seed=1)
    assert result["overall_pass"] is False
