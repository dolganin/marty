from __future__ import annotations

import json

from mars_rover_agents.survival_guard import verify_survival_separation


def _write_eval(path, *, adaptive: bool) -> None:
    split = path / "test"
    split.mkdir(parents=True)
    rows = []
    for seed in range(10):
        for episode in range(4):
            fatal = episode == 0 if adaptive else seed < 5
            rows.append(
                {
                    "biome_id": "ledge",
                    "seed": seed,
                    "episode_index": episode,
                    "fatal_error": fatal,
                    "distance": float(episode + 1),
                }
            )
    (split / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_survival_gate_requires_adaptive_rise_and_robust_flatness(tmp_path) -> None:
    adaptive = tmp_path / "adaptive"
    robust = tmp_path / "robust"
    _write_eval(adaptive, adaptive=True)
    _write_eval(robust, adaptive=False)
    result = verify_survival_separation(
        adaptive,
        robust,
        bootstrap_samples=500,
        seed=4,
    )
    assert result["adaptive_survival_curve"] == [0.0, 1.0, 1.0, 1.0]
    assert result["robust_survival_curve"] == [0.5, 0.5, 0.5, 0.5]
    assert result["overall_pass"]
