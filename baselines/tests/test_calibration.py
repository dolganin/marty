from __future__ import annotations

import json
from pathlib import Path

from mars_rover_env.bank import DEFAULT_MANIFEST

from mars_rover_agents.calibration import freeze_anchor_references, renormalize_evaluation


def _write_eval(path: Path, bank_version: str, mechanics: list[str], value: float) -> None:
    path.mkdir()
    summary = {
        "bank_version": bank_version,
        "split": "anchor",
        "seeds": [11],
        "episodes_per_trial": 2,
        "mechanics": mechanics,
        "normalized_return_by_episode": None,
        "trial_auc": None,
        "adaptation_delta": None,
    }
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rows = []
    for biome_id in mechanics:
        for episode_index in range(2):
            rows.append({
                "biome_id": biome_id,
                "episode_index": episode_index,
                "raw_return": value,
                "normalized_return": None,
            })
    (path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_freeze_and_renormalize_complete_anchor_set(tmp_path) -> None:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    manifest["reference_version"] = "sha256:test-reference"
    manifest_path = tmp_path / "bank.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    mechanics = list(manifest["anchors"])
    random_dir = tmp_path / "random"
    robust_dir = tmp_path / "robust"
    _write_eval(random_dir, manifest["bank_version"], mechanics, 1.0)
    _write_eval(robust_dir, manifest["bank_version"], mechanics, 3.0)

    frozen = freeze_anchor_references(
        manifest_path=manifest_path,
        random_evaluation=random_dir,
        robust_evaluation=robust_dir,
        reference_version="sha256:test-reference",
    )
    assert set(frozen["items"]) == set(mechanics)
    payload = renormalize_evaluation(robust_dir, manifest_path)
    assert payload["summary"]["normalized_return_by_episode"] == [1.0, 1.0]
    assert payload["summary"]["trial_auc"] == 1.0
    assert payload["summary"]["adaptation_delta"] == 0.0
