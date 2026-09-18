"""Regression checks for the offline frozen-stack growth evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mars_rover_env.tools.grow_stack_bank import (
    Candidate,
    _apply_accepted_stacks,
    _make_native_env,
    _stop_reason,
)


def test_candidate_contract_renders_engine_compilable_stack() -> None:
    candidate = Candidate.from_payload(
        {"name": "ice_low_gravity_wind", "components": ["ice", "low_gravity", "wind"]}
    )
    code = candidate.cpp()
    assert "MechanicType::Ice" in code
    assert "MechanicType::LowGravity" in code
    assert "MechanicType::Wind" in code
    assert "3, BiomeSplit::Train" in code


def test_evaluation_override_exercises_candidate_stack_at_spawn() -> None:
    batch = _make_native_env(("ice", "low_gravity", "wind"), max_steps=4)
    observation = np.zeros((1, batch.obs_dim), dtype=np.float32)
    batch.reset_at(0, 17, True, observation[0])
    assert batch.debug_info(0)["active_layer_names"] == ["Ice", "LowGravity", "Wind"]


def test_apply_updates_only_a_copied_bank_and_manifest(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    header = tmp_path / "biome_bank.hpp"
    manifest = tmp_path / "biome_bank.json"
    header.write_text(
        (repository / "environment/cpp/include/mars/biome_bank.hpp").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    manifest.write_text(
        (repository / "environment/python/mars_rover_env/configs/biome_bank.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    version = _apply_accepted_stacks(
        manifest,
        [{"components": ["ice", "wind", "crust"], "split": "train"}],
        header_path=header,
    )
    assert version.startswith("sha256:")
    assert "std::array<FrozenMechanismStack, 15>" in header.read_text(encoding="utf-8")
    updated = json.loads(manifest.read_text(encoding="utf-8"))
    assert updated["bank_version"] == version
    assert updated["frozen_stacks"][-1] == {"mechanisms": ["ice", "wind", "crust"], "split": "train"}


def test_stop_condition_requires_train_and_test_quotas(tmp_path: Path) -> None:
    manifest = {
        "frozen_stacks": [
            *({"mechanisms": ["sand"], "split": "train"} for _ in range(20)),
            *({"mechanisms": ["ice"], "split": "held_out"} for _ in range(10)),
        ]
    }
    assert _stop_reason(tmp_path, manifest) is not None
