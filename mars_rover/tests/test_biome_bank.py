from pathlib import Path

import mars_rover_env  # noqa: F401 - registers the MinGW runtime DLL directory on Windows
import _mars_rover_cpp as native

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import load_manifest, require_compiled_bank


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_biomes_are_exposed_as_a_catalog() -> None:
    ids = [item["id"] for item in native.biome_catalog()]
    assert ids[:8] == ["normal", "sand", "ice", "mud", "wind", "low_gravity", "crust", "liquid"]


def test_compiled_bank_matches_versioned_manifest() -> None:
    assert require_compiled_bank(load_manifest()).startswith("sha256:")


def test_hidden_mechanics_seed_is_frozen_within_trial() -> None:
    env = MarsRoverEnv(config_path=ROOT / "python/mars_rover_env/configs/play.yaml")
    env.reset(seed=0, options={"trial_start": True})
    first = env.debug_info()["mechanic"]
    env.reset(seed=5, options={"trial_start": False})
    assert env.debug_info()["mechanic"] == first

    env.reset(seed=5, options={"trial_start": True})
    assert env.debug_info()["mechanic"] != first
    env.close()


def test_bank_contract_has_disjoint_modes_and_skill_floors() -> None:
    catalog = [dict(item) for item in native.biome_catalog()]
    anchors = [item for item in catalog if item["is_anchor"]]
    train = [item for item in catalog if item["split"] == 1]
    test = [item for item in catalog if item["split"] == 2]

    assert len(anchors) >= 5
    assert len(test) >= 10
    assert not {item["id"] for item in anchors} & {item["id"] for item in train}
    expected = {
        "traction_loss",
        "lateral_force",
        "inertia_hysteresis",
        "gravity_change",
        "energy_mode",
        "dynamic_obstacle",
    }
    assert {item["skill_stratum"] for item in train} == expected
    assert {item["skill_stratum"] for item in test} == expected


def test_sampling_modes_select_exact_expected_biome_sets() -> None:
    catalog = [dict(item) for item in native.biome_catalog()]
    expected_by_mode = {
        MarsRoverEnv.BIOME_MODE_TRAIN: {
            item["name"]
            for item in catalog
            if item["split"] == 1 and not item["is_anchor"]
        },
        MarsRoverEnv.BIOME_MODE_TEST: {
            item["name"] for item in catalog if item["split"] == 2
        },
        MarsRoverEnv.BIOME_MODE_ANCHOR: {
            item["name"] for item in catalog if item["is_anchor"]
        },
    }
    for mode, expected in expected_by_mode.items():
        env = MarsRoverEnv(biome_split=mode)
        seen = set()
        for seed in range(400):
            env.reset(seed=seed, options={"trial_start": True})
            debug = env.debug_info()
            seen.add(debug["mechanic"])
            assert debug["zone_begin_x"] <= -999_999.0
            assert debug["zone_end_x"] >= 999_999.0
        env.close()
        assert seen == expected
