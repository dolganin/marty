from pathlib import Path

import mars_rover_env  # noqa: F401 - registers the MinGW runtime DLL directory on Windows
import _mars_rover_cpp as native

from mars_rover_env import MarsRoverEnv


ROOT = Path(__file__).resolve().parents[1]


def test_builtin_biomes_are_exposed_as_a_catalog() -> None:
    ids = [item["id"] for item in native.biome_catalog()]
    assert ids[:8] == ["normal", "sand", "ice", "mud", "wind", "low_gravity", "crust", "liquid"]


def test_hidden_mechanics_seed_is_frozen_within_trial() -> None:
    env = MarsRoverEnv(config_path=ROOT / "python/mars_rover_env/configs/play.yaml")
    env.reset(seed=0, options={"trial_start": True})
    first = env.debug_info()["mechanic"]
    env.reset(seed=5, options={"trial_start": False})
    assert env.debug_info()["mechanic"] == first

    env.reset(seed=5, options={"trial_start": True})
    assert env.debug_info()["mechanic"] != first
    env.close()
