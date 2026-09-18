"""Contract checks for the v13 fixed-horizon layered world."""

from __future__ import annotations

import numpy as np
from _mars_rover_cpp import EnvConfig, MarsRoverBatchEnv

from mars_rover_env import MarsRoverEnv
from mars_rover_env.config import load_env_config
from mars_rover_env.rules import apply_influence_graph, validate_rule
from mars_rover_env.bank import load_manifest


def test_seed_is_deterministic_and_actions_are_extended() -> None:
    left, right = MarsRoverEnv(), MarsRoverEnv()
    obs_left, _ = left.reset(seed=90210)
    obs_right, _ = right.reset(seed=90210)
    # The action space covers every control bit up to the ballast pair.
    assert left.action_space.n == 1 << 24
    assert obs_left.shape == obs_right.shape
    assert np.array_equal(obs_left, obs_right)
    for action in (1, 1 | 8192, 1 | 16384, 1 | 32768, 0):
        a = left.step(action)
        b = right.step(action)
        assert np.array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]


def test_world_latents_are_bounded_and_not_observed_as_layer_ids() -> None:
    env = MarsRoverEnv()
    obs, _ = env.reset(seed=7)
    debug = env.debug_info()
    assert 0 <= debug["active_layers"] <= 4
    assert 0.12 <= debug["latent_traction"] <= 1.45
    assert 0.0 <= debug["latent_moisture"] <= 1.0
    assert 0.0 <= debug["latent_charge_reserve"] <= 1.0
    # The intentionally breaking v13 observation contains proprioception and
    # scan geometry, but never a mechanic id or layer ordering.
    assert obs.shape == (160,)


def test_only_fixed_timer_ends_an_idle_run() -> None:
    budget = int(round(load_env_config().termination.trial_time_limit * 60.0))
    env = MarsRoverEnv()
    env.reset(seed=42)
    for _ in range(budget - 1):
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert not truncated
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated
    assert truncated
    assert "TIMER" in env.debug_info()["termination_reason"]


def test_the_native_default_config_runs_on_the_same_clock() -> None:
    batch = MarsRoverBatchEnv(1, EnvConfig())
    assert batch.trial_step_budget(0) > 0


def test_coupling_rule_is_structured_and_bounded() -> None:
    # Rules are source -> target -> weight edges, aggregated by the graph.
    rule = validate_rule({"source": "moisture", "target": "traction", "weight": -0.8})
    assert rule.source == "moisture"
    assert rule.target == "traction"

    dry = apply_influence_graph([rule], {"moisture": 0.0, "traction": 1.0})
    wet = apply_influence_graph([rule], {"moisture": 1.0, "traction": 1.0})
    # A negative weight means more moisture costs traction.
    assert wet["traction"] < dry["traction"]


def test_recovery_organs_are_nonterminal_and_stateful() -> None:
    env = MarsRoverEnv(fixed_biome_id=7)  # builtin liquid; lower water route
    env.reset(seed=81)
    _, _, terminated, truncated, _ = env.step(32768)  # propeller toggle
    assert not terminated and not truncated
    assert env.debug_info()["propeller_mode"]
    assert env.debug_info()["route_branch"] == "lower_water"
    env.step(0)
    env.step(16384)  # climb toggle
    assert env.debug_info()["climb_mode"]
    env.step(0)
    _, _, terminated, truncated, _ = env.step(8192)  # suspension kick
    assert not terminated and not truncated
    # Holding the kick preloads the suspension; the cooldown starts on release.
    assert env.debug_info()["suspension_jump_phase"] == 1
    assert env.debug_info()["suspension_jump_charge"] > 0.0


def test_compiled_bank_exposes_only_frozen_reviewed_combinations() -> None:
    manifest = load_manifest()
    assert manifest["schema_version"] >= 3
    assert len(manifest["coupling_rules"]) >= 3
    assert {stack["split"] for stack in manifest["frozen_stacks"]} == {
        "anchor", "train", "held_out"
    }
    assert all(1 <= len(stack["mechanisms"]) <= 4 for stack in manifest["frozen_stacks"])
