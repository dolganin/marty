"""Deterministic regression coverage for the native v15 physics runtime.

The trace deliberately touches every action bit.  Its public observation,
reward/termination outputs and complete debug state are hashed after rounding
only enough to make the check portable across standard C++ math libraries.
Changing physics therefore requires an intentional update of the fingerprint.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

from mars_rover_env.envs.mars_rover_vec_env import MarsRoverVecEnv


CONTROL_JUMP = 1 << 13
CONTROL_JUMP_FRONT = 1 << 17
CONTROL_JUMP_REAR = 1 << 18
CONTROL_GAS = 1 << 0
CONTROL_SHIFT_UP = 1 << 6
CONTROL_REVERSE = 1 << 2
CONTROL_TOGGLE_PROPELLER = 1 << 15
CONTROL_LEGACY_BOTH_PISTONS = 1 << 16
CONTROL_PISTON_FRONT = 1 << 19
CONTROL_PISTON_REAR = 1 << 20


def _actions() -> np.ndarray:
    """A fixed drive sequence that activates each of the 23 action flags."""
    actions = np.zeros(240, dtype=np.int32)
    for step in range(actions.size):
        action = 1  # forward throttle keeps contacts/drivetrain active.
        if 10 <= step < 18:
            action |= 1 << 3  # clutch pedal
        if step in (12, 42):
            action |= 1 << 6  # upshift
        if step in (20, 50):
            action |= 1 << 7  # downshift
        if 26 <= step < 31:
            action |= 1 << 1  # brake
        if 35 <= step < 48:
            action |= CONTROL_JUMP_FRONT
        if 60 <= step < 96:
            action |= CONTROL_JUMP
        if 105 <= step < 124:
            action |= CONTROL_JUMP_REAR
        # Every remaining flag is pulsed twice where it has edge semantics.
        for bit, first in ((2, 2), (4, 4), (5, 6), (8, 8), (9, 52), (10, 54),
                           (11, 134), (12, 136), (14, 138),
                           (15, 144), (16, 148), (19, 152), (20, 156),
                           (21, 160), (22, 164)):
            if step in (first, first + 2):
                action |= 1 << bit
        actions[step] = action
    return actions


def _normalise(value):
    if isinstance(value, (float, np.floating)):
        return round(float(value), 4)
    if isinstance(value, (int, np.integer, bool, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    raise TypeError(f"unsupported debug value: {type(value)!r}")


def _trace(seed: int = 9173):
    env = MarsRoverVecEnv(1)
    initial = env.reset(seed).copy()
    rows = [(initial[0].round(4).tolist(), 0.0, 0, 0, _normalise(env.debug_info(0)))]
    for action in _actions():
        obs, reward, terminated, truncated, _ = env.step_uint8(np.array([action], dtype=np.int32))
        rows.append((obs[0].round(4).tolist(), round(float(reward[0]), 4),
                     int(terminated[0]), int(truncated[0]), _normalise(env.debug_info(0))))
    return rows


def _fingerprint(trace) -> str:
    payload = json.dumps(trace, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def test_fixed_seed_trace_is_repeatable_and_matches_physics_baseline():
    first = _trace()
    second = _trace()
    assert first == second
    # Rebaselined for kinematic engine coupling, the rev-limited gears and the
    # six-speed progressive ladder.
    assert _fingerprint(first) == "263a55410d72a09f1e7e7dbcbc7d0d06321fb873ba2f22dfcf180bd04743a5a4"


def test_preload_has_inertia_and_release_uses_the_stored_compression():
    env = MarsRoverVecEnv(1)
    env.reset(42)
    charges: list[float] = []
    velocities: list[float] = []
    for _ in range(24):
        env.step_uint8(np.array([CONTROL_JUMP], dtype=np.int32))
        debug = env.debug_info(0)
        charges.append(float(debug["suspension_jump_charge"]))
        velocities.append(float(debug["suspension_jump_preload_velocity"]))

    increments = np.diff(charges)
    assert charges[0] > 0.0
    assert increments[4] > increments[0]  # starts from rest, then accelerates.
    assert velocities[0] > 0.0
    assert velocities[-1] < max(velocities)  # damping approaches the travel stop.

    stored_charge = charges[-1]
    env.step_uint8(np.array([0], dtype=np.int32))
    released = env.debug_info(0)
    assert released["suspension_jump_phase"] == 2
    assert released["suspension_jump_charge"] == stored_charge
    assert released["suspension_jump_preload_velocity"] == 0.0


def test_shift_uses_a_loaded_rpm_window_and_consumes_energy():
    env = MarsRoverVecEnv(1)
    env.reset(2024)
    for _ in range(180):
        env.step_uint8(np.array([CONTROL_GAS], dtype=np.int32))
    before = env.debug_info(0)
    assert before["gear"] == 1
    # First gear is deliberately short now, so three seconds of launch sits
    # near the limiter; what must hold is that it never exceeds it.
    assert before["engine_rpm"] <= 9000.0

    env.step_uint8(np.array([CONTROL_GAS | CONTROL_SHIFT_UP], dtype=np.int32))
    shifted = env.debug_info(0)
    assert shifted["gear"] == 2
    assert shifted["shift_energy_cost"] >= 0.75
    assert before["energy"] - shifted["energy"] >= shifted["shift_energy_cost"]
    assert shifted["shift_clutch_cut"] > 0.0
    assert shifted["clutch_engagement"] == 0.0

    # Keeping X down cannot buy further automatic shifts after this one pulse.
    for _ in range(60):
        env.step_uint8(np.array([CONTROL_GAS | CONTROL_SHIFT_UP], dtype=np.int32))
    assert env.debug_info(0)["gear"] == 2


def test_batch_stepping_matches_independent_environments():
    seeds = (71, 10044)
    batch = MarsRoverVecEnv(2)
    batch.reset(seeds[0])  # BatchEnv assigns the documented +9973 seed stride.
    singles = [MarsRoverVecEnv(1), MarsRoverVecEnv(1)]
    for env, seed in zip(singles, seeds):
        env.reset(seed)

    for action in _actions()[:80]:
        actions = np.array([action, action], dtype=np.int32)
        batch_obs, batch_reward, batch_term, batch_trunc, _ = batch.step_uint8(actions)
        for index, env in enumerate(singles):
            obs, reward, term, trunc, _ = env.step_uint8(np.array([action], dtype=np.int32))
            np.testing.assert_array_equal(batch_obs[index], obs[0])
            assert batch_reward[index] == reward[0]
            assert batch_term[index] == term[0]
            assert batch_trunc[index] == trunc[0]


def test_parallel_batch_is_bitwise_repeatable():
    """The OpenMP path must not alter independent environment results."""
    first = MarsRoverVecEnv(64)
    second = MarsRoverVecEnv(64)
    np.testing.assert_array_equal(first.reset(8181), second.reset(8181))
    for action in _actions()[:48]:
        inputs = np.full(64, action, dtype=np.int32)
        first_result = first.step_uint8(inputs)
        second_result = second.step_uint8(inputs)
        for actual, expected in zip(first_result[:4], second_result[:4]):
            np.testing.assert_array_equal(actual, expected)


def test_held_out_split_is_a_full_endgame_world_while_train_stays_progressive():
    train = MarsRoverVecEnv(1, biome_split=1)
    test = MarsRoverVecEnv(1, biome_split=2)
    train.reset(2025)
    test.reset(2025)

    train_debug = train.debug_info(0)
    test_debug = test.debug_info(0)
    assert train_debug["endgame_test_world"] is False
    assert train_debug["course_difficulty"] < 0.01
    assert test_debug["endgame_test_world"] is True
    assert test_debug["course_difficulty"] == 1.0

    # Endgame terrain has no five-metre flattened spawn strip: unlike train,
    # it is generated entirely from the maximum-distance distribution.
    assert test_debug["safe_start_m"] == 0.0
    assert test_debug["generated_pit_count"] >= 240
    assert test_debug["generated_step_count"] >= 240
    assert test_debug["terrain_surprise_mode"] > 0


def test_wheels_stop_rotating_without_ground_contact():
    env = MarsRoverVecEnv(1)
    env.reset(42)
    for _ in range(48):
        env.step_uint8(np.array([CONTROL_JUMP], dtype=np.int32))
    for _ in range(30):
        env.step_uint8(np.array([0], dtype=np.int32))
        debug = env.debug_info(0)
        if debug["airborne"]:
            assert debug["wheel_angular_velocities"] == [0.0, 0.0]
            break
    else:
        raise AssertionError("charged suspension launch never became airborne")


def test_pistons_are_strictly_individual_and_legacy_both_bit_is_inert():
    def piston_state(action: int):
        env = MarsRoverVecEnv(1)
        env.reset(42)
        for _ in range(5):
            env.step_uint8(np.array([action], dtype=np.int32))
        return env.debug_info(0)

    legacy = piston_state(CONTROL_LEGACY_BOTH_PISTONS)
    front = piston_state(CONTROL_PISTON_FRONT)
    rear = piston_state(CONTROL_PISTON_REAR)
    assert legacy["roof_piston_extension"] == 0.0
    assert legacy["roof_piston_mask"] == 0
    assert front["roof_piston_mask"] == 1 and front["roof_piston_extension"] > 0.0
    assert rear["roof_piston_mask"] == 2 and rear["roof_piston_extension"] > 0.0


def _submerged_env(tmp_path, seeds=range(60)):
    """A single-biome test world that is actually a body of water.

    Chaining is switched off so the whole course is the drawn biome: which
    world a seed yields shifts whenever the bank gains a biome.
    """
    config = tmp_path / "single_zone.yaml"
    config.write_text("env:\n  chain_biomes: false\n", encoding="utf-8")
    for seed in seeds:
        env = MarsRoverVecEnv(1, config_path=str(config), biome_split=2)
        env.reset(seed)
        for _ in range(600):
            env.step_uint8(np.array([CONTROL_GAS], dtype=np.int32))
            if env.debug_info(0)["water_depth"] > 0.5:
                return env
    raise AssertionError("no liquid test world within the scanned seeds")


def test_propeller_cannot_reverse_thrust(tmp_path):
    env = _submerged_env(tmp_path)
    env.step_uint8(np.array([CONTROL_GAS | CONTROL_TOGGLE_PROPELLER], dtype=np.int32))
    thrust = 0.0
    for _ in range(300):
        env.step_uint8(np.array([CONTROL_GAS], dtype=np.int32))
        thrust = max(thrust, env.debug_info(0)["propeller_thrust"])
    assert thrust > 0.0

    env.step_uint8(np.array([CONTROL_GAS | CONTROL_REVERSE], dtype=np.int32))
    assert env.debug_info(0)["propeller_thrust"] == 0.0


def test_physics_stress_trace_stays_finite_in_train_and_endgame():
    """Catches NaN/Inf leaks through forces, contacts and all action bits."""
    rng = np.random.default_rng(8371)
    for split in (1, 2):
        env = MarsRoverVecEnv(4, biome_split=split)
        obs = env.reset(500 + split)
        assert np.isfinite(obs).all()
        for _ in range(180):
            actions = rng.integers(0, 1 << 23, size=4, dtype=np.int32)
            obs, rewards, _, _, _ = env.step_uint8(actions)
            assert np.isfinite(obs).all()
            assert np.isfinite(rewards).all()
            for env_id in range(4):
                debug = env.debug_info(env_id)
                for field in ("x", "y", "vx", "vy", "angle", "energy", "damage",
                              "engine_rpm", "engine_temperature", "energy_cost_rate"):
                    assert np.isfinite(debug[field]), (split, env_id, field, debug[field])
