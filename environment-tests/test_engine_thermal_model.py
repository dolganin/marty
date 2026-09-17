from __future__ import annotations

import pytest
import yaml

from mars_rover_env import MarsRoverEnv
from mars_rover_env.config import load_env_config

GAS, UP, IGN = 1, 1 << 6, 1 << 9
DT = 1.0 / 60.0


def _config(tmp_path, name="thermal.yaml", **physics):
    base = {
        "env": {"episodes_per_trial": 0, "biome_split": 3, "chain_biomes": False,
                "fixed_biome_id": 0},
        "physics": {
            "dt": DT,
            "initial_engine_temperature": 20.0,
            "engine_thermal_mass": 1.0,
            "engine_idle_heat": 2.2,
            "engine_heat_per_fuel": 730.0,
            "engine_cooling_conductance": 0.08,
            "engine_cooling_airflow": 0.012,
            "initial_energy": 25.0,
            **physics,
        },
        "termination": {"max_steps": 0, "trial_time_limit": 600.0, "stuck_steps": 1000000},
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def _drive(env, seconds, action_source):
    prev_up = False
    for _ in range(int(seconds / DT)):
        debug = env.debug_info()
        action, prev_up = action_source(debug, prev_up)
        env.step(action)
    return env.debug_info()


def _drive_recording(env, seconds, action_source):
    """Drive and report the peak of each reading.

    The engine can overheat and cut out part-way through a run, so the final
    frame alone does not say what the thermal model did while it was running.
    """
    prev_up = False
    peak_fuel = 0.0
    peak_temperature = -1.0e9
    for _ in range(int(seconds / DT)):
        debug = env.debug_info()
        action, prev_up = action_source(debug, prev_up)
        env.step(action)
        debug = env.debug_info()
        peak_fuel = max(peak_fuel, float(debug["drive_fuel_rate"]))
        peak_temperature = max(peak_temperature, float(debug["engine_temperature"]))
    return peak_fuel, peak_temperature


def _throttle(debug, prev_up):
    if not debug["engine_running"] and not debug["engine_overheated"]:
        return IGN, False
    if str(debug["gear"]) == "N" or (debug["upshift_recommended"] and debug["can_shift_up"]):
        return (GAS, False) if prev_up else (GAS | UP, True)
    return GAS, False


def _idle(debug, prev_up):
    if not debug["engine_running"] and not debug["engine_overheated"]:
        return IGN, False
    return 0, False


def test_config_exposes_the_thermal_parameters(tmp_path):
    config = load_env_config(_config(tmp_path))
    assert config.physics.engine_heat_per_fuel == pytest.approx(730.0)
    assert config.physics.engine_cooling_conductance == pytest.approx(0.08)
    assert config.physics.engine_cooling_airflow == pytest.approx(0.012)
    assert config.physics.engine_thermal_mass == pytest.approx(1.0)


def test_heat_follows_the_fuel_actually_burned(tmp_path):
    env = MarsRoverEnv(config_path=str(_config(tmp_path)))
    env.reset(seed=5, options={"trial_start": True})
    burned, warm = _drive_recording(env, 12.0, _throttle)
    assert burned > 0.0
    env.close()

    env = MarsRoverEnv(config_path=str(_config(tmp_path, engine_heat_per_fuel=0.0)))
    env.reset(seed=5, options={"trial_start": True})
    _, cold = _drive_recording(env, 12.0, _throttle)
    env.close()
    assert warm > cold + 20.0


def test_idle_settles_near_the_predicted_offset(tmp_path):
    env = MarsRoverEnv(config_path=str(_config(tmp_path)))
    env.reset(seed=6, options={"trial_start": True})
    debug = _drive(env, 120.0, _idle)
    if debug["engine_running"]:
        offset = debug["engine_temperature"] - debug["ambient_temperature"]
        assert 15.0 < offset < 40.0
    env.close()


def test_ambient_decides_the_steady_state(tmp_path):
    результаты = {}
    for biome_id in range(8):
        path = _config(tmp_path, name=f"biome_{biome_id}.yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        payload["env"]["fixed_biome_id"] = biome_id
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")
        env = MarsRoverEnv(config_path=str(path))
        env.reset(seed=8, options={"trial_start": True})
        начало = env.debug_info()["ambient_temperature"]
        # Idle, not full throttle: a throttled engine saturates at the overheat
        # cutoff in every biome, which hides the ambient term entirely.
        конец = _drive(env, 60.0, _idle)["engine_temperature"]
        env.close()
        результаты[biome_id] = (начало, конец)

    окружение = [значения[0] for значения in результаты.values()]
    # The builtin biomes span a few degrees of ambient temperature, not the
    # wide band the pre-v13 bank used.
    assert max(окружение) - min(окружение) > 3.0
    самый_тёплый = max(результаты.items(), key=lambda item: item[1][0])[1]
    самый_холодный = min(результаты.items(), key=lambda item: item[1][0])[1]
    assert самый_тёплый[1] > самый_холодный[1]


def test_stopped_engine_relaxes_towards_ambient_within_seconds(tmp_path):
    env = MarsRoverEnv(config_path=str(_config(tmp_path)))
    env.reset(seed=9, options={"trial_start": True})
    горячо = _drive(env, 40.0, _throttle)
    старт = горячо["engine_temperature"]
    окружение = горячо["ambient_temperature"]
    for _ in range(int(12.0 / DT)):
        env.step(0)
    через_12 = env.debug_info()["engine_temperature"]
    env.close()
    остаток = (через_12 - окружение) / max(1e-6, старт - окружение)
    assert 0.2 < остаток < 0.6


def test_faster_travel_cools_the_engine(tmp_path):
    обычный = _config(tmp_path, name="flow.yaml", engine_cooling_airflow=0.012)
    без_обдува = _config(tmp_path, name="noflow.yaml", engine_cooling_airflow=0.0)

    температуры = {}
    for метка, path in (("обдув", обычный), ("без обдува", без_обдува)):
        env = MarsRoverEnv(config_path=str(path))
        env.reset(seed=10, options={"trial_start": True})
        температуры[метка] = _drive(env, 45.0, _throttle)["engine_temperature"]
        env.close()
    assert температуры["обдув"] <= температуры["без обдува"]
