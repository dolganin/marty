from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _mars_rover_cpp import (
    BodyRigConfig,
    CollisionShapeConfig,
    CollisionType,
    EnvConfig,
    RoverRig,
    Vec2,
    WheelRigConfig,
)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install PyYAML to load YAML config files") from exc
    data = yaml.safe_load(text)
    return {} if data is None else data


def _vec2(value: Any, default: Vec2 | None = None) -> Vec2:
    if value is None:
        return Vec2() if default is None else default
    return Vec2(float(value[0]), float(value[1]))


def _collision_type(value: str | None) -> CollisionType:
    if value == "box":
        return CollisionType.Box
    if value == "circle":
        return CollisionType.Circle
    return CollisionType.None_


def _collision(data: dict[str, Any] | None) -> CollisionShapeConfig:
    out = CollisionShapeConfig()
    if not data:
        return out
    out.type = _collision_type(data.get("type"))
    out.size = _vec2(data.get("size"), out.size)
    out.radius = float(data.get("radius", out.radius))
    return out


def _apply_section(obj: Any, data: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in data:
            setattr(obj, field, type(getattr(obj, field))(data[field]))


def load_env_config(config_path: str | Path | None = None, rig_path: str | Path | None = None) -> EnvConfig:
    cfg = EnvConfig()

    if config_path is not None:
        data = _load_mapping(config_path)
        if "env" in data:
            env = data["env"]
            cfg.episodes_per_trial = int(env.get("episodes_per_trial", cfg.episodes_per_trial))
            cfg.biome_split = int(env.get("biome_split", cfg.biome_split))
            cfg.fixed_biome_id = int(env.get("fixed_biome_id", cfg.fixed_biome_id))
            cfg.debug = bool(env.get("debug", cfg.debug))
        if "terrain" in data:
            _apply_section(
                cfg.terrain,
                data["terrain"],
                ("sample_count", "dx", "base_height", "amplitude", "roughness", "crater_count", "step_count", "length"),
            )
        if "physics" in data:
            _apply_section(
                cfg.physics,
                data["physics"],
                (
                    "dt",
                    "gravity",
                    "wheel_friction",
                    "motor_torque",
                    "final_drive_ratio",
                    "engine_inertia",
                    "engine_drag_torque",
                    "clutch_sync_rate",
                    "initial_engine_temperature",
                    "cold_start_temperature",
                    "minimum_operating_temperature",
                    "full_power_temperature",
                    "overheat_temperature",
                    "overheat_restart_temperature",
                    "engine_heat_rate",
                    "engine_cooling_rate",
                    "initial_energy",
                    "energy_capacity",
                    "panel_deploy_time",
                    "panel_retract_time",
                    "lidar_energy_cost",
                    "lidar_scan_duration",
                    "lidar_cooldown",
                    "lidar_base_range",
                    "brake_strength",
                    "body_tilt_torque",
                    "linear_damping",
                    "angular_damping",
                ),
            )
        if "termination" in data:
            _apply_section(
                cfg.termination,
                data["termination"],
                ("finish_x", "min_energy", "flip_angle", "stuck_steps", "max_steps"),
            )
        if "reward" in data:
            _apply_section(
                cfg.reward,
                data["reward"],
                (
                    "progress_scale",
                    "energy_cost_scale",
                    "flip_penalty",
                    "stuck_penalty",
                    "hard_contact_penalty",
                    "finish_bonus",
                ),
            )

    if rig_path is not None:
        cfg.rig = load_rover_rig(rig_path)

    return cfg


def load_rover_rig(path: str | Path) -> RoverRig:
    data = _load_mapping(path).get("rover", {})
    rig = RoverRig.default_two_wheel()

    body_data = data.get("body")
    if body_data:
        body = BodyRigConfig()
        body.mass = float(body_data.get("mass", body.mass))
        body.inertia = float(body_data.get("inertia", body.inertia))
        body.size = _vec2(body_data.get("size"), body.size)
        body.local_position = _vec2(body_data.get("local_position"), body.local_position)
        body.collision = _collision(body_data.get("collision"))
        rig.body = body

    wheels = []
    for wheel_data in data.get("wheels", []):
        wheel = WheelRigConfig()
        wheel.radius = float(wheel_data.get("radius", wheel.radius))
        wheel.mass = float(wheel_data.get("mass", wheel.mass))
        wheel.local_anchor = _vec2(wheel_data.get("local_anchor"), wheel.local_anchor)
        suspension = wheel_data.get("suspension", {})
        wheel.suspension.rest_length = float(
            suspension.get("rest_length", wheel.suspension.rest_length)
        )
        wheel.suspension.min_length = float(suspension.get("min_length", wheel.suspension.min_length))
        wheel.suspension.max_length = float(suspension.get("max_length", wheel.suspension.max_length))
        wheel.suspension.stiffness = float(suspension.get("stiffness", wheel.suspension.stiffness))
        wheel.suspension.damping = float(suspension.get("damping", wheel.suspension.damping))
        wheels.append(wheel)
    if wheels:
        rig.wheels = wheels

    return rig
