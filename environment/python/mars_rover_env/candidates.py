from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from _mars_rover_cpp import MechanicType

from mars_rover_env.config import load_env_config


DEFAULT_CATALOG = Path(__file__).resolve().parent / "configs" / "candidate_worlds.json"
DEFAULT_CANDIDATE_CONFIG = Path(__file__).resolve().parent / "configs" / "play.yaml"
_MECHANICS = {
    "normal": MechanicType.Normal,
    "sand": MechanicType.Sand,
    "ice": MechanicType.Ice,
    "mud": MechanicType.Mud,
    "wind": MechanicType.Wind,
    "low_gravity": MechanicType.LowGravity,
    "crust": MechanicType.Crust,
    "liquid": MechanicType.Liquid,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    revision: int
    description: str
    mechanics: tuple[str, ...]
    helpful_actions: tuple[str, ...]
    support: str
    limitation: str
    parameter_ranges: dict[str, Any]
    generation: dict[str, Any]
    seeds: tuple[int, ...]


def load_catalog(path: str | Path = DEFAULT_CATALOG) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"unsupported candidate catalog schema: {data.get('schema_version')!r}")
    return data


def candidates(path: str | Path = DEFAULT_CATALOG) -> dict[str, Candidate]:
    data = load_catalog(path)
    common_ranges = data["mechanic_parameter_ranges"]
    result: dict[str, Candidate] = {}
    for raw in data["candidates"]:
        item = Candidate(
            name=raw["name"], revision=int(raw["revision"]), description=raw["description_ru"],
            mechanics=tuple(raw["mechanics"]), helpful_actions=tuple(raw["helpful_actions"]),
            support=raw["physics_support"], limitation=raw.get("limitation_ru", ""),
            parameter_ranges={name: common_ranges[name] for name in raw["mechanics"]},
            generation=dict(raw["generation"]), seeds=tuple(int(seed) for seed in raw["seeds"]),
        )
        if item.name in result:
            raise ValueError(f"duplicate candidate name: {item.name}")
        result[item.name] = item
    return result


def candidate_config(candidate: Candidate, config_path=None, rig_path=None):
    if config_path is None:
        config_path = DEFAULT_CANDIDATE_CONFIG
    cfg = load_env_config(config_path, rig_path)
    if not 1 <= len(candidate.mechanics) <= 4:
        raise ValueError(f"{candidate.name}: stack must contain 1..4 mechanics")
    types = [_MECHANICS[name] for name in candidate.mechanics]
    cfg.evaluation_stack_types = tuple(types + [MechanicType.Normal] * (4 - len(types)))
    cfg.evaluation_stack_count = len(types)
    # Candidate trials always start at maximum difficulty and last five minutes.
    # Use stable anchor implementations while independently enabling the
    # maximum-difficulty terrain profile from the spawn point.
    cfg.biome_split = 0
    cfg.force_endgame_difficulty = True
    cfg.chain_biomes = True
    cfg.termination.trial_time_limit = 300.0
    cfg.termination.max_steps = 0
    for key, value in candidate.generation.items():
        if key.startswith("terrain."):
            setattr(cfg.terrain, key.removeprefix("terrain."), value)
        elif key.startswith("physics."):
            setattr(cfg.physics, key.removeprefix("physics."), value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"{candidate.name}: unknown generation setting {key!r}")
    return cfg
