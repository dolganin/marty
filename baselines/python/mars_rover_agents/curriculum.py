"""Configuration schedule for chain-length and consequence curriculum PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CurriculumStage:
    zone_count: int
    fraction: float
    crash_penalty_scale: float
    finish_x: float | None = None
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.zone_count < 1 or self.fraction <= 0.0:
            raise ValueError("zone_count and fraction must be positive")
        if not 0.0 < self.crash_penalty_scale <= 1.0:
            raise ValueError("crash_penalty_scale must be in (0, 1]")


DEFAULT_CURRICULUM = (
    CurriculumStage(1, 0.10, 0.05, finish_x=45.0, max_steps=900),
    CurriculumStage(2, 0.15, 0.15, finish_x=90.0, max_steps=1200),
    CurriculumStage(4, 0.20, 0.35, finish_x=190.0, max_steps=1600),
    CurriculumStage(8, 0.25, 0.65, finish_x=390.0, max_steps=2200),
    CurriculumStage(14, 0.30, 1.00),
)


def allocate_timesteps(total: int, stages: tuple[CurriculumStage, ...]) -> list[int]:
    """Allocate an exact transition budget while keeping every stage non-empty."""
    if total < len(stages):
        raise ValueError("total timesteps must be at least the number of stages")
    weights = [stage.fraction for stage in stages]
    weight_sum = sum(weights)
    raw = [total * weight / weight_sum for weight in weights]
    result = [max(1, int(value)) for value in raw]
    remainder = total - sum(result)
    order = sorted(range(len(stages)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    while remainder > 0:
        for index in order:
            if remainder == 0:
                break
            result[index] += 1
            remainder -= 1
    while remainder < 0:
        for index in reversed(order):
            if remainder == 0:
                break
            if result[index] > 1:
                result[index] -= 1
                remainder += 1
    return result


def stage_config(base: dict[str, Any], stage: CurriculumStage) -> dict[str, Any]:
    """Return a stage config without mutating the authoritative base mapping."""
    import copy

    result = copy.deepcopy(base)
    env = result.setdefault("env", {})
    termination = result.setdefault("termination", {})
    reward = result.setdefault("reward", {})
    env["chain_biomes"] = True
    env["chain_zone_count"] = stage.zone_count
    if stage.finish_x is not None:
        termination["finish_x"] = stage.finish_x
    if stage.max_steps is not None:
        termination["max_steps"] = stage.max_steps
    for key in ("flip_penalty", "stuck_penalty"):
        if key in reward:
            reward[key] = float(reward[key]) * stage.crash_penalty_scale
    return result


def write_stage_configs(
    base_path: Path, output_dir: Path, stages: tuple[CurriculumStage, ...]
) -> list[Path]:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, stage in enumerate(stages, start=1):
        path = output_dir / f"stage_{index:02d}_zones_{stage.zone_count}.yaml"
        path.write_text(
            yaml.safe_dump(stage_config(base, stage), sort_keys=False), encoding="utf-8"
        )
        paths.append(path)
    return paths


def curriculum_payload(stages: tuple[CurriculumStage, ...]) -> list[dict[str, Any]]:
    return [asdict(stage) for stage in stages]
