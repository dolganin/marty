from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, require_compiled_bank

from .agent import AgentFactory


Split = Literal["train", "test", "anchor"]
_SPLIT_MODES = {
    "train": MarsRoverEnv.BIOME_MODE_TRAIN,
    "test": MarsRoverEnv.BIOME_MODE_TEST,
    "anchor": MarsRoverEnv.BIOME_MODE_ANCHOR,
}


@dataclass(frozen=True)
class EvaluationSpec:
    split: Split
    seeds: tuple[int, ...]
    episodes_per_trial: int = 4
    max_steps: int = 3000
    config_path: str | None = None
    render_trials: int = 0
    render_stride: int = 4
    render_max_frames: int = 180

    def __post_init__(self) -> None:
        if self.split not in _SPLIT_MODES:
            raise ValueError(f"unknown split: {self.split}")
        if not self.seeds:
            raise ValueError("at least one evaluation seed is required")
        if self.episodes_per_trial < 1 or self.max_steps < 1:
            raise ValueError("episodes_per_trial and max_steps must be positive")
        if self.render_trials < 0 or self.render_stride < 1 or self.render_max_frames < 1:
            raise ValueError("invalid render settings")


@dataclass
class EpisodeRecord:
    bank_version: str
    split: str
    biome_id: str
    biome_index: int
    seed: int
    trial_index: int
    episode_index: int
    raw_return: float
    normalized_return: float | None
    steps: int
    success: bool
    flipped: bool
    fatal_error: bool
    landing_fatal: bool
    battery_consumed: float
    distance: float
    lidar_scan_count: int
    lidar_active_steps: int
    lidar_energy_spent: float
    lidar_airborne_attempt_count: int
    solar_toggle_count: int
    charging_active_steps: int
    solar_energy_gained: float
    ballistic_flight_count: int
    airborne_steps: int
    safe_landing_count: int
    action_entropy: float | None


def _catalog_for_split(split: Split) -> list[dict[str, Any]]:
    import _mars_rover_cpp as native

    catalog = [dict(item) for item in native.biome_catalog()]
    if split == "anchor":
        return [item for item in catalog if bool(item["is_anchor"])]
    split_value = {"train": 1, "test": 2}[split]
    return [
        item
        for item in catalog
        if not bool(item["is_anchor"]) and int(item["split"]) == split_value
    ]


def _normalization_bounds(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Per-biome (floor, ceiling) used to put every mechanic on one scale.

    The ceiling is the privileged ORACLE, not the memoryless robust reference. Once the
    bank was made to require adaptation, robust stopped clearing random on the very biomes
    that force it (r_robust -0.56 vs r_random -0.35 on momentum_draft_locks), so a
    robust-based band inverted its own sign there and was undefined on 2 of 9 test biomes.
    The oracle represents what is achievable once the mechanic is known, which is exactly
    the ceiling an adaptive agent should be measured against, and it is well-defined on all
    of them. Robust remains recorded in the manifest as a reference point, but it no longer
    sets the scale.
    """
    result: dict[str, tuple[float, float]] = {}
    for item in manifest.get("biomes", []):
        low = item.get("r_random")
        high = item.get("r_solve")
        if high is None:
            high = item.get("r_robust")
        if low is not None and high is not None and float(high) > float(low) + 1.0e-8:
            result[str(item["id"])] = (float(low), float(high))
    for biome_id, item in manifest.get("anchor_references", {}).get("items", {}).items():
        low, high = item.get("r_random"), item.get("r_robust")
        if low is not None and high is not None and float(high) > float(low) + 1.0e-8:
            result[str(biome_id)] = (float(low), float(high))
    return result


def _public_info(*, terminated: bool, truncated: bool, episode_index: int) -> dict[str, Any]:
    # This dict is intentionally small and contains no evaluator/debug state.
    return {
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode_index": int(episode_index),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def evaluate(
    agent_factory: AgentFactory,
    spec: EvaluationSpec,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate every agent through one public API and one trial protocol.

    Debug state is read only after actions have been selected and is sent solely
    to evaluator-owned trace/reliability records. It is never included in
    ``Agent.act`` or ``Agent.observe``.
    """
    manifest = load_manifest(manifest_path)
    bank_version = require_compiled_bank(manifest)
    bounds = _normalization_bounds(manifest)
    catalog = _catalog_for_split(spec.split)
    if not catalog:
        raise RuntimeError(f"compiled bank has no {spec.split} mechanics")

    output = Path(output_dir) if output_dir is not None else None
    episode_records: list[EpisodeRecord] = []
    trace_rows: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    effective_render_stride = max(
        spec.render_stride,
        math.ceil(spec.max_steps * spec.episodes_per_trial / spec.render_max_frames),
    )
    trial_counter = 0

    for biome in catalog:
        biome_id = str(biome["id"])
        biome_index = int(biome["index"])
        for trial_index, trial_seed in enumerate(spec.seeds):
            agent = agent_factory(int(trial_seed))
            env = MarsRoverEnv(
                config_path=spec.config_path,
                render_mode="rgb_array" if trial_counter < spec.render_trials else None,
                biome_split=MarsRoverEnv.BIOME_MODE_ALL,
                fixed_biome_id=biome_index,
            )
            for episode_index in range(spec.episodes_per_trial):
                trial_start = episode_index == 0
                episode_seed = int(trial_seed) + episode_index * 1_000_003
                obs, _ = env.reset(seed=episode_seed, options={"trial_start": trial_start})
                agent.reset(trial_start=trial_start)
                start_debug = env.debug_info()
                start_x = float(start_debug.get("x", 0.0))
                start_energy = float(start_debug.get("energy", 0.0))
                raw_return = 0.0
                entropy_samples: list[float] = []
                terminated = truncated = False
                last_debug = start_debug
                landing_fatal = False
                lidar_scan_count = 0
                lidar_active_steps = 0
                lidar_energy_spent = 0.0
                lidar_airborne_attempt_count = 0
                solar_toggle_count = 0
                charging_active_steps = 0
                solar_energy_gained = 0.0
                ballistic_flight_count = 0
                airborne_steps = 0
                safe_landing_count = 0
                previous_action = 0
                previous_airborne = bool(start_debug.get("airborne", False))
                previous_energy = start_energy
                for step in range(spec.max_steps):
                    action = int(agent.act(obs))
                    if action < 0 or action >= env.action_space.n:
                        env.close()
                        raise ValueError(f"agent returned invalid action {action}")
                    obs, reward, terminated, truncated, _ = env.step(action)
                    step_debug = env.debug_info()
                    lidar_pressed = bool(action & (1 << 11)) and not bool(previous_action & (1 << 11))
                    solar_toggle_pressed = bool(action & (1 << 10)) and not bool(previous_action & (1 << 10))
                    was_airborne = previous_airborne
                    is_airborne = bool(step_debug.get("airborne", False))
                    is_charging = bool(step_debug.get("charging_active", False))
                    step_energy = float(step_debug.get("energy", previous_energy))
                    lidar_cost = float(step_debug.get("lidar_energy_cost", 0.0))
                    lidar_scan_count += int(lidar_cost > 0.0)
                    lidar_energy_spent += max(0.0, lidar_cost)
                    lidar_active_steps += int(bool(step_debug.get("lidar_active", False)))
                    lidar_airborne_attempt_count += int(lidar_pressed and was_airborne)
                    solar_toggle_count += int(solar_toggle_pressed)
                    charging_active_steps += int(is_charging)
                    # While charging the drivetrain is locked, so a positive energy delta
                    # is attributable to the solar system rather than propulsion.
                    solar_energy_gained += max(0.0, step_energy - previous_energy) if is_charging else 0.0
                    ballistic_flight_count += int(is_airborne and not was_airborne)
                    airborne_steps += int(is_airborne)
                    safe_landing_count += int(
                        bool(step_debug.get("landing_event", False))
                        and not bool(step_debug.get("landing_fatal", False))
                    )
                    landing_fatal = landing_fatal or bool(step_debug.get("landing_fatal", False))
                    done = terminated or truncated
                    agent.observe(
                        float(reward),
                        done,
                        _public_info(
                            terminated=terminated,
                            truncated=truncated,
                            episode_index=episode_index,
                        ),
                    )
                    raw_return += float(reward)
                    diagnostics = agent.diagnostics()
                    if "action_entropy" in diagnostics:
                        entropy_samples.append(float(diagnostics["action_entropy"]))
                    if output is not None and trial_counter < spec.render_trials:
                        last_debug = step_debug
                        trace_rows.append(
                            {
                                "bank_version": bank_version,
                                "split": spec.split,
                                "biome_id": biome_id,
                                "seed": int(trial_seed),
                                "trial_index": trial_index,
                                "episode_index": episode_index,
                                "step": step,
                                "action": action,
                                "reward": float(reward),
                                "x": float(last_debug.get("x", 0.0)),
                                "y": float(last_debug.get("y", 0.0)),
                                "speed": float(last_debug.get("speed", 0.0)),
                                "angle": float(last_debug.get("angle", 0.0)),
                                "energy": float(last_debug.get("energy", 0.0)),
                                "damage": float(last_debug.get("damage", 0.0)),
                                "airborne": is_airborne,
                                "landing_event": bool(last_debug.get("landing_event", False)),
                                "landing_fatal": bool(last_debug.get("landing_fatal", False)),
                                "lidar_active": bool(last_debug.get("lidar_active", False)),
                                "lidar_energy_cost": lidar_cost,
                                "charging_active": is_charging,
                                "solar_panel_deployment": float(last_debug.get("solar_panel_deployment", 0.0)),
                                "done": done,
                            }
                        )
                        global_episode_step = episode_index * spec.max_steps + step
                        if (
                            global_episode_step % effective_render_stride == 0
                            and len(frames) < spec.render_max_frames
                        ):
                            frame = env.render()
                            if frame is not None:
                                # Keep behavior recording bounded: 320x180x180 is ~31 MB,
                                # versus >500 MB for the former unbounded 640x360 sequence.
                                frames.append(frame[::2, ::2].copy())
                    if done:
                        break
                    previous_action = action
                    previous_airborne = is_airborne
                    previous_energy = step_energy
                last_debug = env.debug_info()
                low_high = bounds.get(biome_id)
                normalized = None
                if low_high is not None:
                    low, high = low_high
                    normalized = (raw_return - low) / (high - low)
                finish_x = float(env._config.termination.finish_x)
                end_x = float(last_debug.get("x", 0.0))
                success = end_x >= finish_x
                # A physical crash/fall is explicit in debug state. Running out
                # of energy or getting irrecoverably stuck is also fatal to the
                # mission; a time-limit truncation alone is not.
                fatal_error = bool(last_debug.get("fatal_error", False)) or bool(
                    terminated and not success
                )
                episode_records.append(
                    EpisodeRecord(
                        bank_version=bank_version,
                        split=spec.split,
                        biome_id=biome_id,
                        biome_index=biome_index,
                        seed=int(trial_seed),
                        trial_index=trial_index,
                        episode_index=episode_index,
                        raw_return=raw_return,
                        normalized_return=normalized,
                        steps=step + 1,
                        success=success,
                        flipped=abs(float(last_debug.get("angle", 0.0))) > np.pi / 2,
                        fatal_error=fatal_error,
                        landing_fatal=landing_fatal,
                        battery_consumed=max(0.0, start_energy - float(last_debug.get("energy", 0.0))),
                        distance=end_x - start_x,
                        lidar_scan_count=lidar_scan_count,
                        lidar_active_steps=lidar_active_steps,
                        lidar_energy_spent=lidar_energy_spent,
                        lidar_airborne_attempt_count=lidar_airborne_attempt_count,
                        solar_toggle_count=solar_toggle_count,
                        charging_active_steps=charging_active_steps,
                        solar_energy_gained=solar_energy_gained,
                        ballistic_flight_count=ballistic_flight_count,
                        airborne_steps=airborne_steps,
                        safe_landing_count=safe_landing_count,
                        action_entropy=(float(np.mean(entropy_samples)) if entropy_samples else None),
                    )
                )
            env.close()
            trial_counter += 1

    rows = [asdict(record) for record in episode_records]
    episode_indices = range(spec.episodes_per_trial)
    raw_curve = [
        float(np.mean([row["raw_return"] for row in rows if row["episode_index"] == index]))
        for index in episode_indices
    ]
    normalized_curve: list[float] | None = None
    if all(row["normalized_return"] is not None for row in rows):
        normalized_curve = [
            float(np.mean([row["normalized_return"] for row in rows if row["episode_index"] == index]))
            for index in episode_indices
        ]
    entropy_curve = [
        float(np.mean(values)) if values else None
        for index in episode_indices
        for values in [[
            row["action_entropy"]
            for row in rows
            if row["episode_index"] == index and row["action_entropy"] is not None
        ]]
    ]
    fatal_curve = [
        float(np.mean([row["fatal_error"] for row in rows if row["episode_index"] == index]))
        for index in episode_indices
    ]
    survival_curve = [1.0 - value for value in fatal_curve]
    surviving_distance_curve = [
        float(np.mean(values)) if values else 0.0
        for index in episode_indices
        for values in [[
            row["distance"]
            for row in rows
            if row["episode_index"] == index and not row["fatal_error"]
        ]]
    ]
    summary = {
        "schema_version": 2,
        "bank_version": bank_version,
        "split": spec.split,
        "seeds": list(spec.seeds),
        "episodes_per_trial": spec.episodes_per_trial,
        "mechanics": [str(item["id"]) for item in catalog],
        "episode_count": len(rows),
        "raw_return_by_episode": raw_curve,
        "normalized_return_by_episode": normalized_curve,
        "trial_auc": float(np.mean(normalized_curve)) if normalized_curve is not None else None,
        "adaptation_delta": (
            normalized_curve[-1] - normalized_curve[0] if normalized_curve is not None else None
        ),
        "raw_adaptation_delta": raw_curve[-1] - raw_curve[0],
        "action_entropy_by_episode": entropy_curve,
        "fatal_error_rate_by_episode": fatal_curve,
        "survival_rate_by_episode": survival_curve,
        "survival_auc": float(np.mean(survival_curve)),
        "survival_adaptation_delta": survival_curve[-1] - survival_curve[0],
        "surviving_distance_by_episode": surviving_distance_curve,
        "fatal_error_rate": float(np.mean([row["fatal_error"] for row in rows])),
        "fatal_error_count": int(sum(bool(row["fatal_error"]) for row in rows)),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "flip_rate": float(np.mean([row["flipped"] for row in rows])),
        "mean_battery_consumed": float(np.mean([row["battery_consumed"] for row in rows])),
        "mean_distance": float(np.mean([row["distance"] for row in rows])),
        "mean_lidar_scan_count": float(np.mean([row["lidar_scan_count"] for row in rows])),
        "mean_lidar_active_fraction": float(
            np.mean([row["lidar_active_steps"] / max(1, row["steps"]) for row in rows])
        ),
        "mean_lidar_energy_spent": float(np.mean([row["lidar_energy_spent"] for row in rows])),
        "mean_lidar_airborne_attempt_count": float(
            np.mean([row["lidar_airborne_attempt_count"] for row in rows])
        ),
        "mean_solar_toggle_count": float(np.mean([row["solar_toggle_count"] for row in rows])),
        "mean_charging_active_fraction": float(
            np.mean([row["charging_active_steps"] / max(1, row["steps"]) for row in rows])
        ),
        "mean_solar_energy_gained": float(np.mean([row["solar_energy_gained"] for row in rows])),
        "mean_ballistic_flight_count": float(
            np.mean([row["ballistic_flight_count"] for row in rows])
        ),
        "mean_airborne_fraction": float(
            np.mean([row["airborne_steps"] / max(1, row["steps"]) for row in rows])
        ),
        "mean_safe_landing_count": float(np.mean([row["safe_landing_count"] for row in rows])),
        "mean_raw_return": float(np.mean([row["raw_return"] for row in rows])),
    }
    payload = {"summary": summary, "episodes": rows}
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _write_jsonl(output / "episodes.jsonl", rows)
        if trace_rows:
            _write_jsonl(output / "behavior.jsonl", trace_rows)
        if frames:
            try:
                from PIL import Image

                images = [Image.fromarray(frame) for frame in frames]
                gif_path = output / "behavior.gif"
                temporary_gif = output / "behavior.gif.tmp"
                images[0].save(
                    temporary_gif,
                    format="GIF",
                    save_all=True,
                    append_images=images[1:],
                    duration=67,
                    loop=0,
                    optimize=False,
                )
                temporary_gif.replace(gif_path)
            except ImportError:
                pass
    return payload
