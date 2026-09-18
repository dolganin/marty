"""Auditable, batch-oriented growth of the frozen mechanic-stack bank.

Candidates may be proposed by the OpenAI Responses API or supplied as a saved
JSON response.  Proposal is deliberately separate from acceptance: this tool
only appends a stack to the production bank after every local filter succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.bank import DEFAULT_MANIFEST, load_manifest, write_json
from mars_rover_env.config import load_env_config


COMPONENTS = ("normal", "sand", "ice", "mud", "wind", "low_gravity", "crust", "liquid")
CPP_TYPES = {
    "normal": "Normal", "sand": "Sand", "ice": "Ice", "mud": "Mud",
    "wind": "Wind", "low_gravity": "LowGravity", "crust": "Crust", "liquid": "Liquid",
}
SCORING_WINDOW_METERS = 250.0
LATENT_KEYS = ("latent_traction", "gravity_multiplier", "latent_viscosity", "ambient_temperature")
FILTER_NAMES = ("validity", "nontriviality", "solvability", "discriminative_power", "uniqueness")

# First-pass proposals authored by the in-session model.  They deliberately
# cover directional order, water, traction, gravity, deformable ground and
# three/four-way interactions instead of exhaustively enumerating permutations.
CURATED_STACKS = (
    ("ice_wind", ("ice", "wind")), ("wind_ice", ("wind", "ice")),
    ("sand_ice", ("sand", "ice")), ("ice_sand", ("ice", "sand")),
    ("mud_low_gravity", ("mud", "low_gravity")), ("low_gravity_mud", ("low_gravity", "mud")),
    ("crust_wind", ("crust", "wind")), ("wind_crust", ("wind", "crust")),
    ("liquid_ice", ("liquid", "ice")), ("ice_liquid", ("ice", "liquid")),
    ("sand_low_gravity", ("sand", "low_gravity")), ("low_gravity_sand", ("low_gravity", "sand")),
    ("mud_ice", ("mud", "ice")), ("ice_mud", ("ice", "mud")),
    ("crust_low_gravity", ("crust", "low_gravity")), ("low_gravity_crust", ("low_gravity", "crust")),
    ("liquid_low_gravity", ("liquid", "low_gravity")), ("low_gravity_liquid", ("low_gravity", "liquid")),
    ("sand_crust", ("sand", "crust")), ("crust_sand", ("crust", "sand")),
    ("mud_wind", ("mud", "wind")), ("wind_mud", ("wind", "mud")),
    ("sand_liquid", ("sand", "liquid")), ("liquid_sand", ("liquid", "sand")),
    ("ice_wind_low_gravity", ("ice", "wind", "low_gravity")),
    ("low_gravity_wind_ice", ("low_gravity", "wind", "ice")),
    ("sand_crust_low_gravity", ("sand", "crust", "low_gravity")),
    ("low_gravity_crust_sand", ("low_gravity", "crust", "sand")),
    ("mud_ice_wind", ("mud", "ice", "wind")),
    ("wind_ice_mud", ("wind", "ice", "mud")),
    ("liquid_wind_low_gravity", ("liquid", "wind", "low_gravity")),
    ("low_gravity_wind_liquid", ("low_gravity", "wind", "liquid")),
    ("sand_mud_ice", ("sand", "mud", "ice")), ("ice_mud_sand", ("ice", "mud", "sand")),
    ("crust_ice_low_gravity", ("crust", "ice", "low_gravity")),
    ("low_gravity_ice_crust", ("low_gravity", "ice", "crust")),
    ("liquid_sand_wind", ("liquid", "sand", "wind")), ("wind_sand_liquid", ("wind", "sand", "liquid")),
    ("sand_ice_wind_crust", ("sand", "ice", "wind", "crust")),
    ("crust_wind_ice_sand", ("crust", "wind", "ice", "sand")),
    ("mud_low_gravity_wind_crust", ("mud", "low_gravity", "wind", "crust")),
    ("crust_wind_low_gravity_mud", ("crust", "wind", "low_gravity", "mud")),
    ("liquid_ice_wind_sand", ("liquid", "ice", "wind", "sand")),
    ("sand_wind_ice_liquid", ("sand", "wind", "ice", "liquid")),
    ("liquid_mud_low_gravity_crust", ("liquid", "mud", "low_gravity", "crust")),
    ("crust_low_gravity_mud_liquid", ("crust", "low_gravity", "mud", "liquid")),
    ("sand_mud_low_gravity_ice", ("sand", "mud", "low_gravity", "ice")),
    ("ice_low_gravity_mud_sand", ("ice", "low_gravity", "mud", "sand")),
    ("wind_crust_liquid_ice", ("wind", "crust", "liquid", "ice")),
    ("ice_liquid_crust_wind", ("ice", "liquid", "crust", "wind")),
)


@dataclass(frozen=True)
class Candidate:
    name: str
    components: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Candidate":
        name = str(payload.get("name", "")).strip().lower().replace(" ", "_")
        parts = tuple(str(value).strip().lower() for value in payload.get("components", ()))
        if not name or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_" for ch in name):
            raise ValueError("name must be a lowercase identifier")
        if not 1 <= len(parts) <= 4:
            raise ValueError("a stack must contain one to four components")
        if len(set(parts)) != len(parts):
            raise ValueError("a stack cannot repeat a component")
        unknown = sorted(set(parts) - set(COMPONENTS))
        if unknown:
            raise ValueError("unknown components: " + ", ".join(unknown))
        return cls(name=name, components=parts)

    def cpp(self, split: str = "train") -> str:
        padded = [CPP_TYPES[item] for item in self.components] + ["Normal"] * (4 - len(self.components))
        cpp_split = "Test" if split == "test" else "Train"
        return (
            "#include \"mars/biome_bank.hpp\"\n\n"
            "namespace mars::candidate_check {\n"
            f"constexpr FrozenMechanismStack kCandidate{{{{MechanicType::{padded[0]}, "
            f"MechanicType::{padded[1]}, MechanicType::{padded[2]}, MechanicType::{padded[3]}}}, "
            f"{len(self.components)}, BiomeSplit::{cpp_split}}};\n"
            "static_assert(kCandidate.count >= 1 && kCandidate.count <= kMaxActiveMechanisms);\n"
            "}\n"
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _candidate_path(root: Path, batch: int, ordinal: int) -> tuple[Path, Path]:
    directory = root / "candidates" / f"batch_{batch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"candidate_{ordinal:03d}"
    return directory / f"{stem}.cpp", directory / f"{stem}.json"


def _response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    if not chunks:
        raise RuntimeError("LLM response contains no text")
    return "".join(chunks)


def request_candidates(count: int, model: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for live candidate generation; use --input-json to replay")
    contract = (Path(__file__).resolve().parents[4] / "contracts" / "stack_contract.md").read_text(encoding="utf-8")
    prompt = (
        "Propose exactly the requested number of distinct Mars Rover mechanic stacks. "
        "Return JSON only, following the contract below. Do not include explanations.\n\n" + contract
    )
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps({
            "model": model,
            "input": prompt + f"\n\nRequested candidates: {count}.",
            "text": {"format": {"type": "json_object"}},
        }).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as raw:
            response = json.loads(raw.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
    payload = json.loads(_response_text(response))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise RuntimeError("LLM response JSON has no candidates list")
    return candidates[:count], response


def curated_candidates(count: int, *, batch: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic, non-repeating proposals authored for offline screening."""
    catalogue = list(CURATED_STACKS)
    seen = {components for _name, components in catalogue}
    for length in range(2, 5):
        for components in itertools.permutations(COMPONENTS[1:], length):
            if components not in seen:
                catalogue.append(("_".join(components), components))
                seen.add(components)
    first = (batch - 1) * count
    selected = catalogue[first:first + count]
    if len(selected) != count:
        raise ValueError("curated catalogue exhausted")
    return ([{"name": name, "components": list(components)} for name, components in selected],
            {"source": "curated-in-session-model", "generated_at": _now(), "count": count,
             "catalogue_offset": first})


def _make_native_env(components: tuple[str, ...], *, max_steps: int, num_envs: int = 1):
    import _mars_rover_cpp as native

    cfg = load_env_config()
    cfg.biome_split = 0
    cfg.chain_biomes = True
    cfg.termination.trial_time_limit = 0.0
    cfg.termination.max_steps = max_steps
    cfg.terrain.length = SCORING_WINDOW_METERS
    cfg.terrain.sample_count = max(2048, int(SCORING_WINDOW_METERS / cfg.terrain.dx) + 8)
    cfg.evaluation_stack_types = (
        [getattr(native.MechanicType, CPP_TYPES[item]) for item in components]
        + [native.MechanicType.Normal] * (4 - len(components))
    )
    cfg.evaluation_stack_count = len(components)
    batch = native.MarsRoverBatchEnv(num_envs, cfg)
    return batch


def _run_policy(
    components: tuple[str, ...], *, seed: int, max_steps: int,
    action_fn: Callable[[np.ndarray, dict[str, Any], int], int],
    observe: Callable[[float, bool], None] | None = None,
) -> dict[str, Any]:
    batch = _make_native_env(components, max_steps=max_steps)
    obs = np.zeros((1, batch.obs_dim), dtype=np.float32)
    reward = np.zeros(1, dtype=np.float32)
    terminated = np.zeros(1, dtype=np.uint8)
    truncated = np.zeros(1, dtype=np.uint8)
    actions = np.zeros(1, dtype=np.int32)
    batch.reset_at(0, seed, True, obs[0])
    start_x = float(batch.debug_info(0)["x"])
    latent_rows: list[list[float]] = []
    total_reward = 0.0
    for step in range(max_steps):
        debug = dict(batch.debug_info(0))
        latent_rows.append([float(debug[key]) for key in LATENT_KEYS])
        actions[0] = int(action_fn(obs[0], debug, step))
        batch.step(actions, obs, reward, terminated, truncated)
        total_reward += float(reward[0])
        if observe is not None:
            observe(float(reward[0]), bool(terminated[0] or truncated[0]))
        if terminated[0] or truncated[0]:
            break
    debug = dict(batch.debug_info(0))
    latent_rows.append([float(debug[key]) for key in LATENT_KEYS])
    values = np.asarray(latent_rows, dtype=np.float64)
    return {
        "progress": max(0.0, float(debug["x"]) - start_x) / SCORING_WINDOW_METERS,
        "reward": total_reward,
        "latents_mean": values.mean(axis=0).tolist(),
        "latents_var": values.var(axis=0).tolist(),
        "latents_min": values.min(axis=0).tolist(),
        "latents_max": values.max(axis=0).tolist(),
        "steps": step + 1,
    }


def _random_action(seed: int) -> Callable[[np.ndarray, dict[str, Any], int], int]:
    rng = np.random.default_rng(seed)
    return lambda _obs, _debug, _step: int(ACTION_MACROS[rng.integers(0, len(ACTION_MACROS))])


def _scripted_action() -> Callable[[np.ndarray, dict[str, Any], int], int]:
    from mars_rover_agents.scripted_driver import scripted_action
    return lambda _obs, debug, _step: int(scripted_action(debug))


def _validate_compile(candidate_cpp: Path) -> tuple[bool, str | None]:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    if not compiler:
        return False, "no C++ compiler found"
    include = Path(__file__).resolve().parents[3] / "cpp" / "include"
    result = subprocess.run(
        [compiler, "-std=c++20", "-fsyntax-only", "-I", str(include), str(candidate_cpp)],
        text=True, capture_output=True, check=False,
    )
    return result.returncode == 0, (result.stderr or result.stdout).strip() or None


def _finite_and_bounded(row: dict[str, Any]) -> tuple[bool, str | None]:
    values = np.asarray(row["latents_min"] + row["latents_max"], dtype=float)
    if not np.isfinite(values).all():
        return False, "non-finite latent"
    minimum, maximum = row["latents_min"], row["latents_max"]
    if minimum[0] < 0.0:
        return False, "negative traction"
    if minimum[1] < 0.25 or maximum[1] > 1.6:
        return False, "gravity outside [0.25, 1.6]"
    if minimum[2] < 0.0:
        return False, "negative viscosity"
    if minimum[3] < -58.0 or maximum[3] > 72.0:
        return False, "temperature outside [-58, 72]"
    return True, None


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 if denom == 0.0 else 1.0 - float(np.dot(left, right) / denom)


def _normalised_l2(left: np.ndarray, right: np.ndarray) -> float:
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
    return float(np.linalg.norm((left - right) / scale) / math.sqrt(left.size))


def _load_agents(robust_model: Path | None, recurrent_model: Path | None):
    if robust_model is None or recurrent_model is None:
        return None, None
    try:
        from stable_baselines3 import PPO
        from mars_rover_agents.agents import PPOAgent
        from mars_rover_agents.rl2_agent import RL2TransformerAgent
    except ImportError as exc:
        raise RuntimeError("baseline dependencies are unavailable") from exc

    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    robust = PPOAgent(PPO.load(robust_model, device=device))
    recurrent = RL2TransformerAgent.load(recurrent_model, device=device, max_steps=1200)
    return robust, recurrent


def _agent_trial_rows(
    agent: Any,
    components: tuple[str, ...],
    seeds: range,
    *,
    episodes_per_trial: int = 2,
    max_steps: int = 1200,
) -> list[dict[str, Any]]:
    """Evaluate all trial seeds in one native batch and preserve RL2 memory.

    The first episode is adaptation experience; filter 4 scores the second
    episode.  Feed-forward PPO sees the identical courses but has no state to
    carry across the episode boundary.
    """
    import torch
    from mars_rover_agents.rl2_agent import RL2TransformerAgent
    from mars_rover_agents.rl2_model import TransformerXLState, rl2_features

    seed_values = tuple(int(seed) for seed in seeds)
    num_envs = len(seed_values)
    batch = _make_native_env(components, max_steps=max_steps, num_envs=num_envs)
    observations = np.zeros((num_envs, batch.obs_dim), dtype=np.float32)
    rewards = np.zeros(num_envs, dtype=np.float32)
    terminated = np.zeros(num_envs, dtype=np.uint8)
    truncated = np.zeros(num_envs, dtype=np.uint8)
    raw_actions = np.zeros(num_envs, dtype=np.int32)
    action_macros = np.asarray(ACTION_MACROS, dtype=np.int32)
    episode_rewards = np.zeros((num_envs, episodes_per_trial), dtype=np.float64)
    recurrent = isinstance(agent, RL2TransformerAgent)
    if recurrent:
        if int(getattr(agent.model, "context_dim", 0)):
            raise RuntimeError("batched stack screening does not support explicit RL2 context")
        device = agent.device
        state = agent.model.initial_state(num_envs, device)
        previous_actions = np.full(num_envs, -1, dtype=np.int64)
        previous_rewards = np.zeros(num_envs, dtype=np.float32)
        previous_dones = np.ones(num_envs, dtype=bool)
    else:
        model = agent._model

    for episode in range(episodes_per_trial):
        trial_start = episode == 0
        for env_index, seed in enumerate(seed_values):
            episode_seed = seed + episode * 1_000_003
            batch.reset_at(env_index, episode_seed, trial_start, observations[env_index])
        active = np.ones(num_envs, dtype=bool)
        for _step in range(max_steps):
            if recurrent:
                features = rl2_features(
                    observations, previous_actions, previous_rewards, previous_dones,
                    device=device,
                )
                old_state = state
                with torch.no_grad():
                    logits, _values, next_state = agent.model.forward_step(features, state)
                    action_indices = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                active_device = torch.as_tensor(active, dtype=torch.bool, device=device)
                keep = active_device.view(-1, 1, 1)
                state = TransformerXLState(
                    memories=[torch.where(keep, new, old)
                              for new, old in zip(next_state.memories, old_state.memories)],
                    valid=torch.where(active_device.view(-1, 1), next_state.valid, old_state.valid),
                )
            else:
                action_indices, _ = model.predict(observations, deterministic=True)
                action_indices = np.asarray(action_indices, dtype=np.int64).reshape(num_envs)
            raw_actions[:] = action_macros[action_indices]
            raw_actions[~active] = 0
            batch.step(raw_actions, observations, rewards, terminated, truncated)
            episode_rewards[active, episode] += rewards[active]
            dones = np.logical_or(terminated != 0, truncated != 0)
            if recurrent:
                previous_actions[active] = action_indices[active]
                previous_rewards[active] = rewards[active]
                previous_dones[active] = dones[active]
            active &= ~dones
            if not bool(active.any()):
                break
        if recurrent:
            previous_dones[:] = True

    return [
        {
            "reward": float(episode_rewards[index, -1]),
            "episode_rewards": episode_rewards[index].tolist(),
        }
        for index in range(num_envs)
    ]


def _split_for(candidate: Candidate, accepted: list[dict[str, Any]]) -> str:
    train = [item for item in accepted if item["split"] == "train"]
    test = [item for item in accepted if item["split"] == "test"]
    train_components = Counter(part for item in train for part in item["components"])
    if any(train_components[part] < 3 for part in candidate.components):
        return "train"
    # The high-level bank contract requires that held-out stacks do not repeat
    # a train composition. Duplicates are rejected before this function.
    return "train" if len(train) < 2 * max(1, len(test)) else "test"


def _existing_stack_records(
    manifest: dict[str, Any], accepted_dir: Path | None = None,
) -> list[dict[str, Any]]:
    audited: dict[tuple[str, ...], dict[str, Any]] = {}
    if accepted_dir is not None and accepted_dir.is_dir():
        for path in accepted_dir.glob("stack_*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                audited[tuple(payload["components"])] = payload
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    records: list[dict[str, Any]] = []
    for index, stack in enumerate(manifest.get("frozen_stacks", []), start=1):
        split = {"held_out": "test", "anchor": "anchor"}.get(stack.get("split"), "train")
        record = {
            "name": f"existing_stack_{index:02d}",
            "components": list(stack["mechanisms"]),
            "split": split,
            "characteristics": {},
            "existing": True,
        }
        prior = audited.get(tuple(stack["mechanisms"]))
        if prior is not None:
            record.update(prior)
            record["split"] = split
            record["existing"] = True
        records.append(record)
    return records


def _filter_candidate(
    candidate: Candidate, candidate_cpp: Path, accepted: list[dict[str, Any]],
    *, robust_model: Path | None, recurrent_model: Path | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    filters: dict[str, Any] = {}
    compiled, message = _validate_compile(candidate_cpp)
    if not compiled:
        filters["validity"] = {"passed": False, "reason": message}
        return filters, None
    validity = [_run_policy(candidate.components, seed=1000 + seed, max_steps=1000,
                            action_fn=_random_action(1000 + seed)) for seed in range(1)]
    valid, message = _finite_and_bounded(validity[0])
    filters["validity"] = {"passed": valid, "random_steps": 1000, "reason": message}
    if not valid:
        return filters, None
    random_rows = [_run_policy(candidate.components, seed=2000 + seed, max_steps=1500,
                               action_fn=_random_action(2000 + seed)) for seed in range(5)]
    mean_progress = float(np.mean([row["progress"] for row in random_rows]))
    mean_latents = np.mean([row["latents_mean"] for row in random_rows], axis=0)
    existing_latents = [item.get("characteristics", {}).get("latent_mean") for item in accepted]
    distances = [_cosine_distance(mean_latents, np.asarray(vector, dtype=float))
                 for vector in existing_latents if vector is not None]
    nontrivial = not (all(row["progress"] < 0.05 for row in random_rows) or
                      all(row["progress"] > 0.95 for row in random_rows))
    if distances and min(distances) < 0.15:
        nontrivial = False
        reason = "latent cosine distance < 0.15"
    else:
        reason = "random progress outside [0.05, 0.95] on every seed" if not nontrivial else None
    filters["nontriviality"] = {"passed": nontrivial, "mean_progress": mean_progress,
                                 "min_latent_cosine_distance": min(distances) if distances else None,
                                 "reason": reason}
    if not nontrivial:
        return filters, None
    scripted_rows = [_run_policy(candidate.components, seed=3000 + seed, max_steps=3000,
                                 action_fn=_scripted_action()) for seed in range(5)]
    scripted_progress = float(np.mean([row["progress"] for row in scripted_rows]))
    solvable = 0.2 <= scripted_progress <= 0.7
    filters["solvability"] = {"passed": solvable, "mean_progress": scripted_progress,
                              "reason": None if solvable else "outside scripted [0.2, 0.7]"}
    if not solvable:
        return filters, None
    if robust_model is None or recurrent_model is None:
        filters["discriminative_power"] = {
            "passed": False, "reason": "robust and recurrent model paths are both required",
        }
        return filters, None
    robust, recurrent = _load_agents(robust_model, recurrent_model)
    # Compare exactly the same ten worlds.  Different seed ranges made the
    # old signed comparison measure terrain luck rather than policy behavior.
    evaluation_seeds = range(4000, 4010)
    robust_rows = _agent_trial_rows(robust, candidate.components, evaluation_seeds)
    recurrent_rows = _agent_trial_rows(recurrent, candidate.components, evaluation_seeds)
    robust_reward = float(np.mean([row["reward"] for row in robust_rows]))
    recurrent_reward = float(np.mean([row["reward"] for row in recurrent_rows]))
    signed_gap = (recurrent_reward - robust_reward) / max(recurrent_reward, 1.0e-6)
    gap = signed_gap
    discriminative = recurrent_reward > 0.0 and gap > 0.30
    filters["discriminative_power"] = {"passed": discriminative, "robust_reward": robust_reward,
                                        "recurrent_reward": recurrent_reward, "gap": gap,
                                        "signed_gap": signed_gap,
                                        "reason": None if discriminative else "gap <= 0.30"}
    if not discriminative:
        return filters, None
    feature = np.asarray(list(mean_latents) + list(np.mean([row["latents_var"] for row in scripted_rows], axis=0)) +
                         [robust_reward, recurrent_reward], dtype=float)
    accepted_features = [item.get("characteristics", {}).get("feature_vector") for item in accepted]
    l2 = [_normalised_l2(feature, np.asarray(vector, dtype=float))
          for vector in accepted_features if vector is not None]
    unique = not l2 or min(l2) >= 0.2
    filters["uniqueness"] = {"passed": unique, "nearest_l2": min(l2) if l2 else None,
                             "reason": None if unique else "normalised L2 < 0.2"}
    if not unique:
        return filters, None
    record = {
        "name": "stack_" + candidate.name,
        "components": list(candidate.components),
        "split": _split_for(candidate, accepted),
        "characteristics": {
            "mean_traction": float(mean_latents[0]), "mean_gravity_mul": float(mean_latents[1]),
            "mean_viscosity": float(mean_latents[2]), "mean_temperature": float(mean_latents[3]),
            "variance_traction": float(np.mean([row["latents_var"][0] for row in scripted_rows])),
            "reward_robust": robust_reward, "reward_recurrent": recurrent_reward,
            "discriminative_gap": gap, "signed_discriminative_gap": signed_gap,
            "latent_mean": mean_latents.tolist(),
            "feature_vector": feature.tolist(),
        },
    }
    return filters, record


def _write_summary(root: Path, candidates: list[dict[str, Any]], accepted: list[dict[str, Any]]) -> None:
    all_records = _prior_candidate_records(root)
    counts = Counter(row["rejection_stage"] for row in all_records if row["verdict"] == "rejected")
    accepted_names = {row["name"] for row in accepted}
    active = [row for row in candidates if row.get("accepted_name") in accepted_names]
    coverage = {part: (sum(part in row["components"] and row["split"] == "train" for row in accepted),
                       sum(part in row["components"] and row["split"] == "test" for row in accepted))
                for part in COMPONENTS}
    near = sorted((row for row in candidates if row["verdict"] == "rejected"),
                  key=lambda row: float(row.get("boundary_distance", float("inf"))))[:20]
    lines = ["# Stack bank growth", "", "## Statistics", "",
             f"- Batches processed: {len({row['batch'] for row in all_records})}",
             f"- Candidates processed: {len(all_records)}", f"- Accepted in this run: {len(active)}",
             f"- Current bank: {sum(row['split'] == 'train' for row in accepted)} train, "
             f"{sum(row['split'] == 'test' for row in accepted)} test", "",
             "## Accepted stacks", "", "| Name | Components | Split | Gap |", "|---|---|---|---:|"]
    lines += [f"| [{row['name']}]({row.get('code_path', '#')}) | {', '.join(row['components'])} | {row['split']} | "
              f"{row['characteristics'].get('discriminative_gap', 0.0):.3f} |" for row in accepted]
    rejected = max(1, sum(counts.values()))
    lines += ["", "## Rejections", "", *[
        f"- filter {stage}: {count} ({count / rejected:.1%} of rejected candidates)"
        for stage, count in sorted(counts.items())
    ],
              "", "## Near-boundary rejections", "", "| Candidate | Stage | Reason |", "|---|---:|---|"]
    lines += [f"| {row['id']} | {row['rejection_stage']} | {row['rejection_reason']} |" for row in near]
    lines += ["", "## Mechanic coverage", "", "| Mechanic | Train | Test |", "|---|---:|---:|"]
    lines += [f"| {part} | {train} | {test} |" for part, (train, test) in coverage.items()]
    exhausted = len(all_records) >= 500 and all(row.get("verdict") != "accepted" for row in all_records[-50:])
    lines += ["", "## Open questions", ""]
    if exhausted:
        lines.append("- Candidate exhaustion reached: 500 candidates were screened and the final 50 had no full pass. "
                     "The recurrent-vs-robust signed gate, not missing checkpoints, is the dominant limiting condition.")
    else:
        lines.append("- Keep the current frozen checkpoint pair for the next batch so filter-4 comparisons remain reproducible.")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prior_candidate_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((root / "candidates").glob("batch_*/*.json")) if (root / "candidates").is_dir() else ():
        if path.name == "llm_response.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "verdict" in payload:
            records.append(payload)
    return records


def _stop_reason(root: Path, manifest: dict[str, Any]) -> str | None:
    stacks = manifest.get("frozen_stacks", [])
    train = sum(stack.get("split") == "train" for stack in stacks)
    test = sum(stack.get("split") == "held_out" for stack in stacks)
    if train + test >= 30 and train >= 20 and test >= 10:
        return "target reached: at least 30 total stacks, including 20 train and 10 test"
    records = _prior_candidate_records(root)
    if len(records) >= 500 and all(row.get("verdict") != "accepted" for row in records[-50:]):
        return "candidate exhaustion: 500 processed and the last 50 had no full pass"
    return None


def _render_frozen_stacks(stacks: list[dict[str, Any]]) -> str:
    lines = [f"inline constexpr std::array<FrozenMechanismStack, {len(stacks)}> kFrozenMechanismStacks{{{{"]
    for stack in stacks:
        parts = list(stack["mechanisms"])
        padded = [CPP_TYPES[item] for item in parts] + ["Normal"] * (4 - len(parts))
        split = {"anchor": "Builtin", "train": "Train", "held_out": "Test"}[stack["split"]]
        lines.append(
            f"    {{{{MechanicType::{padded[0]}, MechanicType::{padded[1]}, MechanicType::{padded[2]}, "
            f"MechanicType::{padded[3]}}}, {len(parts)}, BiomeSplit::{split}}},"
        )
    return "\n".join(lines) + "\n}};"


def _apply_accepted_stacks(
    manifest_path: Path, accepted: list[dict[str, Any]], *, header_path: Path | None = None,
) -> str:
    manifest = load_manifest(manifest_path)
    stacks = list(manifest.get("frozen_stacks", []))
    stacks.extend({
        "mechanisms": list(record["components"]),
        "split": "held_out" if record["split"] == "test" else "train",
    } for record in accepted)
    canonical = json.dumps(stacks, sort_keys=True, separators=(",", ":"))
    bank_version = "sha256:" + _sha256(canonical)
    header = header_path or (Path(__file__).resolve().parents[3] / "cpp" / "include" / "mars" / "biome_bank.hpp")
    source = header.read_text(encoding="utf-8")
    rendered = _render_frozen_stacks(stacks)
    source, replacements = re.subn(
        r"inline constexpr std::array<FrozenMechanismStack, \d+> kFrozenMechanismStacks\{\{.*?\n\}\};",
        rendered, source, count=1, flags=re.DOTALL,
    )
    if replacements != 1:
        raise RuntimeError("could not locate kFrozenMechanismStacks in biome_bank.hpp")
    source, replacements = re.subn(
        r'inline constexpr std::string_view kBiomeBankVersion = "sha256:[0-9a-f]+";',
        f'inline constexpr std::string_view kBiomeBankVersion = "{bank_version}";', source, count=1,
    )
    if replacements != 1:
        raise RuntimeError("could not locate kBiomeBankVersion in biome_bank.hpp")
    header.write_text(source, encoding="utf-8")
    manifest["frozen_stacks"] = stacks
    manifest["bank_version"] = bank_version
    manifest["train_version"] = "sha256:" + _sha256(json.dumps(
        [stack for stack in stacks if stack["split"] == "train"], sort_keys=True, separators=(",", ":")
    ))
    manifest["test_version"] = "sha256:" + _sha256(json.dumps(
        [stack for stack in stacks if stack["split"] == "held_out"], sort_keys=True, separators=(",", ":")
    ))
    # Every model provenance refers to the old distribution and cannot be
    # reused after a bank expansion.
    manifest.pop("reference_version", None)
    manifest.pop("anchor_references", None)
    manifest.pop("difficulty_gates", None)
    write_json(manifest_path, manifest)
    return bank_version


def _commit_batch(root: Path, batch: int, manifest: Path, *, bank_changed: bool) -> None:
    repository = Path(__file__).resolve().parents[4]
    paths = [
        root / "candidates" / f"batch_{batch:03d}",
        root / "snapshots" / f"snapshot_batch_{batch:03d}.json",
        root / "logs" / f"batch_{batch:03d}_log.txt",
        root / "summary.md",
    ]
    if bank_changed:
        paths.extend([
            root / "accepted",
            manifest,
            repository / "environment/cpp/include/mars/biome_bank.hpp",
        ])
    relative_paths = [str(path.resolve().relative_to(repository)) for path in paths if path.exists()]
    stage = subprocess.run(["git", "add", "-f", "--", *relative_paths], cwd=repository, check=False)
    if stage.returncode != 0:
        raise RuntimeError("could not stage stack-bank batch artifacts")
    accepted_count = len(json.loads(
        (root / "snapshots" / f"snapshot_batch_{batch:03d}.json").read_text(encoding="utf-8")
    )["accepted_in_batch"])
    commit = subprocess.run(
        ["git", "commit", "-m", f"stack bank growth: batch {batch}, accepted {accepted_count}"],
        cwd=repository, text=True, capture_output=True, check=False,
    )
    if commit.returncode != 0:
        raise RuntimeError((commit.stderr or commit.stdout).strip() or "could not commit stack-bank batch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and filter one auditable mechanic-stack batch")
    parser.add_argument("--count", type=int, default=50, choices=range(1, 51))
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/stack_bank_growth"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-json", type=Path, help="saved OpenAI response or {candidates: [...]} replay")
    parser.add_argument("--proposal-source", choices=("curated", "openai"), default="curated")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--robust-model", type=Path)
    parser.add_argument("--recurrent-model", type=Path)
    parser.add_argument("--apply", action="store_true", help="append fully accepted stacks to the frozen bank")
    parser.add_argument("--commit", action="store_true", help="commit only this batch's artifacts and bank files")
    args = parser.parse_args()
    if (args.robust_model is None) != (args.recurrent_model is None):
        raise SystemExit("provide both --robust-model and --recurrent-model")
    if args.robust_model is None:
        raise SystemExit("full filtering requires --robust-model and --recurrent-model; no candidates were generated")
    missing_models = [str(path) for path in (args.robust_model, args.recurrent_model) if not path.is_file()]
    if missing_models:
        raise SystemExit("baseline model file(s) not found: " + ", ".join(missing_models))
    try:
        import stable_baselines3  # noqa: F401
        import torch  # noqa: F401
        import mars_rover_agents  # noqa: F401
    except ImportError as exc:
        raise SystemExit("baseline dependencies are unavailable; install baselines[train]") from exc
    manifest = load_manifest(args.manifest)
    stop = _stop_reason(args.root, manifest)
    if stop:
        raise SystemExit("bank-growth stop condition met: " + stop)
    if args.input_json:
        raw = json.loads(args.input_json.read_text(encoding="utf-8"))
        raw_candidates = raw.get("candidates", raw)
        response = {"source": str(args.input_json), "replayed_at": _now()}
    elif args.proposal_source == "curated":
        raw_candidates, response = curated_candidates(args.count, batch=args.batch)
    else:
        raw_candidates, response = request_candidates(args.count, args.model)
    if not isinstance(raw_candidates, list):
        raise SystemExit("candidate input must be a list or an object with candidates")
    root = args.root
    (root / "snapshots").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    accepted = _existing_stack_records(manifest)
    accepted_in_batch: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    (root / "candidates" / f"batch_{args.batch:03d}").mkdir(parents=True, exist_ok=True)
    write_json(root / "candidates" / f"batch_{args.batch:03d}" / "llm_response.json", response)
    existing = {tuple(stack["mechanisms"]) for stack in manifest.get("frozen_stacks", [])}
    for ordinal, payload in enumerate(raw_candidates[:args.count], start=1):
        candidate_id = f"candidate_{ordinal:03d}"
        cpp_path, metadata_path = _candidate_path(root, args.batch, ordinal)
        record: dict[str, Any] = {"id": candidate_id, "batch": args.batch, "generated_at": _now(),
                                  "components": [], "filters": {name: None for name in FILTER_NAMES}, "verdict": "rejected",
                                  "rejection_reason": "invalid", "rejection_stage": 1}
        try:
            candidate = Candidate.from_payload(payload)
            record["components"] = list(candidate.components)
            cpp = candidate.cpp()
            cpp_path.write_text(cpp, encoding="utf-8")
            record["code_hash"] = "sha256:" + _sha256(cpp)
            if candidate.components in existing:
                record["filters"]["validity"] = {"passed": False, "reason": "duplicate frozen stack"}
            else:
                filters, accepted_record = _filter_candidate(
                    candidate, cpp_path, accepted, robust_model=args.robust_model,
                    recurrent_model=args.recurrent_model,
                )
                record["filters"].update(filters)
                if accepted_record is not None:
                    accepted_record["included_at_batch"] = args.batch
                    accepted_record["sibling_candidates"] = [candidate_id]
                    accepted_record["code_path"] = f"accepted/{accepted_record['name']}.cpp"
                    accepted.append(accepted_record)
                    accepted_in_batch.append(accepted_record)
                    record.update({"verdict": "accepted", "rejection_reason": None, "rejection_stage": None,
                                   "accepted_name": accepted_record["name"]})
                    target_cpp = root / accepted_record["code_path"]
                    target_cpp.parent.mkdir(parents=True, exist_ok=True)
                    target_cpp.write_text(cpp, encoding="utf-8")
                    write_json(target_cpp.with_suffix(".json"), accepted_record)
                    existing.add(candidate.components)
                else:
                    stage = next((index + 1 for index, name in enumerate(FILTER_NAMES)
                                  if not filters.get(name, {}).get("passed", False)), 5)
                    failed = FILTER_NAMES[stage - 1]
                    record["rejection_stage"] = stage
                    record["rejection_reason"] = failed
                    value = filters.get(failed, {}).get("gap", filters.get(failed, {}).get("nearest_l2"))
                    if value is not None:
                        record["boundary_distance"] = abs(float(value) - (0.30 if failed == "discriminative_power" else 0.20))
        except (TypeError, ValueError, RuntimeError) as exc:
            record["filters"]["validity"] = {"passed": False, "reason": str(exc)}
        write_json(metadata_path, record)
        candidate_records.append(record)
    write_json(root / "snapshots" / f"snapshot_batch_{args.batch:03d}.json", {
        "batch": args.batch, "created_at": _now(), "base_bank_version": manifest.get("bank_version"),
        "accepted_in_batch": accepted_in_batch, "candidates": candidate_records,
    })
    (root / "logs" / f"batch_{args.batch:03d}_log.txt").write_text(
        "\n".join(f"{row['id']}: {row['verdict']} {row.get('rejection_reason') or row.get('accepted_name')}"
                  for row in candidate_records) + "\n", encoding="utf-8")
    _write_summary(root, candidate_records, accepted)
    bank_version = None
    if args.apply:
        if not accepted_in_batch:
            raise SystemExit("--apply requested, but no candidate passed all five filters")
        bank_version = _apply_accepted_stacks(args.manifest, accepted_in_batch)
    if args.commit:
        _commit_batch(args.root, args.batch, args.manifest, bank_changed=bank_version is not None)
    print(json.dumps({"batch": args.batch, "processed": len(candidate_records), "accepted": len(accepted_in_batch),
                      "bank_version": bank_version}, indent=2))


if __name__ == "__main__":
    main()
