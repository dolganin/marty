"""Test 1: is the RL2 memory channel informative about the mechanic?

One episode's observation stream is replayed twice through the policy, differing
ONLY in the incoming Transformer-XL state:
  (a) the state actually accumulated over earlier episodes of this trial, on this biome;
  (b) a state accumulated on a DIFFERENT biome (swapped in).

Teacher forcing is essential: if we let each condition act, the trajectories diverge
and any difference confounds "memory changed the policy" with "the rover ended up
somewhere else". Replaying a fixed feature stream isolates the state's contribution.

Reading:
  same behaviour under (a) and (b)  -> memory does not encode the mechanic; channel dead.
  different behaviour               -> memory encodes it; a negative adaptation delta then
                                       means the policy applies that information harmfully.

The per-step breakdown matters as much as the average: Transformer-XL memory is a sliding
window of `memory_len` steps, so a contrast that decays to zero around t=memory_len is
evidence that cross-episode information is being evicted rather than ignored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mars_rover_env import MarsRoverEnv

from .actions import macro_action
from .rl2_model import RL2TransformerActorCritic, TransformerXLState, rl2_features
from .trial_context import EPISODE_SUMMARY_DIM, TrialContextTracker


def _tracker(model: RL2TransformerActorCritic, max_steps: int) -> TrialContextTracker | None:
    """Mirror the agent's trial-context bookkeeping so replayed features match rollout."""
    context_dim = int(getattr(model, "context_dim", 0))
    if not context_dim:
        return None
    history, remainder = divmod(context_dim, EPISODE_SUMMARY_DIM + 1)
    if remainder:
        raise ValueError(f"context_dim {context_dim} is not a whole number of summaries")
    return TrialContextTracker(1, history=history, max_steps=max_steps)


def _load_model(path: Path, device: torch.device) -> RL2TransformerActorCritic:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = RL2TransformerActorCritic(**payload["model_config"]).to(device)
    model.load_compatible_state(payload["model_state"])
    model.eval()
    return model


def _clone_state(state: TransformerXLState) -> TransformerXLState:
    return TransformerXLState(
        memories=[memory.clone() for memory in state.memories],
        valid=state.valid.clone(),
    )


def build_context(
    model: RL2TransformerActorCritic,
    device: torch.device,
    *,
    biome_index: int,
    seed: int,
    context_episodes: int,
    max_steps: int,
    config_path: str | None,
) -> tuple[TransformerXLState, dict[str, Any]]:
    """Run the first `context_episodes` episodes of a trial and return the carried state."""
    env = MarsRoverEnv(
        config_path=config_path,
        biome_split=MarsRoverEnv.BIOME_MODE_ALL,
        fixed_biome_id=biome_index,
    )
    state = model.initial_state(1, device)
    context = _tracker(model, max_steps)
    previous_action = -1
    previous_reward = 0.0
    previous_done = True
    returns: list[float] = []
    for episode_index in range(context_episodes):
        obs, _ = env.reset(
            seed=seed + episode_index * 1_000_003,
            options={"trial_start": episode_index == 0},
        )
        episode_return = 0.0
        for _ in range(max_steps):
            observation = np.asarray(obs, dtype=np.float32)
            features = rl2_features(
                observation[None, :],
                np.asarray([previous_action]),
                np.asarray([previous_reward]),
                np.asarray([previous_done]),
                device=device,
                trial_context=context.features() if context is not None else None,
            )
            with torch.no_grad():
                logits, _, state = model.forward_step(features, state)
                action_index = int(torch.argmax(logits, dim=-1).item())
            obs, reward, terminated, truncated, _ = env.step(macro_action(action_index))
            done = bool(terminated or truncated)
            if context is not None:
                context.observe(
                    observation[None, :], np.asarray([float(reward)]), np.asarray([done])
                )
            previous_action, previous_reward, previous_done = action_index, float(reward), done
            episode_return += float(reward)
            if done:
                break
        returns.append(episode_return)
    debug = env.debug_info()
    env.close()
                                                                                          
                                                                                         
    context_features = context.features().copy() if context is not None else None
    return (
        state,
        {
            "context_returns": returns,
            "final_x": float(debug.get("x", 0.0)),
            "context_features": context_features,
        },
    )


def record_probe_episode(
    model: RL2TransformerActorCritic,
    device: torch.device,
    *,
    biome_index: int,
    seed: int,
    episode_index: int,
    state: TransformerXLState,
    max_steps: int,
    config_path: str | None,
    context_features: np.ndarray | None = None,
) -> list[tuple[np.ndarray, int, float, bool]]:
    """Act with the genuine state and record the raw inputs needed to replay the episode."""
    env = MarsRoverEnv(
        config_path=config_path,
        biome_split=MarsRoverEnv.BIOME_MODE_ALL,
        fixed_biome_id=biome_index,
    )
    obs, _ = env.reset(
        seed=seed + episode_index * 1_000_003,
        options={"trial_start": False},
    )
    rolling = _clone_state(state)
    previous_action = -1
    previous_reward = 0.0
    previous_done = True
    recorded: list[tuple[np.ndarray, int, float, bool]] = []
    for _ in range(max_steps):
        observation = np.asarray(obs, dtype=np.float32)
        recorded.append((observation, previous_action, previous_reward, previous_done))
        features = rl2_features(
            observation[None, :],
            np.asarray([previous_action]),
            np.asarray([previous_reward]),
            np.asarray([previous_done]),
            device=device,
            trial_context=context_features,
        )
        with torch.no_grad():
            logits, _, rolling = model.forward_step(features, rolling)
            action_index = int(torch.argmax(logits, dim=-1).item())
        obs, reward, terminated, truncated, _ = env.step(macro_action(action_index))
        done = bool(terminated or truncated)
        previous_action, previous_reward, previous_done = action_index, float(reward), done
        if done:
            break
    env.close()
    return recorded


def build_feature_stream(
    recorded: list[tuple[np.ndarray, int, float, bool]],
    *,
    device: torch.device,
    context_features: np.ndarray | None,
) -> torch.Tensor:
    """Rebuild the feature stream, optionally under a different trial context."""
    observations = np.stack([row[0] for row in recorded], axis=0)
    return rl2_features(
        observations,
        np.asarray([row[1] for row in recorded]),
        np.asarray([row[2] for row in recorded]),
        np.asarray([row[3] for row in recorded]),
        device=device,
        trial_context=(
            np.repeat(context_features, len(recorded), axis=0)
            if context_features is not None
            else None
        ),
    )


def replay_logits(
    model: RL2TransformerActorCritic,
    state: TransformerXLState,
    features: torch.Tensor,
    *,
    chunk_length: int = 128,
) -> torch.Tensor:
    """Push a fixed feature stream through the model from a given state."""
    rolling = _clone_state(state)
    outputs: list[torch.Tensor] = []
    sequence = features.unsqueeze(0)
    with torch.no_grad():
        for start in range(0, sequence.shape[1], chunk_length):
            chunk = sequence[:, start : start + chunk_length]
            logits, _, rolling = model.forward_chunk(chunk, rolling, detach_memory=True)
            outputs.append(logits[0])
    return torch.cat(outputs, dim=0)


def compare(genuine: torch.Tensor, swapped: torch.Tensor) -> dict[str, Any]:
    genuine_log = torch.log_softmax(genuine, dim=-1)
    swapped_log = torch.log_softmax(swapped, dim=-1)
    genuine_probabilities = genuine_log.exp()
    kl = (genuine_probabilities * (genuine_log - swapped_log)).sum(dim=-1)
    agreement = (genuine.argmax(dim=-1) == swapped.argmax(dim=-1)).float()
    steps = kl.shape[0]
    buckets: list[dict[str, float]] = []
    for begin in range(0, steps, 64):
        end = min(steps, begin + 64)
        buckets.append(
            {
                "begin": begin,
                "end": end,
                "mean_kl": float(kl[begin:end].mean()),
                "argmax_agreement": float(agreement[begin:end].mean()),
            }
        )
    return {
        "steps": steps,
        "mean_kl": float(kl.mean()),
        "max_kl": float(kl.max()),
        "argmax_agreement": float(agreement.mean()),
        "by_step_bucket": buckets,
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import _mars_rover_cpp as native

    device = torch.device(args.device)
    catalog = [dict(item) for item in native.biome_catalog()]
    split_code = {"train": 1, "test": 2}[args.split]
    selected = [item for item in catalog if int(item["split"]) == split_code]
    if len(selected) < 2:
        raise SystemExit(f"need at least two {args.split} biomes to swap between")
    selected = selected[: args.biomes]

    results: dict[str, Any] = {
        "split": args.split,
        "context_episodes": args.context_episodes,
        "max_steps": args.max_steps,
        "seeds": list(range(args.seed, args.seed + args.trials)),
        "models": {},
    }
    for label, model_path in args.models:
        model = _load_model(Path(model_path), device)
        results["models"][label] = {
            "path": str(model_path),
            "memory_len": model.memory_len,
            "pairs": [],
        }
        for trial in range(args.trials):
            seed = args.seed + trial
            states: dict[str, TransformerXLState] = {}
            contexts: dict[str, np.ndarray | None] = {}
            for item in selected:
                state, info = build_context(
                    model,
                    device,
                    biome_index=int(item["index"]),
                    seed=seed,
                    context_episodes=args.context_episodes,
                    max_steps=args.max_steps,
                    config_path=args.config,
                )
                states[str(item["id"])] = state
                contexts[str(item["id"])] = info["context_features"]
            for index, item in enumerate(selected):
                biome_id = str(item["id"])
                other = selected[(index + 1) % len(selected)]
                other_id = str(other["id"])
                recorded = record_probe_episode(
                    model,
                    device,
                    biome_index=int(item["index"]),
                    seed=seed,
                    episode_index=args.context_episodes,
                    state=states[biome_id],
                    max_steps=args.max_steps,
                    config_path=args.config,
                    context_features=contexts[biome_id],
                )
                                                                                       
                                                                                       
                                                                                      
                genuine_features = build_feature_stream(
                    recorded, device=device, context_features=contexts[biome_id]
                )
                swapped_features = build_feature_stream(
                    recorded, device=device, context_features=contexts[other_id]
                )
                genuine = replay_logits(model, states[biome_id], genuine_features)
                swapped = replay_logits(model, states[other_id], swapped_features)
                comparison = compare(genuine, swapped)
                comparison.update(
                    {"seed": seed, "biome": biome_id, "swapped_from": other_id}
                )
                results["models"][label]["pairs"].append(comparison)
                print(
                    f"[{label}] seed={seed} {biome_id} <- {other_id}: "
                    f"steps={comparison['steps']} mean_kl={comparison['mean_kl']:.6f} "
                    f"agreement={comparison['argmax_agreement']:.4f}",
                    flush=True,
                )
        pairs = results["models"][label]["pairs"]
        results["models"][label]["summary"] = {
            "mean_kl": float(np.mean([item["mean_kl"] for item in pairs])),
            "argmax_agreement": float(np.mean([item["argmax_agreement"] for item in pairs])),
        }
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL2 memory informativeness probe (test 1)")
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "PATH"),
        dest="models",
        required=True,
        help="Labelled checkpoint; repeat to compare RL2 against its distilled control.",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--biomes", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=80_000)
    parser.add_argument("--context-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--config")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.output}")
    for label, payload in results["models"].items():
        summary = payload["summary"]
        print(
            f"{label}: memory_len={payload['memory_len']} "
            f"mean_kl={summary['mean_kl']:.6f} "
            f"argmax_agreement={summary['argmax_agreement']:.4f}"
        )


if __name__ == "__main__":
    main()
