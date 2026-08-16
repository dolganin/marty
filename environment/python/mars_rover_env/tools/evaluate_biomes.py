from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from mars_rover_env import MarsRoverEnv
from mars_rover_env.actions import ACTION_MACROS
from mars_rover_env.config import DEFAULT_ENV_CONFIG
from mars_rover_env.tools.generate_biomes import fingerprint_of_compiled_biome
from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    load_manifest,
    require_compiled_bank,
    write_json,
)


Policy = Callable[[np.ndarray, dict, int], int]
GATE_PROTOCOL_VERSION = "rollout-v2-public-macro-random"


def policy_provenance(
    policy_name: str,
    *,
    model_path: str | Path | None = None,
    callable_module=None,
    callable_target: str | None = None,
) -> dict:
    if policy_name == "random":
        return {
            "kind": "random",
            "numpy_version": np.__version__,
            "action_macros": list(ACTION_MACROS),
        }
    if policy_name == "scripted":
        source = Path(__file__).read_bytes()
        return {"kind": "scripted", "source_sha256": hashlib.sha256(source).hexdigest()}
    if policy_name == "ppo":
        if model_path is None:
            raise ValueError("PPO provenance requires a model path")
        path = Path(model_path)
        archive = path if path.suffix == ".zip" else path.with_suffix(".zip")
        if not archive.is_file():
            raise FileNotFoundError(archive)
        return {
            "kind": "ppo",
            "model_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        }
    if callable_module is None or callable_target is None:
        raise ValueError("Callable provenance requires its imported module")
    module_path = Path(callable_module.__file__)
    return {
        "kind": "callable",
        "target": callable_target,
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    }


@dataclass
class RolloutSummary:
    mean_return: float
    std_return: float
    mean_score: float
    success_rate: float
    episodes: int
    trials: int


def random_policy(rng: np.random.Generator) -> Policy:
    def act(_obs: np.ndarray, _debug: dict, _step: int) -> int:
        return int(ACTION_MACROS[rng.integers(0, len(ACTION_MACROS))])

    return act


def privileged_gate_oracle_policy(_obs: np.ndarray, debug: dict, step: int) -> int:
    """Internal solvability oracle; never exposed as a comparable baseline."""
    if not debug.get("engine_running"):
        return 1 << 9
                                                                                      
                                                                                          
                                                                                      
                                                                                           
                                                                                     
    hazard = int(debug.get("hazard", 0))
    if hazard == 3:
                                                                                            
                                                                                         
        if float(debug.get("energy", 0.0)) < 12.0:
            return (1 << 1) if float(debug.get("speed", 0.0)) > 0.25 else 0
    if hazard == 1 or int(debug.get("hazard_ahead", 0)) == 1:
                                                                                          
                               
        return (1 << 1) if float(debug.get("speed", 0.0)) > 0.25 else 0
    gear = debug.get("gear", "N")
    if gear == "N" or (isinstance(gear, (int, float)) and gear <= 0):
        return (1 << 3) | (1 << 6)
    if debug.get("should_shift_down") and debug.get("shift_cooldown", 0.0) <= 0.0:
        return (1 << 3) | (1 << 7)
    if debug.get("upshift_recommended") and debug.get("shift_cooldown", 0.0) <= 0.0:
        return (1 << 3) | (1 << 6)
    action = 1
    if abs(float(debug.get("angle", 0.0))) > 0.35:
        action |= 1 << (5 if debug["angle"] > 0 else 4)
    if step % 240 == 0:
        action |= 1 << 11
    return action


def model_policy(model_path: str | Path) -> Policy:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:                    
        raise RuntimeError("Install the benchmark extra to evaluate robust-PPO") from exc
    model_path = Path(model_path)
    action_macros = None
    artifact = model_path.parent / "artifact.json"
    if artifact.is_file():
        artifact_data = json.loads(artifact.read_text(encoding="utf-8"))
        archive_path = (
            model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
        )
        expected_hash = artifact_data.get("model_sha256")
        if expected_hash is not None:
            if not archive_path.is_file():
                raise RuntimeError(f"Model archive is missing: {archive_path}")
            actual_hash = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError("Model archive hash does not match artifact metadata")
        action_macros = artifact_data.get("action_macros")
        environment_version = artifact_data.get("environment_version")
        if environment_version is not None:
            import _mars_rover_cpp as native

            if environment_version != native.environment_version():
                raise RuntimeError("Model artifact targets a different environment version")
    model = PPO.load(str(model_path))

    def act(obs: np.ndarray, _debug: dict, _step: int) -> int:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        return int(action_macros[action]) if action_macros is not None else action

    return act


def evaluate_policy(
    biome_id: int,
    policy_factory: Callable[[int], Policy],
    seeds: list[int],
    max_steps: int,
    config_path: str | None = None,
    *,
    _privileged_gate_oracle: bool = False,
) -> RolloutSummary:
    returns: list[float] = []
    scores: list[float] = []
    successes = 0
    for seed in seeds:
        env = MarsRoverEnv(
            config_path=config_path,
            biome_split=MarsRoverEnv.BIOME_MODE_ALL,
            fixed_biome_id=biome_id,
        )
        obs, _ = env.reset(seed=seed, options={"trial_start": True})
        policy = policy_factory(seed)
        episode_return = 0.0
        terminated = truncated = False
        for step in range(max_steps):
            evaluator_state = env.debug_info() if _privileged_gate_oracle else {}
            action = policy(obs, evaluator_state, step)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        debug = env.debug_info()
        success = float(debug.get("x", 0.0)) >= float(env._config.termination.finish_x)
        successes += int(success)
        scale = max(
            1.0,
            float(env._config.termination.finish_x) + float(env._config.reward.finish_bonus),
        )
        returns.append(episode_return)
        scores.append(float(np.clip(episode_return / scale, 0.0, 1.0)))
        env.close()
    return RolloutSummary(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_score=float(np.mean(scores)),
        success_rate=successes / len(seeds),
        episodes=len(seeds),
        trials=len(seeds),
    )


def evaluate_adaptive_policy(
    biome_id: int,
    policy_factory: Callable[[int], Policy],
    trial_seeds: list[int],
    episodes_per_trial: int,
    max_steps: int,
    config_path: str | None = None,
) -> RolloutSummary:
    """Evaluate full trials while preserving policy and hidden biome state across episodes."""
    if episodes_per_trial < 1:
        raise ValueError("episodes_per_trial must be positive")
    returns: list[float] = []
    scores: list[float] = []
    successes = 0
    for trial_seed in trial_seeds:
        env = MarsRoverEnv(
            config_path=config_path,
            biome_split=MarsRoverEnv.BIOME_MODE_ALL,
            fixed_biome_id=biome_id,
        )
        policy = policy_factory(trial_seed)
        for episode in range(episodes_per_trial):
            episode_seed = trial_seed + episode * 1_000_003
            obs, _ = env.reset(
                seed=episode_seed,
                options={"trial_start": episode == 0},
            )
            episode_return = 0.0
            for step in range(max_steps):
                action = policy(obs, {}, step)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            debug = env.debug_info()
            success = float(debug.get("x", 0.0)) >= float(env._config.termination.finish_x)
            successes += int(success)
            scale = max(
                1.0,
                float(env._config.termination.finish_x)
                + float(env._config.reward.finish_bonus),
            )
            returns.append(episode_return)
            scores.append(float(np.clip(episode_return / scale, 0.0, 1.0)))
        env.close()
    return RolloutSummary(
        mean_return=float(np.mean(returns)),
        std_return=float(np.std(returns)),
        mean_score=float(np.mean(scores)),
        success_rate=successes / len(returns),
        episodes=len(returns),
        trials=len(trial_seeds),
    )


def _catalog_by_id() -> dict[str, dict]:
    import _mars_rover_cpp as native

    return {str(item["id"]): dict(item) for item in native.biome_catalog()}


def solvability_witness(
    biome_index: int,
    profile: dict[str, float],
    oracle: RolloutSummary,
    seeds: list[int],
    max_steps: int,
    config_path,
) -> RolloutSummary:
    """Best of the reactive oracle and the best fixed style, as the solvability witness.

    Solvability is an existence claim, so the honest estimator is a maximum over witnesses,
    not the score of one hand-written controller. It matters here: on the scheduled-collapse
    biomes the reactive oracle scores 0.049-0.072 because it reacts to the window instead of
    anticipating it, while a policy that simply stops on the right rhythm scores three times
    higher. Judging solvability by the oracle alone would reject the biomes as impossible
    when what is actually impossible is crossing them WITHOUT knowing the schedule - which is
    the entire property the benchmark is built to reward.
    """
    from mars_rover_env.tools.policy_divergence import _strategies

    best = oracle
                                                                                          
                                                                                         
                                                                                 
                                                                                            
                                                               
    for gear in (2, 3, 5, 7):
        geared = evaluate_policy(
            biome_index,
            lambda _seed, g=gear: _geared_privileged_oracle(g),
            seeds,
            max_steps,
            config_path,
            _privileged_gate_oracle=True,
        )
        if geared.mean_return > best.mean_return:
            best = geared

    best_style = max(profile, key=profile.get)
    if profile[best_style] > best.mean_return:
        scripted = _strategies()[best_style]
        styled = evaluate_policy(
            biome_index,
            lambda _seed: (lambda _obs, _debug, step: int(scripted(step))),
            seeds,
            max_steps,
            config_path,
        )
        if styled.mean_return > best.mean_return:
            best = styled
    return best


def _geared_privileged_oracle(target_gear: int) -> Policy:
    """The privileged hazard logic, driving in a fixed gear instead of following advice."""
    shifts = frozenset(12 + 40 * i for i in range(target_gear - 1))

    def act(obs: np.ndarray, debug: dict, step: int) -> int:
        if not debug.get("engine_running"):
            return 1 << 9
        hazard = int(debug.get("hazard", 0))
        slow = float(debug.get("speed", 0.0)) > 0.25
        if hazard == 3 and float(debug.get("energy", 0.0)) < 12.0:
            return (1 << 1) if slow else 0
        if hazard == 1 or int(debug.get("hazard_ahead", 0)) == 1:
            return (1 << 1) if slow else 0
        if step in shifts:
            return (1 << 3) | (1 << 6)
        action = 1
        if abs(float(debug.get("angle", 0.0))) > 0.35:
            action |= 1 << (5 if debug["angle"] > 0 else 4)
        return action

    return act


def strategy_profile(biome_index: int, seeds: list[int], max_steps: int, config_path):
    """Score a fixed repertoire of driving styles on one biome.

    A biome earns its place only by changing WHICH behaviour wins. Measured on the previous
    bank, one style (high-gear cruise) won all ten held-out biomes and adaptation headroom was
    0.00 — knowing the biome was worth nothing, so memory could not pay and no agent comparison
    on that bank meant anything. This makes that property a gate instead of a hope.
    """
    from mars_rover_env.tools.policy_divergence import _strategies, score_strategy

    return {
        label: score_strategy(policy, biome_index, seeds, max_steps, config_path)
        for label, policy in _strategies().items()
    }


def strategy_spread(profile: dict[str, float]) -> float:
    """How much the best style beats the median one, relative to the best.

    Reported for diagnosis, but NOT an acceptance condition, because it inverts its own
    meaning in the regime that matters. It is defined against the best score in the probe
    repertoire, so when no probe has a workable answer and every score is negative it
    collapses to 0 — and it collapses hardest on the biomes where a memoryless policy fails
    completely. It rejected collapse_window_scarp, on which memoryless robust scores 0.000
    while a matched rhythm reaches 0.085, and speed_band_talus, where robust gets 25% of what
    is achievable. Both are exactly what the held-out split is for.

    strategy_regret subsumes it and is measured against what is ACHIEVABLE rather than
    against what the probes manage, so it does not degenerate. Replayed over the gen5 bank,
    dropping this condition changes nothing: regret alone still accepts 2 biomes of 15.
    """
    values = sorted(profile.values(), reverse=True)
    best = values[0]
    median = values[len(values) // 2]
    if best <= 0.0:
        return 0.0
    return float((best - median) / abs(best))



def _handwritten_class_name(biome_id: str) -> str:
    """collapse_window_flats -> CollapseWindowFlats, matching the header's naming."""
    return "".join(part.capitalize() for part in biome_id.split("_"))


def strategy_regret(
    profile: dict[str, float],
    peers: list[dict[str, float]],
    achievable: float | None = None,
) -> float:
    """What the bank's current best single reflex gives up on this biome, relative to its best.

    This is the quantity the readiness gate is actually about, measured per candidate instead
    of hoped for bank-wide. Rank disagreement proved too weak a proxy: a biome can reshuffle
    the losing styles enough to look distinct while the SAME style still wins it, and a bank
    admitted that way was still swept by one reflex (gear 2 won 10 of 15 and captured 89% of
    the oracle). A biome earns its place only if the reflex that is currently best across the
    accepted biomes performs materially WORSE here than this biome's own best style.

    Returns 1.0 when no peers exist yet, else (best - incumbent) / |best|.
    """
    if not peers:
        return 1.0
    labels = sorted(profile)
    incumbent = max(labels, key=lambda k: sum(p.get(k, 0.0) for p in peers))
                                                                                            
                                                                                              
                                                                                             
                                                                                             
                                                                                        
    best = profile[max(labels, key=lambda k: profile[k])] if achievable is None else achievable
    if best <= 0.0:
        return 0.0
    return float((best - profile[incumbent]) / abs(best))


def strategy_disagreement(profile: dict[str, float], peers: list[dict[str, float]]) -> float:
    """How differently this biome ranks the driving styles compared to already-accepted ones.

    Measured directly, because the first version of this gate got it wrong: on the previous bank
    every biome had a healthy internal spread (0.40-1.03) yet the SAME style won all ten, so
    adaptation headroom was 0.00. What earns a place in the bank is disagreeing with the others
    about which behaviour is best, not merely being decisive about it.

    Returns 1.0 when no peers exist yet (the first biome cannot disagree with anything), else
    one minus the mean rank correlation with the peers, so higher means more distinct.
    """
    if not peers:
        return 1.0
    labels = sorted(profile)
    order = [sorted(labels, key=lambda k: p[k], reverse=True) for p in [profile] + peers]
    ranks = [{label: position for position, label in enumerate(o)} for o in order]
    mine = ranks[0]
    correlations = []
    for other in ranks[1:]:
        a = np.array([mine[label] for label in labels], dtype=float)
        b = np.array([other[label] for label in labels], dtype=float)
        if a.std() == 0 or b.std() == 0:
            correlations.append(1.0)
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(1.0 - max(correlations))


def gate_bank(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    require_compiled_bank(manifest)
    gate_config_path = Path(args.config) if args.config else DEFAULT_ENV_CONFIG
    gate_config_sha256 = hashlib.sha256(gate_config_path.read_bytes()).hexdigest()
    catalog = _catalog_by_id()
    seeds = list(range(args.seed, args.seed + args.episodes))
    model_factory = None
    if args.split == "test":
        if not args.robust_model:
            raise SystemExit("--robust-model is required for the test difficulty gate")
        model_path = Path(args.robust_model)
        archive_path = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
        if not archive_path.is_file():
            raise SystemExit(f"Missing frozen robust-PPO model: {archive_path}")
        artifact_path = model_path.parent / "artifact.json"
        if not artifact_path.is_file():
            raise SystemExit(f"Missing frozen robust-PPO metadata: {artifact_path}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if artifact.get("artifact_type") != "robust_ppo":
            raise SystemExit("Test gate requires a versioned robust_ppo artifact")
        model_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if artifact.get("model_sha256") != model_sha256:
            raise SystemExit("Robust-PPO model hash does not match artifact metadata")
        if artifact.get("reference_version") != "sha256:" + model_sha256:
            raise SystemExit("Robust-PPO reference_version does not match its model hash")
        if artifact.get("train_version") != manifest.get("train_version"):
            raise SystemExit("Robust-PPO was trained against a different train_version")
        if (
            set(artifact.get("trained_splits", [])) != {"train"}
            or not {"anchor", "test"}.issubset(set(artifact.get("excluded_splits", [])))
        ):
            raise SystemExit("Robust-PPO metadata does not prove train-only training")
        if artifact.get("reference_version") != manifest.get("reference_version"):
            raise SystemExit("Manifest and robust-PPO reference_version disagree")
        import _mars_rover_cpp as native

        if artifact.get("environment_version") != native.environment_version():
            raise SystemExit("Robust-PPO was trained with a different environment version")
        if artifact.get("config_sha256") != gate_config_sha256:
            raise SystemExit("Robust-PPO was trained with a different environment config")
        robust = model_policy(args.robust_model)
        model_factory = lambda _seed: robust
                                                                                     
                                                                                      
                                                                                          
    split_code = {"train": 1, "test": 2}.get(args.split)
    known = {str(item["id"]) for item in manifest.setdefault("biomes", [])}
    by_id = {str(item["id"]): item for item in manifest["biomes"]}
    for biome_id, biome in catalog.items():
        if int(biome["split"]) != split_code:
            continue
        if biome_id in known:
                                                                                       
                                                                                 
            row = by_id[biome_id]
            if row.get("origin") == "handwritten" and not row.get("behavior_fingerprint"):
                row["behavior_fingerprint"] = fingerprint_of_compiled_biome(
                    "handcrafted_biomes::" + _handwritten_class_name(biome_id)
                )
                print(f"backfilled fingerprint for {biome_id}")
            continue
        manifest["biomes"].append(
            {
                "id": biome_id,
                "split": args.split,
                "origin": "handwritten",
                "display_name": biome.get("display_name", biome_id),
                                                                                            
                                                                                         
                "skill_stratum": biome.get("skill_stratum", ""),
                                                                                      
                                                                                            
                "behavior_fingerprint": fingerprint_of_compiled_biome(
                    "handcrafted_biomes::" + _handwritten_class_name(biome_id)
                ),
            }
        )
        print(f"adopted handwritten biome into the manifest: {biome_id}")

    failed: list[str] = []
    accepted_profiles: list[dict[str, float]] = []
    for item in manifest.get("biomes", []):
        if item.get("split") != args.split:
            continue
        biome = catalog.get(str(item["id"]))
        if not biome:
            raise SystemExit(f"Biome {item['id']!r} is not present in the compiled native catalog")
        biome_index = int(biome["index"])
        random_result = evaluate_policy(
            biome_index,
            lambda seed: random_policy(np.random.default_rng(seed)),
            seeds,
            args.max_steps,
            args.config,
        )
        item["r_random"] = random_result.mean_return
        item["random_score"] = random_result.mean_score
        if args.split == "train":
            solve_result = evaluate_policy(
                biome_index,
                lambda _seed: privileged_gate_oracle_policy,
                seeds,
                args.max_steps,
                args.config,
                _privileged_gate_oracle=True,
            )
            profile = strategy_profile(biome_index, seeds[:3], min(args.max_steps, 1200), args.config)
            solve_result = solvability_witness(
                biome_index, profile, solve_result, seeds, args.max_steps, args.config
            )
            item["r_solve"] = solve_result.mean_return
            item["solve_score"] = solve_result.mean_score
            item["strategy_profile"] = profile
            item["strategy_winner"] = max(profile, key=profile.get)
            item["strategy_spread"] = strategy_spread(profile)
            item["strategy_disagreement"] = strategy_disagreement(profile, accepted_profiles)
            item["strategy_regret"] = strategy_regret(
                profile, accepted_profiles, solve_result.mean_return
            )
            accepted = (
                random_result.mean_score < args.tau_low
                and solve_result.mean_score > args.solve_min
                and item["strategy_regret"] > args.min_strategy_regret
            )
        else:
            assert model_factory is not None
            robust_result = evaluate_policy(
                biome_index, model_factory, seeds, args.max_steps, args.config
            )
            item["r_robust"] = robust_result.mean_return
            item["robust_score"] = robust_result.mean_score
                                                                                        
                                                                                         
                                                                                        
                                                                                        
                                                                                       
                                                                    
            solve_result = evaluate_policy(
                biome_index,
                lambda _seed: privileged_gate_oracle_policy,
                seeds,
                args.max_steps,
                args.config,
                _privileged_gate_oracle=True,
            )
            profile = strategy_profile(biome_index, seeds[:3], min(args.max_steps, 1200), args.config)
            solve_result = solvability_witness(
                biome_index, profile, solve_result, seeds, args.max_steps, args.config
            )
            item["r_solve"] = solve_result.mean_return
            item["solve_score"] = solve_result.mean_score
            item["strategy_profile"] = profile
            item["strategy_winner"] = max(profile, key=profile.get)
            item["strategy_spread"] = strategy_spread(profile)
            item["strategy_disagreement"] = strategy_disagreement(profile, accepted_profiles)
            item["strategy_regret"] = strategy_regret(
                profile, accepted_profiles, solve_result.mean_return
            )
            accepted = (
                random_result.mean_score < args.tau_low
                and solve_result.mean_score > args.solve_min
                and robust_result.mean_score < args.robust_max
                and item["strategy_regret"] > args.min_strategy_regret
                and solve_result.mean_return
                > random_result.mean_return + args.min_reference_gap
            )
                                                                                      
                                                                                          
                                                                                          
                      
        reasons: list[str] = []
        if not random_result.mean_score < args.tau_low:
            reasons.append("trivial_for_random")
        if not solve_result.mean_score > args.solve_min:
            reasons.append("unsolvable_by_oracle")
        if not item["strategy_regret"] > args.min_strategy_regret:
            reasons.append("best_reflex_of_the_bank_already_wins_here")
        if args.split == "test":
            if not robust_result.mean_score < args.robust_max:
                reasons.append("memoryless_robust_already_solves_it")
            if not solve_result.mean_return > random_result.mean_return + args.min_reference_gap:
                reasons.append("reference_gap_too_small")
        item["status"] = "accepted" if accepted else "rejected_difficulty"
        item["rejection_reasons"] = reasons
        if accepted and "strategy_profile" in item:
            accepted_profiles.append(item["strategy_profile"])
        print(item["id"], item["status"], ",".join(reasons), json.dumps({
            "random": asdict(random_result),
            "solve_or_robust": asdict(solve_result),
            "robust": asdict(robust_result) if args.split == "test" else None,
            "strategy_winner": item["strategy_winner"],
            "strategy_spread": item["strategy_spread"],
            "strategy_disagreement": item["strategy_disagreement"],
            "strategy_regret": item["strategy_regret"],
        }, sort_keys=True))
        if not accepted:
            failed.append(str(item["id"]))
    manifest.setdefault("difficulty_gates", {})[args.split] = {
        "protocol_version": GATE_PROTOCOL_VERSION,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "seed_begin": args.seed,
        "tau_low": args.tau_low,
        "solve_min": args.solve_min if args.split == "train" else None,
        "robust_max": args.robust_max if args.split == "test" else None,
        "min_reference_gap": args.min_reference_gap if args.split == "test" else None,
        "reference_version": (
            manifest.get("reference_version") if args.split == "test" else None
        ),
        "environment_version": __import__("_mars_rover_cpp").environment_version(),
        "config_sha256": gate_config_sha256,
        "random_action_macros": list(ACTION_MACROS),
    }
    write_json(args.manifest, manifest)
    if failed:
        raise SystemExit("Difficulty gate rejected: " + ", ".join(failed))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rollout difficulty gates for a compiled biome bank")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--config")
    parser.add_argument("--robust-model")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--tau-low", "--random-max", dest="tau_low", type=float, default=0.20)
    parser.add_argument("--solve-min", type=float, default=0.05)
                                                                                     
                                                                                       
                                                                              
    parser.add_argument("--robust-max", type=float, default=0.35)
    parser.add_argument("--min-reference-gap", type=float, default=1.0)
                                                                                  
    parser.add_argument("--min-strategy-spread", type=float, default=0.35)
                                                                                        
                                                                      
    parser.add_argument("--min-strategy-disagreement", type=float, default=0.30)
                                                                                       
                                                                                        
    parser.add_argument("--min-strategy-regret", type=float, default=0.25)
    return parser


def main() -> None:
    gate_bank(build_parser().parse_args())


if __name__ == "__main__":
    main()
