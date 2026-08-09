from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    equated_score,
    load_manifest,
    normalized_test_return,
    write_json,
)
from mars_rover_env.tools.audit_bank import audit
from mars_rover_env.tools.evaluate_biomes import (
    evaluate_adaptive_policy,
    model_policy,
    policy_provenance,
    random_policy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score any agent on a frozen test bank")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument(
        "--policy", choices=("random", "ppo", "callable"), required=True
    )
    parser.add_argument("--model")
    parser.add_argument("--policy-factory")
    parser.add_argument("--config")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes-per-trial", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--equating", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import _mars_rover_cpp as native

    audit(args.manifest, require_reference=True, require_test_gate=True)
    manifest = load_manifest(args.manifest)
    bank_version = str(manifest["bank_version"])
    slug = bank_version.split(":", 1)[-1][:16]
    output = args.output or PROJECT_ROOT / "artifacts" / "bank_scores" / f"{slug}.json"
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("bank_version") != bank_version:
            raise SystemExit("Output belongs to a different bank_version")
    else:
        payload = {
            "schema_version": 2,
            "bank_version": bank_version,
            "test_version": manifest["test_version"],
            "scores": {},
            "equated_scores": {},
            "agents": {},
        }

    if args.policy == "random":
        factory = lambda seed: random_policy(np.random.default_rng(seed))
        provenance = policy_provenance("random")
    elif args.policy == "ppo":
        if not args.model:
            raise SystemExit("--model is required for --policy ppo")
        policy = model_policy(args.model)
        factory = lambda _seed: policy
        provenance = policy_provenance("ppo", model_path=args.model)
    else:
        if not args.policy_factory or ":" not in args.policy_factory:
            raise SystemExit("--policy-factory module:function is required")
        module_name, function_name = args.policy_factory.rsplit(":", 1)
        policy_module = importlib.import_module(module_name)
        external_factory = getattr(policy_module, function_name)
        factory = lambda seed: external_factory(seed)
        provenance = policy_provenance(
            "callable",
            callable_module=policy_module,
            callable_target=args.policy_factory,
        )

    catalog = {str(item["id"]): dict(item) for item in native.biome_catalog()}
    seeds = list(range(args.seed, args.seed + args.trials))
    per_biome = {}
    normalized = []
    for item in manifest["biomes"]:
        if item["split"] != "test":
            continue
        summary = evaluate_adaptive_policy(
            int(catalog[item["id"]]["index"]),
            factory,
            seeds,
            args.episodes_per_trial,
            args.max_steps,
            args.config,
        )
        score = normalized_test_return(manifest, item["id"], summary.mean_return)
        per_biome[item["id"]] = {**asdict(summary), "normalized_return": score}
        normalized.append(score)
        print(item["id"], f"return={summary.mean_return:.6f}", f"normalized={score:.6f}")
    aggregate = float(np.mean(normalized))
    score_eq = None
    if args.equating:
        table = json.loads(args.equating.read_text(encoding="utf-8"))
        score_eq = equated_score(table, bank_version, aggregate)
    payload["scores"][args.agent_id] = aggregate
    if score_eq is not None:
        payload.setdefault("equated_scores", {})[args.agent_id] = score_eq
    payload["agents"][args.agent_id] = {
        "policy": args.policy,
        "policy_provenance": provenance,
        "seeds": seeds,
        "episodes_per_trial": args.episodes_per_trial,
        "raw_normalized_score": aggregate,
        "equated_score": score_eq,
        "per_biome": per_biome,
    }
    write_json(output, payload)
    print(f"bank score={aggregate:.9g} equated={score_eq}; output={output}")


if __name__ == "__main__":
    main()
