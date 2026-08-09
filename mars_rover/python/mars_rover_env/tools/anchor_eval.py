from __future__ import annotations

import argparse
import importlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    load_manifest,
    require_compiled_bank,
    write_json,
)
from mars_rover_env.tools.evaluate_biomes import (
    evaluate_adaptive_policy,
    model_policy,
    policy_provenance,
    random_policy,
)


def _renormalize(payload: dict) -> None:
    evaluations = payload.get("evaluations", {})
    random_eval = next(
        (
            item.get("anchors", {})
            for item in evaluations.values()
            if item.get("reference_role") == "random"
        ),
        {},
    )
    robust_eval = next(
        (
            item.get("anchors", {})
            for item in evaluations.values()
            if item.get("reference_role") == "robust_ppo"
        ),
        {},
    )
    for evaluation in evaluations.values():
        for biome_id, result in evaluation.get("anchors", {}).items():
            low = random_eval.get(biome_id, {}).get("mean_return")
            high = robust_eval.get(biome_id, {}).get("mean_return")
            if low is None or high is None or high - low <= 1.0e-8:
                result["normalized_return"] = None
            else:
                result["normalized_return"] = (result["mean_return"] - low) / (high - low)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an agent on all frozen anchor biomes")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument(
        "--policy", choices=("random", "ppo", "callable"), required=True
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--policy-factory",
        help="For --policy callable: import path module:function returning a Policy for a seed",
    )
    parser.add_argument("--reference-role", choices=("random", "robust_ppo"))
    parser.add_argument("--config")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--episodes-per-trial", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import _mars_rover_cpp as native

    manifest = load_manifest(args.manifest)
    require_compiled_bank(manifest)
    bank_version = str(manifest["bank_version"])
    slug = bank_version.split(":", 1)[-1][:16]
    output = args.output or PROJECT_ROOT / "artifacts" / "anchor_eval" / f"{slug}.json"
    if output.is_file():
        import json

        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("bank_version") != bank_version:
            raise SystemExit("Output belongs to a different bank_version")
    else:
        payload = {"schema_version": 2, "bank_version": bank_version, "evaluations": {}}

    if args.policy == "random":
        factory = lambda seed: random_policy(np.random.default_rng(seed))
        reference_role = args.reference_role or "random"
        provenance = policy_provenance("random")
    elif args.policy == "ppo":
        if not args.model:
            raise SystemExit("--model is required for --policy ppo")
        policy = model_policy(args.model)
        factory = lambda _seed: policy
        reference_role = args.reference_role
        provenance = policy_provenance("ppo", model_path=args.model)
    else:
        if not args.policy_factory or ":" not in args.policy_factory:
            raise SystemExit(
                "--policy-factory module:function is required for --policy callable"
            )
        module_name, function_name = args.policy_factory.rsplit(":", 1)
        policy_module = importlib.import_module(module_name)
        policy_factory = getattr(policy_module, function_name)
        factory = lambda seed: policy_factory(seed)
        reference_role = args.reference_role
        provenance = policy_provenance(
            "callable",
            callable_module=policy_module,
            callable_target=args.policy_factory,
        )

    anchors = [dict(item) for item in native.biome_catalog() if bool(item["is_anchor"])]
    expected = set(manifest.get("anchors", []))
    actual = {str(item["id"]) for item in anchors}
    if actual != expected:
        raise SystemExit(f"Compiled anchors differ from manifest: compiled={sorted(actual)}, manifest={sorted(expected)}")
    seeds = list(range(args.seed, args.seed + args.trials))
    results = {}
    for anchor in anchors:
        summary = evaluate_adaptive_policy(
            int(anchor["index"]),
            factory,
            seeds,
            args.episodes_per_trial,
            args.max_steps,
            args.config,
        )
        results[str(anchor["id"])] = asdict(summary)
        print(anchor["id"], f"return={summary.mean_return:.4f}", f"success={summary.success_rate:.3f}")
    if reference_role is not None:
        for agent_id, evaluation in payload["evaluations"].items():
            if agent_id != args.agent_id and evaluation.get("reference_role") == reference_role:
                evaluation["reference_role"] = None
    payload["evaluations"][args.agent_id] = {
        "policy": args.policy,
        "policy_provenance": provenance,
        "reference_role": reference_role,
        "seeds": seeds,
        "episodes_per_trial": args.episodes_per_trial,
        "anchors": results,
    }
    _renormalize(payload)
    write_json(output, payload)
    print(f"Anchor results: {output}")


if __name__ == "__main__":
    main()
