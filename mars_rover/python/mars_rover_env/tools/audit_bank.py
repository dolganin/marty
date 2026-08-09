from __future__ import annotations

import argparse
import math
from pathlib import Path

from mars_rover_env import MarsRoverEnv
from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    load_manifest,
    require_compiled_bank,
    require_frozen_train_bank,
)


SKILL_STRATA = {
    "traction_loss",
    "lateral_force",
    "inertia_hysteresis",
    "gravity_change",
    "energy_mode",
    "dynamic_obstacle",
}


def fingerprint_distance(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return math.inf
    return math.sqrt(
        sum(
            ((a - b) / (1.0 + abs(a) + abs(b))) ** 2
            for a, b in zip(left, right)
        )
        / len(left)
    )


def audit(manifest_path: Path, require_reference: bool, require_test_gate: bool) -> None:
    import _mars_rover_cpp as native

    manifest = load_manifest(manifest_path)
    require_compiled_bank(manifest)
    require_frozen_train_bank(manifest)
    train_gate = manifest.get("difficulty_gates", {}).get("train", {})
    if train_gate.get("protocol_version") != "rollout-v2-public-macro-random":
        raise RuntimeError("Train bank uses an unknown difficulty-gate protocol")
    config_sha256 = str(train_gate.get("config_sha256", ""))
    if len(config_sha256) != 64:
        raise RuntimeError("Train difficulty gate does not identify its environment config")
    catalog = [dict(item) for item in native.biome_catalog()]
    compiled = {str(item["id"]): item for item in catalog}
    declared = {str(item["id"]): item for item in manifest.get("biomes", [])}
    compiled_generated = {key for key, item in compiled.items() if int(item["split"]) in (1, 2)}
    if compiled_generated != set(declared):
        raise RuntimeError("Compiled generated IDs differ from the manifest")

    anchors = {str(item["id"]) for item in catalog if bool(item["is_anchor"])}
    if anchors != set(manifest.get("anchors", [])) or len(anchors) < 5:
        raise RuntimeError("Compiled anchor set differs from the manifest")
    for biome_id, item in declared.items():
        native_item = compiled[biome_id]
        if ("train" if int(native_item["split"]) == 1 else "test") != item["split"]:
            raise RuntimeError(f"Split mismatch for {biome_id}")
        if str(native_item["skill_stratum"]) != item["skill_stratum"]:
            raise RuntimeError(f"Skill stratum mismatch for {biome_id}")
        fingerprint = item.get("behavior_fingerprint") or []
        if not fingerprint or not all(math.isfinite(float(value)) for value in fingerprint):
            raise RuntimeError(f"Missing/non-finite fingerprint for {biome_id}")

    for split in ("train", "test"):
        items = [item for item in declared.values() if item["split"] == split]
        if split == "test" and len(items) < 10:
            raise RuntimeError("Test bank has fewer than 10 mechanics")
        present = {str(item["skill_stratum"]) for item in items}
        if not SKILL_STRATA <= present:
            raise RuntimeError(f"{split} bank misses strata: {sorted(SKILL_STRATA - present)}")

    train = [item for item in declared.values() if item["split"] == "train"]
    test = [item for item in declared.values() if item["split"] == "test"]
    epsilon = float(manifest.get("behavioral_dedup_epsilon", 0.035))
    nearest = min(
        (
            fingerprint_distance(left["behavior_fingerprint"], right["behavior_fingerprint"]),
            left["id"],
            right["id"],
        )
        for left in test
        for right in train
    )
    if nearest[0] < epsilon:
        raise RuntimeError(
            f"Test biome {nearest[1]} duplicates train biome {nearest[2]}: "
            f"distance={nearest[0]:.6f} < epsilon={epsilon:.6f}"
        )

    # fixed_biome_id must produce exactly one whole-course zone for every item.
    for item in catalog:
        env = MarsRoverEnv(fixed_biome_id=int(item["index"]))
        env.reset(seed=771, options={"trial_start": True})
        debug = env.debug_info()
        env.close()
        if debug["mechanic"] != item["name"]:
            raise RuntimeError(f"Fixed-biome selection failed for {item['id']}")
        if debug["zone_begin_x"] > -999_999.0 or debug["zone_end_x"] < 999_999.0:
            raise RuntimeError(f"Biome {item['id']} does not span the complete environment")

    if require_reference and not manifest.get("reference_version"):
        raise RuntimeError("Bank has no frozen robust-PPO reference_version")
    if require_test_gate:
        test_gate = manifest.get("difficulty_gates", {}).get("test", {})
        if test_gate.get("protocol_version") != "rollout-v2-public-macro-random":
            raise RuntimeError("Test bank has no recorded public-macro rollout-v2 gate")
        if test_gate.get("reference_version") != manifest.get("reference_version"):
            raise RuntimeError("Test gate used a different robust-PPO reference")
        rejected = [
            item["id"]
            for item in test
            if item.get("status") != "accepted"
            or item.get("r_random") is None
            or item.get("r_robust") is None
        ]
        if rejected:
            raise RuntimeError("Test gate is incomplete for: " + ", ".join(rejected))
        # The normalisation ceiling is the oracle (see harness._normalization_bounds):
        # on a bank that requires adaptation, robust legitimately fails some biomes and
        # cannot define a scale there.
        invalid_bands = [
            item["id"]
            for item in test
            if item.get("r_solve") is None
            or not math.isfinite(float(item["r_random"]))
            or not math.isfinite(float(item["r_solve"]))
            or float(item["r_solve"]) <= float(item["r_random"])
        ]
        if invalid_bands:
            raise RuntimeError(
                "Test normalization bands are invalid for: " + ", ".join(invalid_bands)
            )

    print(f"bank_version={manifest['bank_version']}")
    print(f"train={len(train)} test={len(test)} anchors={len(anchors)}")
    print(
        f"nearest_train_test_fingerprint={nearest[0]:.6f} "
        f"({nearest[1]} vs {nearest[2]}; epsilon={epsilon:.6f})"
    )
    print("bank audit: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a compiled bank before training/scoring")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--require-reference", action="store_true")
    parser.add_argument("--require-test-gate", action="store_true")
    args = parser.parse_args()
    audit(args.manifest, args.require_reference, args.require_test_gate)


if __name__ == "__main__":
    main()
