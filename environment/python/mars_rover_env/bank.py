from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from mars_rover_env.rules import validate_graph


# This data is mirrored by kFrozenMechanismStacks in biome_bank.hpp.  It is
# part of the versioned release contract, not a runtime source of randomness.
FROZEN_STACKS = (
    {"split": "anchor", "mechanisms": ("normal", "wind")},
    {"split": "train", "mechanisms": ("sand", "wind")},
    {"split": "train", "mechanisms": ("mud", "crust", "wind")},
    {"split": "train", "mechanisms": ("ice", "low_gravity")},
    {"split": "held_out", "mechanisms": ("liquid", "wind")},
    {"split": "held_out", "mechanisms": ("sand", "mud", "wind", "crust")},
)

DEFAULT_COUPLING_RULES = (
    {"source": "moisture", "target": "traction", "weight": -0.72},
    {"source": "sink", "target": "traction", "weight": -0.58},
    {"source": "temperature", "target": "viscosity", "weight": -0.78},
    {"source": "viscosity", "target": "energy_resistance", "weight": 0.88},
    {"source": "temperature", "target": "thermal_transfer", "weight": -0.72},
)


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
ACTIVE_BANK_POINTER = REPOSITORY_ROOT / "artifacts" / "active_bank.json"


def _default_manifest() -> Path:
    explicit = os.environ.get("MARS_ROVER_BANK_MANIFEST")
    if explicit:
        return Path(explicit)
    if ACTIVE_BANK_POINTER.is_file():
        pointer = json.loads(ACTIVE_BANK_POINTER.read_text(encoding="utf-8"))
        candidate = Path(str(pointer.get("manifest", "")))
        if candidate.is_file():
            return candidate
    return PACKAGE_ROOT / "configs" / "biome_bank.json"


DEFAULT_MANIFEST = _default_manifest()
def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Biome manifest does not exist: {path}. Generate or refresh the bank first."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    # Optional v13 formulas are checked before a frozen bank is used. Runtime
    # never evaluates arbitrary model output.
    validate_graph(manifest.get("coupling_rules", []))
    known = set(manifest.get("anchors", [])) | {str(item.get("id", ""))
                                                   for item in manifest.get("biomes", [])}
    for stack in manifest.get("frozen_stacks", []):
        mechanisms = stack.get("mechanisms", [])
        if stack.get("split") not in {"anchor", "train", "held_out"}:
            raise ValueError("frozen stack has an invalid split")
        if not 1 <= len(mechanisms) <= 4 or any(name not in known for name in mechanisms):
            raise ValueError("frozen stack contains unknown or unsafe mechanisms")
    return manifest


def require_compiled_bank(manifest: dict[str, Any]) -> str:
    import _mars_rover_cpp as native

    expected = str(manifest.get("bank_version", ""))
    actual = str(native.biome_bank_version())
    if not expected or actual != expected:
        raise RuntimeError(
            "Compiled biome bank does not match the manifest; rebuild the native extension "
            f"(compiled={actual!r}, manifest={expected!r})"
        )
    return actual


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_frozen_train_bank(manifest: dict[str, Any]) -> str:
    """Legacy LLM-generated-bank rollout gate.

    Kept importable only so train_robust_ppo.py / audit_bank.py still load -
    it is unreachable when --allow-unfrozen-bank is passed (the current
    hand-authored biome_bank.hpp manifest was never run through this gate
    and lacks its fields, so calling this against it will always raise).
    """
    train = [item for item in manifest.get("biomes", []) if item.get("split") == "train"]
    if not train:
        raise RuntimeError("The generated train bank is empty")
    pending = [item.get("id", "<unknown>") for item in train if item.get("status") != "accepted"]
    if pending:
        raise RuntimeError("Train bank is not frozen; difficulty gate is pending for: " + ", ".join(pending))
    missing_metrics = [
        str(item.get("id", "<unknown>"))
        for item in train
        if item.get("r_random") is None or item.get("r_solve") is None
    ]
    if missing_metrics:
        raise RuntimeError(
            "Frozen train bank misses rollout metrics for: " + ", ".join(missing_metrics)
        )
    if not manifest.get("difficulty_gates", {}).get("train"):
        raise RuntimeError("Frozen train bank has no recorded difficulty-gate protocol")
    import _mars_rover_cpp as native

    gate_environment = manifest["difficulty_gates"]["train"].get("environment_version")
    if gate_environment != native.environment_version():
        raise RuntimeError(
            "Train difficulty gate was measured with a different environment version"
        )
    expected_strata = {
        "traction_loss",
        "lateral_force",
        "inertia_hysteresis",
        "gravity_change",
        "energy_mode",
        "dynamic_obstacle",
    }
    present_strata = {str(item.get("skill_stratum")) for item in train}
    missing = expected_strata - present_strata
    if missing:
        raise RuntimeError("Frozen train bank misses strata: " + ", ".join(sorted(missing)))
    quota = int(manifest.get("quota_per_stratum", 2))
    stratum_counts = Counter(str(item.get("skill_stratum")) for item in train)
    wrong_counts = {
        stratum: stratum_counts[stratum]
        for stratum in sorted(expected_strata)
        if stratum_counts[stratum] != quota
    }
    if wrong_counts:
        raise RuntimeError(
            f"Frozen train bank violates exact quota={quota} per stratum: {wrong_counts}"
        )
    version = str(manifest.get("train_version", ""))
    if not version:
        raise RuntimeError("Manifest has no train_version")
    return version
