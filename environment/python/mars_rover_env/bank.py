from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mars_rover_env.rules import validate_rule


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
    {"inputs": ("moisture", "viscosity"), "target": "traction",
     "coefficients": (-0.62, -0.16), "bias": 1.18},
    {"inputs": ("slip", "speed"), "target": "heat",
     "coefficients": (0.14, 0.02), "bias": 0.18},
    {"inputs": ("moisture", "heat"), "target": "tire_pressure",
     "coefficients": (-0.035, 0.02), "bias": 1.0},
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
    for rule in manifest.get("coupling_rules", []):
        validate_rule(rule)
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
