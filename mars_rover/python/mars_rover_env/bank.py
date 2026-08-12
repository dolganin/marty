from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_MANIFEST = PACKAGE_ROOT / "configs" / "biome_bank.json"
DEFAULT_EQUATING = PACKAGE_ROOT / "configs" / "bank_equating.json"


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Biome manifest does not exist: {path}. Generate or refresh the bank first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def normalized_test_return(
    manifest: dict[str, Any], biome_id: str, agent_return: float
) -> float:
    """Normalize one test score using the rollout-gate random/robust anchors."""
    matches = [
        item
        for item in manifest.get("biomes", [])
        if item.get("split") == "test" and item.get("id") == biome_id
    ]
    if len(matches) != 1:
        raise KeyError(f"Unknown or duplicate test biome: {biome_id}")
    item = matches[0]
    low = item.get("r_random")
    high = item.get("r_robust")
    if low is None or high is None:
        raise RuntimeError(f"Test biome {biome_id!r} has not passed the rollout gate")
    denominator = float(high) - float(low)
    if denominator <= 1.0e-8:
        raise RuntimeError(f"Test biome {biome_id!r} has an invalid normalization band")
    return (float(agent_return) - float(low)) / denominator


def equated_score(equating: dict[str, Any], bank_version: str, score: float) -> float:
    coefficients = equating.get("coefficients", {}).get(bank_version)
    if coefficients is None:
        raise KeyError(f"No equating coefficients for bank version {bank_version}")
    slope = float(coefficients["a"])
    intercept = float(coefficients["b"])
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0.0:
        raise ValueError(f"Invalid equating coefficients for {bank_version}")
    return slope * float(score) + intercept
