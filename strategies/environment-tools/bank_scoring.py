"""Bank scoring helpers kept outside the shipped package.

These normalization and equating helpers used to live in `mars_rover_env.bank`
and were removed from it so a participant cannot read the scoring rule out of
the environment they are given. The internal tooling still needs them, so they
live here next to the tools that score banks.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from mars_rover_env.bank import PACKAGE_ROOT


DEFAULT_EQUATING = PACKAGE_ROOT / "configs" / "bank_equating.json"


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
