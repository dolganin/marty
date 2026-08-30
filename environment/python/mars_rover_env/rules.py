"""Offline, reviewable coupling-rule format used when assembling a bank.

Rules are data, never model-generated C++: an LLM may propose this JSON, but
the validator makes the accepted bank deterministic and bounded.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LATENTS = frozenset({"traction", "moisture", "heat", "charge_reserve", "tire_pressure",
                     "viscosity", "sink", "suspension"})
INPUTS = LATENTS | frozenset({"speed", "slip", "slope", "immersion", "throttle"})
BOUNDS = {
    "traction": (0.12, 1.45), "moisture": (0.0, 1.0), "heat": (0.0, 1.0),
    "charge_reserve": (0.0, 1.0), "tire_pressure": (0.65, 1.25),
    "viscosity": (0.0, 2.5), "sink": (0.0, 1.0), "suspension": (0.65, 1.25),
}


@dataclass(frozen=True)
class CouplingRule:
    inputs: tuple[str, ...]
    target: str
    coefficients: tuple[float, ...]
    bias: float = 0.0
    lower: float | None = None
    upper: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_rule(data: CouplingRule | dict[str, Any]) -> CouplingRule:
    rule = data if isinstance(data, CouplingRule) else CouplingRule(
        inputs=tuple(data["inputs"]), target=str(data["target"]),
        coefficients=tuple(float(v) for v in data["coefficients"]),
        bias=float(data.get("bias", 0.0)), lower=data.get("lower"), upper=data.get("upper"),
    )
    if not rule.inputs or len(rule.inputs) > 2 or any(v not in INPUTS for v in rule.inputs):
        raise ValueError("a coupling rule needs one or two approved inputs")
    if rule.target not in LATENTS or len(rule.coefficients) != len(rule.inputs):
        raise ValueError("target/input coefficient contract is invalid")
    if any(abs(v) > 2.5 for v in (*rule.coefficients, rule.bias)):
        raise ValueError("rule coefficient exceeds safe bound")
    lo, hi = BOUNDS[rule.target]
    lower = lo if rule.lower is None else float(rule.lower)
    upper = hi if rule.upper is None else float(rule.upper)
    if lower < lo or upper > hi or lower > upper:
        raise ValueError("rule bounds exceed target safety envelope")
    return CouplingRule(rule.inputs, rule.target, rule.coefficients, rule.bias, lower, upper)


def apply_rule(rule: CouplingRule | dict[str, Any], values: dict[str, float]) -> float:
    """Evaluate the bounded affine formula deterministically for review/scenarios."""
    checked = validate_rule(rule)
    value = checked.bias + sum(c * float(values.get(name, 0.0))
                               for name, c in zip(checked.inputs, checked.coefficients))
    assert checked.lower is not None and checked.upper is not None
    return max(checked.lower, min(checked.upper, value))
