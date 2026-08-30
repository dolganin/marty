"""Offline, reviewable generation-time influence graph for biome banks.

Rules are data, never model-generated C++: an LLM may propose source -> target
edges, but validation keeps the accepted graph deterministic, acyclic and
bounded. Incoming contributions are summed and compressed with signed log1p.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable


BOUNDS = {
    "traction": (0.12, 1.45),
    "moisture": (0.0, 1.0),
    "viscosity": (0.0, 2.5),
    "sink": (0.0, 1.0),
    "gravity": (0.45, 1.35),
    "wind": (-80.0, 80.0),
    "temperature": (-58.0, 72.0),
    "thermal_transfer": (0.25, 3.0),
    "solar_efficiency": (0.0, 3.0),
    "lidar_cost": (0.4, 3.0),
    "lidar_range": (0.35, 1.5),
    "energy_resistance": (0.4, 3.0),
}
TARGET_SCALES = {
    "traction": 0.38,
    "viscosity": 0.34,
    "energy_resistance": 0.68,
    "thermal_transfer": 0.62,
}


@dataclass(frozen=True)
class CouplingRule:
    source: str
    target: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_rule(data: CouplingRule | dict[str, Any]) -> CouplingRule:
    rule = data if isinstance(data, CouplingRule) else CouplingRule(
        source=str(data["source"]),
        target=str(data["target"]),
        weight=float(data["weight"]),
    )
    if rule.source not in BOUNDS or rule.target not in BOUNDS:
        raise ValueError("influence endpoints must be approved bounded parameters")
    if rule.source == rule.target:
        raise ValueError("an influence cannot target its own source")
    if not math.isfinite(rule.weight) or abs(rule.weight) > 2.5:
        raise ValueError("influence weight exceeds safe bound")
    if rule.target not in TARGET_SCALES:
        raise ValueError("target has no reviewed influence scale")
    return rule


def validate_graph(items: Iterable[CouplingRule | dict[str, Any]]) -> tuple[CouplingRule, ...]:
    rules = tuple(validate_rule(item) for item in items)
    edges = {(rule.source, rule.target) for rule in rules}
    if len(edges) != len(rules):
        raise ValueError("influence graph contains a duplicate edge")
    nodes = {name for rule in rules for name in (rule.source, rule.target)}
    incoming = {node: 0 for node in nodes}
    outgoing = {node: [] for node in nodes}
    for rule in rules:
        outgoing[rule.source].append(rule.target)
        incoming[rule.target] += 1
    ready = [node for node, count in incoming.items() if count == 0]
    visited = 0
    while ready:
        source = ready.pop()
        visited += 1
        for target in outgoing[source]:
            incoming[target] -= 1
            if incoming[target] == 0:
                ready.append(target)
    if visited != len(nodes):
        raise ValueError("influence graph must be acyclic")
    return rules


def apply_influence_graph(
    items: Iterable[CouplingRule | dict[str, Any]], base_values: dict[str, float]
) -> dict[str, float]:
    """Evaluate a reviewed graph with sum + signed-log aggregation."""
    rules = validate_graph(items)
    values = {name: float(base_values.get(name, lo)) for name, (lo, _) in BOUNDS.items()}
    pending = {rule.target for rule in rules}
    while pending:
        progressed = False
        for target in tuple(pending):
            incoming = [rule for rule in rules if rule.target == target]
            if any(rule.source in pending for rule in incoming):
                continue
            total = 0.0
            for rule in incoming:
                lo, hi = BOUNDS[rule.source]
                normalized = max(0.0, min(1.0, (values[rule.source] - lo) / (hi - lo)))
                total += rule.weight * normalized
            adjustment = math.copysign(math.log1p(abs(total)), total)
            lo, hi = BOUNDS[target]
            values[target] = max(
                lo,
                min(hi, values[target] + TARGET_SCALES[target] * adjustment),
            )
            pending.remove(target)
            progressed = True
        if not progressed:
            raise ValueError("could not topologically evaluate influence graph")
    return values
