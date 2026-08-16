from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mars_rover_env.bank import DEFAULT_EQUATING, write_json


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def matched_anchor_points(
    reference: dict, new: dict, score_field: str = "normalized_return"
) -> tuple[np.ndarray, np.ndarray]:
    reference_eval = reference.get("evaluations", {})
    new_eval = new.get("evaluations", {})
    agents = sorted(set(reference_eval) & set(new_eval))
    if len(agents) < 3:
        raise ValueError("Need at least three reference agents shared by both banks")
    reference_protocols = {
        (
            tuple(reference_eval[agent].get("seeds", [])),
            reference_eval[agent].get("episodes_per_trial"),
        )
        for agent in agents
    }
    new_protocols = {
        (
            tuple(new_eval[agent].get("seeds", [])),
            new_eval[agent].get("episodes_per_trial"),
        )
        for agent in agents
    }
    if len(reference_protocols) != 1 or len(new_protocols) != 1:
        raise ValueError("All reference agents must use one shared fixed-seed trial protocol")
    x: list[float] = []
    y: list[float] = []
    for agent in agents:
        for field in ("policy_provenance", "seeds", "episodes_per_trial"):
            if reference_eval[agent].get(field) != new_eval[agent].get(field):
                raise ValueError(
                    f"Agent {agent!r} uses different {field} across bank versions"
                )
        ref_anchors = reference_eval[agent].get("anchors", {})
        new_anchors = new_eval[agent].get("anchors", {})
        anchors = sorted(set(ref_anchors) & set(new_anchors))
        if len(anchors) < 8:
            raise ValueError(f"Agent {agent!r} has fewer than eight matched anchors")
        for anchor in anchors:
            new_value = new_anchors[anchor].get(score_field)
            reference_value = ref_anchors[anchor].get(score_field)
            if new_value is None or reference_value is None:
                raise ValueError(
                    f"Agent {agent!r}, anchor {anchor!r} misses {score_field}"
                )
            x.append(float(new_value))
            y.append(float(reference_value))
    return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)


def fit_linear_equating(
    reference: dict, new: dict, score_field: str = "normalized_return"
) -> dict:
    x, y = matched_anchor_points(reference, new, score_field)
    if x.size < 24:
        raise ValueError("Need at least 24 matched (agent, anchor) points")
    if float(np.var(x)) < 1.0e-12:
        raise ValueError("New-bank anchor scores have zero variance")
    design = np.column_stack((x, np.ones_like(x)))
    a, b = np.linalg.lstsq(design, y, rcond=None)[0]
    if a <= 0.0:
        raise ValueError("Equating slope must be positive to preserve score ordering")
    residual = y - (a * x + b)
    total = y - np.mean(y)
    r2 = 1.0 - float(residual @ residual) / max(1.0e-12, float(total @ total))
    return {"a": float(a), "b": float(b), "points": int(x.size), "r2": r2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit two-parameter common-item equating")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_EQUATING)
    parser.add_argument("--score-field", default="normalized_return")
    args = parser.parse_args()
    reference = _load(args.reference)
    new = _load(args.new)
    reference_version = str(reference["bank_version"])
    new_version = str(new["bank_version"])
    fit = fit_linear_equating(reference, new, args.score_field)
    payload = _load(args.output) if args.output.is_file() else {
        "schema_version": 1,
        "reference_version": reference_version,
        "score_field": args.score_field,
        "coefficients": {},
    }
    if payload.get("reference_version") != reference_version:
        raise SystemExit("Equating table is tied to a different reference bank")
    if payload.get("score_field") != args.score_field:
        raise SystemExit("Equating table uses a different score field")
    payload["coefficients"][reference_version] = {"a": 1.0, "b": 0.0, "points": 0, "r2": 1.0}
    payload["coefficients"][new_version] = fit
    write_json(args.output, payload)
    print(f"score_eq = {fit['a']:.9g} * score_new + {fit['b']:.9g}; points={fit['points']} r2={fit['r2']:.6f}")


if __name__ == "__main__":
    main()
