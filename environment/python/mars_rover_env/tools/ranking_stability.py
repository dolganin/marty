from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def kendall_tau_b(left: dict[str, float], right: dict[str, float]) -> float:
    agents = sorted(set(left) & set(right))
    if len(agents) < 2:
        raise ValueError("Need at least two agents shared by both banks")
    concordant = discordant = left_ties = right_ties = 0
    for first, second in combinations(agents, 2):
        dx = left[first] - left[second]
        dy = right[first] - right[second]
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            left_ties += 1
        elif dy == 0:
            right_ties += 1
        elif dx * dy > 0:
            concordant += 1
        else:
            discordant += 1
    denominator = ((concordant + discordant + left_ties) *
                   (concordant + discordant + right_ties)) ** 0.5
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator


def _load_scores(path: Path) -> tuple[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores")
    if not isinstance(scores, dict):
        raise ValueError(f"{path} must contain an agent-to-number 'scores' object")
    return str(payload["bank_version"]), {str(k): float(v) for k, v in scores.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline Kendall rank-stability check across 2-3 regenerated banks"
    )
    parser.add_argument("banks", nargs="+", type=Path)
    parser.add_argument("--equating", type=Path)
    parser.add_argument("--min-tau", type=float, default=0.8)
    args = parser.parse_args()
    if not 2 <= len(args.banks) <= 3:
        raise SystemExit("Provide exactly 2 or 3 bank score files")
    banks = [_load_scores(path) for path in args.banks]
    coefficients = {}
    if args.equating:
        coefficients = json.loads(args.equating.read_text(encoding="utf-8")).get(
            "coefficients", {}
        )
        missing = [version for version, _ in banks if version not in coefficients]
        if missing:
            raise SystemExit(
                "Equating table misses bank versions: " + ", ".join(missing)
            )
    failed = False
    for (left_version, left_raw), (right_version, right_raw) in combinations(banks, 2):
        left_fit = coefficients.get(left_version, {"a": 1.0, "b": 0.0})
        right_fit = coefficients.get(right_version, {"a": 1.0, "b": 0.0})
        left = {key: left_fit["a"] * value + left_fit["b"] for key, value in left_raw.items()}
        right = {key: right_fit["a"] * value + right_fit["b"] for key, value in right_raw.items()}
        tau = kendall_tau_b(left, right)
        print(f"{left_version} vs {right_version}: tau_b={tau:.6f}")
        failed |= tau < args.min_tau
    if failed:
        raise SystemExit(f"Ranking stability is below min_tau={args.min_tau}")


if __name__ == "__main__":
    main()
