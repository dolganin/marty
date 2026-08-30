"""Create and curate deterministic manual-review scenarios for coupling rules."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from mars_rover_env.rules import validate_graph, validate_rule


def _rule_id(rule: dict[str, Any]) -> str:
    payload = json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and manually approve offline coupling rules")
    parser.add_argument("candidates", type=Path, help="JSON list of structured rule candidates")
    parser.add_argument("--output", type=Path, required=True, help="versioned review-bank JSON")
    parser.add_argument("--approve", action="append", default=[], metavar="RULE_ID")
    parser.add_argument("--reject", action="append", default=[], metavar="RULE_ID")
    args = parser.parse_args()
    raw = json.loads(args.candidates.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit("candidate file must contain a JSON list")
    approved, rejected, pending = [], [], []
    for item in raw:
        try:
            rule = validate_rule(item).to_dict()
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"rule": item, "reason": str(exc)})
            continue
        rid = _rule_id(rule)
        # Every candidate owns a stable seed so the reviewer can launch the
        # same HUD scenario before approving it.
        record = {"id": rid, "seed": int(rid[:8], 16), "rule": rule}
        if rid in args.approve:
            approved.append(record)
        elif rid in args.reject:
            rejected.append({**record, "reason": "manual rejection"})
        else:
            pending.append(record)
    validate_graph(record["rule"] for record in (*approved, *pending))
    output = {"schema_version": 2, "approved_rules": approved,
              "pending_rules": pending, "rejected_rules": rejected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for record in pending:
        print(f"review {record['id']}: python -m mars_rover_env.tools.play --seed {record['seed']}")
    print(f"approved={len(approved)} pending={len(pending)} rejected={len(rejected)}")


if __name__ == "__main__":
    main()
