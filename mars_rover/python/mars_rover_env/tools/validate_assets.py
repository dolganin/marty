from __future__ import annotations

import json
from pathlib import Path


def validate_atlas(path: str | Path) -> None:
    data = json.loads(Path(path).read_text())
    assert "image" in data
    assert "sprites" in data and isinstance(data["sprites"], dict)
    for name, sprite in data["sprites"].items():
        for key in ("x", "y", "w", "h", "pivot", "pixels_per_meter"):
            assert key in sprite, f"{name}: missing {key}"
        assert len(sprite["pivot"]) == 2, f"{name}: pivot must be [x, y]"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    for atlas in (root / "assets" / "atlas").glob("*.json"):
        validate_atlas(atlas)
        print(f"ok {atlas}")
