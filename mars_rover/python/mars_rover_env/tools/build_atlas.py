from __future__ import annotations

from pathlib import Path


def main() -> None:
    src = Path("assets/src")
    dst = Path("assets/atlas")
    dst.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        f"Atlas packing is intentionally external for MVP. Pack PNGs from {src} into {dst} "
        "and emit rover_atlas.json/terrain_atlas.json with pivot and pixels_per_meter."
    )


if __name__ == "__main__":
    main()
