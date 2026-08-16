from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


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
