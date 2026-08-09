from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mars_rover_env.bank import (
    DEFAULT_MANIFEST,
    PROJECT_ROOT,
    load_manifest,
    require_compiled_bank,
    write_json,
)
from mars_rover_env.config import DEFAULT_ENV_CONFIG


BIOME_HEADER = PROJECT_ROOT / "cpp" / "include" / "mars" / "biome_bank.hpp"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def snapshot(manifest_path: Path, output_root: Path) -> Path:
    import _mars_rover_cpp as native

    manifest = load_manifest(manifest_path)
    require_compiled_bank(manifest)
    version = str(manifest["bank_version"])
    slug = version.split(":", 1)[-1]
    header_bytes = BIOME_HEADER.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    config_bytes = DEFAULT_ENV_CONFIG.read_bytes()
    manifest_state_sha256 = _sha256(manifest_bytes)
    output = output_root / slug / manifest_state_sha256[:16]
    record = {
        "schema_version": 1,
        "bank_version": version,
        "train_version": manifest["train_version"],
        "test_version": manifest["test_version"],
        "environment_version": native.environment_version(),
        "manifest_state_sha256": manifest_state_sha256,
        "files": {
            "biome_bank.hpp": _sha256(header_bytes),
            "biome_bank.json": _sha256(manifest_bytes),
            "env.yaml": _sha256(config_bytes),
        },
    }
    record_path = output / "snapshot.json"
    if record_path.is_file():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError(f"Immutable bank snapshot conflicts with current files: {output}")
        for name, expected_hash in record["files"].items():
            path = output / name
            if not path.is_file() or _sha256(path.read_bytes()) != expected_hash:
                raise RuntimeError(f"Immutable bank snapshot is corrupted: {path}")
        print(f"Bank snapshot already verified: {output}")
        return output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Refusing to populate non-empty snapshot directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_bytes_atomic(output / "biome_bank.hpp", header_bytes)
    _write_bytes_atomic(output / "biome_bank.json", manifest_bytes)
    _write_bytes_atomic(output / "env.yaml", config_bytes)
    write_json(record_path, record)
    print(f"Frozen bank snapshot: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive an immutable compiled bank version")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "bank_snapshots",
    )
    args = parser.parse_args()
    snapshot(args.manifest, args.output_root)


if __name__ == "__main__":
    main()
