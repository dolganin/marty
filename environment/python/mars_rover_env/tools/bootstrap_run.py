"""Refresh the LLM-generated biome slice before a training or validation run.

A trial is a chain of zones (see EnvConfig.chain_biomes): a stable anchor/
hand-written backbone plus a slice of generated biomes that fills the rest of
the chain. The backbone must stay put so runs are comparable; the generated
slice is what makes every run's course different from the last one. This tool
is the one step that has to happen before you start training or validating:

    mars-rover-bootstrap-run --count 8 --split train
    mars-rover-bootstrap-run --count 8 --split test

It (1) calls generate_biomes.py with --replace so the previous generated
slice is discarded and a fresh one takes its place, then (2) rebuilds the
native extension, because the bank is compiled into the binary and nothing
downstream sees a new biome until that happens.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(
            f"Command failed with exit code {result.returncode}: {' '.join(command)}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenerate the LLM biome slice and rebuild the native extension "
        "before a training or validation run."
    )
    parser.add_argument("--count", type=int, default=8, help="new biomes to generate")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument(
        "--bank-dir", type=Path,
        help="run artifact directory; required to keep generated banks out of source files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "python" / "mars_rover_env" / "configs" / "biome_generator.yaml",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="generate only; useful when chaining train+test calls before one rebuild",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="skip the post-rebuild bank audit",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    python = sys.executable
    if args.bank_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.bank_dir = ROOT.parent / "artifacts" / "banks" / f"manual_{stamp}"
    bank_dir = args.bank_dir.resolve()
    bank_include = bank_dir / "include"
    run_env = dict(os.environ)
    run_env["MARS_ROVER_BANK_MANIFEST"] = str(bank_dir / "biome_bank.json")
    run_env["MARS_ROVER_BANK_INCLUDE"] = str(bank_include)

    _run(
        [
            python,
            "-m",
            "mars_rover_env.tools.generate_biomes",
            "--config",
            str(args.config),
            "--split",
            args.split,
            "--count",
            str(args.count),
            "--replace",
            "--bank-dir",
            str(bank_dir),
        ],
        cwd=ROOT,
        env=run_env,
    )

    if args.skip_rebuild:
        print("Skipping rebuild (--skip-rebuild); the compiled bank is now stale.")
        return

    _run([python, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-q"], cwd=ROOT, env=run_env)

    _run(
        [python, "-m", "mars_rover_env.tools.evaluate_biomes", "--split", args.split],
        cwd=ROOT, env=run_env,
    )

    if not args.skip_audit:
        _run([python, "-m", "mars_rover_env.tools.audit_bank"], cwd=ROOT, env=run_env)

    pointer = ROOT.parent / "artifacts" / "active_bank.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=pointer.parent, delete=False) as handle:
        json.dump({"manifest": str(bank_dir / "biome_bank.json"), "include": str(bank_include)}, handle)
        temporary = Path(handle.name)
    temporary.replace(pointer)

    print(
        f"\nBootstrap done: {args.count} fresh {args.split} biomes generated, "
        "extension rebuilt. The chain assembled at Env::reset() now draws from this bank."
    )


if __name__ == "__main__":
    main()
