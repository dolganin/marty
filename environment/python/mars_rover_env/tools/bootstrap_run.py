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
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], *, cwd: Path) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd)
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
        ],
        cwd=ROOT,
    )

    if args.skip_rebuild:
        print("Skipping rebuild (--skip-rebuild); the compiled bank is now stale.")
        return

    _run([python, "-m", "pip", "install", "-e", ".", "--no-build-isolation", "-q"], cwd=ROOT)

    _run(
        [python, "-m", "mars_rover_env.tools.evaluate_biomes", "--split", args.split],
        cwd=ROOT,
    )

    if not args.skip_audit:
        _run([python, "-m", "mars_rover_env.tools.audit_bank"], cwd=ROOT)

    print(
        f"\nBootstrap done: {args.count} fresh {args.split} biomes generated, "
        "extension rebuilt. The chain assembled at Env::reset() now draws from this bank."
    )


if __name__ == "__main__":
    main()
