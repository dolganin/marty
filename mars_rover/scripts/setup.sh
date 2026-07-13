#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${1:-Debug}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

case "$BUILD_TYPE" in
  Debug|Release|RelWithDebInfo) ;;
  *)
    printf 'Unsupported build type: %s (use Debug, Release or RelWithDebInfo)\n' "$BUILD_TYPE" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
if [[ "${RECREATE_VENV:-0}" == "1" ]]; then
  rm -rf .venv
fi
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install pybind11
MARS_ROVER_BUILD_TYPE="$BUILD_TYPE" \
  .venv/bin/python -m pip install --editable '.[dev]' --no-build-isolation
.venv/bin/python -m mars_rover_env.tools.doctor
printf 'Ready (%s). Play: .venv/bin/mars-rover-play --debug\n' "$BUILD_TYPE"
