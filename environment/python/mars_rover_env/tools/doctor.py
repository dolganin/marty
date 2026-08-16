from __future__ import annotations

import platform
import sys


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Platform: {platform.platform()}")
    try:
        import _mars_rover_cpp as core
        import gymnasium
        import numpy
        import yaml

        print(f"Native module: {core.__file__}")
        print(f"NumPy: {numpy.__version__}")
        print(f"Gymnasium: {gymnasium.__version__}")
        print(f"PyYAML: {yaml.__version__}")
    except ImportError as exc:
        raise SystemExit(f"BROKEN: {exc}\nRun the platform setup script again.") from exc

    from mars_rover_env import MarsRoverEnv
    from mars_rover_env.bank import load_manifest, require_compiled_bank




    manifest = load_manifest()
    bank_version = require_compiled_bank(manifest)

    env = MarsRoverEnv(render_mode="rgb_array", render_width=64, render_height=36)
    obs, _ = env.reset(seed=123)
    obs, reward, terminated, truncated, _ = env.step(0)
    frame = env.render()
    assert obs.shape == env.observation_space.shape
    assert frame.shape == (36, 64, 3)
    print(
        "Smoke test: OK "
        f"(reward={reward:.4f}, terminated={terminated}, truncated={truncated}, "
        f"bank={bank_version})"
    )


if __name__ == "__main__":
    main()
