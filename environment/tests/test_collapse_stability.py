from __future__ import annotations

import _mars_rover_cpp as native

from mars_rover_env import MarsRoverEnv


def _biome_index(biome_id: str) -> int:
    return next(
        int(item["index"])
        for item in native.biome_catalog()
        if item["id"] == biome_id
    )


def test_molten_window_does_not_inject_a_phase_wide_rollover() -> None:
    """A collapse may remove support; it must not rotate a parked rover by itself."""
    env = MarsRoverEnv(
        fixed_biome_id=_biome_index("collapse_window_flats"), biome_split=0
    )
    env.reset(seed=7, options={"trial_start": True})
    peak_angle = 0.0
    for _ in range(180):
                                                                            
                                                                               
        _, _, terminated, truncated, _ = env.step(1 << 11)
        debug = env.debug_info()
        peak_angle = max(peak_angle, abs(float(debug["angle"])))
        assert not (terminated or truncated), debug
    env.close()
    assert peak_angle < 0.2
