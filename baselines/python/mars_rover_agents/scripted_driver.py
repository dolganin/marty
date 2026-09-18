"""A five-rule scripted driver: the honest floor any learned policy must beat.

No RL, no learning - just react to debug_info() every step:
  1. Engine not running -> ignition.
  2. In neutral, or an upshift is recommended -> throttle + shift-up (releases
     the clutch pedal so the powered-upshift path in physics.cpp engages).
  3. Load has dropped enough that the gearbox wants to drop a gear -> clutch +
     shift-down.
  4. Otherwise -> throttle.

This never reasons about *which* biome mechanic it is in - it only reacts to
drivetrain state. Any adaptation-capable agent should clear it by a wide
margin; if it does not, the bottleneck is drivetrain bootstrapping / physics,
not "adaptation to biomes" - see AGENT_TASK_ADAPTATION.md.
"""

from __future__ import annotations

                                                                              
                                                                               
CONTROL_GAS = 1 << 0
CONTROL_CLUTCH_PEDAL = 1 << 3
CONTROL_SHIFT_UP = 1 << 6
CONTROL_SHIFT_DOWN = 1 << 7
CONTROL_IGNITION = 1 << 9


def scripted_action(debug: dict) -> int:
    if not debug.get("engine_running", False):
        return CONTROL_IGNITION
    if debug.get("gear") == "N" or debug.get("upshift_recommended", False):
        return CONTROL_GAS | CONTROL_SHIFT_UP
    if debug.get("should_shift_down", False):
        return CONTROL_CLUTCH_PEDAL | CONTROL_SHIFT_DOWN
    return CONTROL_GAS
