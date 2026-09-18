from __future__ import annotations

from mars_rover_env.actions import ACTION_MACROS


def macro_action(index: int) -> int:
    if index < 0 or index >= len(ACTION_MACROS):
        raise ValueError(f"macro action index out of bounds: {index}")
    return ACTION_MACROS[index]
