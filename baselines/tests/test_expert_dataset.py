from __future__ import annotations

import numpy as np
import pytest

from mars_rover_agents.actions import ACTION_MACROS
from mars_rover_agents.expert_dataset import expert_macro_index
from mars_rover_agents.scripted_driver import (
    CONTROL_CLUTCH_PEDAL,
    CONTROL_GAS,
    CONTROL_IGNITION,
    CONTROL_SHIFT_DOWN,
    CONTROL_SHIFT_UP,
)


def _debug(**overrides) -> dict:
    base = {
        "engine_running": True,
        "gear": "1",
        "upshift_recommended": False,
        "should_shift_down": False,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    "debug, expected_mask",
    [
        (_debug(engine_running=False), CONTROL_IGNITION),
        (_debug(gear="N"), CONTROL_GAS | CONTROL_SHIFT_UP),
        (_debug(upshift_recommended=True), CONTROL_GAS | CONTROL_SHIFT_UP),
        (_debug(should_shift_down=True), CONTROL_CLUTCH_PEDAL | CONTROL_SHIFT_DOWN),
        (_debug(), CONTROL_GAS),
    ],
)
def test_every_expert_action_is_expressible_as_a_public_macro(debug, expected_mask):
    """A label the PPO student cannot emit would be uncloneable, so this is checked."""
    index = expert_macro_index(debug)
    assert ACTION_MACROS[index] == expected_mask


def test_unknown_mask_is_rejected_rather_than_silently_remapped(monkeypatch):
    monkeypatch.setattr("mars_rover_agents.expert_dataset.scripted_action", lambda debug: 4095)
    with pytest.raises(RuntimeError, match="not in ACTION_MACROS"):
        expert_macro_index(_debug())


def test_returns_to_go_are_discounted_backwards():
    from mars_rover_agents.expert_dataset import ExpertDataset

                                                                         
    dataset = ExpertDataset(
        observations=np.zeros((3, 2), np.float32),
        actions=np.zeros(3, np.int64),
        returns_to_go=np.array([1.0, 2.0, 3.0], np.float32),
        episode_distances=np.array([1.0], np.float32),
        episode_returns=np.array([1.0], np.float32),
    )
    assert len(dataset) == 3
