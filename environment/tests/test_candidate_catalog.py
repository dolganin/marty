import pytest

from mars_rover_env.candidates import candidate_config, candidates
from mars_rover_env.actions import ACTION_MACROS


def test_catalog_contains_60_stable_candidates():
    catalogue = candidates()
    assert len(catalogue) == 60
    assert len(set(catalogue)) == 60
    assert all(len(item.seeds) >= 3 for item in catalogue.values())


def test_candidate_config_is_five_minute_endgame_override():
    item = candidates()["blind_relief"]
    cfg = candidate_config(item)
    assert cfg.force_endgame_difficulty
    assert cfg.evaluation_stack_count == len(item.mechanics)
    assert cfg.termination.trial_time_limit == 300.0
    assert cfg.physics.lidar_energy_cost == pytest.approx(1.8)


def test_agent_macros_include_candidate_specific_actuators():
    assert 1 << 14 in ACTION_MACROS  # climb mode
    assert 1 << 15 in ACTION_MACROS  # propeller


def test_every_candidate_has_a_manual_briefing_description():
    for item in candidates().values():
        assert item.description.strip()
        assert item.mechanics
        assert item.helpful_actions
        assert item.support in {"implemented", "partial"}
