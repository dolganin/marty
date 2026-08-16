from __future__ import annotations

from mars_rover_agents.curriculum import CurriculumStage, allocate_timesteps, stage_config


def test_allocate_timesteps_is_exact_and_nonempty() -> None:
    stages = (
        CurriculumStage(1, 0.2, 0.1),
        CurriculumStage(4, 0.3, 0.5),
        CurriculumStage(14, 0.5, 1.0),
    )
    budgets = allocate_timesteps(101, stages)
    assert sum(budgets) == 101
    assert all(value > 0 for value in budgets)
    assert budgets[2] > budgets[1] > budgets[0]


def test_stage_config_does_not_mutate_base_and_scales_only_curriculum_fields() -> None:
    base = {
        "env": {"chain_zone_count": 14},
        "termination": {"finish_x": 800.0, "max_steps": 3000},
        "reward": {"flip_penalty": 300.0, "stuck_penalty": 300.0, "finish_bonus": 100.0},
    }
    result = stage_config(base, CurriculumStage(2, 1.0, 0.1, finish_x=90.0, max_steps=1200))
    assert base["env"]["chain_zone_count"] == 14
    assert result["env"]["chain_zone_count"] == 2
    assert result["termination"] == {"finish_x": 90.0, "max_steps": 1200}
    assert result["reward"]["flip_penalty"] == 30.0
    assert result["reward"]["stuck_penalty"] == 30.0
    assert result["reward"]["finish_bonus"] == 100.0
