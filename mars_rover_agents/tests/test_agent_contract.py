from __future__ import annotations

import numpy as np

from mars_rover_agents.agent import Agent
from mars_rover_agents.agents import RandomAgent
from mars_rover_agents.harness import EvaluationSpec, evaluate


def test_random_agent_uses_public_action_contract() -> None:
    agent = RandomAgent(7)
    agent.reset(trial_start=True)
    action = agent.act(np.zeros(124, dtype=np.float32))
    assert 0 <= action < 4096
    agent.observe(1.0, False, {"episode_index": 0})


class MemoryProbe(Agent):
    def __init__(self) -> None:
        self.trial_memory = 0
        self.episode_memory = 0

    def reset(self, trial_start: bool) -> None:
        if trial_start:
            self.trial_memory = 0
        self.episode_memory = 0

    def act(self, obs: np.ndarray) -> int:
        del obs
        self.episode_memory += 1
        return 0

    def observe(self, reward: float, done: bool, info: dict) -> None:
        del reward, done, info
        self.trial_memory += 1


def test_reset_false_preserves_trial_memory() -> None:
    agent = MemoryProbe()
    agent.reset(True)
    agent.act(np.zeros(1))
    agent.observe(0.0, True, {})
    agent.reset(False)
    assert agent.trial_memory == 1
    assert agent.episode_memory == 0


def test_harness_never_passes_debug_state_to_agent(tmp_path) -> None:
    agents: list[MemoryProbe] = []

    class StrictProbe(MemoryProbe):
        def observe(self, reward: float, done: bool, info: dict) -> None:
            assert set(info) == {"terminated", "truncated", "episode_index"}
            assert "mechanic" not in info
            assert "friction_mul" not in info
            super().observe(reward, done, info)

    def factory(_seed: int) -> StrictProbe:
        agent = StrictProbe()
        agents.append(agent)
        return agent

    payload = evaluate(
        factory,
        EvaluationSpec(
            split="test",
            seeds=(19,),
            episodes_per_trial=2,
            max_steps=1,
            render_trials=0,
        ),
        output_dir=tmp_path,
    )
    # Derived from the compiled bank rather than hard-coded: this asserted 10 held-out
    # biomes and broke the moment the split grew to 11, which says nothing about the
    # contract it is meant to police.
    import _mars_rover_cpp as native

    held_out = sum(1 for item in native.biome_catalog() if int(item["split"]) == 2)
    assert payload["summary"]["episode_count"] == held_out * 2
    # Research telemetry is recorded from evaluator-only debug state, while the
    # agent-facing contract above remains deliberately minimal.
    for key in (
        "mean_lidar_scan_count",
        "mean_lidar_active_fraction",
        "mean_lidar_energy_spent",
        "mean_lidar_airborne_attempt_count",
        "mean_solar_toggle_count",
        "mean_charging_active_fraction",
        "mean_solar_energy_gained",
        "mean_ballistic_flight_count",
        "mean_airborne_fraction",
        "mean_safe_landing_count",
    ):
        assert key in payload["summary"]
    assert {"lidar_scan_count", "solar_energy_gained", "ballistic_flight_count"} <= set(
        payload["episodes"][0]
    )
    assert len(agents) == held_out
    assert all(agent.trial_memory == 2 for agent in agents)
