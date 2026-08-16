import pytest

from mars_rover_env.tools.equate_banks import fit_linear_equating
from mars_rover_env.tools.ranking_stability import kendall_tau_b
from mars_rover_env.tools.train_robust_ppo import _latest_checkpoint
from mars_rover_env.tools.evaluate_biomes import evaluate_adaptive_policy
from mars_rover_env.bank import equated_score, normalized_test_return


def test_common_item_equating_recovers_linear_scale() -> None:
    reference = {"evaluations": {}}
    new = {"evaluations": {}}
    for agent_index in range(3):
        agent = f"agent_{agent_index}"
        new_scores = {
            f"anchor_{anchor}": {
                "mean_return": float(anchor + agent_index),
                "normalized_return": float(anchor + agent_index),
            }
            for anchor in range(8)
        }
        reference_scores = {
            anchor: {
                "mean_return": 2.0 * result["mean_return"] + 1.0,
                "normalized_return": 2.0 * result["normalized_return"] + 1.0,
            }
            for anchor, result in new_scores.items()
        }
        protocol = {
            "policy_provenance": {"kind": "test", "hash": agent},
            "seeds": [1, 2],
            "episodes_per_trial": 4,
        }
        new["evaluations"][agent] = {**protocol, "anchors": new_scores}
        reference["evaluations"][agent] = {**protocol, "anchors": reference_scores}
    fit = fit_linear_equating(reference, new)
    assert fit["a"] == pytest.approx(2.0)
    assert fit["b"] == pytest.approx(1.0)
    assert fit["r2"] == pytest.approx(1.0)


def test_kendall_tau_b_detects_stable_and_reversed_rankings() -> None:
    scores = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert kendall_tau_b(scores, scores) == pytest.approx(1.0)
    assert kendall_tau_b(scores, {"a": 3.0, "b": 2.0, "c": 1.0}) == pytest.approx(-1.0)


def test_test_return_uses_gate_baselines_for_normalization() -> None:
    manifest = {
        "biomes": [
            {
                "id": "held_out",
                "split": "test",
                "r_random": 10.0,
                "r_robust": 30.0,
            }
        ]
    }
    assert normalized_test_return(manifest, "held_out", 20.0) == pytest.approx(0.5)


def test_equated_score_applies_the_versioned_linear_map() -> None:
    table = {"coefficients": {"v1": {"a": 1.5, "b": -2.0}}}
    assert equated_score(table, "v1", 10.0) == pytest.approx(13.0)


def test_latest_ppo_checkpoint_uses_numeric_step_order(tmp_path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "robust_ppo_900_steps.zip").touch()
    latest = checkpoint_dir / "robust_ppo_1200_steps.zip"
    latest.touch()
    (checkpoint_dir / "unrelated.zip").touch()
    assert _latest_checkpoint(tmp_path) == latest


def test_adaptive_evaluation_keeps_one_policy_instance_per_trial(tmp_path) -> None:
    config = tmp_path / "short.yaml"
    config.write_text("termination:\n  max_steps: 1\n", encoding="utf-8")
    factory_calls = []
    action_calls = []

    def factory(seed):
        factory_calls.append(seed)

        def policy(_obs, _debug, _step):
            action_calls.append(seed)
            return 0

        return policy

    summary = evaluate_adaptive_policy(
        biome_id=0,
        policy_factory=factory,
        trial_seeds=[10, 20],
        episodes_per_trial=3,
        max_steps=1,
        config_path=str(config),
    )
    assert factory_calls == [10, 20]
    assert action_calls == [10, 10, 10, 20, 20, 20]
    assert summary.trials == 2
    assert summary.episodes == 6
