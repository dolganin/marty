import json
import sys

from mars_rover_env.bank import write_json
from mars_rover_env.tools import equate_banks, ranking_stability


def _anchor_payload(version: str, scale: float, shift: float) -> dict:
    evaluations = {}
    for agent_index in range(3):
        anchors = {}
        for anchor_index in range(8):
            reference_value = float(agent_index * 10 + anchor_index)
            value = (reference_value - shift) / scale
            anchors[f"anchor_{anchor_index}"] = {
                "mean_return": value,
                "normalized_return": value,
            }
        evaluations[f"agent_{agent_index}"] = {
            "policy_provenance": {"kind": "fixture", "agent": agent_index},
            "seeds": [4000, 4001, 4002, 4003, 4004],
            "episodes_per_trial": 4,
            "anchors": anchors,
        }
    return {"schema_version": 2, "bank_version": version, "evaluations": evaluations}


def test_three_bank_equating_and_kendall_cli_workflow(tmp_path, monkeypatch, capsys) -> None:
    reference = tmp_path / "v0-anchors.json"
    new_one = tmp_path / "v1-anchors.json"
    new_two = tmp_path / "v2-anchors.json"
    table = tmp_path / "equating.json"
    write_json(reference, _anchor_payload("v0", 1.0, 0.0))
    write_json(new_one, _anchor_payload("v1", 2.0, 1.0))
    write_json(new_two, _anchor_payload("v2", 0.5, -3.0))

    for new_path in (new_one, new_two):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "mars-rover-equate-banks",
                "--reference",
                str(reference),
                "--new",
                str(new_path),
                "--output",
                str(table),
            ],
        )
        equate_banks.main()

    coefficients = json.loads(table.read_text(encoding="utf-8"))["coefficients"]
    assert coefficients["v0"]["a"] == 1.0
    assert abs(coefficients["v1"]["a"] - 2.0) < 1.0e-10
    assert abs(coefficients["v1"]["b"] - 1.0) < 1.0e-10
    assert abs(coefficients["v2"]["a"] - 0.5) < 1.0e-10
    assert abs(coefficients["v2"]["b"] + 3.0) < 1.0e-10

    score_paths = []
    raw_scores = {
        "v0": {"slow": 0.1, "middle": 0.5, "fast": 0.9},
        "v1": {"slow": -0.45, "middle": -0.25, "fast": -0.05},
        "v2": {"slow": 6.2, "middle": 7.0, "fast": 7.8},
    }
    for version, scores in raw_scores.items():
        path = tmp_path / f"{version}-scores.json"
        write_json(path, {"bank_version": version, "scores": scores})
        score_paths.append(path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mars-rover-ranking-stability",
            *(str(path) for path in score_paths),
            "--equating",
            str(table),
            "--min-tau",
            "0.99",
        ],
    )
    ranking_stability.main()
    output = capsys.readouterr().out
    assert output.count("tau_b=1.000000") == 3
