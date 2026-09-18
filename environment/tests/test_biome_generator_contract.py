import pytest

from mars_rover_env.tools.generate_biomes import CandidateRejected, _normalize_candidate


MINIMAL = """class SafeBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return \"safe_biome\"; }
};"""


def test_candidate_normalization_allows_only_one_class() -> None:
    candidate = _normalize_candidate(
        {"class_name": "SafeBiome", "skill_stratum": "traction_loss", "source": MINIMAL}
    )
    assert candidate["source"] == MINIMAL

    with pytest.raises(CandidateRejected, match="globals"):
        _normalize_candidate(
            {
                "class_name": "SafeBiome",
                "skill_stratum": "traction_loss",
                "source": MINIMAL + "\nint hidden_state = 0;",
            }
        )
