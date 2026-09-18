#include "mars/biome_bank.hpp"

namespace mars::candidate_check {
constexpr FrozenMechanismStack kCandidate{{MechanicType::Sand, MechanicType::Wind, MechanicType::Mud, MechanicType::Crust}, 4, BiomeSplit::Train};
static_assert(kCandidate.count >= 1 && kCandidate.count <= kMaxActiveMechanisms);
}
