#include "mars/biome_bank.hpp"

namespace mars::candidate_check {
constexpr FrozenMechanismStack kCandidate{{MechanicType::Ice, MechanicType::Wind, MechanicType::LowGravity, MechanicType::Normal}, 3, BiomeSplit::Train};
static_assert(kCandidate.count >= 1 && kCandidate.count <= kMaxActiveMechanisms);
}
