#include "mars/biome_bank.hpp"

namespace mars::candidate_check {
constexpr FrozenMechanismStack kCandidate{{MechanicType::Ice, MechanicType::Wind, MechanicType::Normal, MechanicType::Normal}, 2, BiomeSplit::Train};
static_assert(kCandidate.count >= 1 && kCandidate.count <= kMaxActiveMechanisms);
}
