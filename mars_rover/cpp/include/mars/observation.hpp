#pragma once

#include <cstddef>

#include "mars/state.hpp"

namespace mars {

constexpr int kTerrainSamplesAhead = 24;
constexpr int kObservationDim =
    8 +                 // body position, velocity, angle, angular velocity, energy
    kMaxWheels * 3 +    // contact, slip, normal force per wheel slot
    kTerrainSamplesAhead * 2 +
    3;                 // previous action, previous reward, episode_in_trial

struct ObservationView {
  float* data = nullptr;
  int dim = kObservationDim;
};

}  // namespace mars
