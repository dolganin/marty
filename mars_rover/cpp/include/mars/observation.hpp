#pragma once

#include <cstddef>

#include "mars/state.hpp"

namespace mars {

constexpr int kTerrainSamplesAhead = 24;
constexpr int kBiomeSamplesAhead = 8;
constexpr int kObservationDim =
    8 +                 // body position, velocity, angle, angular velocity, energy
    kMaxWheels * 3 +    // contact, slip, normal force per wheel slot
    kTerrainSamplesAhead * 2 +
    kBiomeSamplesAhead * 3 +  // raw surface echo, discontinuity and vibration ahead
    20;                // transmission, thermal/solar/lidar state, action/reward and episode

struct ObservationView {
  float* data = nullptr;
  int dim = kObservationDim;
};

}  // namespace mars
