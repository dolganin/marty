#pragma once

#include <cstddef>

#include "mars/state.hpp"

namespace mars {

constexpr int kTerrainSamplesAhead = 24;
constexpr int kBiomeSamplesAhead = 8;
constexpr int kObservationDim =
    8 +                 
    kMaxWheels * 3 +    
    kTerrainSamplesAhead * 2 +
    kBiomeSamplesAhead * 3 +  
    20;                

struct ObservationView {
  float* data = nullptr;
  int dim = kObservationDim;
};

}  
