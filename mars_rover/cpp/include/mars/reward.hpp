#pragma once

#include "mars/state.hpp"

namespace mars {

struct RewardConfig {
  float progress_scale = 1.0f;
  float energy_cost_scale = 0.01f;
  float flip_penalty = 5.0f;
  float stuck_penalty = 0.05f;
  float hard_contact_penalty = 0.0f;
  float finish_bonus = 100.0f;
};

struct TerminationConfig {
  float finish_x = 100.0f;
  float min_energy = 0.0f;
  float flip_angle = 2.2f;
  int stuck_steps = 180;
  int max_steps = 3000;
};

float compute_reward(const RewardConfig& config, const RoverState& state, float energy_cost,
                     bool finished, bool flipped, bool stuck);

}  // namespace mars
