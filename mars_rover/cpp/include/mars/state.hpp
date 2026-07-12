#pragma once

#include <array>
#include <cstdint>

#include "mars/math.hpp"

namespace mars {

constexpr int kMaxWheels = 8;
constexpr int kDefaultWheelCount = 2;

struct RigidBodyState {
  Vec2 position{};
  Vec2 velocity{};
  float angle = 0.0f;
  float angular_velocity = 0.0f;
  float mass = 1.0f;
  float inv_mass = 1.0f;
  float inertia = 1.0f;
  float inv_inertia = 1.0f;
};

struct WheelState {
  Vec2 position{};
  Vec2 velocity{};
  float angle = 0.0f;
  float angular_velocity = 0.0f;
  float radius = 0.24f;
  float mass = 1.0f;
  float inv_mass = 1.0f;
  bool in_contact = false;
  float slip = 0.0f;
  float normal_force = 0.0f;
};

struct RoverState {
  RigidBodyState body{};
  std::array<WheelState, kMaxWheels> wheels{};
  int wheel_count = kDefaultWheelCount;
  float energy = 1.0f;
  float damage = 0.0f;
  float previous_x = 0.0f;
  float last_reward = 0.0f;
  int previous_action = 0;
  int step_index = 0;
  int episode_in_trial = 0;
  bool trial_start = true;
};

struct EpisodeResult {
  float reward = 0.0f;
  bool terminated = false;
  bool truncated = false;
};

}  // namespace mars
