#pragma once

#include "mars/contacts.hpp"
#include "mars/math.hpp"

namespace mars {

enum class MechanicType : int {
  Normal = 0,
  Sand = 1,
  Ice = 2,
  Mud = 3,
  Wind = 4,
  LowGravity = 5,
  Crust = 6,
};

struct MechanicParams {
  float friction_mul = 1.0f;
  float sink_rate = 0.0f;
  float viscosity = 0.0f;
  float wind_force = 0.0f;
  float gravity_mul = 1.0f;
  float energy_drain_mul = 1.0f;
  float crust_deform = 0.0f;
};

struct MechanicContext {
  WheelContact* contact = nullptr;
  Vec2* wheel_force = nullptr;
  Vec2* body_force = nullptr;
  float* energy_cost = nullptr;
  float dt = 1.0f / 60.0f;
  float wheel_radius = 0.24f;
  float base_friction = 1.0f;
  float drive_force = 0.0f;
  float wheel_speed = 0.0f;
};

void apply_mechanic(MechanicType type, const MechanicParams& params, MechanicContext& ctx);

}  // namespace mars
