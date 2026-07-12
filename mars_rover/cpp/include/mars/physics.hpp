#pragma once

#include "mars/action.hpp"
#include "mars/contacts.hpp"
#include "mars/mechanics.hpp"
#include "mars/rover_rig.hpp"
#include "mars/state.hpp"
#include "mars/terrain.hpp"

namespace mars {

struct PhysicsConfig {
  float dt = 1.0f / 60.0f;
  float gravity = -3.71f;
  float wheel_friction = 1.2f;
  float motor_torque = 18.0f;
  float brake_strength = 20.0f;
  float body_tilt_torque = 8.0f;
  float linear_damping = 0.01f;
  float angular_damping = 0.02f;
};

struct PhysicsStepStats {
  float energy_cost = 0.0f;
  float hard_contact = 0.0f;
  ContactArray contacts{};
};

class PhysicsEngine {
 public:
  explicit PhysicsEngine(PhysicsConfig config = {});

  void initialize_state(const RoverRig& rig, RoverState& state, Vec2 spawn) const;
  PhysicsStepStats step(const RoverRig& rig, const Terrain& terrain, RoverState& state,
                        int discrete_action, MechanicType mechanic,
                        const MechanicParams& mechanic_params) const;

 private:
  void apply_body_impulse(RigidBodyState& body, Vec2 impulse, Vec2 point) const;

  PhysicsConfig config_{};
};

}  // namespace mars
