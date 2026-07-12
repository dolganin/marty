#include "mars/physics.hpp"

#include <cmath>

namespace mars {

PhysicsEngine::PhysicsEngine(PhysicsConfig config) : config_(config) {}

void PhysicsEngine::initialize_state(const RoverRig& rig, RoverState& state, Vec2 spawn) const {
  state = RoverState{};
  state.body.position = spawn;
  state.body.mass = rig.body.mass;
  state.body.inv_mass = rig.body.mass > 0.0f ? 1.0f / rig.body.mass : 0.0f;
  state.body.inertia = rig.body.inertia;
  state.body.inv_inertia = rig.body.inertia > 0.0f ? 1.0f / rig.body.inertia : 0.0f;
  state.wheel_count = static_cast<int>(rig.wheels.size());
  for (int i = 0; i < state.wheel_count; ++i) {
    const auto& wr = rig.wheels[static_cast<size_t>(i)];
    auto& w = state.wheels[static_cast<size_t>(i)];
    w.radius = wr.radius;
    w.mass = wr.mass;
    w.inv_mass = wr.mass > 0.0f ? 1.0f / wr.mass : 0.0f;
    w.position = spawn + wr.local_anchor + Vec2{0.0f, -wr.suspension.rest_length};
  }
  state.energy = 1.0f;
  state.previous_x = state.body.position.x;
}

PhysicsStepStats PhysicsEngine::step(const RoverRig& rig, const Terrain& terrain, RoverState& state,
                                     int discrete_action, MechanicType mechanic,
                                     const MechanicParams& mechanic_params) const {
  PhysicsStepStats stats{};
  const float dt = config_.dt;
  const ControlInput control =
      decode_discrete_action(discrete_action, config_.motor_torque, config_.body_tilt_torque);

  const float gravity = config_.gravity *
                        (mechanic == MechanicType::LowGravity ? mechanic_params.gravity_mul : 1.0f);
  Vec2 body_force{0.0f, state.body.mass * gravity};
  float body_torque = control.body_torque;

  for (int i = 0; i < state.wheel_count; ++i) {
    auto& wheel = state.wheels[static_cast<size_t>(i)];
    const auto& wr = rig.wheels[static_cast<size_t>(i)];
    Vec2 wheel_force{0.0f, wheel.mass * gravity};

    const Vec2 anchor_world = state.body.position + rotate(wr.local_anchor, state.body.angle);
    const Vec2 spring_vec = wheel.position - anchor_world;
    const float spring_len = length(spring_vec);
    const Vec2 spring_dir = spring_len > 1.0e-5f ? spring_vec / spring_len : Vec2{0.0f, -1.0f};
    const float spring_vel = dot(wheel.velocity - state.body.velocity, spring_dir);
    const float spring_force_mag =
        -wr.suspension.stiffness * (spring_len - wr.suspension.rest_length) -
        wr.suspension.damping * spring_vel;
    const Vec2 spring_force = spring_dir * spring_force_mag;
    wheel_force += spring_force;
    body_force -= spring_force;
    body_torque += cross(anchor_world - state.body.position, -spring_force);

    auto terrain_sample = terrain.query(wheel.position.x);
    const float penetration = terrain_sample.height + wheel.radius - wheel.position.y;
    WheelContact contact{};
    contact.wheel_index = i;
    if (penetration > 0.0f) {
      contact.active = true;
      contact.point = {wheel.position.x, terrain_sample.height};
      contact.normal = terrain_sample.normal;
      contact.tangent = normalized({contact.normal.y, -contact.normal.x});
      contact.penetration = penetration;
      contact.ground_height = terrain_sample.height;
      contact.slope = terrain_sample.slope;

      const float normal_vel = dot(wheel.velocity, contact.normal);
      const float normal_force = penetration * 1800.0f - normal_vel * 80.0f;
      contact.normal_force = std::max(0.0f, normal_force);
      wheel_force += contact.normal * contact.normal_force;

      const float drive_force = control.motor_torque / std::max(0.05f, wheel.radius);
      MechanicContext ctx{};
      ctx.contact = &contact;
      ctx.wheel_force = &wheel_force;
      ctx.body_force = &body_force;
      ctx.energy_cost = &stats.energy_cost;
      ctx.dt = dt;
      ctx.wheel_radius = wheel.radius;
      ctx.base_friction = config_.wheel_friction;
      ctx.drive_force = drive_force;
      ctx.wheel_speed = dot(wheel.velocity, contact.tangent);
      apply_mechanic(mechanic, mechanic_params, ctx);

      if (control.brake > 0.0f) {
        const float brake_force =
            -clamp(dot(wheel.velocity, contact.tangent) * config_.brake_strength,
                   -contact.normal_force, contact.normal_force);
        wheel_force += contact.tangent * brake_force;
      }
    }

    wheel.in_contact = contact.active;
    wheel.normal_force = contact.normal_force;
    wheel.slip = contact.slip;
    stats.contacts[static_cast<size_t>(i)] = contact;

    wheel.velocity += wheel_force * wheel.inv_mass * dt;
    wheel.velocity *= (1.0f - config_.linear_damping);
    wheel.position += wheel.velocity * dt;
    wheel.angular_velocity += control.motor_torque * wheel.inv_mass * dt;
    wheel.angle += wheel.angular_velocity * dt;
    stats.energy_cost += std::abs(control.motor_torque) * 0.0005f * dt;
  }

  if (rig.body.collision.type == CollisionType::Box) {
    const float hw = rig.body.collision.size.x * 0.5f;
    const float hh = rig.body.collision.size.y * 0.5f;
    const Vec2 samples[3] = {{-hw, -hh}, {0.0f, -hh}, {hw, -hh}};
    for (const Vec2 local : samples) {
      const Vec2 point = state.body.position + rotate(local, state.body.angle);
      const auto ground = terrain.query(point.x);
      const float penetration = ground.height - point.y;
      if (penetration > 0.0f) {
        const Vec2 r = point - state.body.position;
        const Vec2 point_velocity = state.body.velocity + perp(r) * state.body.angular_velocity;
        const float normal_vel = dot(point_velocity, ground.normal);
        const float force_mag = std::max(0.0f, penetration * 3000.0f - normal_vel * 120.0f);
        const Vec2 force = ground.normal * force_mag;
        body_force += force;
        body_torque += cross(r, force);
        stats.hard_contact = std::max(stats.hard_contact, force_mag);
      }
    }
  }

  state.body.velocity += body_force * state.body.inv_mass * dt;
  state.body.velocity *= (1.0f - config_.linear_damping);
  state.body.position += state.body.velocity * dt;
  state.body.angular_velocity += body_torque * state.body.inv_inertia * dt;
  state.body.angular_velocity *= (1.0f - config_.angular_damping);
  state.body.angle += state.body.angular_velocity * dt;
  state.energy = std::max(0.0f, state.energy - stats.energy_cost);
  return stats;
}

void PhysicsEngine::apply_body_impulse(RigidBodyState& body, Vec2 impulse, Vec2 point) const {
  body.velocity += impulse * body.inv_mass;
  body.angular_velocity += cross(point - body.position, impulse) * body.inv_inertia;
}

}  // namespace mars
