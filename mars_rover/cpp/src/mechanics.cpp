#include "mars/mechanics.hpp"

namespace mars {

void apply_mechanic(MechanicType type, const MechanicParams& params, MechanicContext& ctx) {
  float friction_mul = params.friction_mul;
  switch (type) {
    case MechanicType::Sand:
      friction_mul *= 0.65f;
      if (ctx.contact) {
        ctx.contact->penetration += params.sink_rate * ctx.dt;
      }
      if (ctx.energy_cost) {
        *ctx.energy_cost += 0.02f * params.energy_drain_mul * ctx.dt;
      }
      break;
    case MechanicType::Ice:
      friction_mul *= 0.18f;
      break;
    case MechanicType::Mud:
      friction_mul *= 0.5f;
      if (ctx.wheel_force) {
        *ctx.wheel_force += Vec2{-params.viscosity * ctx.wheel_speed, 0.0f};
      }
      if (ctx.energy_cost) {
        *ctx.energy_cost += 0.04f * params.energy_drain_mul * ctx.dt;
      }
      break;
    case MechanicType::Wind:
      if (ctx.body_force) {
        *ctx.body_force += Vec2{params.wind_force, 0.0f};
      }
      break;
    case MechanicType::LowGravity:
    case MechanicType::Crust:
    case MechanicType::Normal:
    default:
      break;
  }

  if (ctx.wheel_force && ctx.contact && ctx.contact->active) {
    const float max_drive = ctx.contact->normal_force * ctx.base_friction * friction_mul;
    const float drive = clamp(ctx.drive_force, -max_drive, max_drive);
    *ctx.wheel_force += ctx.contact->tangent * drive;
    ctx.contact->slip = std::abs(ctx.drive_force - drive) / (std::abs(ctx.drive_force) + 1.0f);
  }
}

}  // namespace mars
