#pragma once

// This is the complete LLM-facing biome contract and implementation bank.
// MechanicParams is immutable input for a step and exposes friction_mul, sink_rate,
// viscosity, wind_force, gravity_mul, energy_drain_mul, crust_deform,
// ambient_temperature, thermal_transfer, solar_charge_rate, lidar_energy_mul
// and lidar_range_mul. MechanicContext
// exposes only the current wheel contact/forces, energy accumulator, dt, wheel
// radius/speed, base friction, requested/minimum drive force and immersion. A
// biome must not retain references to it. MechanicBodyContext additionally lets
// it apply arbitrary body force/torque and energy cost using velocity, mass,
// gravity, dt and step index. BiomeVisuals independently controls RGB colors,
// sky/ground/liquid tint, screen brightness, wheel-particle rate/lift/spread/size
// and ambient particles. Keep
// generated implementations between the markers near the bottom.

#include <cmath>
#include <cstdint>
#include <string_view>
#include <vector>

#include "mars/mechanics.hpp"

namespace mars {

enum class BiomeSplit : int { Builtin = 0, Train = 1, Test = 2 };

struct BiomeColor { uint8_t r = 42, g = 35, b = 30; };

struct BiomeVisuals {
  BiomeColor sky{188, 112, 72};
  BiomeColor ground{42, 35, 30};
  BiomeColor particles{150, 118, 82};
  BiomeColor liquid{62, 145, 181};
  float particle_rate = 7.0f;
  float particle_lift = 1.0f;
  float particle_spread = 1.0f;
  int base_particles = 0;
  int max_particles = 36;
  int particle_size = 2;
  int ambient_particles = 0;
  float ambient_drift = 0.0f;
  float screen_brightness = 1.0f;
  bool liquid_surface = false;
};

inline float biome_random01(uint64_t seed, uint64_t stream = 0) noexcept {
  uint64_t x = seed + 0x9e3779b97f4a7c15ULL * (stream + 1ULL);
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  x ^= x >> 31U;
  return static_cast<float>(x >> 40U) * (1.0f / 16777216.0f);
}

class Biome {
 public:
  virtual ~Biome() = default;
  virtual std::string_view id() const noexcept = 0;
  virtual std::string_view display_name() const noexcept = 0;
  virtual MechanicType visual_type() const noexcept = 0;
  virtual BiomeSplit split() const noexcept { return BiomeSplit::Builtin; }
  virtual MechanicParams sample_params(uint64_t seed) const noexcept = 0;
  virtual BiomeVisuals visuals() const noexcept {
    uint32_t hash = 2166136261u;
    for (char ch : id()) hash = (hash ^ static_cast<uint8_t>(ch)) * 16777619u;
    BiomeVisuals v;
    v.ground = {static_cast<uint8_t>(45u + (hash & 127u)),
                static_cast<uint8_t>(35u + ((hash >> 8u) & 127u)),
                static_cast<uint8_t>(30u + ((hash >> 16u) & 127u))};
    v.particles = {static_cast<uint8_t>(110u + (hash & 111u)),
                   static_cast<uint8_t>(100u + ((hash >> 7u) & 111u)),
                   static_cast<uint8_t>(90u + ((hash >> 15u) & 111u))};
    v.sky = {static_cast<uint8_t>(135u + (hash & 63u)),
             static_cast<uint8_t>(65u + ((hash >> 9u) & 63u)),
             static_cast<uint8_t>(40u + ((hash >> 18u) & 47u))};
    v.particle_rate = 4.0f + static_cast<float>((hash >> 5u) & 15u);
    v.particle_lift = 0.45f + static_cast<float>((hash >> 12u) & 15u) * 0.12f;
    v.particle_spread = 0.55f + static_cast<float>((hash >> 20u) & 7u) * 0.18f;
    v.ambient_particles = (hash & 3u) == 0u ? 24 + static_cast<int>((hash >> 24u) & 63u) : 0;
    v.ambient_drift = 1.0f + static_cast<float>((hash >> 16u) & 7u);
    return v;
  }
  virtual float friction_scale(const MechanicParams& p) const noexcept {
    return p.friction_mul;
  }
  virtual void apply_effects(const MechanicParams&, MechanicContext&) const noexcept {}
  virtual void apply_body_effects(const MechanicParams&, MechanicBodyContext&) const noexcept {}

  void apply(const MechanicParams& p, MechanicContext& ctx) const noexcept {
    apply_effects(p, ctx);
    if (!ctx.wheel_force || !ctx.contact || !ctx.contact->active) return;
    float friction = friction_scale(p);
    if (visual_type() == MechanicType::Liquid) {
      friction = 1.0f + (friction - 1.0f) * ctx.immersion;
    }
    const float limit = std::max(ctx.minimum_drive_limit,
                                 ctx.contact->normal_force * ctx.base_friction * friction);
    const float drive = clamp(ctx.drive_force, -limit, limit);
    *ctx.wheel_force += ctx.contact->tangent * drive;
    ctx.contact->slip = std::abs(ctx.drive_force - drive) / (std::abs(ctx.drive_force) + 1.0f);
  }
};

class NormalBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "normal"; }
  std::string_view display_name() const noexcept override { return "Normal"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeVisuals visuals() const noexcept override { return {}; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.ambient_temperature = -65.0f + 30.0f * biome_random01(s);
    p.thermal_transfer = 0.9f + 0.2f * biome_random01(s, 1);
    p.solar_charge_rate = 0.9f + 0.5f * biome_random01(s, 2); return p;
  }
};

class SandBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "sand"; }
  std::string_view display_name() const noexcept override { return "Sand"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.friction_mul = 0.50f + 0.28f * biome_random01(s);
    p.sink_rate = 0.018f + 0.035f * biome_random01(s, 1);
    p.energy_drain_mul = 1.25f + 0.65f * biome_random01(s, 2);
    p.ambient_temperature = -25.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.55f + 0.25f * biome_random01(s, 4);
    p.solar_charge_rate = 0.25f + 0.25f * biome_random01(s, 5); return p;
  }
  float friction_scale(const MechanicParams& p) const noexcept override { return p.friction_mul * 0.58f; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {184, 139, 76}; v.particles = {218, 178, 112};
    v.particle_rate = 10.0f; v.ambient_particles = 72; v.ambient_drift = 3.0f; return v;
  }
  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) c.contact->penetration += p.sink_rate * c.dt * (1.0f + std::abs(c.wheel_speed));
    if (c.wheel_force && c.contact) {
      *c.wheel_force += c.contact->tangent * (-4.5f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.06f);
    }
    if (c.energy_cost) *c.energy_cost += (0.025f + std::abs(c.wheel_speed) * 0.018f) * p.energy_drain_mul * c.dt;
  }
  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!c.body_force) return;
    c.body_force->x -= c.velocity.x * c.mass * 0.45f;
    c.body_force->x += 0.8f + std::sin(static_cast<float>(c.step_index) * 0.071f) * 0.55f;
    c.body_force->y += c.mass * c.gravity * 0.05f;
  }
};

class IceBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "ice"; }
  std::string_view display_name() const noexcept override { return "Ice"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.friction_mul = 0.18f + 0.20f * biome_random01(s);
    p.ambient_temperature = -105.0f + 30.0f * biome_random01(s, 1);
    p.thermal_transfer = 1.6f + 0.5f * biome_random01(s, 2);
    p.solar_charge_rate = 2.2f + 0.8f * biome_random01(s, 3); return p;
  }
  float friction_scale(const MechanicParams& p) const noexcept override { return p.friction_mul * 0.70f; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {112, 172, 204}; v.particles = {190, 225, 238};
    v.particle_rate = 3.0f; return v;
  }
};

class MudBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "mud"; }
  std::string_view display_name() const noexcept override { return "Mud"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.friction_mul = 0.38f + 0.28f * biome_random01(s);
    p.viscosity = 1.5f + 4.0f * biome_random01(s, 1); p.energy_drain_mul = 1.5f + biome_random01(s, 2);
    p.ambient_temperature = -55.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.2f + 0.4f * biome_random01(s, 4);
    p.solar_charge_rate = 0.55f + 0.35f * biome_random01(s, 5); return p;
  }
  float friction_scale(const MechanicParams& p) const noexcept override { return p.friction_mul * 0.46f; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {67, 49, 39}; v.particles = {92, 68, 50};
    v.particle_rate = 8.0f; v.particle_lift = 0.65f; return v;
  }
  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      *c.wheel_force += c.contact->tangent * (-p.viscosity * 2.4f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.10f);
    }
    if (c.energy_cost) *c.energy_cost += (0.05f + std::abs(c.wheel_speed) * 0.025f) * p.energy_drain_mul * c.dt;
  }
  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      c.body_force->x -= c.velocity.x * c.mass * 1.15f;
      c.body_force->y -= c.velocity.y * c.mass * 0.35f;
    }
    if (c.energy_cost) *c.energy_cost += std::abs(c.velocity.x) * 0.004f * c.dt;
  }
};

class WindBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "wind"; }
  std::string_view display_name() const noexcept override { return "Wind"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.wind_force = 5.0f + 10.0f * biome_random01(s);
    p.ambient_temperature = -90.0f + 35.0f * biome_random01(s, 1);
    p.thermal_transfer = 1.9f + 0.6f * biome_random01(s, 2);
    p.solar_charge_rate = 0.65f + 0.35f * biome_random01(s, 3); return p;
  }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {92, 67, 52}; v.particles = {174, 130, 91};
    v.ambient_particles = 28; v.ambient_drift = 5.0f; return v;
  }
};

class LowGravityBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "low_gravity"; }
  std::string_view display_name() const noexcept override { return "Low gravity"; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.gravity_mul = 0.45f + 0.25f * biome_random01(s);
    p.ambient_temperature = -70.0f + 30.0f * biome_random01(s, 1);
    p.thermal_transfer = 0.65f + 0.25f * biome_random01(s, 2);
    p.solar_charge_rate = 1.2f + 0.5f * biome_random01(s, 3); return p;
  }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {83, 69, 91}; v.particles = {145, 122, 151};
    v.particle_lift = 1.8f; return v;
  }
};

class CrustBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "crust"; }
  std::string_view display_name() const noexcept override { return "Crust"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.friction_mul = 0.75f + 0.25f * biome_random01(s);
    p.crust_deform = 0.006f + 0.02f * biome_random01(s, 1);
    p.ambient_temperature = -45.0f + 25.0f * biome_random01(s, 2);
    p.thermal_transfer = 0.7f + 0.25f * biome_random01(s, 3);
    p.solar_charge_rate = 1.1f + 0.5f * biome_random01(s, 4); return p;
  }
  float friction_scale(const MechanicParams& p) const noexcept override { return p.friction_mul * 1.12f; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {137, 72, 42}; v.particles = {185, 105, 67}; return v;
  }
  void apply_effects(const MechanicParams&, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact && c.contact->penetration > 0.012f)
      *c.wheel_force += c.contact->tangent * (-1.4f * c.wheel_speed);
  }
};

class LiquidBiome final : public Biome {
 public:
  std::string_view id() const noexcept override { return "liquid"; }
  std::string_view display_name() const noexcept override { return "Liquid"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p; p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.viscosity = 0.55f + 0.80f * biome_random01(s, 1); p.energy_drain_mul = 1.15f + 0.45f * biome_random01(s, 2);
    p.ambient_temperature = -35.0f + 23.0f * biome_random01(s, 3);
    p.thermal_transfer = 4.0f + 1.5f * biome_random01(s, 4);
    p.solar_charge_rate = 0.25f + 0.25f * biome_random01(s, 5); return p;
  }
  float friction_scale(const MechanicParams& p) const noexcept override { return p.friction_mul * 0.28f; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v; v.ground = {39, 54, 57}; v.particles = {86, 151, 178};
    v.liquid = {62, 145, 181}; v.liquid_surface = true; v.particle_rate = 13.0f;
    v.base_particles = 3; v.particle_lift = 2.4f; return v;
  }
  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact)
      *c.wheel_force += c.contact->tangent * (-p.viscosity * 3.8f * c.wheel_speed * c.immersion);
    if (c.energy_cost) *c.energy_cost += c.immersion * (0.08f + std::abs(c.wheel_speed) * 0.04f) * p.energy_drain_mul * c.dt;
  }
};

namespace generated_biomes {
// <MARS_GENERATED_BIOMES>
class FracturedScree final : public Biome {
 public:
  std::string_view id() const noexcept override { return "fractured_scree"; }
  std::string_view display_name() const noexcept override { return "Fractured Scree"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.60f + 0.25f * biome_random01(s);
    p.sink_rate = 0.003f + 0.008f * biome_random01(s, 1);
    p.crust_deform = 0.012f + 0.025f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.5f + 2.0f * biome_random01(s, 4);
    p.ambient_temperature = -95.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.40f + 0.20f * biome_random01(s, 6);
    p.solar_charge_rate = 0.30f + 0.20f * biome_random01(s, 7);
    p.gravity_mul = 0.85f + 0.18f * biome_random01(s, 8);
    p.viscosity = 0.02f + 0.04f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.30f + 0.20f * biome_random01(s, 10);
    p.lidar_range_mul = 0.40f + 0.20f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 5.0f * std::tanh(c.contact->penetration * 25.0f);
      *c.wheel_force += c.contact->tangent * (-1.8f * c.wheel_speed * depth_factor);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.08f + c.contact->normal_force * 0.15f * depth_factor * c.contact->penetration * 20.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.018f + c.contact->penetration * 0.25f + std::abs(c.wheel_speed) * 0.012f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float slide = std::abs(c.velocity.x) > 0.3f ? (c.velocity.x > 0.0f ? c.velocity.x : -c.velocity.x) * 0.02f : 0.0f;
      float shift = std::sin(t * 0.031f + c.velocity.x * 0.5f) * 0.04f * c.mass;
      c.body_force->x -= c.velocity.x * c.mass * (0.15f + 0.20f * std::sin(t * 0.017f + 1.2f));
      c.body_force->x += shift + slide * c.mass;
      if (std::abs(c.velocity.x) > 0.1f) {
        c.body_force->x += (c.velocity.x > 0.0f ? -0.3f : 0.3f) * c.mass * std::sin(t * 0.059f + c.velocity.x);
      }
      c.body_force->y += c.mass * c.gravity * 0.025f - c.velocity.y * c.mass * 0.15f;
      if (c.body_torque) {
        *c.body_torque -= std::sin(t * 0.047f + c.velocity.x) * 0.02f * c.mass * c.gravity;
      }
    }
    if (c.energy_cost) {
      *c.energy_cost += (std::abs(c.velocity.x) * 0.003f + std::abs(c.velocity.y) * 0.001f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {96, 84, 72};
    v.particles = {62, 55, 48};
    v.liquid = {45, 42, 38};
    v.sky = {115, 98, 82};
    v.particle_rate = 0.5f;
    v.particle_lift = 0.15f;
    v.particle_spread = 0.20f;
    v.ambient_particles = 0;
    v.screen_brightness = 0.50f;
    return v;
  }
};

class TectonicPendulum final : public Biome {
 public:
  std::string_view id() const noexcept override { return "tectonic_pendulum"; }
  std::string_view display_name() const noexcept override { return "Tectonic Pendulum"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.20f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.00f;
    p.energy_drain_mul = 1.00f + 0.50f * biome_random01(s, 2);
    p.wind_force = 4.0f + 10.0f * biome_random01(s, 3);
    p.ambient_temperature = -100.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.20f + 0.40f * biome_random01(s, 5);
    p.solar_charge_rate = 1.50f + 0.80f * biome_random01(s, 6);
    p.gravity_mul = 0.75f + 0.20f * biome_random01(s, 7);
    p.lidar_energy_mul = 1.20f + 0.50f * biome_random01(s, 8);
    p.lidar_range_mul = 1.80f + 0.60f * biome_random01(s, 9);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.40f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      *c.wheel_force += c.contact->tangent * (-0.6f * c.wheel_speed / (1.0f + 15.0f * c.contact->penetration));
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.008f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float phase = t * 0.0081f + c.velocity.x * 0.002f;
      float swing = std::sin(phase) * std::cos(phase * 0.37f);
      float wind_sway = p.wind_force * 0.25f * swing;
      float pendulum_torque = swing * 0.20f * c.mass;
      c.body_force->x += wind_sway + pendulum_torque - c.velocity.x * c.mass * 0.08f;
      c.body_force->y -= c.velocity.y * c.mass * 0.10f;
      if (c.body_torque) {
        float torque = swing * 0.15f * c.mass * c.gravity * 0.3f;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.04f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float swing_cost = std::abs(std::sin(t * 0.0081f)) * 0.002f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + swing_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {48, 42, 38};
    v.particles = {145, 120, 90};
    v.liquid = {55, 45, 35};
    v.sky = {180, 155, 110};
    v.particle_rate = 1.5f;
    v.particle_lift = 0.25f;
    v.particle_spread = 0.30f;
    v.base_particles = 0;
    v.max_particles = 18;
    v.particle_size = 1;
    v.ambient_particles = 8;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 1.2f;
    v.liquid_surface = false;
    return v;
  }
};

class ThermalGradient final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_gradient"; }
  std::string_view display_name() const noexcept override { return "Thermal Gradient"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.25f * biome_random01(s);
    p.sink_rate = 0.001f + 0.005f * biome_random01(s, 1);
    p.crust_deform = 0.003f + 0.010f * biome_random01(s, 2);
    p.energy_drain_mul = 1.50f + 0.60f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -120.0f + 40.0f * biome_random01(s, 4);
    p.thermal_transfer = 3.5f + 1.2f * biome_random01(s, 5);
    p.solar_charge_rate = 0.20f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.viscosity = 0.0f;
    p.lidar_energy_mul = 1.8f + 0.6f * biome_random01(s, 8);
    p.lidar_range_mul = 1.2f + 0.4f * biome_random01(s, 9);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.5f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      float depth_penalty = 1.0f + 2.0f * c.contact->penetration * 20.0f;
      *c.wheel_force += c.contact->tangent * (-0.8f * c.wheel_speed * depth_penalty);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f + c.contact->normal_force * 0.06f * c.contact->penetration * 25.0f);
    }
    if (c.energy_cost) {
      float thermal_load = std::max(0.0f, -p.ambient_temperature - 50.0f) * 0.003f * p.thermal_transfer;
      *c.energy_cost += (0.015f + c.contact->penetration * 0.10f + std::abs(c.wheel_speed) * 0.005f + thermal_load) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float cold_creep = std::max(0.0f, -p.ambient_temperature - 60.0f) * 0.005f * std::sin(t * 0.025f + c.velocity.x * 0.3f);
      c.body_force->x -= c.velocity.x * c.mass * (0.10f + 0.05f * std::sin(t * 0.019f));
      c.body_force->x += cold_creep * c.mass * 0.2f;
      c.body_force->y += c.mass * c.gravity * 0.02f * std::sin(t * 0.037f + 0.5f) - c.velocity.y * c.mass * 0.12f;
      if (c.body_torque) {
        float thermal_torque = std::sin(t * 0.041f) * 0.01f * c.mass * c.gravity * 0.2f;
        *c.body_torque += thermal_torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float thermal_oscillation = std::abs(std::sin(t * 0.025f)) * 0.002f * std::max(0.0f, -p.ambient_temperature - 60.0f) * 0.01f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + thermal_oscillation) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {88, 78, 68};
    v.particles = {160, 140, 112};
    v.liquid = {60, 52, 44};
    v.sky = {100, 86, 72};
    v.particle_rate = 1.0f;
    v.particle_lift = 0.15f;
    v.particle_spread = 0.20f;
    v.base_particles = 0;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.25f;
    return v;
  }
};

class DarkSinkhole final : public Biome {
 public:
  std::string_view id() const noexcept override { return "dark_sinkhole"; }
  std::string_view display_name() const noexcept override { return "Dark Sinkhole"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.20f * biome_random01(s);
    p.sink_rate = 0.005f + 0.015f * biome_random01(s, 1);
    p.viscosity = 0.80f + 1.20f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -60.0f + 50.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.50f + 0.50f * biome_random01(s, 5);
    p.solar_charge_rate = 0.15f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 1.10f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.005f + 0.012f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.15f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 0.60f + 0.25f * biome_random01(s, 10);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 2.0f * std::abs(c.wheel_speed) / (1.0f + 3.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 4.0f * std::tanh(c.contact->penetration * 20.0f);
      float viscous_drag = 2.0f * p.viscosity * depth_factor * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-viscous_drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.06f + c.contact->normal_force * 0.20f * depth_factor * c.contact->penetration * 20.0f);
    }
    if (c.energy_cost) {
      float sink_energy = c.contact->penetration * 0.30f;
      *c.energy_cost += (0.020f + sink_energy + std::abs(c.wheel_speed) * 0.015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float suction = std::min(0.0f, std::sin(t * 0.037f + 0.8f) * 0.5f - 0.3f) * 0.5f * c.mass;
      c.body_force->x -= c.velocity.x * c.mass * (0.30f + 0.10f * std::sin(t * 0.023f));
      c.body_force->x += suction * (c.velocity.x / (std::abs(c.velocity.x) + 0.1f));
      c.body_force->y -= c.mass * c.gravity * 0.04f * (1.0f + 0.2f * std::sin(t * 0.029f));
      c.body_force->y -= c.velocity.y * c.mass * 0.20f;
      if (c.body_torque) {
        float sink_torque = -std::sin(t * 0.043f) * 0.02f * c.mass * c.gravity * 0.2f;
        *c.body_torque += sink_torque;
        *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float suction_cost = std::max(0.0f, std::sin(t * 0.037f + 0.8f) * 0.5f - 0.3f) * 0.006f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.003f + suction_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {15, 15, 18};
    v.particles = {25, 28, 35};
    v.liquid = {10, 10, 12};
    v.sky = {8, 8, 12};
    v.particle_rate = 0.0f;
    v.particle_lift = 0.0f;
    v.particle_spread = 0.0f;
    v.base_particles = 0;
    v.max_particles = 0;
    v.particle_size = 1;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.05f;
    return v;
  }
};

class TidalBasin final : public Biome {
 public:
  std::string_view id() const noexcept override { return "tidal_basin"; }
  std::string_view display_name() const noexcept override { return "Tidal Basin"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.20f * biome_random01(s);
    p.sink_rate = 0.005f + 0.018f * biome_random01(s, 1);
    p.viscosity = 1.2f + 1.8f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -40.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.8f + 1.0f * biome_random01(s, 5);
    p.solar_charge_rate = 0.30f + 0.20f * biome_random01(s, 6);
    p.gravity_mul = 0.60f + 0.20f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.50f + 0.30f * biome_random01(s, 8);
    p.lidar_range_mul = 0.45f + 0.20f * biome_random01(s, 9);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.32f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 2.0f * std::abs(c.wheel_speed) / (1.0f + 3.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float immersion_factor = c.immersion * (1.0f + 0.6f * c.contact->penetration * 25.0f);
      float drag = (3.0f * p.viscosity * immersion_factor + 0.5f * immersion_factor * p.viscosity) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.09f + c.contact->normal_force * 0.30f * c.contact->penetration * 30.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.050f + std::abs(c.wheel_speed) * 0.030f + c.immersion * 0.45f + c.contact->penetration * 0.55f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float tide = 0.5f + 0.5f * std::sin(t * 0.003f + 1.2f);
      float surge = 0.6f + 0.4f * std::sin(t * 0.061f) * std::cos(t * 0.113f);
      float buoyancy_boost = tide * 0.08f * c.mass * c.gravity;
      float drag_force = c.velocity.x * c.mass * (0.45f + 0.25f * tide);
      float lateral_push = std::sin(t * 0.049f + tide * 3.14f) * 0.06f * c.mass * tide;
      c.body_force->x -= drag_force * surge;
      c.body_force->x += lateral_push;
      c.body_force->y += buoyancy_boost - c.velocity.y * c.mass * 0.22f * surge;
      if (c.body_torque) {
        float roll = std::sin(t * 0.047f + tide * 2.0f) * 0.015f * c.mass * c.gravity * tide;
        *c.body_torque += roll - c.angular_velocity * c.mass * 0.03f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float tide_cost = std::abs(std::sin(t * 0.003f + 1.2f)) * 0.004f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.005f + tide_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {28, 52, 48};
    v.particles = {80, 130, 110};
    v.liquid = {20, 90, 75};
    v.sky = {110, 135, 120};
    v.particle_rate = 14.0f;
    v.particle_lift = 1.80f;
    v.particle_spread = 0.90f;
    v.base_particles = 4;
    v.max_particles = 54;
    v.particle_size = 3;
    v.ambient_particles = 22;
    v.ambient_drift = 1.2f;
    v.liquid_surface = true;
    v.screen_brightness = 0.50f;
    return v;
  }
};

class CyclicBorealis final : public Biome {
 public:
  std::string_view id() const noexcept override { return "cyclic_borealis"; }
  std::string_view display_name() const noexcept override { return "Cyclic Borealis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.18f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 3);
    p.wind_force = 2.0f + 6.0f * biome_random01(s, 4);
    p.ambient_temperature = -120.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 2.0f + 0.6f * biome_random01(s, 6);
    p.solar_charge_rate = 0.80f + 0.40f * biome_random01(s, 7);
    p.gravity_mul = 0.85f + 0.18f * biome_random01(s, 8);
    p.viscosity = 0.05f + 0.10f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.30f + 0.15f * biome_random01(s, 10);
    p.lidar_range_mul = 0.60f + 0.25f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 2.5f * std::tanh(c.contact->penetration * 30.0f);
      float drag = (0.8f + p.viscosity * 5.0f * depth_factor) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f + c.contact->normal_force * 0.10f * c.contact->penetration * 35.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.010f + c.contact->penetration * 0.15f + std::abs(c.wheel_speed) * 0.006f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float cycle = std::sin(t * 0.071f);
      float gust = 0.3f + 0.7f * (0.5f + 0.5f * cycle);
      float lateral_force = p.wind_force * 0.20f * gust + std::sin(t * 0.053f + 1.3f) * 0.05f * c.mass;
      c.body_force->x += lateral_force - c.velocity.x * c.mass * (0.10f + 0.12f * gust);
      c.body_force->y += c.mass * c.gravity * (0.015f + 0.015f * gust) - c.velocity.y * c.mass * (0.08f + 0.06f * gust);
      if (c.body_torque) {
        float torque = std::sin(t * 0.083f + cycle * 0.5f) * 0.015f * c.mass * c.gravity * gust;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float cycle_cost = std::abs(std::sin(t * 0.071f)) * 0.003f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + cycle_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {32, 70, 82};
    v.particles = {142, 200, 218};
    v.liquid = {28, 90, 108};
    v.sky = {32, 60, 95};
    v.particle_rate = 5.0f;
    v.particle_lift = 0.80f;
    v.particle_spread = 0.90f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.60f;
    return v;
  }
};

class RimeCascade final : public Biome {
 public:
  std::string_view id() const noexcept override { return "rime_cascade"; }
  std::string_view display_name() const noexcept override { return "Rime Cascade"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.40f + 0.25f * biome_random01(s);
    p.sink_rate = 0.001f + 0.005f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 2);
    p.energy_drain_mul = 1.60f + 0.60f * biome_random01(s, 3);
    p.wind_force = 1.5f + 4.0f * biome_random01(s, 4);
    p.ambient_temperature = -40.0f + 35.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 6);
    p.solar_charge_rate = 0.15f + 0.10f * biome_random01(s, 7);
    p.gravity_mul = 1.05f + 0.20f * biome_random01(s, 8);
    p.viscosity = 0.10f + 0.20f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.10f + 0.10f * biome_random01(s, 10);
    p.lidar_range_mul = 0.55f + 0.25f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.60f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.8f * std::abs(c.wheel_speed) / (1.0f + 3.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 3.0f * c.contact->penetration * 22.0f;
      float tilt_effect = c.contact->tangent.y * 0.15f * c.contact->normal_force * (c.contact->penetration * 10.0f);
      *c.wheel_force += c.contact->tangent * (-0.8f * c.wheel_speed * depth_factor + tilt_effect);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.10f * c.contact->penetration * 28.0f);
    }
    if (c.energy_cost) {
      float temp_penalty = std::max(0.0f, 10.0f - p.ambient_temperature) * 0.0015f * p.thermal_transfer;
      *c.energy_cost += (0.012f + c.contact->penetration * 0.18f + std::abs(c.wheel_speed) * 0.008f + temp_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float melt = std::max(0.0f, std::sin(t * 0.031f + 0.3f) * 0.3f + 0.1f);
      float cascade = std::sin(t * 0.067f) * std::cos(t * 0.123f) * 0.3f + 0.5f;
      float energy_deficit = 1.0f - std::tanh(std::abs(c.velocity.x) * 0.1f);
      float gravity_assist = c.mass * c.gravity * 0.02f * std::sin(t * 0.041f + c.velocity.x * 0.2f);
      float surge = (p.wind_force * 0.10f * cascade + gravity_assist * energy_deficit) * melt;
      c.body_force->x -= c.velocity.x * c.mass * (0.12f + 0.15f * cascade);
      c.body_force->x += surge;
      c.body_force->y += c.mass * c.gravity * 0.01f * std::sin(t * 0.053f + 1.0f) * melt - c.velocity.y * c.mass * (0.06f + 0.08f * melt);
      if (c.body_torque) {
        float melt_torque = std::sin(t * 0.023f + t * 0.001f * c.angular_velocity) * 0.012f * c.mass * c.gravity * melt;
        *c.body_torque += melt_torque - c.angular_velocity * c.mass * 0.02f * (1.0f + melt);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float melt_rhythm = std::abs(std::sin(t * 0.031f + 0.3f)) * 0.004f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.003f + melt_rhythm) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {42, 72, 78};
    v.particles = {174, 208, 220};
    v.liquid = {30, 100, 112};
    v.sky = {28, 52, 72};
    v.particle_rate = 8.0f;
    v.particle_lift = 1.10f;
    v.particle_spread = 0.80f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.32f;
    return v;
  }
};

class PhosphorescentMiasma final : public Biome {
 public:
  std::string_view id() const noexcept override { return "phosphorescent_miasma"; }
  std::string_view display_name() const noexcept override { return "Phosphorescent Miasma"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.02f + 0.06f * biome_random01(s, 2);
    p.energy_drain_mul = 1.10f + 0.35f * biome_random01(s, 3);
    p.wind_force = 1.5f + 4.0f * biome_random01(s, 4);
    p.ambient_temperature = -20.0f + 30.0f * biome_random01(s, 5);
    p.thermal_transfer = 1.80f + 0.50f * biome_random01(s, 6);
    p.solar_charge_rate = 1.20f + 0.40f * biome_random01(s, 7);
    p.gravity_mul = 0.80f + 0.15f * biome_random01(s, 8);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.08f + 0.06f * biome_random01(s, 10);
    p.lidar_range_mul = 1.20f + 0.40f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.82f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      *c.wheel_force += c.contact->tangent * (-0.6f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.008f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float miasma = 0.5f + 0.5f * std::sin(t * 0.037f + std::cos(t * 0.059f) * 1.2f);
      float drift = std::sin(t * 0.023f + 0.7f) * 0.3f + 0.5f;
      float glow = std::max(0.0f, std::sin(t * 0.047f) * std::cos(t * 0.091f) * 0.5f + 0.5f);
      float thrust = glow * 0.06f * c.mass * c.gravity;
      c.body_force->x -= c.velocity.x * c.mass * (0.06f + 0.04f * miasma);
      c.body_force->x += p.wind_force * 0.12f * miasma * drift;
      c.body_force->y += thrust * (1.0f + 0.3f * std::sin(t * 0.031f)) - c.velocity.y * c.mass * (0.05f + 0.03f * glow);
      if (c.body_torque) {
        *c.body_torque += std::sin(t * 0.043f + glow * 1.5f) * 0.008f * c.mass * c.gravity;
        *c.body_torque -= c.angular_velocity * c.mass * 0.015f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float glow_cost = std::max(0.0f, std::sin(t * 0.047f) * std::cos(t * 0.091f)) * 0.002f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + glow_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {35, 30, 42};
    v.particles = {80, 200, 130};
    v.liquid = {25, 160, 80};
    v.sky = {20, 180, 90};
    v.particle_rate = 14.0f;
    v.particle_lift = 2.8f;
    v.particle_spread = 1.8f;
    v.base_particles = 5;
    v.max_particles = 48;
    v.particle_size = 3;
    v.ambient_particles = 40;
    v.ambient_drift = 3.5f;
    v.screen_brightness = 0.20f;
    return v;
  }
};

class CompactedLoess final : public Biome {
 public:
  std::string_view id() const noexcept override { return "compacted_loess"; }
  std::string_view display_name() const noexcept override { return "Compacted Loess"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.65f + 0.22f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.crust_deform = 0.008f + 0.016f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 3);
    p.wind_force = 2.5f + 5.0f * biome_random01(s, 4);
    p.ambient_temperature = -80.0f + 28.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.55f + 0.20f * biome_random01(s, 6);
    p.solar_charge_rate = 1.15f + 0.50f * biome_random01(s, 7);
    p.gravity_mul = 1.15f + 0.22f * biome_random01(s, 8);
    p.viscosity = 0.02f + 0.05f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.40f + 0.25f * biome_random01(s, 10);
    p.lidar_range_mul = 0.45f + 0.20f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.88f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::abs(c.wheel_speed) / (1.0f + 5.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 4.5f * std::tanh(c.contact->penetration * 22.0f);
      *c.wheel_force += c.contact->tangent * (-1.1f * c.wheel_speed * depth_factor);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.12f * depth_factor * c.contact->penetration * 25.0f);
    }
    if (c.energy_cost) {
      float load_penalty = std::abs(c.wheel_speed) * 0.010f * (1.0f + 2.0f * c.minimum_drive_limit / 10.0f);
      *c.energy_cost += (0.012f + c.contact->penetration * 0.20f + load_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float compression = 0.7f + 0.3f * std::sin(t * 0.027f + c.velocity.x * 0.15f);
      float wind_rhythm = p.wind_force * 0.08f * (0.5f + 0.5f * std::cos(t * 0.011f));
      c.body_force->x -= c.velocity.x * c.mass * (0.25f + 0.15f * compression);
      c.body_force->x += wind_rhythm + std::sin(t * 0.063f + 1.5f) * 0.03f * c.mass;
      c.body_force->y += c.mass * c.gravity * (0.02f + 0.02f * std::sin(t * 0.031f)) - c.velocity.y * c.mass * (0.12f + 0.06f * compression);
      if (c.body_torque) {
        *c.body_torque += std::sin(t * 0.051f + c.velocity.x * 0.1f) * 0.012f * c.mass * c.gravity - c.angular_velocity * c.mass * 0.025f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float rhythm_cost = std::abs(std::sin(t * 0.027f)) * 0.0015f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0025f + rhythm_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {98, 82, 65};
    v.particles = {132, 112, 90};
    v.liquid = {55, 48, 38};
    v.sky = {150, 125, 95};
    v.particle_rate = 2.5f;
    v.particle_lift = 0.30f;
    v.particle_spread = 0.40f;
    v.base_particles = 0;
    v.max_particles = 20;
    v.particle_size = 1;
    v.ambient_particles = 8;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.82f;
    return v;
  }
};

class StratifiedTailings final : public Biome {
 public:
  std::string_view id() const noexcept override { return "stratified_tailings"; }
  std::string_view display_name() const noexcept override { return "Stratified Tailings"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.25f + 0.20f * biome_random01(s);
    p.sink_rate = 0.003f + 0.010f * biome_random01(s, 1);
    p.viscosity = 0.30f + 0.50f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 3);
    p.wind_force = 2.0f + 5.0f * biome_random01(s, 4);
    p.ambient_temperature = -50.0f + 40.0f * biome_random01(s, 5);
    p.thermal_transfer = 1.60f + 0.40f * biome_random01(s, 6);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 7);
    p.gravity_mul = 0.70f + 0.20f * biome_random01(s, 8);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.30f + 0.20f * biome_random01(s, 10);
    p.lidar_range_mul = 0.50f + 0.25f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.50f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 3.5f * std::tanh(c.contact->penetration * 25.0f);
      *c.wheel_force += c.contact->tangent * (-1.2f * c.wheel_speed * depth_factor);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.15f * depth_factor * c.contact->penetration * 20.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.015f + c.contact->penetration * 0.20f + std::abs(c.wheel_speed) * 0.010f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float stratification = 0.6f + 0.4f * std::abs(std::sin(t * 0.047f + c.velocity.x * 0.1f));
      float wind_gust = p.wind_force * (0.5f + 0.5f * std::sin(t * 0.031f + 1.2f)) * 0.12f;
      c.body_force->x -= c.velocity.x * c.mass * (0.15f + 0.08f * stratification);
      c.body_force->x += wind_gust + std::sin(t * 0.059f + c.velocity.x * 0.3f) * 0.04f * c.mass;
      float lift = c.mass * c.gravity * (0.025f + 0.015f * std::sin(t * 0.043f + c.velocity.x * 0.2f));
      c.body_force->y += lift - c.velocity.y * c.mass * (0.10f + 0.06f * stratification);
      if (c.body_torque) {
        float tilt = std::sin(t * 0.067f + c.velocity.x * 0.15f) * 0.012f * c.mass * c.gravity * stratification;
        *c.body_torque += tilt - c.angular_velocity * c.mass * 0.02f * (1.0f + 0.3f * stratification);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float strat_cost = std::abs(std::sin(t * 0.047f)) * 0.002f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + strat_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {68, 58, 48};
    v.particles = {118, 98, 78};
    v.liquid = {45, 38, 32};
    v.sky = {138, 115, 90};
    v.particle_rate = 14.0f;
    v.particle_lift = 0.80f;
    v.particle_spread = 0.90f;
    v.base_particles = 4;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 30;
    v.ambient_drift = 2.8f;
    v.screen_brightness = 0.55f;
    v.liquid_surface = true;
    return v;
  }
};

class ThermalCore final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_core"; }
  std::string_view display_name() const noexcept override { return "Thermal Core"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.70f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 2);
    p.energy_drain_mul = 1.40f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = 45.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 4.0f + 1.5f * biome_random01(s, 5);
    p.solar_charge_rate = 0.15f + 0.10f * biome_random01(s, 6);
    p.gravity_mul = 1.10f + 0.20f * biome_random01(s, 7);
    p.viscosity = 0.0f;
    p.lidar_energy_mul = 0.25f + 0.15f * biome_random01(s, 8);
    p.lidar_range_mul = 0.40f + 0.20f * biome_random01(s, 9);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      *c.wheel_force += c.contact->tangent * (-0.6f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      float heat_load = std::max(0.0f, p.ambient_temperature - 30.0f) * 0.008f * p.thermal_transfer;
      *c.energy_cost += (0.008f + std::abs(c.wheel_speed) * 0.004f + heat_load) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float heat_pulse = std::max(0.0f, p.ambient_temperature - 30.0f) * 0.01f;
      float thermal_thrust = heat_pulse * std::sin(t * 0.073f) * 0.5f * c.mass;
      float expansion = std::sin(t * 0.041f + 0.8f) * 0.02f * c.mass;
      c.body_force->x -= c.velocity.x * c.mass * (0.08f + 0.04f * std::sin(t * 0.029f));
      c.body_force->x += thermal_thrust + expansion;
      c.body_force->y += c.mass * c.gravity * 0.01f * std::sin(t * 0.053f) - c.velocity.y * c.mass * 0.06f;
      if (c.body_torque) {
        float thermal_torque = heat_pulse * std::sin(t * 0.061f) * 0.008f * c.mass * c.gravity;
        *c.body_torque += thermal_torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float pulse_cost = std::max(0.0f, p.ambient_temperature - 30.0f) * 0.002f * std::abs(std::sin(t * 0.073f));
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + pulse_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {55, 25, 15};
    v.particles = {200, 80, 30};
    v.liquid = {40, 15, 8};
    v.sky = {220, 60, 20};
    v.particle_rate = 12.0f;
    v.particle_lift = 2.2f;
    v.particle_spread = 0.8f;
    v.base_particles = 4;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 1.8f;
    return v;
  }
};

class ObscuredCinderfield final : public Biome {
 public:
  std::string_view id() const noexcept override { return "obscured_cinderfield"; }
  std::string_view display_name() const noexcept override { return "Obscured Cinderfield"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.50f + 0.25f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 2);
    p.energy_drain_mul = 0.90f + 0.30f * biome_random01(s, 3);
    p.wind_force = 0.5f + 2.0f * biome_random01(s, 4);
    p.ambient_temperature = -90.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.50f + 0.20f * biome_random01(s, 6);
    p.solar_charge_rate = 0.10f + 0.08f * biome_random01(s, 7);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 8);
    p.viscosity = 0.02f + 0.04f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.08f + 0.05f * biome_random01(s, 10);
    p.lidar_range_mul = 0.20f + 0.15f * biome_random01(s, 11);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.68f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.6f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 2.5f * c.contact->penetration * 20.0f;
      *c.wheel_force += c.contact->tangent * (-0.7f * c.wheel_speed * depth_factor);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.08f * c.contact->penetration * 25.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.010f + c.contact->penetration * 0.12f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float flicker = 0.5f + 0.5f * std::sin(t * 0.081f) * std::cos(t * 0.149f);
      float draught = p.wind_force * 0.10f * flicker;
      c.body_force->x -= c.velocity.x * c.mass * (0.10f + 0.05f * flicker);
      c.body_force->x += draught + std::sin(t * 0.047f + c.velocity.x * 0.2f) * 0.02f * c.mass;
      c.body_force->y += c.mass * c.gravity * 0.015f * flicker - c.velocity.y * c.mass * (0.08f + 0.04f * flicker);
      if (c.body_torque) {
        *c.body_torque += std::sin(t * 0.053f + c.velocity.x * 0.1f) * 0.008f * c.mass * c.gravity - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float flicker_cost = std::abs(std::sin(t * 0.081f) * std::cos(t * 0.149f)) * 0.0012f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + flicker_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {38, 30, 25};
    v.particles = {58, 46, 36};
    v.liquid = {22, 18, 15};
    v.sky = {12, 10, 8};
    v.particle_rate = 4.0f;
    v.particle_lift = 0.20f;
    v.particle_spread = 0.25f;
    v.ambient_particles = 0;
    v.screen_brightness = 0.06f;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const FracturedScree biome_0; out.push_back(&biome_0);
  static const TectonicPendulum biome_1; out.push_back(&biome_1);
  static const ThermalGradient biome_2; out.push_back(&biome_2);
  static const DarkSinkhole biome_3; out.push_back(&biome_3);
  static const TidalBasin biome_4; out.push_back(&biome_4);
  static const CyclicBorealis biome_5; out.push_back(&biome_5);
  static const RimeCascade biome_6; out.push_back(&biome_6);
  static const PhosphorescentMiasma biome_7; out.push_back(&biome_7);
  static const CompactedLoess biome_8; out.push_back(&biome_8);
  static const StratifiedTailings biome_9; out.push_back(&biome_9);
  static const ThermalCore biome_10; out.push_back(&biome_10);
  static const ObscuredCinderfield biome_11; out.push_back(&biome_11);
}
// </MARS_GENERATED_BIOMES>
}  // namespace generated_biomes

inline const std::vector<const Biome*>& biome_registry() {
  static const NormalBiome normal; static const SandBiome sand; static const IceBiome ice;
  static const MudBiome mud; static const WindBiome wind; static const LowGravityBiome low_gravity;
  static const CrustBiome crust; static const LiquidBiome liquid;
  static const std::vector<const Biome*> registry = [] {
    std::vector<const Biome*> out{&normal, &sand, &ice, &mud, &wind, &low_gravity, &crust, &liquid};
    generated_biomes::append(out); return out;
  }();
  return registry;
}

inline const Biome& biome_by_id(int id) noexcept {
  const auto& bank = biome_registry();
  return *bank[static_cast<size_t>(id >= 0 && id < static_cast<int>(bank.size()) ? id : 0)];
}

}  // namespace mars
