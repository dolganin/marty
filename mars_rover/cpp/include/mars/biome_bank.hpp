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
  virtual std::string_view skill_stratum() const noexcept = 0;
  virtual MechanicType visual_type() const noexcept = 0;
  virtual BiomeSplit split() const noexcept { return BiomeSplit::Builtin; }
  virtual bool is_anchor() const noexcept { return false; }
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
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  bool is_anchor() const noexcept override { return true; }
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
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  bool is_anchor() const noexcept override { return true; }
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
class RutileSinkhole final : public Biome {
 public:
  std::string_view id() const noexcept override { return "rutile_sinkhole"; }
  std::string_view display_name() const noexcept override { return "Rutile Sinkhole"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.025f + 0.040f * biome_random01(s, 1);
    p.crust_deform = 0.015f + 0.025f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -40.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.90f + 0.40f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 7);
    p.viscosity = 0.10f + 0.20f * biome_random01(s, 8);
    p.lidar_energy_mul = 1.10f + 0.40f * biome_random01(s, 9);
    p.lidar_range_mul = 1.40f + 0.50f * biome_random01(s, 10);
    // Reshape the actual terrain: denser but shallower craters, sharper steps
    p.terrain_amplitude_mul = 0.80f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.20f + 0.60f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.60f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.42f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.6f * std::abs(c.wheel_speed) / (1.0f + 3.5f * c.contact->penetration * 40.0f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration;
      float collapse = 1.0f + 7.0f * depth * 35.0f;
      float cohesion = 0.4f + 0.6f * std::tanh(depth * 45.0f);
      // High-slip digging: spinning wheels sink dramatically faster
      float slipFactor = 1.0f + 2.0f * std::tanh(std::abs(c.wheel_speed) * 0.15f);
      *c.wheel_force += c.contact->tangent * (-1.2f * c.wheel_speed * collapse * slipFactor * (1.0f - 0.2f * cohesion));
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.30f * depth * 40.0f);
    }
    if (c.energy_cost) {
      float depthPenalty = 1.0f + 3.0f * c.contact->penetration * 50.0f;
      *c.energy_cost += (0.020f + c.contact->penetration * 0.30f + std::abs(c.wheel_speed) * 0.010f * depthPenalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float rut_phase = 0.6f + 0.4f * std::sin(t * 0.089f + c.velocity.x * 0.11f);
      float collapseDrag = c.velocity.x * c.mass * (0.08f + 0.14f * rut_phase * std::tanh(speed * 0.06f));
      // Terrain heterogeneity: randomly shifting micro-slopes create lateral biases
      float lateralGround = std::sin(t * 0.053f + c.velocity.x * 0.23f) * 0.04f * c.mass * rut_phase;
      c.body_force->x -= collapseDrag;
      c.body_force->x += lateralGround;
      c.body_force->y += c.mass * c.gravity * 0.004f * rut_phase * std::sin(t * 0.031f + 0.9f) - c.velocity.y * c.mass * (0.05f + 0.03f * rut_phase);
      if (c.body_torque) {
        float tilt = std::sin(t * 0.067f + c.velocity.x * 0.11f) * 0.013f * c.mass * c.gravity * rut_phase;
        *c.body_torque += tilt - c.angular_velocity * c.mass * 0.015f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float rut_cost = std::abs(std::sin(t * 0.089f)) * 0.002f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.003f + rut_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {112, 84, 62};
    v.particles = {158, 128, 96};
    v.liquid = {66, 46, 30};
    v.sky = {178, 148, 108};
    v.particle_rate = 9.0f;
    v.particle_lift = 0.70f;
    v.particle_spread = 1.30f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 22;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 1.10f;
    v.liquid_surface = false;
    return v;
  }
};

class MagneticAnomalyDrift final : public Biome {
 public:
  std::string_view id() const noexcept override { return "magnetic_anomaly_drift"; }
  std::string_view display_name() const noexcept override { return "Magnetic Anomaly Drift"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.70f + 0.25f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 3);
    p.wind_force = 1.0f + 3.0f * biome_random01(s, 4);
    p.ambient_temperature = -50.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 6);
    p.solar_charge_rate = 0.90f + 0.30f * biome_random01(s, 7);
    p.gravity_mul = 1.00f + 0.10f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.80f + 0.30f * biome_random01(s, 9);
    p.lidar_range_mul = 1.20f + 0.40f * biome_random01(s, 10);
    // Terrain: broad rolling hills with subtle craters, moderate roughness
    p.terrain_amplitude_mul = 1.20f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.00f + 0.40f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.00f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.60f + 0.30f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Normal grip but with slight magnetic drag
      *c.wheel_force += c.contact->tangent * (-0.15f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.01f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Magnetic field rotates slowly, pushing laterally with varying intensity
      float field_phase = t * 0.007f + c.velocity.x * 0.03f;
      float field_mag = 0.6f + 0.4f * std::sin(t * 0.013f + 0.5f);
      float lateral_accel = std::sin(field_phase) * field_mag * 0.15f * c.mass;
      // Counteracting force requires throttle/steering, but high speed increases drift
      float drift_penalty = 1.0f + std::tanh(speed * 0.05f) * 1.5f;
      float lateral_force = lateral_accel * drift_penalty;
      // Damping prevents runaway
      c.body_force->x -= c.velocity.x * c.mass * 0.08f;
      c.body_force->x += lateral_force;
      // Slight vertical oscillation mimicking magnetic pull
      c.body_force->y += std::sin(t * 0.021f + field_phase) * 0.005f * c.mass * c.gravity;
      c.body_force->y -= c.velocity.y * c.mass * 0.05f;
      if (c.body_torque) {
        // Magnetic torque tries to rotate rover toward field
        float torque = std::sin(field_phase * 0.5f + 1.0f) * 0.02f * c.mass * c.gravity * field_mag;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.03f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float field_cost = 0.002f * (0.5f + 0.5f * std::sin(t * 0.013f));
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + field_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {68, 62, 58};  // iron-rich soil
    v.particles = {142, 134, 128}; // metallic dust
    v.liquid = {40, 36, 34};
    v.sky = {168, 155, 130};  // hazy but not dark
    v.particle_rate = 6.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 0.7f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 16;
    v.ambient_drift = 2.5f;
    v.screen_brightness = 0.85f;
    v.liquid_surface = false;
    return v;
  }
};

class MomentumMireBackwater final : public Biome {
 public:
  std::string_view id() const noexcept override { return "momentum_mire_backwater"; }
  std::string_view display_name() const noexcept override { return "Momentum Mire Backwater"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.60f + 0.25f * biome_random01(s);
    p.sink_rate = 0.004f + 0.012f * biome_random01(s, 1);
    p.viscosity = 1.0f + 1.5f * biome_random01(s, 2);
    p.energy_drain_mul = 1.35f + 0.45f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -35.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.40f + 0.40f * biome_random01(s, 5);
    p.solar_charge_rate = 0.45f + 0.25f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.60f + 0.30f * biome_random01(s, 9);
    p.lidar_range_mul = 0.70f + 0.25f * biome_random01(s, 10);
    // Terrain: dense broad craters and gentle steps, creating basins that hold momentum
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.70f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.80f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 1.20f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.8f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration * 30.0f));
    }
    if (c.wheel_force && c.contact) {
      // Hysteretic drag: resistance builds with speed and penetration, but releases in bursts
      float speed = std::abs(c.wheel_speed);
      float depth_factor = 1.0f + 5.0f * c.contact->penetration * 30.0f;
      float hyst = 0.4f + 0.6f * std::tanh(speed * 0.35f);
      // Viscous drag plus sudden grip loss at critical speeds (slippery layer)
      float drag = (p.viscosity * 1.8f * depth_factor + hyst * 0.5f) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.10f * c.contact->penetration * 25.0f);
    }
    if (c.energy_cost) {
      float speed = std::abs(c.wheel_speed);
      float hyst_cost = 0.02f + 0.04f * std::tanh(speed * 0.3f);
      *c.energy_cost += (0.015f + c.contact->penetration * 0.25f + hyst_cost) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);
      // Memory of recent speed creates delayed resistance and release
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.4f) * 5.0f);
      float phase = t * 0.013f + speed * 0.04f;
      // Delayed drag: resists acceleration more than deceleration (hysteresis loop)
      float drag = c.velocity.x * c.mass * (0.15f + 0.25f * memory);
      // Sudden release when momentum exceeds threshold: burst of forward force
      float release = memory * 0.03f * c.mass * c.gravity * std::sin(phase);
      // Lateral slosh: residual motion from stored momentum
      float slosh = std::sin(t * 0.029f + speed * 0.06f + memory * 2.0f) * 0.04f * c.mass * (1.0f + 0.5f * memory);
      c.body_force->x -= drag;
      c.body_force->x += release + slosh;
      c.body_force->y += c.mass * c.gravity * 0.008f * memory * std::sin(t * 0.021f + 0.5f) - c.velocity.y * c.mass * (0.06f + 0.05f * memory);
      if (c.body_torque) {
        float torque = slosh * 0.35f + std::sin(t * 0.037f + speed * 0.04f + memory * 1.5f) * 0.008f * c.mass * c.gravity * memory;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f * (1.0f + 0.3f * memory);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.4f) * 5.0f);
      // Energy cost has memory-dependent component: stored momentum drains slowly
      float memory_cost = memory * 0.004f + speed_norm * 0.002f;
      *c.energy_cost += (memory_cost + std::abs(c.velocity.x) * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {56, 44, 36};  // dark boggy soil
    v.particles = {96, 74, 56};  // murky spray
    v.liquid = {34, 26, 20};  // stagnant water
    v.sky = {96, 104, 84};  // dim greenish overcast
    v.particle_rate = 10.0f;
    v.particle_lift = 0.70f;
    v.particle_spread = 0.90f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.40f;
    v.liquid_surface = true;
    return v;
  }
};

class InvertedSuspensionRavine final : public Biome {
 public:
  std::string_view id() const noexcept override { return "inverted_suspension_ravine"; }
  std::string_view display_name() const noexcept override { return "Inverted Suspension Ravine"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.10f + 0.40f * biome_random01(s, 2);
    p.wind_force = 1.0f + 3.0f * biome_random01(s, 3);
    p.ambient_temperature = -70.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 5);
    p.solar_charge_rate = 0.35f + 0.20f * biome_random01(s, 6);
    p.gravity_mul = 0.35f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 1.50f + 0.50f * biome_random01(s, 9);
    p.lidar_range_mul = 1.30f + 0.40f * biome_random01(s, 10);
    // Terrain: rugged, crater-dense with sharp steps and moderate amplitude
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.50f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.80f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 1.60f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.75f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Low gravity means less normal force, so traction is scarce; add dynamic grip recovery
      float speed = std::abs(c.wheel_speed);
      float grip_boost = 0.6f + 0.4f * std::exp(-speed * 0.2f);
      float depth_factor = 1.0f + 2.0f * c.contact->penetration * 20.0f;
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed * depth_factor * grip_boost);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f + c.contact->normal_force * 0.08f * c.contact->penetration * 25.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.008f + c.contact->penetration * 0.10f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Low gravity makes suspension bounce longer; add spring-like oscillatory lift
      float bounce = 0.5f + 0.5f * std::sin(t * 0.045f + c.velocity.x * 0.03f);
      float lift = p.gravity_mul * 0.06f * c.mass * c.gravity * bounce * std::exp(-speed * 0.02f);
      // Gravity dips and surges periodically, forcing throttle/brake management
      float gravity_phase = t * 0.011f + c.velocity.x * 0.02f;
      float gravity_dip = 0.7f + 0.3f * std::sin(gravity_phase);
      float effective_g = p.gravity_mul * gravity_dip;
      // Vertical component of gravity changes, affecting suspension loading
      float vertical_force = c.mass * c.gravity * effective_g * 0.02f * std::sin(gravity_phase);
      // Lateral drift from terrain interaction at low gravity
      float lateral = std::sin(t * 0.029f + c.velocity.x * 0.06f + gravity_phase) * 0.03f * c.mass * (1.0f - effective_g);
      c.body_force->x -= c.velocity.x * c.mass * (0.05f + 0.03f * bounce);
      c.body_force->x += lateral + 0.02f * c.mass * c.gravity * (1.0f - effective_g) * std::sin(gravity_phase);
      c.body_force->y += lift + vertical_force - c.velocity.y * c.mass * 0.04f;
      if (c.body_torque) {
        // Gravity changes induce rocking; suspension must compensate
        float torque = std::sin(gravity_phase + 0.8f) * 0.015f * c.mass * c.gravity * effective_g * bounce;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float gravity_phase = t * 0.011f + std::abs(c.velocity.x) * 0.02f;
      float gravity_var = 0.5f + 0.5f * std::sin(gravity_phase);
      // Energy cost scales with gravity variation: anticipate dips to save battery
      float gravity_cost = 0.002f * gravity_var * (1.0f + (1.0f - p.gravity_mul) * 2.0f);
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + gravity_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {52, 48, 58};  // dark rocky soil
    v.particles = {120, 110, 135};  // pale dust
    v.liquid = {32, 30, 42};  // murky pools
    v.sky = {85, 75, 105};  // dim violet overcast
    v.particle_rate = 7.0f;
    v.particle_lift = 2.5f;  // floating dust due to low gravity
    v.particle_spread = 1.4f;
    v.base_particles = 3;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.45f;  // dim, forcing lidar
    v.liquid_surface = false;
    return v;
  }
};

class EmberColumn final : public Biome {
 public:
  std::string_view id() const noexcept override { return "ember_column"; }
  std::string_view display_name() const noexcept override { return "Ember Column"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 2);
    p.energy_drain_mul = 1.35f + 0.45f * biome_random01(s, 3);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 4);
    p.ambient_temperature = 30.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 3.0f + 1.0f * biome_random01(s, 6);
    p.solar_charge_rate = 1.4f + 0.6f * biome_random01(s, 7);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.15f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 0.55f + 0.25f * biome_random01(s, 10);
    // Terrain: undulating heat-tempered plateaus with sharp ridges and sparse craters
    p.terrain_amplitude_mul = 1.10f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.30f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 1.50f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.95f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Heat-induced thermal expansion creates subtle rolling resistance that peaks at low speed
      float speed = std::abs(c.wheel_speed);
      float thermal_grip = 1.0f + 0.3f * std::exp(-speed * 0.4f);
      *c.wheel_force += c.contact->tangent * (-0.4f * c.wheel_speed * thermal_grip);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Ambient heat adds a constant thermal load to the drivetrain
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.004f * p.thermal_transfer;
      *c.energy_cost += (0.008f + heat_load + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal convection cells: rhythmic updrafts and downdrafts that vary with speed
      float cell_phase = t * 0.021f + speed * 0.05f;
      float convection = std::sin(cell_phase) * 0.5f + 0.5f;
      float lift = convection * 0.03f * c.mass * c.gravity * p.thermal_transfer;
      // Hot air pushes laterally in counter-rotating vortices
      float lateral_phase = t * 0.017f + speed * 0.03f;
      float lateral = std::sin(lateral_phase) * 0.02f * c.mass * (1.0f + convection);
      // Speed-dependent damping: faster speeds compress the thermal cushion
      float damping = 0.06f + 0.04f * std::exp(-speed * 0.02f);
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->x += lateral;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.02f * convection);
      if (c.body_torque) {
        // Convection cells induce mild rocking on the rover body
        float torque = std::sin(t * 0.031f + speed * 0.04f + convection * 2.0f) * 0.012f * c.mass * c.gravity * convection;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal pulsing: energy drain oscillates with convection, rewarding steady cruise
      float convection = 0.5f + 0.5f * std::sin(t * 0.021f + speed * 0.05f);
      float pulse_cost = convection * 0.003f * p.thermal_transfer;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + pulse_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {88, 46, 28};  // baked iron-rich soil
    v.particles = {255, 120, 30};  // glowing ember dust
    v.liquid = {40, 24, 16};  // lava-tinted pools
    v.sky = {200, 90, 30};  // hot orange haze
    v.particle_rate = 12.0f;
    v.particle_lift = 1.8f;
    v.particle_spread = 1.2f;
    v.base_particles = 4;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 24;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.70f;  // bright but hazy
    v.liquid_surface = false;
    return v;
  }
};

class ChromaticPulse final : public Biome {
 public:
  std::string_view id() const noexcept override { return "chromatic_pulse"; }
  std::string_view display_name() const noexcept override { return "Chromatic Pulse"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -60.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.20f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 1.30f + 0.50f * biome_random01(s, 5);
    p.gravity_mul = 1.00f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.15f + 0.10f * biome_random01(s, 8);
    p.lidar_range_mul = 0.80f + 0.30f * biome_random01(s, 9);
    // Terrain: frequent sharp ridges and rugged bumps form natural phases
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.80f + 0.30f * biome_random01(s, 12);
    p.terrain_step_mul = 1.60f + 0.50f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Chromatic traction: varies with a deterministic phase that depends on rover speed
      float phase = static_cast<float>(c.step_index) * 0.071f + std::abs(c.wheel_speed) * 0.04f;
      float pulse = 0.6f + 0.4f * std::sin(phase);
      *c.wheel_force += c.contact->tangent * (-0.3f * c.wheel_speed * (1.0f + 0.5f * (1.0f - pulse)));
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      float phase = static_cast<float>(c.step_index) * 0.071f + std::abs(c.wheel_speed) * 0.04f;
      float pulse = 0.6f + 0.4f * std::sin(phase);
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f * (1.0f + 0.5f * (1.0f - pulse))) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Two coupled harmonic phases, one spatial and one temporal
      float phase = t * 0.013f + c.velocity.x * 0.017f;
      float phase2 = t * 0.007f + c.velocity.x * 0.009f;
      // Ground resonance: vertical heave that grows with speed, but only during pulse peaks
      float pulse1 = 0.5f + 0.5f * std::sin(phase);
      float pulse2 = 0.5f + 0.5f * std::sin(phase2 + 1.3f);
      float combined = pulse1 * pulse2;
      float heave = combined * (0.08f + 0.12f * std::tanh(speed * 0.08f)) * c.mass * c.gravity;
      // Lateral surge follows the same phase, stronger when both pulses align
      float lateral = std::sin(phase + 0.7f) * combined * 0.04f * c.mass * (1.0f + std::tanh(speed * 0.06f));
      // Damping is lower during pulse peaks, making the rover more responsive but harder to control
      float damping = 0.06f + 0.04f * (1.0f - combined);
      c.body_force->x += lateral - c.velocity.x * c.mass * damping;
      c.body_force->y += heave - c.velocity.y * c.mass * (0.03f + 0.02f * (1.0f - combined));
      if (c.body_torque) {
        // Torsional pulse tries to yaw the rover, strongest during alignment
        float torque = std::sin(phase - 0.9f) * combined * 0.012f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.013f + speed * 0.017f;
      float phase2 = t * 0.007f + speed * 0.009f;
      float combined = (0.5f + 0.5f * std::sin(phase)) * (0.5f + 0.5f * std::sin(phase2 + 1.3f));
      // Energy drain peaks when pulses align, rewarding steady speed that keeps phases locked
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + combined * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {66, 44, 58};
    v.particles = {188, 84, 150};
    v.liquid = {42, 30, 52};
    v.sky = {112, 82, 141};
    v.particle_rate = 8.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 1.1f;
    v.base_particles = 3;
    v.max_particles = 34;
    v.particle_size = 2;
    v.ambient_particles = 16;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.60f;
    v.liquid_surface = false;
    return v;
  }
};

class OrbitalPendulum final : public Biome {
 public:
  std::string_view id() const noexcept override { return "orbital_pendulum"; }
  std::string_view display_name() const noexcept override { return "Orbital Pendulum"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Low gravity with a strong oscillating vertical component
    p.friction_mul = 0.40f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 2.0f * biome_random01(s, 3);
    p.ambient_temperature = -80.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.90f + 0.25f * biome_random01(s, 5);
    p.solar_charge_rate = 0.30f + 0.20f * biome_random01(s, 6);
    p.gravity_mul = 0.30f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 1.00f + 0.40f * biome_random01(s, 9);
    p.lidar_range_mul = 0.70f + 0.30f * biome_random01(s, 10);
    // Terrain: high-amplitude rolling hills with craters, few sharp steps
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.60f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.50f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.70f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Low normal force means less grip; add speed-dependent recovery
      float speed = std::abs(c.wheel_speed);
      float grip_boost = 0.5f + 0.5f * std::exp(-speed * 0.2f);
      *c.wheel_force += c.contact->tangent * (-0.4f * c.wheel_speed * grip_boost);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Orbital mechanics: gravity oscillates sinusoidally, creating a pendulum-like force
      float orbital_phase = t * 0.009f + c.velocity.x * 0.02f;
      float gravity_mod = 0.7f + 0.6f * std::sin(orbital_phase);
      float effective_g = p.gravity_mul * gravity_mod;

      // Pendulum swing: lateral force proportional to gravity oscillation and speed
      float swing = std::sin(orbital_phase + 0.8f) * 0.04f * c.mass * c.gravity * effective_g;
      // Damping: suspension absorbs but also stores energy like a pendulum
      float damping = 0.05f + 0.02f * std::sin(orbital_phase * 0.5f);

      c.body_force->x += swing - c.velocity.x * c.mass * damping;
      // Vertical force: gravity variation directly affects normal load
      c.body_force->y += c.mass * c.gravity * (effective_g - p.gravity_mul) * 0.1f - c.velocity.y * c.mass * 0.04f;
      if (c.body_torque) {
        // Torque proportional to rate of gravity change, driving suspension resonance
        float torque = std::cos(orbital_phase) * 0.012f * c.mass * c.gravity * effective_g;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.009f + speed * 0.02f;
      float grav_var = 0.5f + 0.5f * std::sin(phase);
      // Energy cost peaks when gravity is lowest (harder to keep traction)
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + grav_var * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {45, 40, 55};  // dark regolith
    v.particles = {120, 105, 140};  // faint violet dust
    v.liquid = {30, 28, 40};
    v.sky = {80, 70, 100};  // dim twilight
    v.particle_rate = 5.0f;
    v.particle_lift = 2.2f;  // low-gravity float
    v.particle_spread = 1.5f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.35f;  // dark, lidar useful but costly
    v.liquid_surface = false;
    return v;
  }
};

class PrecessionRavine final : public Biome {
 public:
  std::string_view id() const noexcept override { return "precession_ravine"; }
  std::string_view display_name() const noexcept override { return "Precession Ravine"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -70.0f + 30.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.80f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.35f + 0.20f * biome_random01(s, 5);
    p.gravity_mul = 0.55f + 0.25f * biome_random01(s, 6);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.20f + 0.15f * biome_random01(s, 8);
    p.lidar_range_mul = 1.10f + 0.40f * biome_random01(s, 9);
    // Terrain: high-amplitude wavy ridges with moderate craters, few sharp steps
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.10f + 0.40f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.20f + 0.50f * biome_random01(s, 12);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Low gravity reduces normal force; add speed-dependent grip recovery
      float speed = std::abs(c.wheel_speed);
      float grip_boost = 0.6f + 0.4f * std::exp(-speed * 0.15f);
      float depth_factor = 1.0f + 3.0f * c.contact->penetration * 20.0f;
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed * depth_factor * grip_boost);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f + c.contact->normal_force * 0.08f * c.contact->penetration * 30.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.010f + c.contact->penetration * 0.12f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Precessing gravity vector: rotates in the x-y plane at a slow frequency
      float precess_phase = t * 0.004f + c.velocity.x * 0.01f;
      float gravity_x = std::sin(precess_phase) * 0.30f;
      float gravity_y = std::cos(precess_phase) * 0.20f;
      // Effective gravity becomes a rotating vector that tilts the rover
      float lateral_force = gravity_x * c.mass * c.gravity * p.gravity_mul;
      float vertical_force = gravity_y * c.mass * c.gravity * p.gravity_mul;
      // Suspension must adapt to changing vertical load; damping oscillates inversely
      float damping = 0.06f + 0.04f * (0.5f + 0.5f * std::sin(precess_phase * 2.0f));
      c.body_force->x += lateral_force - c.velocity.x * c.mass * damping;
      c.body_force->y += vertical_force - c.velocity.y * c.mass * (0.05f + 0.03f * (0.5f + 0.5f * std::cos(precess_phase)));
      if (c.body_torque) {
        // Gravity gradient induces a slow yawing torque
        float torque = std::cos(precess_phase * 0.7f) * 0.02f * c.mass * c.gravity * p.gravity_mul;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float precess_phase = t * 0.004f + speed * 0.01f;
      // Energy drain peaks when gravity tilts most, forcing careful throttle timing
      float tilt_cost = std::abs(std::sin(precess_phase)) * 0.004f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + tilt_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {48, 42, 58};  // dark indigo regolith
    v.particles = {128, 112, 148};  // pale violet dust
    v.liquid = {32, 28, 44};  // deep purple pools
    v.sky = {88, 76, 118};  // twilight violet
    v.particle_rate = 6.0f;
    v.particle_lift = 2.4f;  // low-gravity floating dust
    v.particle_spread = 1.4f;
    v.base_particles = 2;
    v.max_particles = 32;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.38f;  // dim, encourages lidar use (cheap here)
    v.liquid_surface = false;
    return v;
  }
};

class InertialReedbed final : public Biome {
 public:
  std::string_view id() const noexcept override { return "inertial_reedbed"; }
  std::string_view display_name() const noexcept override { return "Inertial Reedbed"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.50f + 0.25f * biome_random01(s);
    p.sink_rate = 0.002f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.30f + 0.50f * biome_random01(s, 2);
    p.energy_drain_mul = 1.10f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -35.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.60f + 0.50f * biome_random01(s, 5);
    p.solar_charge_rate = 0.70f + 0.30f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.40f + 0.25f * biome_random01(s, 9);
    p.lidar_range_mul = 1.00f + 0.40f * biome_random01(s, 10);
    // Terrain: broad shallow basins with dense gentle craters and low steps, creating momentum pockets
    p.terrain_amplitude_mul = 1.60f + 0.50f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.90f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 0.70f + 0.30f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.75f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::abs(c.wheel_speed) / (1.0f + 5.0f * c.contact->penetration * 30.0f));
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth_factor = 1.0f + 4.0f * c.contact->penetration * 25.0f;
      // Hysteretic friction: high speed gives less grip (reeds flatten), low speed more grip
      float speed_factor = 1.0f - 0.4f * std::tanh(speed * 0.20f);
      float drag = (p.viscosity * 1.2f * depth_factor + 0.4f * speed_factor) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.12f * c.contact->penetration * 28.0f);
    }
    if (c.energy_cost) {
      float speed = std::abs(c.wheel_speed);
      float hyst_factor = 0.6f + 0.4f * std::tanh(speed * 0.15f);
      *c.energy_cost += (0.012f + c.contact->penetration * 0.18f + hyst_factor * 0.015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      // Memory of recent speed: reeds bend and stay bent, creating delayed resistance
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);
      float phase = t * 0.017f + speed * 0.05f;

      // Delayed elastic recoil: resisting acceleration, releasing on deceleration
      float recoil = memory * (0.02f + 0.03f * std::sin(phase)) * c.mass * c.gravity;
      // Inertial drag grows with memory but saturates, rewarding momentum conservation
      float drag = c.velocity.x * c.mass * (0.10f + 0.20f * memory);
      // Lateral sway: reeds push sideways with a hysteresis-dependent phase
      float sway = std::sin(t * 0.023f + speed * 0.07f + memory * 2.0f) * 0.035f * c.mass * (0.5f + memory);

      c.body_force->x -= drag;
      c.body_force->x += recoil + sway;
      c.body_force->y += c.mass * c.gravity * 0.006f * memory * std::sin(t * 0.019f + 0.3f) - c.velocity.y * c.mass * (0.05f + 0.04f * memory);
      if (c.body_torque) {
        float torque = sway * 0.3f + std::sin(t * 0.029f + speed * 0.04f + memory * 1.5f) * 0.009f * c.mass * c.gravity * memory;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f * (1.0f + 0.3f * memory);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);
      // Energy cost ramps with stored momentum, but steady cruise is efficient
      float memory_cost = memory * 0.0035f + speed_norm * 0.0015f;
      *c.energy_cost += (memory_cost + std::abs(c.velocity.x) * 0.0012f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {48, 72, 58};  // muted green marsh soil
    v.particles = {92, 142, 105};  // pale reed dust
    v.liquid = {30, 85, 65};  // shallow murky water
    v.sky = {120, 145, 125};  // overcast greenish light
    v.particle_rate = 9.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 0.8f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.60f;  // dim but visible, lidar moderately useful
    v.liquid_surface = true;
    return v;
  }
};

class SwaybackLedger final : public Biome {
 public:
  std::string_view id() const noexcept override { return "swayback_ledger"; }
  std::string_view display_name() const noexcept override { return "Swayback Ledger"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip, but the ground itself tilts laterally in long, predictable waves
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.15f + 0.35f * biome_random01(s, 2);
    p.wind_force = 1.0f + 3.0f * biome_random01(s, 3);
    p.ambient_temperature = -70.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.8f + 0.25f * biome_random01(s, 5);
    p.solar_charge_rate = 0.9f + 0.3f * biome_random01(s, 6);
    p.gravity_mul = 1.05f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.6f + 0.3f * biome_random01(s, 9);
    p.lidar_range_mul = 1.4f + 0.5f * biome_random01(s, 10);
    // Long rolling hills, few craters/steps, but gentle lateral slopes (via body force below)
    p.terrain_amplitude_mul = 1.6f + 0.4f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.7f + 0.2f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.4f + 0.2f * biome_random01(s, 13);
    p.terrain_step_mul = 0.3f + 0.2f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Adequate grip, but not excessive; body force does the work
    return p.friction_mul * 0.72f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Mild rolling resistance to prevent freewheeling
      *c.wheel_force += c.contact->tangent * (-0.4f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);

      // Long-wavelength lateral tilt: slowly changing cross-slope, deterministic from position
      float spatial_phase = c.velocity.x * 0.012f + t * 0.006f;
      float lateral_slope = 0.7f * std::sin(spatial_phase) + 0.3f * std::sin(spatial_phase * 0.37f + 1.1f);
      // The slope pushes the rover sideways, stronger at speed due to momentum
      float lateral_force = lateral_slope * c.mass * c.gravity * 0.15f * (1.0f + 0.3f * std::tanh(speed * 0.08f));

      // Slight banking: when the slope changes, there's a brief inertial yaw
      float slope_rate = std::cos(spatial_phase) * 0.7f + 0.3f * 0.37f * std::cos(spatial_phase * 0.37f + 1.1f);
      float yaw_torque = slope_rate * 0.02f * c.mass * (1.0f + 0.5f * std::tanh(speed * 0.04f));

      // Damping: less at low speed to allow deliberate control, more at high speed to prevent spins
      float damping = 0.05f + 0.08f * std::tanh(speed * 0.1f);

      c.body_force->x += lateral_force - c.velocity.x * c.mass * damping;
      c.body_force->y += c.mass * c.gravity * 0.004f * std::sin(spatial_phase * 1.3f + 0.5f) - c.velocity.y * c.mass * 0.05f;
      if (c.body_torque) {
        // Counter-steering required; torque encourages body rotation to align with slope
        *c.body_torque += yaw_torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Energy cost scales with lateral deflection, encouraging steady, straight-line speed
      float spatial_phase = speed * 0.012f + t * 0.006f;
      float lateral_mag = std::abs(0.7f * std::sin(spatial_phase) + 0.3f * std::sin(spatial_phase * 0.37f + 1.1f));
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + lateral_mag * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {72, 68, 60};  // muted grey-brown, like weathered slate
    v.particles = {142, 132, 112};  // fine dust
    v.liquid = {48, 44, 38};  // dark pools
    v.sky = {152, 142, 118};  // hazy beige
    v.particle_rate = 5.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.6f;
    v.base_particles = 0;
    v.max_particles = 20;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 2.5f;
    v.screen_brightness = 0.95f;  // clear visibility, lidar optional but useful for long-range planning
    v.liquid_surface = false;
    return v;
  }
};

class ConvectionCanyon final : public Biome {
 public:
  std::string_view id() const noexcept override { return "convection_canyon"; }
  std::string_view display_name() const noexcept override { return "Convection Canyon"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 2);
    p.wind_force = 1.0f + 3.0f * biome_random01(s, 3);
    p.ambient_temperature = 10.0f + 35.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.8f + 1.0f * biome_random01(s, 5);
    p.solar_charge_rate = 0.20f + 0.10f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.30f + 0.20f * biome_random01(s, 9);
    p.lidar_range_mul = 0.60f + 0.30f * biome_random01(s, 10);
    // Terrain: deep rugged canyons with moderate craters and sharp steps, forcing careful energy management
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.30f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.00f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 1.40f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.75f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Heat-softened crust: grip improves with speed (wheels bite in) but rolling resistance rises
      float speed = std::abs(c.wheel_speed);
      float bite = 1.0f - 0.3f * std::exp(-speed * 0.1f);
      *c.wheel_force += c.contact->tangent * (-0.3f * c.wheel_speed * bite);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Thermal load: hot air forces constant cooling drain, worsens with speed
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.006f * p.thermal_transfer;
      *c.energy_cost += (0.010f + std::abs(c.wheel_speed) * 0.006f + heat_load) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);

      // Convection cells: rising hot air creates periodic updrafts that reduce effective gravity
      float cell_phase = t * 0.017f + c.velocity.x * 0.04f;
      float updraft = 0.5f + 0.5f * std::sin(cell_phase);
      float lift_force = updraft * 0.10f * c.mass * c.gravity * p.thermal_transfer;

      // Thermal wind: hot air pushes laterally, stronger during updraft peaks
      float lateral_phase = t * 0.011f + c.velocity.x * 0.03f;
      float lateral_push = std::sin(lateral_phase) * 0.04f * c.mass * (1.0f + updraft * 0.8f);

      // Speed-dependent drag: faster speeds compress the thermal cushion, increasing resistance
      float damping = 0.05f + 0.04f * std::tanh(speed * 0.08f);

      c.body_force->x += lateral_push - c.velocity.x * c.mass * damping;
      c.body_force->y += lift_force - c.velocity.y * c.mass * (0.04f + 0.02f * updraft);

      if (c.body_torque) {
        // Updraft asymmetry creates rocking torque that peaks between cells
        float torque = std::sin(cell_phase + 0.4f) * updraft * 0.015f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.015f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);

      // Thermal pulsing: energy drain oscillates with convection, rewarding steady speed that rides the waves
      float cell_phase = t * 0.017f + speed * 0.04f;
      float updraft = 0.5f + 0.5f * std::sin(cell_phase);
      float thermal_cost = updraft * 0.004f * p.thermal_transfer;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + thermal_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {92, 58, 42};  // heat-baked red rock
    v.particles = {210, 110, 60};  // glowing dust rising with convection
    v.liquid = {48, 32, 24};  // dark thermal pools
    v.sky = {180, 120, 50};  // hot amber haze
    v.particle_rate = 10.0f;
    v.particle_lift = 2.0f;
    v.particle_spread = 1.2f;
    v.base_particles = 3;
    v.max_particles = 42;
    v.particle_size = 2;
    v.ambient_particles = 22;
    v.ambient_drift = 2.2f;
    v.screen_brightness = 0.55f;  // dim but visible, lidar useful for terrain
    v.liquid_surface = false;
    return v;
  }
};

class TorridKiln final : public Biome {
 public:
  std::string_view id() const noexcept override { return "torrid_kiln"; }
  std::string_view display_name() const noexcept override { return "Torrid Kiln"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = 55.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 4.5f + 1.0f * biome_random01(s, 5);
    p.solar_charge_rate = 0.05f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 1.05f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.10f + 0.06f * biome_random01(s, 9);
    p.lidar_range_mul = 0.50f + 0.20f * biome_random01(s, 10);
    // Terrain: broad heat-swelled plateaus with frequent shallow craters and few sharp steps
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.50f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.50f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Heat-softened ground: grip worsens with speed (tires overheat and slide)
      float speed = std::abs(c.wheel_speed);
      float thermal_slip = 1.0f / (1.0f + 0.25f * speed);
      *c.wheel_force += c.contact->tangent * (-0.6f * c.wheel_speed * thermal_slip);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      // Extreme heat forces constant cooling: energy drains even at rest, spikes with engine load
      float heat_load = std::max(0.0f, p.ambient_temperature - 40.0f) * 0.008f * p.thermal_transfer;
      *c.energy_cost += (0.015f + heat_load + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal expansion of ground creates slow, rhythmic heaves that push the rover upward
      float heave_phase = t * 0.009f + c.velocity.x * 0.02f;
      float heave = 0.5f + 0.5f * std::sin(heave_phase);
      float lift = heave * 0.06f * c.mass * c.gravity * p.thermal_transfer;
      // Heat shimmer creates lateral pseudo-wind that deflects the rover
      float shimmer_phase = t * 0.013f + c.velocity.x * 0.03f;
      float lateral = std::sin(shimmer_phase) * 0.03f * c.mass * (1.0f + p.thermal_transfer * 0.2f);
      // Overheating dampens suspension, making control less responsive at high speed
      float damping = 0.05f + 0.05f * std::tanh(speed * 0.06f);
      c.body_force->x += lateral - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.03f * heave);
      if (c.body_torque) {
        // Thermal gradients induce slow rocking that worsens with heat
        float torque = std::sin(t * 0.019f + speed * 0.02f + heave * 1.5f) * 0.014f * c.mass * c.gravity * p.thermal_transfer;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal pulsing: energy drain oscillates with ground heave, and high speed worsens heat build-up
      float heave_phase = t * 0.009f + speed * 0.02f;
      float heave = 0.5f + 0.5f * std::sin(heave_phase);
      float heat_drain = heave * 0.004f * p.thermal_transfer + speed * 0.001f;
      *c.energy_cost += (heat_drain) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {112, 44, 20};  // scorched brick-red soil
    v.particles = {255, 90, 30};  // glowing ember dust
    v.liquid = {60, 20, 10};  // molten pools
    v.sky = {220, 70, 20};  // intense orange haze
    v.particle_rate = 12.0f;
    v.particle_lift = 1.6f;
    v.particle_spread = 1.1f;
    v.base_particles = 4;
    v.max_particles = 44;
    v.particle_size = 2;
    v.ambient_particles = 20;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.10f;  // very dark; solar nearly zero, lidar is cheap but short-range
    v.liquid_surface = false;
    return v;
  }
};

class GeothermalSiphon final : public Biome {
 public:
  std::string_view id() const noexcept override { return "geothermal_siphon"; }
  std::string_view display_name() const noexcept override { return "Geothermal Siphon"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.50f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = 25.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 3.0f + 1.0f * biome_random01(s, 4);
    p.solar_charge_rate = 0.10f + 0.05f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.15f + 0.10f * biome_random01(s, 8);
    p.lidar_range_mul = 0.45f + 0.20f * biome_random01(s, 9);
    // Terrain: moderate amplitude, rough but few craters, abundant sharp steps (thermal vents)
    p.terrain_amplitude_mul = 1.20f + 0.40f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.40f + 0.50f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.50f + 0.20f * biome_random01(s, 12);
    p.terrain_step_mul = 1.70f + 0.50f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.82f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; heat-softened ground gives slightly less grip at speed
      float speed = std::abs(c.wheel_speed);
      float thermal_grip = 1.0f / (1.0f + 0.2f * speed);
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed * thermal_grip);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Constant thermal load, worsens with speed and ambient heat
      float heat_load = std::max(0.0f, p.ambient_temperature - 20.0f) * 0.005f * p.thermal_transfer;
      *c.energy_cost += (0.010f + std::abs(c.wheel_speed) * 0.004f + heat_load) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Geothermal vents: periodic updrafts and thrusts tied to spatial position
      float vent_phase = t * 0.011f + c.velocity.x * 0.025f;
      float vent_pulse = 0.5f + 0.5f * std::sin(vent_phase);
      float vent_strength = vent_pulse * vent_pulse;
      // Updraft reduces effective gravity and adds vertical lift
      float lift = vent_strength * 0.10f * c.mass * c.gravity * p.thermal_transfer;
      // Lateral vent push (asymmetric plumes) creates side forces
      float lateral = std::sin(vent_phase + 0.8f) * vent_strength * 0.05f * c.mass;
      // Speed-dependent damping: faster speeds compress the hot air cushion, increasing drag
      float damping = 0.05f + 0.06f * std::tanh(speed * 0.08f);

      c.body_force->x += lateral - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.03f * vent_strength);
      if (c.body_torque) {
        // Vent asymmetry induces rocking torque, strongest during vent peaks
        float torque = std::sin(vent_phase + 0.4f) * vent_strength * 0.018f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Energy drain oscillates with vent cycles; steady speed that rides the waves is efficient
      float vent_phase = t * 0.011f + speed * 0.025f;
      float vent_strength = (0.5f + 0.5f * std::sin(vent_phase));
      vent_strength *= vent_strength;
      float vent_cost = vent_strength * 0.006f * p.thermal_transfer;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + vent_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {72, 48, 32};  // dark baked rock
    v.particles = {230, 120, 40};  // glowing vent dust
    v.liquid = {44, 24, 12};  // molten pools
    v.sky = {180, 90, 35};  // hot amber haze
    v.particle_rate = 9.0f;
    v.particle_lift = 1.4f;
    v.particle_spread = 0.9f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 18;
    v.ambient_drift = 1.6f;
    v.screen_brightness = 0.25f;  // dark, forces lidar (cheap but short-range)
    v.liquid_surface = false;
    return v;
  }
};

class WanderingDune final : public Biome {
 public:
  std::string_view id() const noexcept override { return "wandering_dune"; }
  std::string_view display_name() const noexcept override { return "Wandering Dune"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.012f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 2);
    p.wind_force = 4.0f + 8.0f * biome_random01(s, 3);
    p.ambient_temperature = -55.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.70f + 0.30f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.006f + 0.015f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.80f + 0.30f * biome_random01(s, 9);
    p.lidar_range_mul = 1.30f + 0.40f * biome_random01(s, 10);
    // Terrain: broad rolling dunes with sparse craters and low steps
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.70f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.8f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      float depth_factor = 1.0f + 4.0f * c.contact->penetration * 30.0f;
      *c.wheel_force += c.contact->tangent * (-0.8f * c.wheel_speed * depth_factor);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f + c.contact->normal_force * 0.18f * c.contact->penetration * 35.0f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.012f + c.contact->penetration * 0.15f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Slow, large-scale dune migration: a moving lateral slope
      float phase = t * 0.0035f + c.velocity.x * 0.008f;
      float dune_slope = std::sin(phase) * 0.7f + 0.3f * std::sin(phase * 0.23f + 1.2f);
      // Temporal drift makes the slope slowly shift, so the optimal path changes
      float drift = 0.5f + 0.5f * std::sin(t * 0.0011f + 0.7f);
      float lateral_force = dune_slope * c.mass * c.gravity * 0.12f * (0.7f + 0.6f * drift);
      // Wind pushes sand, adding a smaller faster oscillation
      float wind_gust = p.wind_force * 0.06f * std::sin(t * 0.041f + c.velocity.x * 0.05f);
      // Damping increases with speed, making it hard to correct at high velocity
      float damping = 0.04f + 0.12f * std::tanh(speed * 0.12f);
      c.body_force->x += lateral_force + wind_gust - c.velocity.x * c.mass * damping;
      c.body_force->y += c.mass * c.gravity * 0.003f * std::sin(phase * 1.7f + 0.4f) * drift - c.velocity.y * c.mass * (0.03f + 0.02f * drift);
      if (c.body_torque) {
        // Dune slope induces rolling torque, stronger when drift is high
        float torque = dune_slope * drift * 0.012f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.0035f + speed * 0.008f;
      float dune_mag = std::abs(std::sin(phase) * 0.7f + 0.3f * std::sin(phase * 0.23f + 1.2f));
      // Energy drain scales with lateral deflection, rewarding steady path
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + dune_mag * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {170, 120, 70};
    v.particles = {210, 160, 95};
    v.liquid = {110, 80, 50};
    v.sky = {190, 150, 90};
    v.particle_rate = 8.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 20;
    v.ambient_drift = 3.0f;
    v.screen_brightness = 0.85f;
    v.liquid_surface = false;
    return v;
  }
};

class LoessDrift final : public Biome {
 public:
  std::string_view id() const noexcept override { return "loess_drift"; }
  std::string_view display_name() const noexcept override { return "Loess Drift"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.010f + 0.025f * biome_random01(s, 1);
    p.viscosity = 0.05f + 0.10f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 3);
    p.wind_force = 1.0f + 3.0f * biome_random01(s, 4);
    p.ambient_temperature = -45.0f + 20.0f * biome_random01(s, 5);
    p.thermal_transfer = 0.55f + 0.20f * biome_random01(s, 6);
    p.solar_charge_rate = 1.10f + 0.40f * biome_random01(s, 7);
    p.gravity_mul = 0.80f + 0.15f * biome_random01(s, 8);
    p.crust_deform = 0.012f + 0.020f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.90f + 0.30f * biome_random01(s, 10);
    p.lidar_range_mul = 1.40f + 0.50f * biome_random01(s, 11);
    // Loess: fine, highly compactable dust. Moderate amplitude, high roughness, sparse craters, low steps.
    p.terrain_amplitude_mul = 1.20f + 0.40f * biome_random01(s, 12);
    p.terrain_roughness_mul = 1.60f + 0.50f * biome_random01(s, 13);
    p.terrain_crater_mul = 0.50f + 0.25f * biome_random01(s, 14);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.45f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Loess is fine and compacts quickly, but also collapses under load.
      // Penetration grows faster with wheel speed, but also re-compacts if speed is low.
      float speed = std::abs(c.wheel_speed);
      float compaction = 1.0f + 2.0f * std::tanh(speed * 0.1f);
      c.contact->penetration += p.sink_rate * c.dt * compaction;
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration;
      float collapse = 1.0f + 8.0f * depth * 30.0f;
      // High-speed spinning causes rapid loss of grip as the loess fluidizes.
      float fluidize = 1.0f / (1.0f + 0.3f * std::abs(c.wheel_speed));
      // But there's a small 'bite' at moderate speed where the dust compacts and grips.
      float bite = 0.7f + 0.3f * std::exp(-std::abs(c.wheel_speed - 1.5f) * 0.5f);
      *c.wheel_force += c.contact->tangent * (-1.1f * c.wheel_speed * collapse * fluidize * bite);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.25f * depth * 30.0f);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration;
      float speedPenalty = 1.0f + 0.5f * std::abs(c.wheel_speed);
      *c.energy_cost += (0.015f + depth * 0.25f + std::abs(c.wheel_speed) * 0.008f * speedPenalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Loess drifts laterally in slow, large-scale waves, like a slow-moving dust sea.
      float phase = t * 0.0027f + c.velocity.x * 0.006f;
      float drift = std::sin(phase) * 0.5f + 0.3f * std::sin(phase * 0.41f + 1.7f);
      // The drift is stronger at higher speeds (more momentum to carry you sideways).
      float drift_force = drift * c.mass * c.gravity * 0.10f * (1.0f + 0.5f * std::tanh(speed * 0.06f));
      // Fine dust also creates a subtle, speed-dependent drag that resists acceleration.
      float drag = c.velocity.x * c.mass * (0.06f + 0.08f * std::tanh(speed * 0.04f));
      // Occasional small 'dust devils' add brief lateral kicks.
      float devil_phase = t * 0.071f + c.velocity.x * 0.03f;
      float devil = std::sin(devil_phase) * 0.02f * c.mass * (1.0f + std::sin(t * 0.013f) * 0.5f);
      c.body_force->x -= drag;
      c.body_force->x += drift_force + devil;
      // Slight vertical settling: the rover sinks a tiny bit into the loess, reducing bounce.
      c.body_force->y -= c.mass * c.gravity * 0.01f * (1.0f + 0.5f * std::sin(t * 0.019f + 0.4f));
      c.body_force->y -= c.velocity.y * c.mass * (0.06f + 0.04f * std::sin(t * 0.031f + 0.8f));
      if (c.body_torque) {
        // Drift induces a slow rolling torque that the rover must counter.
        float torque = drift * 0.015f * c.mass * c.gravity * (1.0f + 0.4f * std::tanh(speed * 0.05f));
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Energy cost includes a component proportional to lateral drift, rewarding steady heading.
      float phase = t * 0.0027f + speed * 0.006f;
      float drift = std::abs(std::sin(phase) * 0.5f + 0.3f * std::sin(phase * 0.41f + 1.7f));
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + drift * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {128, 96, 62};  // pale yellowish loess
    v.particles = {174, 142, 90};  // fine dust
    v.liquid = {72, 54, 36};  // dark dusty pools
    v.sky = {168, 138, 92};  // hazy tan sky
    v.particle_rate = 14.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.6f;
    v.base_particles = 4;
    v.max_particles = 44;
    v.particle_size = 2;
    v.ambient_particles = 30;
    v.ambient_drift = 2.5f;
    v.screen_brightness = 0.75f;  // dim but visible, lidar useful for long-range planning
    v.liquid_surface = false;
    return v;
  }
};

class NocturneScribe final : public Biome {
 public:
  std::string_view id() const noexcept override { return "nocturne_scribe"; }
  std::string_view display_name() const noexcept override { return "Nocturne Scribe"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.25f + 0.18f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -80.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.30f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.03f + 0.02f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.004f + 0.008f * biome_random01(s, 7);
    p.lidar_energy_mul = 4.00f + 1.50f * biome_random01(s, 8);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 9);
    // Terrain: moderate amplitude, high roughness, moderate craters, low steps
    p.terrain_amplitude_mul = 1.10f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.60f + 0.40f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.20f + 0.40f * biome_random01(s, 12);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.38f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      // Traction is extremely low at low speed, but improves slightly with speed
      float speed_grip = 0.25f + 0.75f * std::tanh(speed * 0.5f);
      float depth = 1.0f + 5.0f * c.contact->penetration * 30.0f;
      float drag = 0.9f * c.wheel_speed * depth * speed_grip;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.12f * c.contact->penetration * 35.0f);
    }
    if (c.energy_cost) {
      float depth = 1.0f + c.contact->penetration * 20.0f;
      *c.energy_cost += (0.012f + std::abs(c.wheel_speed) * 0.006f * depth) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);

      // "Scribe" effect: the ground is etched with near-invisible grooves that
      // only become tangible when the rover moves. Traction is low at rest,
      // but the grooves provide a brief grip burst when rolling over them.
      // However, the grooves are chaotic, so the burst direction is unpredictable.
      float groove_phase = t * 0.061f + c.velocity.x * 0.13f;
      float groove_grip = 0.5f + 0.5f * std::sin(groove_phase);
      float grip_force = groove_grip * 0.05f * c.mass * c.gravity * speed_norm;

      // Darkness: the rover can barely see, so it cannot predict the grooves.
      // Lidar is extremely expensive and short-range, so it is rarely used,
      // leaving the agent to learn the pattern through proprioception.
      float dark_drag = 0.08f + 0.10f * speed_norm;
      c.body_force->x += grip_force - c.velocity.x * c.mass * dark_drag;

      // Lateral "scribe": the grooves also push the rover sideways, stronger at speed.
      float lateral = std::sin(groove_phase + 1.3f) * 0.03f * c.mass * speed_norm;
      c.body_force->x += lateral;

      // Vertical micro-bumps from the etched ground, causing suspension jitter.
      c.body_force->y += std::sin(t * 0.047f + c.velocity.x * 0.09f) * 0.004f * c.mass * c.gravity;
      c.body_force->y -= c.velocity.y * c.mass * 0.05f;

      if (c.body_torque) {
        // The grooves induce a rocking torque that grows with speed.
        float torque = std::sin(groove_phase * 0.7f + 0.5f) * 0.012f * c.mass * c.gravity * speed_norm;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Energy cost is heavily speed-dependent, discouraging reckless speeding
      // and rewarding careful, steady pacing that reads the grooves.
      float speed_cost = speed * 0.012f + speed * speed * 0.003f;
      *c.energy_cost += (speed_cost + 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {12, 10, 14};  // near-black, etched rock
    v.particles = {25, 22, 30};  // faint dust
    v.liquid = {8, 7, 10};
    v.sky = {3, 3, 4};  // almost total darkness
    v.particle_rate = 2.0f;
    v.particle_lift = 0.1f;
    v.particle_spread = 0.2f;
    v.base_particles = 0;
    v.max_particles = 8;
    v.particle_size = 1;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.015f;  // extremely dark, solar nearly zero
    v.liquid_surface = false;
    return v;
  }
};

class ScorchedCrosstide final : public Biome {
 public:
  std::string_view id() const noexcept override { return "scorched_crosstide"; }
  std::string_view display_name() const noexcept override { return "Scorched Crosstide"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.45f * biome_random01(s, 2);
    p.wind_force = 2.0f + 6.0f * biome_random01(s, 3);
    p.ambient_temperature = -35.0f + 45.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.2f + 0.8f * biome_random01(s, 5);
    p.solar_charge_rate = 0.08f + 0.05f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.25f + 0.15f * biome_random01(s, 9);
    p.lidar_range_mul = 0.70f + 0.30f * biome_random01(s, 10);
    // Terrain: moderate amplitude, rough with sharp ridges, sparse craters, moderate steps
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.40f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.25f * biome_random01(s, 13);
    p.terrain_step_mul = 1.30f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Heat-softened ground: moderate grip, but worsens with speed as tires overheat
      float speed = std::abs(c.wheel_speed);
      float thermal_grip = 1.0f / (1.0f + 0.15f * speed);
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed * thermal_grip);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Thermal load: ambient heat causes extra drain, scaling with heat and speed
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.005f * p.thermal_transfer;
      *c.energy_cost += (0.008f + heat_load + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);

      // Thermal load: heat from ambient temperature creates warm air currents
      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 20.0f) * 0.03f);

      // Intermittent cross-force: alternates between calm windows and strong lateral gusts,
      // with gust intensity coupled to thermal load
      float long_cycle = 0.5f + 0.5f * std::sin(t * 0.007f + c.velocity.x * 0.02f);
      float short_gust = 0.3f + 0.7f * std::sin(t * 0.039f + c.velocity.x * 0.06f) * std::cos(t * 0.061f);
      float cross_mag = (p.wind_force * 0.10f + heat_factor * 0.18f) * long_cycle * short_gust;
      float cross_force = cross_mag * c.mass;

      // Calm windows: when long cycle is low, cross-force nearly vanishes, allowing progress
      float calm = 1.0f - std::max(0.0f, (long_cycle - 0.35f) * 2.0f) * (short_gust - 0.35f) * 1.8f;

      // Damping: stronger during gusts, lighter in calm windows for efficient cruising
      float damping = 0.04f + 0.07f * (1.0f - calm);
      c.body_force->x += cross_force - c.velocity.x * c.mass * damping;

      // Vertical thermal shimmer: subtle lift during heat peaks, affecting suspension
      c.body_force->y += heat_factor * calm * 0.006f * c.mass * c.gravity * std::sin(t * 0.013f + 0.3f);
      c.body_force->y -= c.velocity.y * c.mass * (0.03f + 0.02f * (1.0f - calm));

      if (c.body_torque) {
        // Cross-force creates yaw torque; stronger during gusts, requiring counter-steering
        float torque = cross_force * 0.35f - c.angular_velocity * c.mass * 0.02f;
        *c.body_torque += torque;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 20.0f) * 0.03f);

      // Energy cost tied to gust activity: calm windows are cheaper, gusts drain more
      float long_cycle = 0.5f + 0.5f * std::sin(t * 0.007f + speed * 0.02f);
      float short_gust = 0.3f + 0.7f * std::sin(t * 0.039f + speed * 0.06f) * std::cos(t * 0.061f);
      float gust_cost = long_cycle * short_gust * heat_factor * 0.008f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + gust_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {92, 52, 38};  // scorched reddish-brown rock
    v.particles = {200, 110, 55};  // glowing ember dust
    v.liquid = {54, 30, 20};  // dark thermal pools
    v.sky = {170, 95, 45};  // hazy amber-orange
    v.particle_rate = 9.0f;
    v.particle_lift = 1.4f;
    v.particle_spread = 1.1f;
    v.base_particles = 3;
    v.max_particles = 38;
    v.particle_size = 2;
    v.ambient_particles = 20;
    v.ambient_drift = 2.2f;
    v.screen_brightness = 0.40f;  // dim, lidar useful but moderately priced
    v.liquid_surface = false;
    return v;
  }
};

class DrivetrainHysteresisSluice final : public Biome {
 public:
  std::string_view id() const noexcept override { return "drivetrain_hysteresis_sluice"; }
  std::string_view display_name() const noexcept override { return "Drivetrain Hysteresis Sluice"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.40f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.50f + 0.80f * biome_random01(s, 2);
    p.energy_drain_mul = 1.10f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -40.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.80f + 0.60f * biome_random01(s, 5);
    p.solar_charge_rate = 0.50f + 0.30f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.20f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 1.10f + 0.40f * biome_random01(s, 10);
    // Terrain: dense broad craters with moderate amplitude and roughness, few sharp steps
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.80f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 0.50f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.0f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration * 30.0f));
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth = c.contact->penetration * 25.0f;
      // Drivetrain hysteresis: resistance is higher when accelerating than when cruising
      // Simulated by making drag depend on wheel acceleration proxy (speed change)
      float speed_mem = 0.4f + 0.6f * std::tanh(speed * 0.25f);
      // Sudden release when momentum exceeds threshold: burst of forward force
      float release = 0.8f + 0.2f * std::sin(speed * 0.5f + c.contact->penetration * 30.0f);
      float drag = (p.viscosity * 2.2f * (1.0f + depth) + 0.35f * speed_mem) * c.wheel_speed * release;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.12f * depth);
    }
    if (c.energy_cost) {
      float speed = std::abs(c.wheel_speed);
      // Hysteresis energy: extra cost when changing speed (acceleration/deceleration), less on steady cruise
      float speed_rate = std::abs(speed - 0.5f);  // proxy for speed change, peak at moderate speeds
      float hyst_cost = 0.015f + 0.05f * std::exp(-speed_rate * 0.8f);
      *c.energy_cost += (0.010f + c.contact->penetration * 0.20f + hyst_cost) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      // Memory of recent speed: drivetrain stores momentum like a flywheel, releasing on deceleration
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);
      // Delayed elastic recoil: resists acceleration, assists on deceleration
      float phase = t * 0.015f + speed * 0.04f;
      float recoil = memory * (0.02f + 0.04f * std::sin(phase)) * c.mass * c.gravity;
      // Inertial drag grows with memory but saturates, rewarding steady speed
      float drag = c.velocity.x * c.mass * (0.10f + 0.20f * memory);
      // Lateral slosh from stored momentum and buoyancy coupling
      float slosh = std::sin(t * 0.027f + speed * 0.06f + memory * 2.0f) * 0.035f * c.mass * (1.0f + 0.5f * memory);
      // Buoyancy coupling: vertical force depends on momentum state
      float buoy = memory * 0.05f * c.mass * c.gravity * std::sin(t * 0.019f + 0.4f);
      c.body_force->x -= drag;
      c.body_force->x += recoil + slosh;
      c.body_force->y += buoy - c.velocity.y * c.mass * (0.05f + 0.04f * memory);
      if (c.body_torque) {
        // Torque tied to momentum state, causing sway that must be counteracted
        float torque = std::sin(t * 0.033f + speed * 0.04f + memory * 1.5f) * 0.010f * c.mass * c.gravity * memory;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f * (1.0f + 0.3f * memory);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);
      // Energy cost scales with stored momentum, but steady cruise is efficient
      float memory_cost = memory * 0.004f + speed_norm * 0.0015f;
      *c.energy_cost += (memory_cost + std::abs(c.velocity.x) * 0.0018f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {40, 52, 46};  // dark greenish muddy soil
    v.particles = {70, 128, 98};  // murky green spray
    v.liquid = {22, 90, 68};  // shallow greenish water
    v.sky = {92, 128, 108};  // dim greenish overcast
    v.particle_rate = 10.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 0.9f;
    v.base_particles = 3;
    v.max_particles = 38;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.55f;  // dim but visible, lidar moderately useful
    v.liquid_surface = true;
    return v;
  }
};

class TidalGravitySluice final : public Biome {
 public:
  std::string_view id() const noexcept override { return "tidal_gravity_sluice"; }
  std::string_view display_name() const noexcept override { return "Tidal Gravity Sluice"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.40f + 0.70f * biome_random01(s, 2);
    p.energy_drain_mul = 1.40f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -50.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.00f + 0.60f * biome_random01(s, 5);
    p.solar_charge_rate = 0.15f + 0.10f * biome_random01(s, 6);
    p.gravity_mul = 0.50f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.30f + 0.20f * biome_random01(s, 9);
    p.lidar_range_mul = 0.90f + 0.30f * biome_random01(s, 10);
    // Terrain: dense broad basins with moderate amplitude, low roughness, heavy craters, low steps
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.90f + 0.60f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.45f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration * 30.0f));
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth = c.contact->penetration * 25.0f;
      float tide_phase = static_cast<float>(c.step_index) * 0.017f + speed * 0.03f;
      float tide = 0.5f + 0.5f * std::sin(tide_phase);
      float drag = (p.viscosity * 2.5f * (1.0f + depth) + 0.3f * tide) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.15f * depth);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 25.0f;
      // Water drag plus tidal energy cost that oscillates with gravity phase
      float tide_cost = 0.01f + 0.04f * std::abs(std::sin(static_cast<float>(c.step_index) * 0.017f));
      *c.energy_cost += (0.015f + depth * 0.20f + tide_cost) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Tidal gravity cycle: effective gravity oscillates spatially and temporally
      float phase = t * 0.013f + c.velocity.x * 0.025f;
      float tide = 0.6f + 0.4f * std::sin(phase);
      float effective_g = p.gravity_mul * tide;
      // When gravity is low, water immersion is more buoyant; when high, more drag
      float buoy = (1.0f - effective_g) * c.mass * c.gravity * 0.12f * std::sin(phase + 0.5f);
      float drag = c.velocity.x * c.mass * (0.08f + 0.12f * effective_g);
      float lateral = std::sin(phase + 1.2f) * 0.04f * c.mass * (1.0f - effective_g * 0.3f);
      c.body_force->x += lateral - drag;
      c.body_force->y += buoy - c.velocity.y * c.mass * (0.10f + 0.05f * effective_g);
      if (c.body_torque) {
        float torque = std::sin(phase * 1.3f + 0.8f) * 0.012f * c.mass * c.gravity * (1.0f - effective_g * 0.5f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.013f + speed * 0.025f;
      float tide = 0.5f + 0.5f * std::sin(phase);
      float scarcity = (1.0f - p.gravity_mul) * 0.006f * (1.0f + tide);
      *c.energy_cost += (std::abs(c.velocity.x) * 0.002f + scarcity) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {34, 50, 44};  // dark teal muddy soil
    v.particles = {64, 128, 100};  // murky green spray
    v.liquid = {20, 92, 70};  // shallow greenish water
    v.sky = {88, 120, 100};  // dim teal overcast
    v.particle_rate = 11.0f;
    v.particle_lift = 1.3f;
    v.particle_spread = 0.9f;
    v.base_particles = 3;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 1.1f;
    v.screen_brightness = 0.45f;  // dim, lidar useful but moderately priced
    v.liquid_surface = true;
    return v;
  }
};

class ChargeNullGeyser final : public Biome {
 public:
  std::string_view id() const noexcept override { return "charge_null_geyser"; }
  std::string_view display_name() const noexcept override { return "Charge Null Geyser"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.20f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -40.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.90f + 0.30f * biome_random01(s, 5);
    p.solar_charge_rate = 0.12f + 0.08f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.008f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.10f + 0.06f * biome_random01(s, 9);
    p.lidar_range_mul = 1.00f + 0.40f * biome_random01(s, 10);
    // Terrain: rugged ridges with frequent sharp steps and moderate craters,
    // forcing careful timing to exploit vent windows
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.40f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.00f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 1.60f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance, slightly higher at speed to reward steady cruise
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      // Charge null zones: periodic “dead zones” where solar charging is suppressed,
      // but geyser vents between them provide brief upward impulses that can be ridden
      // to cross the zone with less energy. The vents are invisible without lidar,
      // which is cheap here but reveals the pattern and positions.
      float zone_phase = t * 0.009f + c.velocity.x * 0.021f;
      float charge_null = 0.5f + 0.5f * std::sin(zone_phase);
      float vent_phase = zone_phase + 0.7f;
      float vent_strength = 0.5f + 0.5f * std::sin(vent_phase);
      vent_strength *= vent_strength;  // narrow, strong peaks

      // Geyser updraft: gives a boost when airborne, helping to hop over obstacles
      float aerial = std::min(1.0f, std::max(0.0f, c.velocity.y * 0.8f));
      float upforce = vent_strength * aerial * 0.12f * c.mass * c.gravity;
      float forward_surge = vent_strength * aerial * 0.08f * c.mass * c.gravity;
      float lateral = std::sin(vent_phase * 1.7f + 0.5f) * vent_strength * aerial * 0.03f * c.mass;

      // Damping: low when airborne (less ground contact), higher on ground
      float damping = 0.04f + 0.06f * (1.0f - aerial);
      c.body_force->x += forward_surge + lateral - c.velocity.x * c.mass * damping;
      c.body_force->y += upforce - c.velocity.y * c.mass * (0.03f + 0.02f * aerial);

      // Small thermal drift from the vents
      c.body_force->x += std::sin(t * 0.013f + speed * 0.02f) * 0.01f * c.mass;

      if (c.body_torque) {
        // Vents induce a pitching torque that can flip the rover if not managed
        float torque = vent_strength * aerial * std::sin(vent_phase * 0.9f + 1.2f) * 0.015f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      // Charge null zones: inside them, solar is suppressed, but outside they recharge
      // quickly. Vent windows provide a small energy bonus, but only when airborne.
      float zone_phase = t * 0.009f + speed * 0.021f;
      float charge_null = 0.5f + 0.5f * std::sin(zone_phase);
      float vent_phase = zone_phase + 0.7f;
      float vent_strength = 0.5f + 0.5f * std::sin(vent_phase);
      vent_strength *= vent_strength;
      float aerial = std::min(1.0f, std::max(0.0f, c.velocity.y * 0.8f));

      // Energy gain when outside null zones (charging), drain inside them
      float charge_gain = (1.0f - charge_null) * p.solar_charge_rate * 0.10f;
      float null_drain = charge_null * 0.006f;
      float vent_bonus = vent_strength * aerial * p.solar_charge_rate * 0.03f;

      *c.energy_cost += (null_drain + std::abs(c.velocity.x) * 0.0015f - charge_gain - vent_bonus) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {68, 62, 55};  // dark grey-brown rock
    v.particles = {180, 200, 220};  // pale cyan dust that marks the vent zones
    v.liquid = {35, 40, 48};
    v.sky = {120, 140, 170};  // hazy blue-grey daylight
    v.particle_rate = 8.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 1.3f;
    v.base_particles = 3;
    v.max_particles = 42;
    v.particle_size = 2;
    v.ambient_particles = 18;
    v.ambient_drift = 2.2f;
    v.screen_brightness = 0.55f;  // dim but visible; solar is weak overall, vents are slightly brighter
    v.liquid_surface = false;
    return v;
  }
};

class QuicksandPulse final : public Biome {
 public:
  std::string_view id() const noexcept override { return "quicksand_pulse"; }
  std::string_view display_name() const noexcept override { return "Quicksand Pulse"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.18f * biome_random01(s);
    p.sink_rate = 0.020f + 0.035f * biome_random01(s, 1);
    p.viscosity = 1.50f + 2.00f * biome_random01(s, 2);
    p.energy_drain_mul = 1.35f + 0.45f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -70.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.00f + 0.30f * biome_random01(s, 5);
    p.solar_charge_rate = 0.18f + 0.10f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.010f * biome_random01(s, 8);
    p.lidar_energy_mul = 0.50f + 0.25f * biome_random01(s, 9);
    p.lidar_range_mul = 0.65f + 0.25f * biome_random01(s, 10);

    // Terrain: broad basins with pulsating quicksand zones (moderate craters, low steps)
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.45f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.30f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.8f * std::abs(c.wheel_speed) / (1.0f + 4.0f * c.contact->penetration * 30.0f));
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth = c.contact->penetration * 30.0f;
      // Time-varying quicksand liquefaction: friction drops sharply during pulses
      float pulse_phase = static_cast<float>(c.step_index) * 0.021f + speed * 0.04f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float liquefy = 1.0f - 0.6f * pulse * std::tanh(depth * 1.2f);
      float drag = (p.viscosity * 2.2f * (1.0f + depth * 0.8f) + 0.4f) * c.wheel_speed * liquefy;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.25f * depth);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      float pulse_phase = static_cast<float>(c.step_index) * 0.021f + std::abs(c.wheel_speed) * 0.04f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      *c.energy_cost += (0.020f + depth * 0.30f + pulse * 0.015f + std::abs(c.wheel_speed) * 0.010f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);

      // Pulsing quicksand: basins rhythmically liquefy, creating strong suction
      // that grows with penetration and is most dangerous at moderate speeds.
      float phase = t * 0.017f + c.velocity.x * 0.04f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      float pulse_width = 0.75f + 0.25f * std::sin(phase * 0.5f + 0.8f);
      float suction = pulse * pulse_width * speed_norm * 0.10f * c.mass * c.gravity;

      // Counter-intuitively, high speed is safer: momentum carries the rover across
      // the basin before the quicksand fully grips.
      float suction_speed_factor = 1.0f / (1.0f + 1.5f * speed_norm);
      suction *= suction_speed_factor;

      // Lateral sloshing of the liquefied sand adds a sideways push
      float slosh = std::sin(phase * 1.3f + 0.5f) * pulse * 0.04f * c.mass * (1.0f - 0.5f * speed_norm);

      // Damping is high when stuck, low when moving fast (freeing the rover)
      float damping = 0.10f + 0.30f * (1.0f - speed_norm) * pulse;
      c.body_force->x += slosh - c.velocity.x * c.mass * damping;
      c.body_force->y -= suction + c.velocity.y * c.mass * (0.20f + 0.10f * pulse);

      if (c.body_torque) {
        // Asymmetric suction forces cause rocking torque during pulses
        float torque = std::sin(phase + 1.1f) * pulse * 0.015f * c.mass * c.gravity * (1.0f - 0.3f * speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);
      float phase = t * 0.017f + speed * 0.04f;
      float pulse = 0.5f + 0.5f * std::sin(phase);

      // Energy drain peaks during pulses and at low speed (fighting the suction)
      float pulse_cost = pulse * (1.0f - 0.5f * speed_norm) * 0.008f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + pulse_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {55, 45, 38};  // dark muddy soil
    v.particles = {110, 90, 65};  // murky brown spray
    v.liquid = {28, 22, 16};  // stagnant pools
    v.sky = {92, 78, 62};  // dim overcast
    v.particle_rate = 9.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 0.9f;
    v.base_particles = 3;
    v.max_particles = 34;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.25f;  // quite dark, lidar moderately useful
    v.liquid_surface = true;
    return v;
  }
};

class GlintScribe final : public Biome {
 public:
  std::string_view id() const noexcept override { return "glint_scribe"; }
  std::string_view display_name() const noexcept override { return "Glint Scribe"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Baseline friction is moderate, but the scribed channels invert grip during lidar blind windows.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.10f + 0.30f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -70.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.10f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 5);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.003f + 0.008f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.30f + 0.15f * biome_random01(s, 8);
    p.lidar_range_mul = 0.35f + 0.15f * biome_random01(s, 9);
    // Terrain: rough, cratered, with sharp steps that require careful placement.
    p.terrain_amplitude_mul = 1.20f + 0.40f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.50f + 0.50f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.20f + 0.40f * biome_random01(s, 12);
    p.terrain_step_mul = 1.50f + 0.50f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Baseline grip is decent, but body effects invert it during blind windows.
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the scribed channels, speed-dependent.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.8f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Higher rolling resistance than baseline, especially at speed.
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.010f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Lidar blind windows: a slow, position-dependent phase where the scribed
      // channels invert their grip. When the phase is high, the rover gains
      // extra traction but takes lateral kicks; when low, it slides and drifts.
      float blind_phase = t * 0.008f + c.velocity.x * 0.018f;
      float blind = 0.5f + 0.5f * std::sin(blind_phase);

      // Additional faster shimmer within the blind window, forcing careful throttle.
      float shimmer_phase = t * 0.023f + c.velocity.x * 0.05f;
      float shimmer = 0.5f + 0.5f * std::sin(shimmer_phase);

      // Effective grip: high during bright phases, low during blind windows.
      float grip_factor = 0.20f + 0.80f * blind;

      // Sliding resistance: when grip is low, the rover slides and drifts.
      float slide_drag = c.velocity.x * c.mass * (0.03f + 0.08f * (1.0f - grip_factor));
      float lateral_drift = std::sin(t * 0.027f + c.velocity.x * 0.06f) * (1.0f - grip_factor) * 0.06f * c.mass;

      // During high-grip phases, the scribed channels add a small forward assist
      // but also a slight lateral pull, rewarding alignment with the channels.
      float channel_assist = grip_factor * (0.02f + 0.02f * shimmer) * c.mass * c.gravity;
      float channel_lateral = grip_factor * std::sin(shimmer_phase + 1.0f) * 0.02f * c.mass;

      // Moderate vertical chatter from the rough scribed ground, stronger at speed.
      float chatter = (0.5f + 0.5f * std::sin(t * 0.041f + speed * 0.08f)) * 0.005f * c.mass * c.gravity * speed_norm;

      // Damping is lower during blind windows, making control harder.
      float damping = 0.05f + 0.05f * grip_factor;

      c.body_force->x -= slide_drag;
      c.body_force->x += lateral_drift + channel_assist + channel_lateral;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y += chatter - c.velocity.y * c.mass * (0.04f + 0.02f * grip_factor);

      if (c.body_torque) {
        // Low-grip windows cause yaw instability; high-grip windows have a
        // channel-guided torque that can help or hinder depending on alignment.
        float torque = std::sin(blind_phase + 1.2f) * (1.0f - grip_factor) * 0.02f * c.mass;
        torque += grip_factor * std::sin(shimmer_phase + 1.7f) * 0.01f * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float blind_phase = t * 0.008f + speed * 0.018f;
      float blind = 0.5f + 0.5f * std::sin(blind_phase);

      // Energy drain is higher during blind windows (fighting sliding) and
      // at moderate speeds; very slow or very fast is more efficient.
      float grip_penalty = (1.0f - blind) * 0.008f;
      float speed_penalty = 0.002f * std::exp(-std::abs(speed - 3.0f) * 0.3f);
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + grip_penalty + speed_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {70, 66, 62};  // pale grey-brown, like weathered tuff
    v.particles = {140, 132, 120};  // fine dust
    v.liquid = {42, 38, 34};  // dark hollows
    v.sky = {150, 148, 138};  // hazy warm grey, not too dark
    v.particle_rate = 6.0f;
    v.particle_lift = 0.6f;
    v.particle_spread = 0.8f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.65f;  // moderately dark, lidar useful but not mandatory
    v.liquid_surface = false;
    return v;
  }
};

class PulsingMireVent final : public Biome {
 public:
  std::string_view id() const noexcept override { return "pulsing_mire_vent"; }
  std::string_view display_name() const noexcept override { return "Pulsing Mire Vent"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.20f * biome_random01(s);
    p.sink_rate = 0.015f + 0.030f * biome_random01(s, 1);
    p.viscosity = 1.00f + 1.50f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -50.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.50f + 0.50f * biome_random01(s, 5);
    p.solar_charge_rate = 0.10f + 0.06f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.010f * biome_random01(s, 8);
    p.lidar_energy_mul = 1.20f + 0.40f * biome_random01(s, 9);
    p.lidar_range_mul = 0.50f + 0.20f * biome_random01(s, 10);
    // Terrain: undulating mud flats with dense broad craters and gentle steps.
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.70f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.70f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.60f + 0.25f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.40f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sinking increases with wheel speed and with a periodic "pulse" that
      // simulates pockets of liquefied mud.
      float phase = static_cast<float>(c.step_index) * 0.041f + std::abs(c.wheel_speed) * 0.07f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      pulse *= pulse;
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * std::abs(c.wheel_speed) * (1.0f + 2.0f * pulse));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 30.0f;
      // Mud viscosity increases with depth, but periodic "pockets" reduce grip suddenly.
      float phase = static_cast<float>(c.step_index) * 0.041f + std::abs(c.wheel_speed) * 0.07f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      pulse *= pulse;
      float grip_loss = 1.0f - 0.5f * pulse * std::tanh(depth * 1.5f);
      float drag = (p.viscosity * 2.0f * (1.0f + depth * 0.8f) + 0.3f) * c.wheel_speed * grip_loss;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.05f + c.contact->normal_force * 0.20f * depth);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      float phase = static_cast<float>(c.step_index) * 0.041f + std::abs(c.wheel_speed) * 0.07f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      pulse *= pulse;
      *c.energy_cost += (0.020f + depth * 0.25f + pulse * 0.020f + std::abs(c.wheel_speed) * 0.008f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Pulsing mud pockets: periodic suction that is strongest at moderate speeds.
      // Fast momentum can carry the rover through before the pocket fully grips.
      float phase = t * 0.031f + c.velocity.x * 0.06f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      pulse *= pulse;
      float suction = pulse * speed_norm * 0.12f * c.mass * c.gravity;
      suction *= 1.0f / (1.0f + 1.2f * speed_norm);

      // Asymmetric mud collapse: lateral push that varies with the pulse.
      float lateral = std::sin(phase * 1.3f + 0.8f) * pulse * 0.05f * c.mass * (1.0f - 0.5f * speed_norm);

      // Damping is high when stuck in a pulse, low when moving fast to escape.
      float damping = 0.08f + 0.25f * pulse * (1.0f - 0.5f * speed_norm);
      c.body_force->x += lateral - c.velocity.x * c.mass * damping;
      c.body_force->y -= suction - c.velocity.y * c.mass * (0.15f + 0.10f * pulse);

      if (c.body_torque) {
        // Pulse-induced rocking torque, stronger during pulse peaks.
        float torque = std::sin(phase + 1.1f) * pulse * 0.018f * c.mass * c.gravity * (1.0f - 0.3f * speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float phase = t * 0.031f + speed * 0.06f;
      float pulse = 0.5f + 0.5f * std::sin(phase);
      pulse *= pulse;
      // Energy drain peaks during pulses and at moderate speeds; steady fast cruise is efficient.
      float pulse_cost = pulse * (1.0f - 0.4f * speed_norm) * 0.010f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + pulse_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {58, 48, 40};  // dark greasy mud
    v.particles = {118, 96, 70};  // brown spray
    v.liquid = {30, 24, 18};  // murky pools
    v.sky = {88, 74, 58};  // dim overcast
    v.particle_rate = 10.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.0f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 0;
    v.ambient_drift = 0.0f;
    v.screen_brightness = 0.28f;  // dark, lidar is expensive and short-range, so crucial decisions must be made on limited info
    v.liquid_surface = true;
    return v;
  }
};

class EclipseTractionWeave final : public Biome {
 public:
  std::string_view id() const noexcept override { return "eclipse_traction_weave"; }
  std::string_view display_name() const noexcept override { return "Eclipse Traction Weave"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.40f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.15f + 0.35f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -80.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.20f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.12f + 0.08f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.60f + 0.25f * biome_random01(s, 8);
    p.lidar_range_mul = 0.50f + 0.20f * biome_random01(s, 9);
    // Terrain: high roughness, moderate craters, low steps. The woven texture
    // dominates, requiring careful traction management rather than obstacle avoidance.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.70f + 0.40f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.10f + 0.30f * biome_random01(s, 12);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Baseline traction is low, but the weave provides periodic grip
      // depending on wheel speed. Slow wheels get a small bite, faster
      // wheels slide more, but there are windows of grip at specific speeds.
      float speed = std::abs(c.wheel_speed);
      float weave_phase = speed * 0.8f;
      float weave = 0.6f + 0.4f * std::sin(weave_phase);
      float grip = 0.5f + 0.5f * weave;
      float drag = (0.7f + 0.3f * (1.0f - grip)) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      // Energy cost is lower when in a grip window, higher when sliding.
      float speed = std::abs(c.wheel_speed);
      float weave = 0.6f + 0.4f * std::sin(speed * 0.8f);
      float grip = 0.5f + 0.5f * weave;
      float sliding_penalty = (1.0f - grip) * 0.02f;
      *c.energy_cost += (0.008f + sliding_penalty + speed * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);

      // Eclipse cycle: a slow, position-dependent oscillation that modulates
      // the weave's grip. When the eclipse is "active" (value near 1),
      // the weave is tighter and provides more grip but also more lateral
      // pull; when "inactive" (near 0), the ground is slicker and the rover
      // drifts more.
      float eclipse_phase = t * 0.0045f + c.velocity.x * 0.012f;
      float eclipse = 0.5f + 0.5f * std::sin(eclipse_phase);

      // Lidar couples to the eclipse: a pulse "illuminates" the weave,
      // temporarily sharpening the grip windows. We cannot read the lidar
      // state directly, but we approximate with a faster oscillation that
      // the agent can learn to match by timing its lidar pulses.
      float lidar_phase = t * 0.019f + c.velocity.x * 0.04f;
      float lidar_window = 0.5f + 0.5f * std::sin(lidar_phase);
      lidar_window *= lidar_window;

      // Effective grip depends on the eclipse and lidar windows.
      float grip_base = 0.30f + 0.50f * eclipse * lidar_window;
      // The weave pushes laterally, stronger when the eclipse is active and
      // the lidar window is high (more visible texture).
      float lateral_force = std::sin(eclipse_phase + 0.8f) * eclipse * lidar_window * 0.06f * c.mass * c.gravity;
      // Sliding drag is high when grip is low.
      float slide_drag = c.velocity.x * c.mass * (0.06f + 0.12f * (1.0f - grip_base));
      // Damping is lower during high-grip windows (allowing momentum), higher
      // during slide windows (making control harder).
      float damping = 0.05f + 0.05f * (1.0f - grip_base);

      c.body_force->x += lateral_force - slide_drag - c.velocity.x * c.mass * damping;
      // Vertical micro-bumps from the rough woven ground, more pronounced at
      // speed and during grip windows (the rover is actually touching).
      float chatter = (0.5f + 0.5f * std::sin(t * 0.037f + c.velocity.x * 0.09f)) * 0.006f * c.mass * c.gravity * speed_norm * grip_base;
      c.body_force->y += chatter - c.velocity.y * c.mass * (0.04f + 0.02f * grip_base);

      if (c.body_torque) {
        // The weave induces yaw torque, stronger when the eclipse is active
        // and the lidar window is high (more visible texture causes more
        // asymmetric forces). During low-grip windows, the rover yaws
        // unpredictably.
        float torque = std::sin(eclipse_phase + 1.2f) * (1.0f - grip_base) * 0.02f * c.mass;
        torque += grip_base * std::sin(lidar_phase + 1.5f) * 0.012f * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float eclipse_phase = t * 0.0045f + speed * 0.012f;
      float eclipse = 0.5f + 0.5f * std::sin(eclipse_phase);
      float lidar_phase = t * 0.019f + speed * 0.04f;
      float lidar_window = 0.5f + 0.5f * std::sin(lidar_phase);
      lidar_window *= lidar_window;
      float grip_base = 0.30f + 0.50f * eclipse * lidar_window;
      // Energy cost is lower during grip windows (using the assist) and
      // higher during slide windows (fighting the drag).
      float grip_saving = grip_base * 0.008f;
      float speed_penalty = 0.001f * speed * speed * 0.1f;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + speed_penalty - grip_saving) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {96, 88, 84};  // pale grey-brown, woven rock texture
    v.particles = {160, 148, 138};  // fine dust
    v.liquid = {58, 54, 50};  // dark hollows
    v.sky = {70, 66, 62};  // dim overcast, not total darkness
    v.particle_rate = 5.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.7f;
    v.base_particles = 0;
    v.max_particles = 20;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.35f;  // dark enough to encourage lidar use, but not completely blind
    v.liquid_surface = false;
    return v;
  }
};

class PolarTractionFade final : public Biome {
 public:
  std::string_view id() const noexcept override { return "polar_traction_fade"; }
  std::string_view display_name() const noexcept override { return "Polar Traction Fade"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.20f + 0.15f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -110.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 2.2f + 0.6f * biome_random01(s, 4);
    p.solar_charge_rate = 1.0f + 0.5f * biome_random01(s, 5);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 7);
    p.lidar_energy_mul = 0.20f + 0.10f * biome_random01(s, 8);
    p.lidar_range_mul = 1.20f + 0.40f * biome_random01(s, 9);
    // Reshape terrain: broad smooth rolling plains with sparse craters and low steps,
    // but high amplitude to force speed choice. The fade is the real hazard.
    p.terrain_amplitude_mul = 1.50f + 0.50f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.60f + 0.25f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.60f + 0.30f * biome_random01(s, 12);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.30f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      // Traction is highest at low speed, fades as speed increases, but recovers slightly
      // at very high speed due to thermal melting giving a thin water film (counter-intuitive).
      float low_speed_grip = 0.7f / (1.0f + 0.5f * speed);
      float high_speed_recovery = 0.2f * std::tanh((speed - 4.0f) * 0.5f);
      float grip = 0.15f + low_speed_grip + high_speed_recovery;
      *c.wheel_force += c.contact->tangent * (-grip * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Energy cost is mild, but thermal load from friction increases with speed,
      // encouraging efficient pacing rather than constant high speed.
      float thermal_friction = std::abs(c.wheel_speed) * 0.003f * p.thermal_transfer;
      *c.energy_cost += (0.005f + thermal_friction) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Polar day/night cycle: brightness oscillates slowly, affecting both visibility
      // and traction (colder dark phases tighten ice, warmer bright phases loosen it).
      float day_phase = t * 0.0042f + c.velocity.x * 0.011f;
      float daylight = 0.5f + 0.5f * std::sin(day_phase);

      // During dark phases, the ice is more rigid and provides better grip,
      // but the rover cannot see far without lidar (which is cheap and long-range here).
      // During bright phases, visibility is good but the surface becomes slicker.
      float grip_mod = 0.8f + 0.2f * (1.0f - daylight);

      // The "fade" effect: a slow, position-dependent modulation that makes traction
      // oscillate between icy-slick and slightly-grippy. It is predictable but requires
      // timing acceleration to the grippy windows.
      float fade_phase = t * 0.0067f + c.velocity.x * 0.018f;
      float fade = 0.5f + 0.5f * std::sin(fade_phase);
      float effective_grip = 0.3f + 0.4f * fade * grip_mod;

      // Lateral drift is stronger when grip is low, but also has a slow oscillation
      // tied to the day/night cycle (thermal expansion of the ice).
      float lateral_drift = std::sin(t * 0.031f + c.velocity.x * 0.07f + day_phase * 0.5f)
        * (1.0f - effective_grip) * 0.05f * c.mass;

      // Damping is lower during high-grip windows (allowing momentum to be conserved)
      // and higher during slick phases (making control harder).
      float damping = 0.05f + 0.08f * (1.0f - effective_grip);

      // Gentle vertical chatter from the ice surface, more pronounced at speed.
      float chatter = (0.5f + 0.5f * std::sin(t * 0.047f + c.velocity.x * 0.11f))
        * 0.004f * c.mass * c.gravity * speed_norm;

      c.body_force->x += lateral_drift - c.velocity.x * c.mass * damping;
      c.body_force->y += chatter - c.velocity.y * c.mass * (0.04f + 0.01f * effective_grip);

      if (c.body_torque) {
        // Ice induces yaw instability, stronger during slick phases. Also a small
        // torque from the day/night thermal gradient.
        float torque = std::sin(fade_phase + 1.3f) * (1.0f - effective_grip) * 0.025f * c.mass;
        torque += (1.0f - daylight) * std::sin(t * 0.021f) * 0.005f * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float day_phase = t * 0.0042f + speed * 0.011f;
      float daylight = 0.5f + 0.5f * std::sin(day_phase);
      float fade_phase = t * 0.0067f + speed * 0.018f;
      float fade = 0.5f + 0.5f * std::sin(fade_phase);
      float effective_grip = 0.3f + 0.4f * fade * (0.8f + 0.2f * (1.0f - daylight));

      // Energy drain is lower during high-grip windows (efficient cruising) and
      // higher during slick phases (fighting the drift). Also a small thermal cost
      // from the day/night cycle.
      float grip_saving = effective_grip * 0.008f;
      float thermal_cost = (1.0f - daylight) * 0.002f * p.thermal_transfer;
      *c.energy_cost += (std::abs(c.velocity.x) * 0.0015f + thermal_cost - grip_saving) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {180, 200, 220};  // ice-blue, brighter than typical dark biomes
    v.particles = {220, 235, 245};  // fine snow
    v.liquid = {90, 140, 170};  // meltwater
    v.sky = {130, 160, 190};  // pale polar sky, but not too bright
    v.particle_rate = 3.0f;
    v.particle_lift = 0.3f;
    v.particle_spread = 0.5f;
    v.base_particles = 0;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 6;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.45f;  // moderate visibility, not total darkness
    v.liquid_surface = false;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const RutileSinkhole biome_0; out.push_back(&biome_0);
  static const MagneticAnomalyDrift biome_1; out.push_back(&biome_1);
  static const MomentumMireBackwater biome_2; out.push_back(&biome_2);
  static const InvertedSuspensionRavine biome_3; out.push_back(&biome_3);
  static const EmberColumn biome_4; out.push_back(&biome_4);
  static const ChromaticPulse biome_5; out.push_back(&biome_5);
  static const OrbitalPendulum biome_6; out.push_back(&biome_6);
  static const PrecessionRavine biome_7; out.push_back(&biome_7);
  static const InertialReedbed biome_8; out.push_back(&biome_8);
  static const SwaybackLedger biome_9; out.push_back(&biome_9);
  static const ConvectionCanyon biome_10; out.push_back(&biome_10);
  static const TorridKiln biome_11; out.push_back(&biome_11);
  static const GeothermalSiphon biome_12; out.push_back(&biome_12);
  static const WanderingDune biome_13; out.push_back(&biome_13);
  static const LoessDrift biome_14; out.push_back(&biome_14);
  static const NocturneScribe biome_15; out.push_back(&biome_15);
  static const ScorchedCrosstide biome_16; out.push_back(&biome_16);
  static const DrivetrainHysteresisSluice biome_17; out.push_back(&biome_17);
  static const TidalGravitySluice biome_18; out.push_back(&biome_18);
  static const ChargeNullGeyser biome_19; out.push_back(&biome_19);
  static const QuicksandPulse biome_20; out.push_back(&biome_20);
  static const GlintScribe biome_21; out.push_back(&biome_21);
  static const PulsingMireVent biome_22; out.push_back(&biome_22);
  static const EclipseTractionWeave biome_23; out.push_back(&biome_23);
  static const PolarTractionFade biome_24; out.push_back(&biome_24);
}
// </MARS_GENERATED_BIOMES>
}  // namespace generated_biomes

inline constexpr std::string_view kBiomeBankVersion = "sha256:6ba0127dafae25cff72d54e776986efd88f761917d84d6365c17ca151c2b0a0d";

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
