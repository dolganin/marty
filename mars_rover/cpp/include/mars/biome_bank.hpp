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
class GranularThrottleTrap final : public Biome {
 public:
  std::string_view id() const noexcept override { return "granular_throttle_trap"; }
  std::string_view display_name() const noexcept override { return "Granular Throttle Trap"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.010f + 0.025f * biome_random01(s, 1);
    p.viscosity = 0.05f + 0.10f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -40.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.30f + 0.20f * biome_random01(s, 6);
    p.gravity_mul = 0.80f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 3.50f + 1.50f * biome_random01(s, 9);
    p.lidar_range_mul = 0.20f + 0.15f * biome_random01(s, 10);
    // Terrain: bumpy, cratered, moderate steps. The trap is the granular bed itself.
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.50f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.30f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.50f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.50f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Granular bed: penetration grows with wheel speed, but only up to a saturation.
      float speed = std::abs(c.wheel_speed);
      float sat = std::tanh(speed * 0.25f);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 2.5f * sat);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration;
      float speed = std::abs(c.wheel_speed);
      // Critical: traction collapses at high speed (grains fluidize), but there is
      // a narrow 'bite' band at low speed where the grains interlock.
      float fluidize = 1.0f / (1.0f + 0.35f * speed * speed);
      float bite = 0.7f + 0.3f * std::exp(-(speed - 1.2f) * (speed - 1.2f) * 0.8f);
      float depth_factor = 1.0f + 6.0f * depth * 30.0f;
      float drag = 1.4f * c.wheel_speed * depth_factor * fluidize * bite;
      *c.wheel_force += c.contact->tangent * (-drag);
      // Deeper penetration also increases normal drag, making recovery harder.
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.20f * depth * 30.0f);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration;
      float speed = std::abs(c.wheel_speed);
      // Energy cost spikes when sliding (high speed) and when deep, rewarding
      // a steady low-speed 'bite' crawl.
      float slide_penalty = 0.02f * std::tanh(speed * 0.3f);
      *c.energy_cost += (0.010f + depth * 0.25f + slide_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);
      // 'Trap': when the rover slides, the granular bed pushes back laterally
      // and drags it down. The more it slides, the stronger the trap.
      float slide = std::max(0.0f, speed_norm - 0.35f);
      float trap_phase = t * 0.031f + c.velocity.x * 0.09f;
      float lateral_trap = std::sin(trap_phase) * slide * 0.08f * c.mass * c.gravity;
      float downward_trap = slide * 0.10f * c.mass * c.gravity;
      // However, there is a subtle 'grain lock' assist: at very low speed, the
      // grains interlock and give a tiny forward push, rewarding slow crawling.
      float lock_assist = (1.0f - speed_norm) * 0.02f * c.mass * c.gravity;

      // Damping is high when sliding, low when crawling (so momentum is not wasted).
      float damping = 0.05f + 0.15f * slide;

      c.body_force->x += lateral_trap + lock_assist - c.velocity.x * c.mass * damping;
      c.body_force->y -= downward_trap - c.velocity.y * c.mass * (0.04f + 0.03f * slide);

      if (c.body_torque) {
        // The granular trap also induces a rocking torque when sliding.
        float torque = std::sin(trap_phase + 1.1f) * slide * 0.02f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);
      float slide = std::max(0.0f, speed_norm - 0.35f);
      // Energy drain is much higher when sliding, very low when crawling.
      float slide_cost = slide * 0.02f;
      float lock_bonus = (1.0f - speed_norm) * 0.002f;
      *c.energy_cost += (0.005f + slide_cost - lock_bonus) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {88, 70, 50};  // brown granular regolith
    v.particles = {158, 130, 90};  // dusty spray
    v.liquid = {52, 38, 26};
    v.sky = {168, 148, 118};  // hazy, muted daylight
    v.particle_rate = 12.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 1.4f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 18;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.22f;  // dark: forces lidar, which is expensive and short-range
    v.liquid_surface = false;
    return v;
  }
};

class CantileverGale final : public Biome {
 public:
  std::string_view id() const noexcept override { return "cantilever_gale"; }
  std::string_view display_name() const noexcept override { return "Cantilever Gale"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate default grip, but the wind and body forces dominate.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 2);
    p.wind_force = 3.0f + 6.0f * biome_random01(s, 3);
    p.ambient_temperature = -50.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Expensive, short-range lidar forces inference from proprioception rather than forward scans.
    p.lidar_energy_mul = 3.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.20f + 0.10f * biome_random01(s, 10);
    // Terrain: low, rolling dunes with sparse craters and almost no steps \u2014
    // the hazard is the wind, not the ground shape.
    p.terrain_amplitude_mul = 1.10f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.50f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Good grip \u2014 enough that the wind can't simply blow the rover away,
    // but it must actively counter-steer and manage throttle to hold a line.
    return p.friction_mul * 1.10f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Light rolling resistance and slight lateral scrub to keep the rover planted.
      *c.wheel_force += c.contact->tangent * (-0.3f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.01f);
    }
    if (c.energy_cost) {
      // Base drain is low; the energy cost comes from wind resistance and correction below.
      *c.energy_cost += (0.004f + std::abs(c.wheel_speed) * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // A slow, large-scale "cantilever" wind: the lateral force grows and fades
      // with a long period, but also has a faster gust component. The rover must
      // lean into the wind by steering and adjusting throttle \u2014 but the wind
      // is invisible without lidar, so the agent must infer it from the drift.
      float slow_phase = t * 0.0023f + c.velocity.x * 0.006f;
      float fast_phase = t * 0.041f + c.velocity.x * 0.05f;
      float slow_mag = 0.5f + 0.5f * std::sin(slow_phase);
      float fast_mag = 0.3f + 0.7f * std::sin(fast_phase);
      // The wind pushes laterally, stronger at higher speed (more surface area exposed).
      float wind_push = p.wind_force * (0.12f + 0.18f * slow_mag * fast_mag) * c.mass * (0.6f + 0.5f * speed_norm);

      // The wind also creates a slight vertical lift that varies with the gust, reducing
      // normal load and thus traction \u2014 so the rover must manage speed to stay planted.
      float lift = 0.02f * c.mass * c.gravity * slow_mag * fast_mag;

      // Damping is moderate; the rover does not want to be too twitchy.
      float damping = 0.05f + 0.03f * slow_mag;

      c.body_force->x += wind_push - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * 0.05f;

      if (c.body_torque) {
        // The wind exerts a torque that tries to yaw the rover into the wind,
        // especially during gusts. Counter-steering is essential.
        float torque = fast_mag * (0.5f + 0.5f * std::sin(slow_phase + 0.8f)) * 0.015f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float slow_phase = t * 0.0023f + speed * 0.006f;
      float fast_phase = t * 0.041f + speed * 0.05f;
      float slow_mag = 0.5f + 0.5f * std::sin(slow_phase);
      float fast_mag = 0.3f + 0.7f * std::sin(fast_phase);

      // Energy drain is proportional to the wind's intensity and to how fast the rover
      // is going (fighting the wind costs more at speed). Steady, moderate cruising
      // with periodic throttle adjustments is the most efficient strategy.
      float wind_cost = (0.5f + 0.5f * slow_mag * fast_mag) * speed_norm * 0.010f;
      float base_cost = 0.004f;
      *c.energy_cost += (base_cost + wind_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, wind-blasted rock and dust \u2014 visually distinct but not revealing the wind pattern.
    v.ground = {142, 126, 118};  // light grey-tan
    v.particles = {190, 178, 168};  // fine pale dust
    v.liquid = {70, 60, 55};  // dark hollows
    v.sky = {190, 170, 150};  // hazy, dusty daylight
    v.particle_rate = 10.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 2.0f;
    v.base_particles = 3;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 30;
    v.ambient_drift = 4.0f;  // strong wind-blown dust, but the actual lateral force is hidden
    v.screen_brightness = 0.20f;  // very dark: lidar is expensive and short-range, so the agent must infer the wind from drift
    v.liquid_surface = false;
    return v;
  }
};

class BarometricBrakeWells final : public Biome {
 public:
  std::string_view id() const noexcept override { return "barometric_brake_wells"; }
  std::string_view display_name() const noexcept override { return "Barometric Brake Wells"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate baseline grip, but strong pressure-coupled hysteresis dominates.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 2);
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 3);
    p.wind_force = 1.0f + 2.0f * biome_random01(s, 4);
    p.ambient_temperature = -55.0f + 20.0f * biome_random01(s, 5);
    p.thermal_transfer = 1.10f + 0.30f * biome_random01(s, 6);
    p.solar_charge_rate = 0.80f + 0.30f * biome_random01(s, 7);
    p.gravity_mul = 1.00f + 0.10f * biome_random01(s, 8);
    p.viscosity = 0.10f + 0.20f * biome_random01(s, 9);
    // Short-range, moderate-cost lidar: the pressure wells are invisible until you're in them.
    p.lidar_energy_mul = 2.5f + 1.0f * biome_random01(s, 10);
    p.lidar_range_mul = 0.30f + 0.10f * biome_random01(s, 11);
    // Terrain: dense, shallow craters with low roughness and few steps — the wells are the hazard,
    // not the geometry. This prevents a generic terrain-avoidance policy from scoring.
    p.terrain_amplitude_mul = 1.50f + 0.40f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 13);
    p.terrain_crater_mul = 1.70f + 0.50f * biome_random01(s, 14);
    p.terrain_step_mul = 0.35f + 0.15f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Decent baseline grip, but the body's pressure coupling changes effective traction drastically.
    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very slight, speed-independent sink into the porous crust; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.2f;
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.25f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Barometric pressure wells: a position-dependent scalar that creates delayed,
      // momentum-dependent resistance. When the rover enters a well, friction rises
      // gradually; when it leaves, friction releases only after the rover has built
      // enough momentum to "blow through" the pressure boundary.
      float well_phase = t * 0.006f + c.velocity.x * 0.018f;
      float well_pressure = 0.5f + 0.5f * std::sin(well_phase);
      well_pressure = well_pressure * well_pressure * 1.5f;  // sharp, distinct wells

      // Hysteresis memory: builds with speed, but lags behind — so acceleration into a well
      // is punished more than steady cruising through it.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 7.0f);

      // The "brake": a delayed, speed-dependent drag that is stronger when the rover is
      // accelerating into a well (high memory) and weaker when it is already moving fast
      // through the well (momentum carries it through).
      float brake_drag = c.velocity.x * c.mass * (0.10f + 0.30f * well_pressure * memory);

      // When momentum is high, a sharp "release" gives a forward assist, simulating
      // the pressure boundary breaking — but it's phase-coupled, so timing matters.
      float release = memory * (0.015f + 0.030f * std::sin(well_phase + 0.8f)) * c.mass * c.gravity;

      // Lateral slosh: the pressure wells push sideways with a delayed, memory-dependent
      // oscillation. This is the true signature — steady, moderate speed is safest.
      float lateral = std::sin(t * 0.021f + speed * 0.05f + memory * 2.0f) * 0.035f * c.mass * (0.5f + well_pressure * 0.8f);

      // Damping is lower when momentum is high (efficient cruising) and higher when
      // speed is low (the wells grip harder).
      float damping = 0.05f + 0.10f * (1.0f - memory) * (1.0f + well_pressure);

      c.body_force->x -= brake_drag;
      c.body_force->x += release + lateral;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: the pressure wells cause slight downward force when speed is low,
      // increasing effective normal load and thus rolling resistance — another reason
      // to keep moving.
      c.body_force->y -= c.mass * c.gravity * (0.02f + 0.04f * (1.0f - memory) * well_pressure);
      c.body_force->y -= c.velocity.y * c.mass * (0.05f + 0.03f * memory);

      if (c.body_torque) {
        // Pressure gradients induce a slow oscillation that tries to yaw the rover;
        // stronger at low speed, weaker when momentum is high.
        float torque = std::sin(t * 0.025f + speed * 0.03f + memory * 1.5f) * 0.012f * c.mass * c.gravity * (1.0f - memory) * well_pressure;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (1.0f + 0.3f * memory);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float well_phase = t * 0.006f + speed * 0.018f;
      float well_pressure = 0.5f + 0.5f * std::sin(well_phase);
      well_pressure = well_pressure * well_pressure * 1.5f;
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 7.0f);

      // Energy cost: peaks when fighting the wells (low speed, high pressure) and when
      // accelerating hard (high memory). Steady moderate cruise is efficient.
      float well_cost = well_pressure * (1.0f - 0.5f * memory) * 0.008f;
      float accel_cost = memory * 0.004f + speed_norm * 0.0015f;
      *c.energy_cost += (well_cost + accel_cost + std::abs(c.velocity.x) * 0.0012f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, wind-scoured rock with faint blue-grey tint — suggests thin atmosphere, not mud or ice.
    v.ground = {112, 118, 124};  // light grey-blue rock
    v.particles = {190, 196, 202};  // fine pale dust
    v.liquid = {48, 56, 64};  // dark hollows
    v.sky = {145, 155, 165};  // hazy, overcast but not dark
    v.particle_rate = 5.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.6f;
    v.base_particles = 0;
    v.max_particles = 18;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.30f;  // dark: lidar is moderately costly and short-range, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class GravityPendulumHollow final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gravity_pendulum_hollow"; }
  std::string_view display_name() const noexcept override { return "Gravity Pendulum Hollow"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Low gravity with a strong oscillating vertical component
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -60.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.40f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Short-range, expensive lidar: the gravity phase cannot be read from afar;
    // it must be felt through the rover's own motion and suspension.
    p.lidar_energy_mul = 3.00f + 1.00f * biome_random01(s, 9);
    p.lidar_range_mul = 0.20f + 0.10f * biome_random01(s, 10);
    // Terrain: broad, smooth rolling hollows with gentle craters and almost no steps,
    // so the gravity modulation is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip, but the oscillation changes effective normal load.
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very slight, speed-independent sink into the porous regolith; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.15f;
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Pendulum gravity: the effective gravity oscillates in a slow, position-
      // dependent cycle, varying both magnitude and direction (slight x tilt).
      // The rover must synchronize throttle with the "heavy" phases when it has
      // traction, and ease off during the "light" phases when it floats and
      // cannot steer effectively.
      float phase = t * 0.006f + c.velocity.x * 0.017f;
      float grav_mag = 0.65f + 0.50f * std::sin(phase);
      float grav_tilt = 0.20f * std::cos(phase + 0.8f);
      float effective_g = p.gravity_mul * grav_mag;

      // Vertical force: when gravity is low, buoyancy lifts the rover, reducing
      // traction; when high, it presses down, increasing normal load and grip.
      // This is faster than the tide in TidalGravitySluice, so the agent must
      // learn a rhythmic pedal pattern rather than just cruising.
      float buoyancy = (1.0f - effective_g) * c.mass * c.gravity * 0.18f * std::sin(phase + 0.5f);
      float weight = (effective_g - p.gravity_mul) * c.mass * c.gravity * 0.12f;

      // Lateral "pendulum" push: the gravity vector tilts, creating a sideways
      // force that grows with speed (more momentum to carry the tilt).
      float lateral_force = grav_tilt * (0.5f + 0.5f * speed_norm) * 0.06f * c.mass * c.gravity;

      // Damping is higher when gravity is high (more grip, better control),
      // lower when gravity is low (easier to coast, but also easier to lose
      // control and slide).
      float damping = 0.05f + 0.10f * (grav_mag - 0.65f) * 0.5f + 0.03f * speed_norm;

      c.body_force->x += lateral_force - c.velocity.x * c.mass * damping;
      c.body_force->y += buoyancy + weight - c.velocity.y * c.mass * (0.05f + 0.03f * (1.0f - effective_g * 0.2f));

      if (c.body_torque) {
        // The tilting gravity vector induces a pitching torque that tries to
        // tip the rover, strongest when the gravity is changing fastest.
        float grav_rate = std::cos(phase) * 0.50f;
        float torque = grav_rate * std::sin(phase + 1.2f) * 0.012f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float phase = t * 0.006f + speed * 0.017f;
      float grav_mag = 0.65f + 0.50f * std::sin(phase);

      // Energy cost is much higher during the "light" phase when control is
      // poor and the rover must fight to keep traction; during the "heavy"
      // phase it can cruise efficiently with high grip.
      float scarcity = (1.0f - grav_mag) * 0.012f + speed_norm * 0.004f;
      *c.energy_cost += (0.006f + scarcity) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, sun-bleached rock with a faint golden tint — suggests low gravity,
    // but not specifically a pendulum; the hazard is hidden in the dynamics.
    v.ground = {132, 118, 92};  // pale tan regolith
    v.particles = {190, 178, 148};  // fine dust
    v.liquid = {60, 52, 40};  // dark hollows (not water)
    v.sky = {188, 168, 138};  // hazy, dusty daylight
    v.particle_rate = 7.0f;
    v.particle_lift = 2.2f;  // low-gravity floating dust
    v.particle_spread = 1.6f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.28f;  // dark: lidar is expensive and short-range, so the agent must feel the gravity phase
    v.liquid_surface = false;
    return v;
  }
};

class AvalancheProbe final : public Biome {
 public:
  std::string_view id() const noexcept override { return "avalanche_probe"; }
  std::string_view display_name() const noexcept override { return "Avalanche Probe"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip, but the avalanche triggers are the real hazard.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.012f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -45.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.45f + 0.25f * biome_random01(s, 6);
    p.gravity_mul = 1.00f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.006f + 0.014f * biome_random01(s, 8);
    // Lidar is expensive and short-range: the avalanche phase must be inferred from motion.
    p.lidar_energy_mul = 3.50f + 1.50f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 10);
    // Terrain: broad rolling slopes with low roughness and sparse craters —
    // the avalanche is the hazard, not the geometry. This prevents a generic
    // terrain-avoidance policy from scoring.
    p.terrain_amplitude_mul = 1.50f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.50f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.75f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Loose snowpack: slight sink, but avalanche forces dominate in body.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.8f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the avalanche triggers are hidden.
      *c.wheel_force += c.contact->tangent * (-0.4f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.03f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Avalanche trigger phase: a slow, position-dependent cycle that determines
      // whether the snowpack is "stable" (low risk) or "loaded" (high risk). When
      // the rover drives fast into a loaded phase, it triggers an avalanche that
      // pushes it backward and sideways, punishes throttle, and drains energy.
      // When it slows down during loaded phases, it avoids triggering and can
      // pass through safely. The phase is invisible without lidar (which is
      // expensive and short here), so the agent must learn the pattern from
      // the rover's own motion and vice-versa.
      float load_phase = t * 0.004f + c.velocity.x * 0.011f;
      float load = 0.5f + 0.5f * std::sin(load_phase);
      load = load * load * 1.5f;  // sharp, distinct loaded zones

      // Avalanche trigger: probability/strength scales with speed and load,
      // but has a sharp threshold — a "probe" that must be measured by
      // cautiously increasing speed and watching for the response.
      float trigger = std::max(0.0f, speed_norm - 0.35f) * load * 1.2f;
      trigger = std::min(1.0f, trigger);

      // When triggered, the avalanche applies a strong backward drag and
      // lateral push, with a delayed release. This is the opposite of
      // momentum-based biomes: it punishes high speed rather than rewarding it.
      float avalanche_force = trigger * (0.20f + 0.10f * std::sin(load_phase + 0.8f)) * c.mass * c.gravity;
      float lateral_force = trigger * std::sin(t * 0.031f + speed * 0.05f) * 0.08f * c.mass * c.gravity;

      // Damping is high when triggered (snow grips), low when stable.
      float damping = 0.04f + 0.15f * trigger;

      c.body_force->x -= avalanche_force + c.velocity.x * c.mass * damping;
      c.body_force->x += lateral_force;
      c.body_force->y -= trigger * 0.05f * c.mass * c.gravity - c.velocity.y * c.mass * (0.03f + 0.02f * trigger);

      if (c.body_torque) {
        // Avalanche induces a rocking torque when triggered, making control harder.
        float torque = trigger * std::sin(load_phase + 1.1f) * 0.02f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float load_phase = t * 0.004f + speed * 0.011f;
      float load = 0.5f + 0.5f * std::sin(load_phase);
      load = load * load * 1.5f;
      float trigger = std::max(0.0f, speed_norm - 0.35f) * load * 1.2f;
      trigger = std::min(1.0f, trigger);

      // Energy cost peaks when triggering an avalanche; stable slow cruising is cheap.
      float trigger_cost = trigger * 0.02f;
      *c.energy_cost += (0.005f + speed * 0.0015f + trigger_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, snow-dusted rock that suggests a cold, unstable slope — but not
    // the avalanche pattern itself.
    v.ground = {168, 178, 188};
    v.particles = {210, 220, 230};
    v.liquid = {90, 100, 110};
    v.sky = {210, 218, 226};
    v.particle_rate = 6.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.8f;
    v.base_particles = 0;
    v.max_particles = 16;
    v.particle_size = 2;
    v.ambient_particles = 6;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.35f;  // dark: lidar is costly and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class StrobePitfall final : public Biome {
 public:
  std::string_view id() const noexcept override { return "strobe_pitfall"; }
  std::string_view display_name() const noexcept override { return "Strobe Pitfall"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip but the hazard is the flashing ground that periodically loses stiffness.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.010f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -60.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 4);
    p.solar_charge_rate = 1.0f + 0.5f * biome_random01(s, 5);
    p.gravity_mul = 1.0f + 0.1f * biome_random01(s, 6);
    p.crust_deform = 0.010f + 0.020f * biome_random01(s, 7);
    // Lidar is expensive and short: the strobe phase must be inferred from wheel/body response.
    p.lidar_energy_mul = 4.0f + 1.5f * biome_random01(s, 8);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 9);
    // Terrain: fairly smooth with broad craters and low steps — the hazard is the strobe, not geometry.
    p.terrain_amplitude_mul = 1.3f + 0.4f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.6f + 0.2f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.6f + 0.5f * biome_random01(s, 12);
    p.terrain_step_mul = 0.3f + 0.15f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Baseline is decent, but the strobe modulates effective traction in body effects.
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the fractured crust; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.5f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from body effects.
      *c.wheel_force += c.contact->tangent * (-0.4f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Strobe cycle: a fast oscillation independent of speed, plus a slower spatial phase.
      // The fast part (0.19 rad/step) is what the rover feels as brief traction loss;
      // the slow part (0.013 rad/step) changes whether the strobe is in a 'hard' or 'soft' zone.
      float fast_phase = t * 0.19f;
      float slow_phase = t * 0.013f + c.velocity.x * 0.031f;

      // Strobe intensity: 0 (hard, normal) to 1 (soft, sinking).
      float strobe = 0.5f + 0.5f * std::sin(fast_phase + 0.3f * std::sin(slow_phase));
      strobe = strobe * strobe; // narrow, strong pulses

      // When strobe is high, the ground turns to quicksand: strong drag, sink, and traction loss.
      // When strobe is low, the ground is solid and grip is normal.
      float soft = strobe;

      // Soft ground: strong backward drag and sinking, but only if the rover is moving.
      // At rest, the ground stiffens and the rover doesn't sink further.
      float sink_drag = soft * speed_norm * 0.15f * c.mass * c.gravity;
      float sink_force = soft * (0.6f + 0.4f * speed_norm) * 0.12f * c.mass * c.gravity;

      // Lateral instability: during soft phases, the ground gives way unevenly,
      // pushing the rover sideways with a force that depends on speed and strobe phase.
      float lateral = soft * (0.5f + 0.5f * std::sin(fast_phase * 0.7f + slow_phase * 2.0f)) * 0.10f * c.mass * (0.5f + speed_norm);

      // Damping is high during soft phases (more resistance), low during hard (efficient).
      float damping = 0.04f + 0.12f * soft;

      c.body_force->x -= sink_drag + c.velocity.x * c.mass * damping;
      c.body_force->x += lateral;
      c.body_force->y -= sink_force + c.velocity.y * c.mass * (0.04f + 0.05f * soft);

      if (c.body_torque) {
        // Soft ground causes uneven sinking, creating a rocking torque that can flip the rover
        // if it's going too fast during a soft pulse.
        float torque = soft * std::sin(fast_phase * 0.5f + 1.3f) * 0.018f * c.mass * c.gravity * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float fast_phase = t * 0.19f;
      float slow_phase = t * 0.013f + speed * 0.031f;
      float strobe = 0.5f + 0.5f * std::sin(fast_phase + 0.3f * std::sin(slow_phase));
      strobe = strobe * strobe;
      // Energy cost peaks during soft phases (fighting the sink) and at moderate speed;
      // very slow or very fast (skipping over soft pulses) is more efficient.
      float soft_cost = strobe * (0.5f + 0.5f * speed_norm) * 0.012f;
      float speed_penalty = 0.001f * std::exp(-std::abs(speed - 2.5f) * 0.4f);
      *c.energy_cost += (0.006f + soft_cost + speed_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, fractured rock with a faint cyan glow — but no visible strobe pattern,
    // so the flashing must be felt, not seen.
    v.ground = {48, 54, 60};
    v.particles = {140, 220, 240};
    v.liquid = {30, 100, 120};
    v.sky = {90, 110, 130};
    v.particle_rate = 4.0f;
    v.particle_lift = 0.4f;
    v.particle_spread = 0.6f;
    v.base_particles = 0;
    v.max_particles = 18;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.10f;  // very dark: lidar is expensive and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class StroboscopicCrust final : public Biome {
 public:
  std::string_view id() const noexcept override { return "stroboscopic_crust"; }
  std::string_view display_name() const noexcept override { return "Stroboscopic Crust"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip, but the crust periodically loses stiffness, creating a rhythm hazard
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.006f + 0.014f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -60.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 4);
    p.solar_charge_rate = 0.20f + 0.10f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.008f + 0.018f * biome_random01(s, 7);
    // Lidar is expensive and very short-range: the strobe must be inferred from motion
    p.lidar_energy_mul = 4.5f + 1.5f * biome_random01(s, 8);
    p.lidar_range_mul = 0.12f + 0.08f * biome_random01(s, 9);
    // Terrain: rolling, cratered, low steps — the hazard is the strobe, not geometry
    p.terrain_amplitude_mul = 1.4f + 0.4f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.7f + 0.2f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.6f + 0.5f * biome_random01(s, 12);
    p.terrain_step_mul = 0.3f + 0.2f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Baseline decent, but body effects modulate effective traction
    return p.friction_mul * 0.95f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the fractured crust; not a mud trap
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.6f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from body effects
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Strobe cycle: fast oscillation (0.17 rad/step) plus a slower spatial phase.
      // The fast part is what the rover feels as brief traction loss; the slow part
      // changes whether the strobe is in a 'hard' or 'soft' zone (spatial periodicity).
      float fast_phase = t * 0.17f;
      float slow_phase = t * 0.011f + c.velocity.x * 0.029f;

      // Strobe intensity: 0 (hard, normal) to 1 (soft, sinking).
      float strobe = 0.5f + 0.5f * std::sin(fast_phase + 0.4f * std::sin(slow_phase));
      strobe = strobe * strobe; // narrow, strong pulses

      // Soft ground: strong backward drag and sinking, but only if the rover is moving.
      // At rest, the ground stiffens and the rover doesn't sink further
      float soft = strobe;

      float sink_drag = soft * speed_norm * 0.18f * c.mass * c.gravity;
      float sink_force = soft * (0.7f + 0.3f * speed_norm) * 0.14f * c.mass * c.gravity;

      // Lateral instability: during soft phases, the ground gives way unevenly,
      // pushing the rover sideways with a force that depends on speed and strobe phase
      float lateral = soft * (0.5f + 0.5f * std::sin(fast_phase * 0.7f + slow_phase * 2.0f)) * 0.12f * c.mass * (0.5f + speed_norm);

      // Damping is high during soft phases (more resistance), low during hard (efficient)
      float damping = 0.04f + 0.14f * soft;

      c.body_force->x -= sink_drag + c.velocity.x * c.mass * damping;
      c.body_force->x += lateral;
      c.body_force->y -= sink_force + c.velocity.y * c.mass * (0.04f + 0.06f * soft);

      if (c.body_torque) {
        // Soft ground causes uneven sinking, creating a rocking torque that can flip the rover
        // if it's going too fast during a soft pulse
        float torque = soft * std::sin(fast_phase * 0.5f + 1.3f) * 0.020f * c.mass * c.gravity * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float fast_phase = t * 0.17f;
      float slow_phase = t * 0.011f + speed * 0.029f;
      float strobe = 0.5f + 0.5f * std::sin(fast_phase + 0.4f * std::sin(slow_phase));
      strobe = strobe * strobe;

      // Energy cost peaks during soft phases (fighting the sink) and at moderate speed;
      // very slow or very fast (skipping over soft pulses) is more efficient
      float soft_cost = strobe * (0.5f + 0.5f * speed_norm) * 0.014f;
      float speed_penalty = 0.001f * std::exp(-std::abs(speed - 2.5f) * 0.4f);
      *c.energy_cost += (0.006f + soft_cost + speed_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, fractured rock with a faint cyan glow — but no visible strobe pattern,
    // so the flashing must be felt, not seen
    v.ground = {52, 58, 64};
    v.particles = {150, 230, 250};
    v.liquid = {34, 110, 130};
    v.sky = {95, 115, 135};
    v.particle_rate = 4.0f;
    v.particle_lift = 0.4f;
    v.particle_spread = 0.6f;
    v.base_particles = 0;
    v.max_particles = 18;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.08f;  // very dark: lidar is costly and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class AccordionGravityLadder final : public Biome {
 public:
  std::string_view id() const noexcept override { return "accordion_gravity_ladder"; }
  std::string_view display_name() const noexcept override { return "Accordion Gravity Ladder"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Low but variable gravity; the ladder changes effective gravity in discrete spatial steps.
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 3);
    p.ambient_temperature = -60.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.30f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.40f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 8);
    // Lidar is expensive and very short-range: the gravity steps are not visible,
    // only felt through suspension and wheel load.
    p.lidar_energy_mul = 3.50f + 1.00f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, smooth terraces with gentle slopes and almost no steps,
    // so the gravity ladder is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.15f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.80f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip, but the gravity ladder changes effective normal load.
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very slight, speed-independent sink into the porous regolith; not a trait.
      c.contact->penetration += p.sink_rate * c.dt * 0.1f;
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.30f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Base drain is modest; the ladder exacts its cost in body effects.
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Accordion gravity ladder: effective gravity changes in discrete spatial "rungs" —
      // a slow, position-dependent staircase function. The rover feels alternating
      // heavy and light rungs. It must shift its weight and throttle to match:
      // on heavy rungs it has traction and can push; on light rungs it floats and
      // cannot steer effectively, so it must coast through them.
      // The phase is not directly observable; the agent must infer it from
      // suspension compression and wheel load changes.
      float phase = t * 0.0042f + c.velocity.x * 0.012f;
      float rung = std::sin(phase) > 0.0f ? 1.0f : -1.0f;  // sharp, discrete steps
      float smooth = 0.5f + 0.5f * rung;  // 0 on light, 1 on heavy
      float grav_mag = 0.45f + 1.10f * smooth;  // oscillates between 0.45 and 1.55
      float effective_g = p.gravity_mul * grav_mag;

      // On heavy rungs, gravity pulls down, increasing normal load and traction,
      // but also adding rolling resistance. On light rungs, buoyancy lifts the
      // rover, reducing traction and making it skittish.
      float weight = (effective_g - p.gravity_mul) * c.mass * c.gravity * 0.20f;
      float buoyancy = (p.gravity_mul - effective_g) * c.mass * c.gravity * 0.15f;

      // Light rungs induce lateral "accordion" sway: the rover drifts sideways
      // as if being pulled by a weak spring, stronger when moving fast.
      float lateral = smooth * (0.5f + 0.5f * std::sin(phase * 3.0f + 1.0f)) * 0.04f * c.mass * (0.4f + speed_norm);

      // Damping is higher on heavy rungs (good grip), lower on light rungs
      // (less control, more drift).
      float damping = 0.05f + 0.10f * smooth + 0.02f * speed_norm;

      // When transitioning from light to heavy, a brief "thump" adds a forward
      // impulse; the reverse transition adds a backward tug. This makes the
      // rhythm important: throttle timing matters.
      float transition = std::cos(phase) > 0.0f ? 1.0f : -1.0f;
      float thump = transition * 0.03f * c.mass * c.gravity * smooth;

      c.body_force->x += lateral + thump - c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * smooth);

      if (c.body_torque) {
        // The transition between rungs induces a pitching torque that can rock
        // the rover, strongest when moving fast.
        float torque = transition * (1.0f - smooth) * 0.018f * c.mass * c.gravity * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.0042f + speed * 0.012f;
      float rung = std::sin(phase) > 0.0f ? 1.0f : -1.0f;
      float smooth = 0.5f + 0.5f * rung;

      // Energy cost is lower on light rungs (coasting), higher on heavy rungs
      // (pushing through). But inefficient throttle during light rungs wastes
      // energy via wheel spin and drift. The agent must learn to coast on light,
      // push on heavy.
      float rung_cost = smooth * 0.006f + (1.0f - smooth) * (0.004f + speed * 0.002f);
      *c.energy_cost += (0.005f + rung_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, sun-bleached rock with faint layered bands — suggests terraces,
    // but not the gravity ladder itself.
    v.ground = {138, 124, 98};  // light tan, stratified
    v.particles = {195, 182, 150};  // fine dust
    v.liquid = {64, 56, 42};  // dark hollows
    v.sky = {192, 172, 140};  // hazy dusty daylight
    v.particle_rate = 6.0f;
    v.particle_lift = 2.4f;  // low-gravity floating dust
    v.particle_spread = 1.7f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 1.4f;
    v.screen_brightness = 0.25f;  // dark: lidar is expensive and short-range, so the agent must feel the rungs
    v.liquid_surface = false;
    return v;
  }
};

class RuttedThrottleBasin final : public Biome {
 public:
  std::string_view id() const noexcept override { return "rutted_throttle_basin"; }
  std::string_view display_name() const noexcept override { return "Rutted Throttle Basin"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.006f + 0.014f * biome_random01(s, 1);
    p.viscosity = 0.05f + 0.10f * biome_random01(s, 2);
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -45.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.55f + 0.25f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.008f + 0.016f * biome_random01(s, 8);
    p.lidar_energy_mul = 4.5f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.12f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, gently sloped basins with dense shallow craters and low steps.
    // The hazard is the rut pattern, not the geometry.
    p.terrain_amplitude_mul = 1.5f + 0.4f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.6f + 0.2f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.6f + 0.5f * biome_random01(s, 13);
    p.terrain_step_mul = 0.3f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sinking into deformable rut material, speed-dependent but saturating.
      float speed = std::abs(c.wheel_speed);
      float sat = std::tanh(speed * 0.2f);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * sat);
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth = c.contact->penetration * 30.0f;
      // Rut pattern: a spatial phase that changes whether the wheel is in a groove
      // (high grip, aligned force) or on a ridge (low grip, sideways force).
      // The phase is slow but position-dependent, invisible without expensive lidar.
      float rut_phase = c.contact->penetration * 40.0f + c.wheel_speed * 0.3f;
      float rut = 0.5f + 0.5f * std::sin(rut_phase);

      // Traction: moderate in groove, poor on ridge, and collapses at high speed
      // (the rut material fluidizes).
      float grip = 0.4f + 0.5f * rut;
      float fluidize = 1.0f / (1.0f + 0.30f * speed * speed);
      float depth_factor = 1.0f + 3.0f * depth;
      float drag = 1.0f * c.wheel_speed * grip * fluidize * depth_factor;
      *c.wheel_force += c.contact->tangent * (-drag);

      // Deep penetration increases normal drag, making recovery harder.
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.04f + c.contact->normal_force * 0.18f * depth);
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      float speed = std::abs(c.wheel_speed);
      float rut_phase = c.contact->penetration * 40.0f + speed * 0.3f;
      float rut = 0.5f + 0.5f * std::sin(rut_phase);
      // Energy cost is higher on ridges (fighting the bump) and at high speed
      // (sliding/fluidizing), lower in grooves at moderate speed.
      float ridge_penalty = (1.0f - rut) * 0.015f;
      float slide_penalty = 0.02f * std::tanh(speed * 0.3f);
      *c.energy_cost += (0.010f + depth * 0.20f + ridge_penalty + slide_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // The rut pattern is primarily felt through body forces: a slow, spatial
      // phase creates alternating 'groove' and 'ridge' states. In a groove, the
      // rover is aligned and gets a forward assist; on a ridge, it gets a strong
      // lateral push and a downward force that worsens with speed.
      float rut_phase = t * 0.011f + c.velocity.x * 0.031f;
      float rut = 0.5f + 0.5f * std::sin(rut_phase);

      // Ridge: lateral instability, strong push, and a downward 'seating' force.
      // Groove: forward assist, small lateral alignment, and slight lift.
      float lateral_force = (1.0f - rut) * std::sin(rut_phase + 1.2f) * 0.10f * c.mass * (0.7f + 0.5f * speed_norm);
      float forward_assist = rut * 0.02f * c.mass * c.gravity;
      float downward = (1.0f - rut) * 0.06f * c.mass * c.gravity * (1.0f + 0.5f * speed_norm);
      float lift = rut * 0.01f * c.mass * c.gravity * speed_norm;

      // Damping is higher on ridges (more resistance), lower in grooves (efficient).
      float damping = 0.05f + 0.08f * (1.0f - rut) + 0.02f * speed_norm;

      c.body_force->x += lateral_force + forward_assist - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - downward - c.velocity.y * c.mass * (0.04f + 0.02f * (1.0f - rut));

      if (c.body_torque) {
        // Ridge states induce a rocking torque that can flip the rover if it's
        // going too fast; groove states provide a stabilizing torque that
        // rewards alignment.
        float torque = (1.0f - rut) * std::sin(rut_phase + 2.0f) * 0.018f * c.mass * c.gravity * (0.6f + speed_norm);
        torque += rut * std::sin(rut_phase * 0.5f) * 0.006f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float rut_phase = t * 0.011f + speed * 0.031f;
      float rut = 0.5f + 0.5f * std::sin(rut_phase);
      // Energy cost is much higher on ridges (fighting the bump) and at high
      // speed (sliding/fluidizing); grooves at moderate speed are efficient.
      float ridge_cost = (1.0f - rut) * 0.012f;
      float slide_cost = speed * 0.002f * (1.0f + (1.0f - rut));
      *c.energy_cost += (0.005f + ridge_cost + slide_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, reddish-brown rutted soil with faint parallel striations; visual
    // suggests 'ruts' but not the exact phase pattern, which is hidden.
    v.ground = {88, 56, 44};  // dark red-brown
    v.particles = {160, 110, 80};  // dusty spray
    v.liquid = {40, 28, 20};  // dark pools
    v.sky = {140, 100, 70};  // dim overcast
    v.particle_rate = 9.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.15f;  // very dark: lidar is very expensive and short, so the rut pattern must be inferred from motion
    v.liquid_surface = false;
    return v;
  }
};

class PendulumHollow final : public Biome {
 public:
  std::string_view id() const noexcept override { return "pendulum_hollow"; }
  std::string_view display_name() const noexcept override { return "Pendulum Hollow"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Low gravity with a strong oscillating vertical component
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -60.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.40f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Short-range, expensive lidar: the gravity phase cannot be read from afar;
    // it must be felt through the rover's own motion and suspension.
    p.lidar_energy_mul = 3.00f + 1.00f * biome_random01(s, 9);
    p.lidar_range_mul = 0.20f + 0.10f * biome_random01(s, 10);
    // Terrain: broad, smooth rolling hollows with gentle craters and almost no steps,
    // so the gravity modulation is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip, but the oscillation changes effective normal load.
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very slight, speed-independent sink into the porous regolith; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.15f;
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Pendulum gravity: the effective gravity oscillates in a slow, position-
      // dependent cycle, varying both magnitude and direction (slight x tilt).
      // The rover must synchronize throttle with the "heavy" phases when it has
      // traction, and ease off during the "light" phases when it floats and
      // cannot steer effectively.
      float phase = t * 0.006f + c.velocity.x * 0.017f;
      float grav_mag = 0.65f + 0.50f * std::sin(phase);
      float grav_tilt = 0.20f * std::cos(phase + 0.8f);
      float effective_g = p.gravity_mul * grav_mag;

      // Vertical force: when gravity is low, buoyancy lifts the rover, reducing
      // traction; when high, it presses down, increasing normal load and grip.
      // This is faster than the tide in TidalGravitySluice, so the agent must
      // learn a rhythmic pedal pattern rather than just cruising.
      float buoyancy = (1.0f - effective_g) * c.mass * c.gravity * 0.18f * std::sin(phase + 0.5f);
      float weight = (effective_g - p.gravity_mul) * c.mass * c.gravity * 0.12f;

      // Lateral "pendulum" push: the gravity vector tilts, creating a sideways
      // force that grows with speed (more momentum to carry the tilt).
      float lateral_force = grav_tilt * (0.5f + 0.5f * speed_norm) * 0.06f * c.mass * c.gravity;

      // Damping is higher when gravity is high (more grip, better control),
      // lower when gravity is low (easier to coast, but also easier to lose
      // control and slide).
      float damping = 0.05f + 0.10f * (grav_mag - 0.65f) * 0.5f + 0.03f * speed_norm;

      c.body_force->x += lateral_force - c.velocity.x * c.mass * damping;
      c.body_force->y += buoyancy + weight - c.velocity.y * c.mass * (0.05f + 0.03f * (1.0f - effective_g * 0.2f));

      if (c.body_torque) {
        // The tilting gravity vector induces a pitching torque that tries to
        // tip the rover, strongest when the gravity is changing fastest.
        float grav_rate = std::cos(phase) * 0.50f;
        float torque = grav_rate * std::sin(phase + 1.2f) * 0.012f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float phase = t * 0.006f + speed * 0.017f;
      float grav_mag = 0.65f + 0.50f * std::sin(phase);

      // Energy cost is much higher during the "light" phase when control is
      // poor and the rover must fight to keep traction; during the "heavy"
      // phase it can cruise efficiently with high grip.
      float scarcity = (1.0f - grav_mag) * 0.012f + speed_norm * 0.004f;
      *c.energy_cost += (0.006f + scarcity) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, sun-bleached rock with a faint golden tint — suggests low gravity,
    // but not specifically a pendulum; the hazard is hidden in the dynamics.
    v.ground = {132, 118, 92};  // pale tan regolith
    v.particles = {190, 178, 148};  // fine dust
    v.liquid = {60, 52, 40};  // dark hollows (not water)
    v.sky = {188, 168, 138};  // hazy, dusty daylight
    v.particle_rate = 7.0f;
    v.particle_lift = 2.2f;  // low-gravity floating dust
    v.particle_spread = 1.6f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.28f;  // dark: lidar is expensive and short-range, so the agent must feel the gravity phase
    v.liquid_surface = false;
    return v;
  }
};

class FlywheelBrakeLag final : public Biome {
 public:
  std::string_view id() const noexcept override { return "flywheel_brake_lag"; }
  std::string_view display_name() const noexcept override { return "Flywheel Brake Lag"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate baseline grip, but the flywheel coupling dominates.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -55.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.00f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 5);
    p.gravity_mul = 1.00f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 7);
    // Expensive, short-range lidar: the flywheel phase must be felt, not seen.
    p.lidar_energy_mul = 4.0f + 1.5f * biome_random01(s, 8);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 9);
    // Terrain: broad, smooth rolling plateaus with sparse craters and low steps,
    // so the flywheel is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.60f + 0.25f * biome_random01(s, 12);
    p.terrain_step_mul = 0.35f + 0.15f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Good grip, but the flywheel coupling changes effective traction drastically.
    return p.friction_mul * 1.00f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.30f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Flywheel brake lag: a massive internal flywheel stores kinetic energy
      // and releases it with a delay. When the rover accelerates, the flywheel
      // 'winds up' and resists; when it decelerates, the flywheel 'unwinds'
      // and pushes forward. This creates a hysteresis loop where braking is
      // punished by a delayed forward surge that can destabilize the rover.
      float windup = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Spatial phase: the flywheel coupling strength varies with position,
      // creating 'stiff' and 'loose' zones. In stiff zones, the flywheel is
      // heavily coupled and its lag is strong; in loose zones, it disengages
      // and the rover coasts freely.
      float phase = t * 0.005f + c.velocity.x * 0.014f;
      float coupling = 0.5f + 0.5f * std::sin(phase);

      // Delayed brake surge: when the rover slows down (speed_norm decreasing),
      // the flywheel releases stored energy as a forward impulse. This is
      // strongest when coupling is high and windup is high.
      float decel = std::max(0.0f, 0.5f - speed_norm);
      float surge = coupling * windup * decel * 0.15f * c.mass * c.gravity;

      // Windup drag: when accelerating, the flywheel resists, stronger in
      // stiff zones.
      float accel_drag = coupling * windup * (1.0f - decel) * 0.10f * c.mass * c.gravity;

      // Lateral 'wobble': the flywheel's gyroscopic effect creates a slow
      // sideways oscillation, stronger when windup is high and coupling is
      // strong.
      float wobble = std::sin(t * 0.023f + speed * 0.05f + windup * 2.0f) * coupling * 0.04f * c.mass * (0.5f + windup);

      // Damping is lower in loose zones (easy coasting), higher in stiff zones
      // (the flywheel adds resistance to everything).
      float damping = 0.04f + 0.08f * coupling + 0.02f * speed_norm;

      c.body_force->x -= accel_drag;
      c.body_force->x += surge + wobble;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: the flywheel's gyroscopic precession causes a slight lift
      // during windup, reducing normal force in stiff zones.
      c.body_force->y += coupling * windup * 0.02f * c.mass * c.gravity * std::sin(t * 0.019f + 0.3f);
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.03f * coupling);

      if (c.body_torque) {
        // The flywheel induces a yaw torque that grows with windup and coupling,
        // especially during deceleration when the surge is released.
        float torque = coupling * windup * (0.015f + 0.010f * decel) * std::sin(t * 0.031f + speed * 0.04f) * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (1.0f + 0.3f * coupling);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float windup = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);
      float phase = t * 0.005f + speed * 0.014f;
      float coupling = 0.5f + 0.5f * std::sin(phase);

      // Energy cost: peaks when fighting the flywheel (high windup, high
      // coupling), and when braking (the surge wastes energy). Steady, moderate
      // cruise in loose zones is the most efficient.
      float windup_cost = windup * coupling * 0.012f;
      float brake_cost = std::max(0.0f, 0.5f - speed_norm) * coupling * 0.008f;
      *c.energy_cost += (0.005f + windup_cost + brake_cost + speed * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, iron-rich rock with faint metallic sheen — suggests heavy machinery,
    // but not the flywheel pattern itself.
    v.ground = {58, 54, 52};  // dark grey-brown
    v.particles = {168, 160, 150};  // metallic dust
    v.liquid = {32, 30, 28};  // dark hollows
    v.sky = {142, 136, 128};  // hazy metallic overcast
    v.particle_rate = 5.0f;
    v.particle_lift = 0.6f;
    v.particle_spread = 0.7f;
    v.base_particles = 0;
    v.max_particles = 16;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.18f;  // very dark: lidar is expensive and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class HystereticRotorField final : public Biome {
 public:
  std::string_view id() const noexcept override { return "hysteretic_rotor_field"; }
  std::string_view display_name() const noexcept override { return "Hysteretic Rotor Field"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate traction, but a dense array of buried rotors creates the dominant hazard.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -55.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 5);
    p.solar_charge_rate = 0.30f + 0.20f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Lidar is expensive and short-range: the rotor phase must be inferred from motion.
    p.lidar_energy_mul = 3.00f + 1.00f * biome_random01(s, 9);
    p.lidar_range_mul = 0.20f + 0.10f * biome_random01(s, 10);
    // Terrain: broad, smooth rolling plains with sparse craters and almost no steps,
    // so the rotor field is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.55f + 0.15f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Good baseline grip; the rotors dominate via body effects.
    return p.friction_mul * 0.95f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the rotor forces are the main challenge.
      *c.wheel_force += c.contact->tangent * (-0.30f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Rotor phase: a slow, position-dependent cycle that determines the
      // rotor array's coupling strength. When the phase is 'high' (value near 1),
      // the rotors are strongly engaged and create strong, delayed forces that
      // resist acceleration but release momentum on deceleration. When 'low',
      // the rotors are disengaged and the rover coasts freely.
      float rotor_phase = t * 0.005f + c.velocity.x * 0.013f;
      float rotor_coupling = 0.5f + 0.5f * std::sin(rotor_phase);

      // Hysteresis memory: builds with speed but lags behind, so accelerating
      // into a rotor zone is punished more than steady cruising through it.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);

      // Delayed elastic recoil: the rotors resist acceleration but, once the
      // rover builds momentum, they release it as a forward assist. This
      // creates a hysteresis loop that rewards steady, moderate speed.
      float accel_drag = rotor_coupling * memory * 0.12f * c.mass * c.gravity;
      float release = rotor_coupling * memory * (0.02f + 0.04f * std::sin(rotor_phase + 0.8f)) * c.mass * c.gravity;

      // Lateral 'rotor wash': the spinning rotors push the rover sideways
      // with a delayed, memory-dependent oscillation. This is the signature
      // hazard — steady, moderate speed is safest.
      float wash = std::sin(t * 0.021f + speed * 0.05f + memory * 2.0f) * rotor_coupling * 0.045f * c.mass * (0.5f + memory);

      // Damping is lower when momentum is high (efficient cruising) and higher
      // when speed is low (the rotors grip harder).
      float damping = 0.05f + 0.08f * rotor_coupling * (1.0f - memory) + 0.02f * speed_norm;

      c.body_force->x -= accel_drag;
      c.body_force->x += release + wash;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: the rotors' downwash creates a slight lift that reduces
      // normal load when memory is high, decreasing traction and making
      // high-speed cornering riskier.
      c.body_force->y += rotor_coupling * memory * 0.025f * c.mass * c.gravity * std::sin(t * 0.019f + 0.3f);
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.03f * memory);

      if (c.body_torque) {
        // Rotor asymmetry induces a rocking torque that grows with memory and
        // coupling, strongest during deceleration when the release is strongest.
        float decel = std::max(0.0f, 0.5f - speed_norm);
        float torque = rotor_coupling * memory * (0.015f + 0.010f * decel) * std::sin(t * 0.031f + speed * 0.04f) * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (1.0f + 0.3f * rotor_coupling);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float rotor_phase = t * 0.005f + speed * 0.013f;
      float rotor_coupling = 0.5f + 0.5f * std::sin(rotor_phase);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);

      // Energy cost: peaks when fighting the rotors (high memory, high
      // coupling) and when braking (the release wastes energy). Steady,
      // moderate cruise in low-coupling zones is the most efficient.
      float rotor_cost = rotor_coupling * memory * 0.012f;
      float brake_cost = std::max(0.0f, 0.5f - speed_norm) * rotor_coupling * 0.008f;
      *c.energy_cost += (0.005f + rotor_cost + brake_cost + speed * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, wind-blown terrain with faint metallic sheen — suggests machinery
    // but not the rotor pattern itself.
    v.ground = {118, 112, 104};  // light grey-beige rock
    v.particles = {188, 182, 172};  // fine metallic dust
    v.liquid = {58, 54, 50};  // dark hollows
    v.sky = {172, 166, 158};  // hazy, dusty daylight
    v.particle_rate = 7.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 1.1f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 16;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.22f;  // dark: lidar is expensive and short-range, so the rotor phase must be inferred from motion
    v.liquid_surface = false;
    return v;
  }
};

class RotorRhythmField final : public Biome {
 public:
  std::string_view id() const noexcept override { return "rotor_rhythm_field"; }
  std::string_view display_name() const noexcept override { return "Rotor Rhythm Field"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip; the rotor array creates the real hazard.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -60.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 5);
    p.solar_charge_rate = 0.20f + 0.12f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Expensive, very short-range lidar: the rotor rhythm must be felt, not seen.
    p.lidar_energy_mul = 4.0f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 10);
    // Terrain: smooth rolling plains with gentle craters and almost no steps,
    // so the rotor hazard is the only real challenge.
    p.terrain_amplitude_mul = 1.40f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.15f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.80f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Good baseline grip; the rotor forces are the hazard, not the ground.
    return p.friction_mul * 0.95f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the real forces come from the rotor field.
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Rotor phase: a slow, position-dependent cycle that determines the
      // rotor array's coupling strength. When the phase is 'high' (value near 1),
      // the rotors are strongly engaged and create strong, delayed forces that
      // resist acceleration but release momentum on deceleration. When 'low',
      // the rotors are disengaged and the rover coasts freely.
      float rotor_phase = t * 0.005f + c.velocity.x * 0.013f;
      float rotor_coupling = 0.5f + 0.5f * std::sin(rotor_phase);

      // Hysteresis memory: builds with speed but lags behind, so accelerating
      // into a rotor zone is punished more than steady cruising through it.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);

      // Delayed elastic recoil: the rotors resist acceleration but, once the
      // rover builds momentum, they release it as a forward assist. This
      // creates a hysteresis loop that rewards steady, moderate speed.
      float accel_drag = rotor_coupling * memory * 0.12f * c.mass * c.gravity;
      float release = rotor_coupling * memory * (0.02f + 0.04f * std::sin(rotor_phase + 0.8f)) * c.mass * c.gravity;

      // Lateral 'rotor wash': the spinning rotors push the rover sideways
      // with a delayed, memory-dependent oscillation. This is the signature
      // hazard — steady, moderate speed is safest.
      float wash = std::sin(t * 0.021f + speed * 0.05f + memory * 2.0f) * rotor_coupling * 0.045f * c.mass * (0.5f + memory);

      // Damping is lower when momentum is high (efficient cruising) and higher
      // when speed is low (the rotors grip harder).
      float damping = 0.05f + 0.08f * rotor_coupling * (1.0f - memory) + 0.02f * speed_norm;

      c.body_force->x -= accel_drag;
      c.body_force->x += release + wash;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: the rotors' downwash creates a slight lift that reduces
      // normal load when memory is high, decreasing traction and making
      // high-speed cornering riskier.
      c.body_force->y += rotor_coupling * memory * 0.025f * c.mass * c.gravity * std::sin(t * 0.019f + 0.3f);
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.03f * memory);

      if (c.body_torque) {
        // Rotor asymmetry induces a rocking torque that grows with memory and
        // coupling, strongest during deceleration when the release is strongest.
        float decel = std::max(0.0f, 0.5f - speed_norm);
        float torque = rotor_coupling * memory * (0.015f + 0.010f * decel) * std::sin(t * 0.031f + speed * 0.04f) * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (1.0f + 0.3f * rotor_coupling);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float rotor_phase = t * 0.005f + speed * 0.013f;
      float rotor_coupling = 0.5f + 0.5f * std::sin(rotor_phase);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 6.0f);

      // Energy cost: peaks when fighting the rotors (high memory, high
      // coupling) and when braking (the release wastes energy). Steady,
      // moderate cruise in low-coupling zones is the most efficient.
      float rotor_cost = rotor_coupling * memory * 0.012f;
      float brake_cost = std::max(0.0f, 0.5f - speed_norm) * rotor_coupling * 0.008f;
      *c.energy_cost += (0.005f + rotor_cost + brake_cost + speed * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, wind-blown terrain with faint metallic sheen — suggests machinery
    // but not the rotor pattern itself.
    v.ground = {118, 112, 104};  // light grey-beige rock
    v.particles = {188, 182, 172};  // fine metallic dust
    v.liquid = {58, 54, 50};  // dark hollows
    v.sky = {172, 166, 158};  // hazy, dusty daylight
    v.particle_rate = 7.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 1.1f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 16;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.22f;  // dark: lidar is expensive and short-range, so the rotor phase must be inferred from motion
    v.liquid_surface = false;
    return v;
  }
};

class TidalBrakeVault final : public Biome {
 public:
  std::string_view id() const noexcept override { return "tidal_brake_vault"; }
  std::string_view display_name() const noexcept override { return "Tidal Brake Vault"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Baseline low gravity, but the tide modulates it far below/above.
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.60f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -75.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.80f + 0.25f * biome_random01(s, 5);
    // Solar is very weak: energy must be conserved for the heavy phase.
    p.solar_charge_rate = 0.05f + 0.04f * biome_random01(s, 6);
    p.gravity_mul = 0.55f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 8);
    // Lidar is cheap but very short-range: the tide phase must be inferred from motion.
    p.lidar_energy_mul = 0.15f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 10);
    // Terrain: high amplitude, high roughness, dense craters, moderate steps.
    // The terrain must be negotiated only during the light phase; the heavy phase is for braking.
    p.terrain_amplitude_mul = 1.60f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.50f + 0.50f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.80f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 1.40f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip; the tide changes effective normal load and thus traction.
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the real forces come from body effects.
      *c.wheel_force += c.contact->tangent * (-0.30f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.008f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);

      // Tidal gravity cycle: a slow, position-dependent sine that alternates between
      // a light phase (effective_g ~0.5x) and a heavy phase (effective_g ~1.5x).
      // The phase is completely hidden (lidar only shows terrain), so the agent must
      // learn to feel the change through suspension compression and wheel load.
      float phase = t * 0.0045f + c.velocity.x * 0.012f;
      float tide = 0.5f + 0.5f * std::sin(phase);  // 0 light, 1 heavy

      // Effective gravity multiplier: light phase is ~0.35, heavy phase is ~1.65.
      // This is a strong modulation that makes the rover nearly weightless in light
      // and crushingly heavy in heavy.
      float grav_mul = 0.35f + 1.30f * tide;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      // Weight force: in heavy phase, gravity pulls the rover down, increasing
      // traction but also increasing rolling resistance and making it hard to climb.
      // In light phase, buoyancy lifts the rover, reducing traction and making it
      // easy to accelerate but hard to brake.
      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.25f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.20f;

      // Lateral drift: during light phase, the rover drifts sideways (low traction);
      // during heavy phase, the rover is stable but sluggish. The drift is stronger
      // at speed and has a spatial-frequency component that the agent can learn.
      float lateral_drift = std::sin(phase + 1.3f) * (1.0f - tide) * 0.08f * c.mass * (0.5f + std::tanh(speed * 0.10f));

      // Damping: heavy phase has high damping (good grip), light phase has low
      // damping (easy to coast but hard to steer). This rewards timing acceleration
      // with the heavy phase and coasting through the light phase.
      float damping = 0.04f + 0.12f * tide + 0.02f * std::tanh(speed * 0.10f);

      c.body_force->x += lateral_drift - c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * tide);

      if (c.body_torque) {
        // The tide change induces a pitching torque: when transitioning from light
        // to heavy, the rover pitches forward (nose-down); when transitioning from
        // heavy to light, it pitches backward (nose-up). This can flip the rover if
        // it is moving too fast during the transition. The torque is strongest when
        // the tide is changing fastest (cos phase > 0).
        float transition = std::cos(phase);
        float torque = transition * (0.5f + 0.5f * std::tanh(speed * 0.10f)) * 0.03f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.0045f + speed * 0.012f;
      float tide = 0.5f + 0.5f * std::sin(phase);

      // Energy cost: heavy phase costs much more (fighting extra weight), light phase
      // is cheap. But accelerating during light phase wastes energy via wheel spin
      // (low traction), so the optimal strategy is to build momentum in heavy phase,
      // then coast through light phase.
      float heavy_cost = tide * 0.018f;
      float light_penalty = (1.0f - tide) * (0.004f + speed * 0.002f);
      *c.energy_cost += (0.010f + heavy_cost + light_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, wind-scoured rock with faint blue-grey tint — suggests thin atmosphere,
    // but the tide is completely invisible.
    v.ground = {112, 118, 124};  // light grey-blue
    v.particles = {190, 196, 202};  // fine pale dust
    v.liquid = {48, 56, 64};  // dark hollows
    v.sky = {145, 155, 165};  // hazy overcast
    v.particle_rate = 5.0f;
    v.particle_lift = 1.8f;  // low-gravity floating dust
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.10f;  // very dark: lidar is cheap but the tide is hidden, forcing inference from motion
    v.liquid_surface = false;
    return v;
  }
};

class LullAndThermalScrub final : public Biome {
 public:
  std::string_view id() const noexcept override { return "lull_thermal_scrub"; }
  std::string_view display_name() const noexcept override { return "Lull Thermal Scrub"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.006f + 0.014f * biome_random01(s, 1);
    p.viscosity = 1.0f + 1.2f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.30f * biome_random01(s, 3);
    p.wind_force = 1.0f + 2.0f * biome_random01(s, 4);
    p.ambient_temperature = 10.0f + 25.0f * biome_random01(s, 5);
    p.thermal_transfer = 2.0f + 0.8f * biome_random01(s, 6);
    p.solar_charge_rate = 0.05f + 0.04f * biome_random01(s, 7);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 8);
    p.crust_deform = 0.004f + 0.010f * biome_random01(s, 9);
    p.lidar_energy_mul = 0.70f + 0.30f * biome_random01(s, 10);
    p.lidar_range_mul = 0.30f + 0.15f * biome_random01(s, 11);
    // Calm, benign-looking terrain with wide basins and little vertical relief.
    // The real threat is the heat-scrub lateral force that appears only in \'lull\' windows.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.55f + 0.20f * biome_random01(s, 13);
    p.terrain_crater_mul = 1.40f + 0.40f * biome_random01(s, 14);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate baseline grip; the lateral scrub is the dominant hazard, not traction loss.
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Slight rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.25f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Base drain plus a small thermal load from the warm, damp ground.
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.004f * p.thermal_transfer;
      *c.energy_cost += (0.006f + heat_load + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Thermal state: warm mud generates both buoyant lift and a lateral scrub.
      // The scrub is entirely hidden: no visual cue, no lidar signature.
      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 15.0f) * 0.04f);

      // Lull phase: a slow, position-dependent oscillation. When \'lull\'
      // is high, the ground looks perfectly calm and flat, but that is exactly
      // when the thermal scrub is strongest. When \'lull\' is low, the ground
      // feels slightly rougher, but the lateral forces abate.
      float lull_phase = t * 0.0045f + c.velocity.x * 0.011f;
      float lull = 0.5f + 0.5f * std::sin(lull_phase);  // 1 = deceptive calm, 0 = real calm

      // A faster sub-oscillation creates brief \'scrub bursts\' inside the lull window.
      float burst_phase = t * 0.027f + c.velocity.x * 0.065f;
      float burst = 0.5f + 0.5f * std::sin(burst_phase);
      burst = burst * burst;  // narrow, strong peaks

      // Combined scrub intensity: highest during lull + burst windows.
      float scrub = lull * (0.45f + 0.85f * burst);

      // The scrub is a strong lateral force that grows with forward speed.
      // In real-calm windows the force vanishes, giving the rover a stable
      // recovery window. In deceptive lulls it can shove the rover off course
      // or roll it over if it is going too fast.
      float lateral_scrub = heat_factor * scrub * (0.12f + 0.22f * speed_norm) * c.mass * p.wind_force;

      // Thermal buoyancy: hot mud lifts the rover slightly, reducing normal
      // load and thus available grip exactly when the scrub shoves hardest.
      float lift = scrub * 0.025f * c.mass * c.gravity;

      // Damping is lower during deceptive lulls (the rover seems to glide
      // easily), which makes the surprise shove more dangerous. Real-calm
      // windows have higher damping, which helps the rover stabilise.
      float damping = 0.04f + 0.045f * (1.0f - lull) + 0.02f * speed_norm;

      c.body_force->x += lateral_scrub - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.02f * scrub);

      if (c.body_torque) {
        // The scrub creates a yaw torque that tries to spin the rover,
        // stronger during bursts and at higher speed.
        float torque = heat_factor * scrub * std::sin(burst_phase + 0.9f) * 0.016f * c.mass * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 15.0f) * 0.04f);

      float lull_phase = t * 0.0045f + speed * 0.011f;
      float lull = 0.5f + 0.5f * std::sin(lull_phase);

      float burst_phase = t * 0.027f + speed * 0.065f;
      float burst = 0.5f + 0.5f * std::sin(burst_phase);
      burst = burst * burst;

      float scrub = lull * (0.45f + 0.85f * burst);

      // Energy drain scales with scrub intensity and speed, punishing
      // throttle during deceptive lull windows. The efficient strategy
      // is to cruise gently through real-calm windows and brake/steer
      // through the deceptive lulls, letting momentum carry the rover.
      float scrub_drain = scrub * heat_factor * 0.012f * p.energy_drain_mul;
      float speed_drain = speed * 0.002f * (1.0f + scrub);
      *c.energy_cost += (0.004f + scrub_drain + speed_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Warm, hazy mudflat with faint heat shimmer. The deceptive lull is
    // completely invisible; the scene looks uniformly calm.
    v.ground = {96, 82, 58};  // pale warm mud
    v.particles = {210, 190, 140};  // fine heat-haze dust
    v.liquid = {46, 38, 24};  // dark thermal pools
    v.sky = {170, 140, 90};  // hazy amber daylight
    v.particle_rate = 8.0f;
    v.particle_lift = 1.1f;
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.15f;  // dark: lidar is moderately costly and short-range, so the lull must be inferred
    v.liquid_surface = false;
    return v;
  }
};

class GravityWellBrine final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gravity_well_brine"; }
  std::string_view display_name() const noexcept override { return "Gravity-Well Brine"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip in liquid, but the gravity wells dominate.
    p.friction_mul = 0.35f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.30f + 0.50f * biome_random01(s, 2);
    p.energy_drain_mul = 1.70f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    // Very cold brine: thermal transfer is high, heat drains rapidly.
    p.ambient_temperature = -85.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.8f + 0.8f * biome_random01(s, 5);
    // Solar is weak in the dark brine; energy scarcity forces conservation.
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 8);
    // Lidar is cheap but very short: the wells must be felt, not seen.
    p.lidar_energy_mul = 0.25f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, deep basins with high amplitude and roughness, dense craters, low steps.
    // The gravity wells are the hazard, not the geometry.
    p.terrain_amplitude_mul = 1.60f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.50f + 0.40f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.70f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.70f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the cold brine, more in deeper immersion.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.0f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      // Viscous drag plus depth-induced normal resistance.
      float depth = c.contact->penetration * 20.0f;
      float drag = (0.20f + p.viscosity * 1.5f * (1.0f + depth) * c.immersion + 0.12f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.10f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      // Energy cost is higher in water (fighting drag) and at speed.
      *c.energy_cost += (0.008f + c.immersion * depth * 0.12f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Gravity-well phase: a slow, position-dependent oscillation that creates
      // alternating 'well' (strong downward pull, high gravity, high drag) and
      // 'rise' (buoyant, low gravity, low drag) zones. The phase is invisible
      // to lidar (very short, cheap but useless) and must be inferred from
      // suspension compression, wheel load, and the rover's vertical motion.
      float well_phase = t * 0.0045f + c.velocity.x * 0.012f;
      float well = 0.5f + 0.5f * std::sin(well_phase); // 1 = well, 0 = rise

      // A faster sub-oscillation creates 'probes' of extreme gravity within the
      // well: brief moments of crushing weight that make forward progress extremely
      // difficult but also grant a strong downward force that improves traction.
      // The agent must learn to brace and push through these probes.
      float probe_phase = t * 0.027f + c.velocity.x * 0.065f;
      float probe = 0.5f + 0.5f * std::sin(probe_phase);
      probe = probe * probe; // narrow, strong peaks

      // Effective gravity multiplier: rise phase ~0.65x, well phase ~1.35x,
      // with probes pushing to ~1.8x. This is the core hazard: the rover has
      // almost no weight in the rise (slides easily) but heavy weight in the
      // well (hard to accelerate).
      float grav_mul = 0.65f + 0.70f * well + 0.45f * probe * well;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      // Weight force: well phase presses down, increasing normal load and
      // traction, but also increasing rolling resistance. Rise phase lifts
      // the rover, reducing traction and making it easier to coast.
      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.25f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.20f;

      // Lateral 'well drift': the gravity vector tilts slightly with the well,
      // creating a sideways force that grows with speed. During the rise phase
      // this is gentle; during the well it can shove the rover off course.
      float lateral_drift = std::sin(well_phase + 1.1f) * (0.5f + 0.5f * well) * 0.055f * c.mass * (0.6f + 0.4f * speed_norm);

      // Damping is higher during the well (more grip, better control), lower
      // during the rise (easy to coast, but harder to steer).
      float damping = 0.05f + 0.09f * well + 0.02f * speed_norm;

      // Probe instability: during extreme gravity probes, the rover pitches
      // and can flip if it is moving too fast. This is the signature hazard
      // - a naive 'drive carefully' policy that just slows down will still
      // get caught.
      float probe_torque = probe * well * (0.5f + 0.5f * speed_norm) * 0.025f * c.mass * c.gravity * std::sin(probe_phase + 0.6f);

      c.body_force->x += lateral_drift - c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * well);

      if (c.body_torque) {
        *c.body_torque += probe_torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Same phase but computed using speed for the spatial component.
      float well_phase = t * 0.0045f + speed * 0.012f;
      float well = 0.5f + 0.5f * std::sin(well_phase);
      float probe_phase = t * 0.027f + speed * 0.065f;
      float probe = 0.5f + 0.5f * std::sin(probe_phase);
      probe = probe * probe;
      float grav_mul = 0.65f + 0.70f * well + 0.45f * probe * well;

      // Energy harvest: very weak, essentially scarce. There is a tiny trickle
      // in the rise phase (solar penetrates the brine), but the well phase
      // consumes heavily through drag and fighting the extra weight.
      float charge_gain = (1.0f - well) * p.solar_charge_rate * 0.10f;

      // Drain: well phase is expensive (fighting gravity), rise phase is cheap.
      // Probes cost extra, punishing throttle during extreme gravity.
      float well_cost = well * (0.020f + 0.010f * speed_norm) * p.energy_drain_mul;
      float rise_cost = (1.0f - well) * (0.004f + 0.002f * speed_norm) * p.energy_drain_mul;
      float probe_penalty = probe * well * 0.015f * p.energy_drain_mul;

      // Net: negative everywhere, but much worse in the well. The agent must
      // learn to coast through the rise and carefully balance throttle to
      // minimize drain, or risk running out of energy entirely.
      *c.energy_cost += (well_cost + rise_cost + probe_penalty - charge_gain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, icy brine with faint blue-green glow - suggests cold liquid
    // but not the gravity-well pattern.
    v.ground = {38, 50, 58};  // dark teal mud
    v.particles = {110, 180, 190};  // pale cyan spray
    v.liquid = {30, 80, 95};  // dark blue-green brine
    v.sky = {60, 90, 105};  // dim overcast
    v.particle_rate = 9.0f;
    v.particle_lift = 1.3f;
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.10f;  // very dark: lidar is cheap but short, so the wells must be inferred from motion
    v.liquid_surface = true;
    return v;
  }
};

class FrictionMirageBelt final : public Biome {
 public:
  std::string_view id() const noexcept override { return "friction_mirage_belt"; }
  std::string_view display_name() const noexcept override { return "Friction Mirage Belt"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate baseline grip, but the mirage inversion is the real challenge.
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.010f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -20.0f + 15.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 4);
    p.solar_charge_rate = 0.06f + 0.04f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.006f + 0.014f * biome_random01(s, 7);
    // Lidar is cheap but extremely short-ranged and has almost no energy cost:
    // it reveals nothing about the mirage, only confirms uniform flat terrain.
    p.lidar_energy_mul = 0.10f + 0.05f * biome_random01(s, 8);
    p.lidar_range_mul = 0.10f + 0.05f * biome_random01(s, 9);
    // Terrain: extremely smooth, no craters/steps, very low amplitude — a flat
    // uniform 'mirage' that gives no geometric hint of the traction inversion.
    // A generic terrain-avoidance policy has nothing to avoid and will be caught.
    p.terrain_amplitude_mul = 0.6f + 0.2f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.3f + 0.1f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.3f + 0.1f * biome_random01(s, 12);
    p.terrain_step_mul = 0.2f + 0.1f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Nominal grip is decent; the mirage imposes its own traction law via body effects.
    return p.friction_mul * 0.80f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very light sink into the deceptive surface, not a mud/sand trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.1f;
    }
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the mirage inversion lives in body effects.
      *c.wheel_force += c.contact->tangent * (-0.25f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.012f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.0015f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Mirage phase: a slow, position-dependent oscillation that flips between
      // a 'stable' state (grip responds positively to throttle) and an 'inverted'
      // state (grip responds negatively to throttle). The phase is completely
      // invisible: the terrain is perfectly flat and uniform, lidar is so short
      // and cheap that it confirms only flatness, and the visual is a single dull
      // mirage-like tint. Only the rover's own motion and energy reveal which
      // state it is in, forcing the agent to actively probe and adapt its throttle.
      float mirage_phase = t * 0.0048f + c.velocity.x * 0.012f;
      float mirage = 0.5f + 0.5f * std::sin(mirage_phase); // 1 = inverted, 0 = stable

      // A fast sub-oscillation creates 'mirage flickers': brief, sharp grip changes
      // that are stronger in the inverted state. These cannot be seen ahead;
      // the rover must learn to ease off before them.
      float flicker_phase = t * 0.033f + c.velocity.x * 0.077f;
      float flicker = 0.5f + 0.5f * std::sin(flicker_phase);
      float flicker_narrow = flicker * flicker; // narrow, strong dips

      // Effective traction multiplier. In the inverted state:
      //   0.42 - 0.34*throttle_norm  (throttle hurts)
      // In the stable state:
      //   0.78 + 0.18*throttle_norm  (throttle helps)
      // The rover cannot directly observe throttle here, so we use speed_norm
      // as a proxy for throttle demand. This makes the inversion feel like a
      // speed-dependent grip loss: going faster in the inverted state makes the
      // ground effectively more slippery.
      float throttle_norm = speed_norm;
      float inverted_traction = mirage * (0.42f - 0.34f * throttle_norm);
      float stable_traction = (1.0f - mirage) * (0.78f + 0.18f * throttle_norm);
      float traction = inverted_traction + stable_traction;
      traction *= (1.0f - 0.50f * mirage * flicker_narrow); // mirage flicker dips
      traction = std::max(0.02f, traction);

      // The traction multiplier directly modulates the force available from the ground.
      // Rather than modify wheel_force (already applied), we add a corrective body force
      // that mimics the grip change: in the inverted phase, extra throttle (speed) causes
      // a backward drag and a lateral shove; in the stable phase, it gives a small assist.
      float grip_correction = traction - 0.82f; // relative to baseline friction_scale
      float forward_correction = grip_correction * 0.22f * c.mass * c.gravity * (0.4f + speed_norm);
      float lateral_correction = grip_correction * std::sin(t * 0.041f + speed * 0.055f) * 0.06f * c.mass * (0.5f + speed_norm);

      // Damping reflects the effective grip: lower in the inverted/low-traction state
      // (the rover slides more), higher in the stable/high-traction state.
      float damping = 0.04f + 0.16f * traction;

      // Vertical: in the inverted phase, the mirage 'lifts' slightly under load,
      // reducing normal force and thus further reducing available traction.
      // In the stable phase, it presses down slightly for more grip.
      float lift = mirage * 0.030f * traction * c.mass * c.gravity * std::sin(flicker_phase + 0.5f);

      c.body_force->x += forward_correction + lateral_correction - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.03f * traction);

      if (c.body_torque) {
        // Mirage asymmetry induces a pitching torque, strongest during inverted-phase
        // flickers and at higher speed. This is the flip hazard: a policy that simply
        // maintains constant speed through the inverted phase will be destabilised.
        float torque = mirage * flicker_narrow * (0.5f + 0.70f * speed_norm) * 0.032f * c.mass * c.gravity * std::sin(flicker_phase + 1.1f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.5f * traction);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float mirage_phase = t * 0.0048f + speed * 0.012f;
      float mirage = 0.5f + 0.5f * std::sin(mirage_phase);
      float flicker_phase = t * 0.033f + speed * 0.077f;
      float flicker = 0.5f + 0.5f * std::sin(flicker_phase);
      float flicker_narrow = flicker * flicker;

      // Energy budget: the inverted phase punishes throttle with a strong energy drain
      // that grows with speed; the stable phase is cheap and rewards steady cruising.
      // This forces the agent to learn to coast through the inverted phase and push
      // only in the stable phase — a completely different policy from sand/mud,
      // where slow crawling is always best, and from the normal biome where constant
      // moderate throttle is efficient.
      float inverted_drain = mirage * (0.011f + 0.030f * speed_norm) * p.energy_drain_mul;
      float stable_drain = (1.0f - mirage) * (0.004f + 0.004f * speed_norm) * p.energy_drain_mul;
      float flicker_penalty = mirage * flicker_narrow * (0.009f + 0.011f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: the cold mirage haze saps battery in the inverted phase.
      float cold_drain = p.thermal_transfer * 0.002f * (1.0f + mirage * 0.8f);

      *c.energy_cost += (inverted_drain + stable_drain + flicker_penalty + cold_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, uniform, mirage-like hazy tan with faint shimmer. No visual cue
    // reveals the polarity phase. Screen brightness is very low and lidar is
    // nearly blind/costly, so the hazard must be inferred entirely from the rover's
    // own motion and energy drain.
    v.ground = {14, 16, 20};        // near-black mesh
    v.particles = {80, 86, 100};     // faint shimmer dust
    v.liquid = {10, 12, 16};         // darker melt pools (rare)
    v.sky = {20, 22, 30};            // near-black hazy sky
    v.particle_rate = 2.0f;
    v.particle_lift = 0.3f;
    v.particle_spread = 0.4f;
    v.base_particles = 0;
    v.max_particles = 10;
    v.particle_size = 1;
    v.ambient_particles = 3;
    v.ambient_drift = 0.6f;
    v.screen_brightness = 0.05f;     // near-total darkness: lidar is very costly and nearly blind
    v.liquid_surface = false;
    return v;
  }
};

class TideSluiceReservoir final : public Biome {
 public:
  std::string_view id() const noexcept override { return "tide_sluice_reservoir"; }
  std::string_view display_name() const noexcept override { return "Tide Sluice Reservoir"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Slippery, dense liquid with strong coupling to gravity and drivetrain inertia.
    p.friction_mul = 0.25f + 0.15f * biome_random01(s);
    p.sink_rate = 0.006f + 0.014f * biome_random01(s, 1);
    p.viscosity = 1.8f + 2.4f * biome_random01(s, 2);
    p.energy_drain_mul = 1.8f + 0.6f * biome_random01(s, 3);
    p.wind_force = 0.6f + 1.2f * biome_random01(s, 4);
    // Very cold brine: thermal transfer high, but energy drain is the real pressure.
    p.ambient_temperature = -90.0f + 20.0f * biome_random01(s, 5);
    p.thermal_transfer = 3.0f + 0.8f * biome_random01(s, 6);
    // Near-total darkness: almost no solar, forcing strict battery budgeting.
    p.solar_charge_rate = 0.02f + 0.02f * biome_random01(s, 7);
    // Gravity oscillates broadly; the tidal phase must be inferred from load and response.
    p.gravity_mul = 0.55f + 0.25f * biome_random01(s, 8);
    p.crust_deform = 0.004f + 0.010f * biome_random01(s, 9);
    // Lidar is very expensive and nearly blind: the tide is invisible ahead.
    p.lidar_energy_mul = 5.5f + 1.0f * biome_random01(s, 10);
    p.lidar_range_mul = 0.10f + 0.05f * biome_random01(s, 11);
    // Reshape terrain into sluice-like basins: gentle rolling with moderate craters and low steps.
    // A smooth low-gravity policy cannot coast through; the tide + inertia coupling dominates.
    p.terrain_amplitude_mul = 1.50f + 0.40f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.70f + 0.25f * biome_random01(s, 13);
    p.terrain_crater_mul = 1.60f + 0.40f * biome_random01(s, 14);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.40f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sinking into the dense brine, more with speed and immersion.
      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.5f * std::tanh(speed * 0.2f) + 0.5f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 25.0f;
      // Strong viscous drag plus depth-induced normal resistance.
      float drag = (0.35f + p.viscosity * 2.6f * (1.0f + depth) * c.immersion + 0.18f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.04f + 0.12f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 25.0f;
      *c.energy_cost += (0.012f + c.immersion * depth * 0.15f + std::abs(c.wheel_speed) * 0.006f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Tide phase: a slow, position-dependent oscillation that alternates between
      // 'high' (effective gravity ~1.8x, crushing weight, but strong grip) and
      // 'low' (effective gravity ~0.35x, buoyant, but low traction). The phase is
      // completely invisible: lidar is nearly blind (very expensive, extremely short),
      // the visual is uniformly dark, and the rover must infer the tide from its own
      // acceleration response and suspension compression.
      float tide_phase = t * 0.0042f + c.velocity.x * 0.011f;
      float tide = 0.5f + 0.5f * std::sin(tide_phase); // 1 = high, 0 = low

      // A faster sub-oscillation creates 'sluice pulses': brief, extreme gravity spikes
      // within the high phase. These are the primary flip/stall hazard. A naive policy
      // that simply drives forward carefully will hit a pulse while heavy and either
      // pitch over or drain its battery catastrophically.
      float pulse_phase = t * 0.025f + c.velocity.x * 0.061f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse; // narrow, strong peaks

      // Effective gravity multiplier: low ~0.35, high ~1.45, pulses push to ~1.9.
      float grav_mul = 0.35f + 1.10f * tide + 0.45f * tide * pulse_narrow;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      // Core inertia hysteresis: 'sluice memory' builds with speed but lags behind.
      // In the high-tide phase, the fluid resists acceleration strongly (lock_drag),
      // but once momentum is high, it releases a delayed forward surge (release).
      // In the low phase, the fluid is free-flowing and gives a small constant assist.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Locked-phase drag: strong when accelerating from low speed in high tide,
      // weaker once momentum is built. This makes constant re-acceleration painful.
      float lock_drag = tide * memory * (0.14f + 0.10f * (1.0f - memory)) * c.mass * c.gravity * 0.55f;

      // Delayed elastic release: once the rover has enough momentum, the locked fluid
      // suddenly gives way, pushing it forward efficiently through the rest of the
      // high phase. The release magnitude oscillates slightly, so timing matters.
      float release = tide * memory * (0.045f + 0.060f * std::sin(tide_phase + 1.0f)) * c.mass * c.gravity * 0.5f;

      // Unlocked-phase assist: free-flowing fluid gives a smooth constant forward push,
      // making it easy to build momentum here.
      float free_assist = (1.0f - tide) * 0.028f * c.mass * c.gravity * 0.5f;

      // Lateral 'sluice drift': the fluid swirls with a delayed, memory-dependent
      // oscillation, stronger at speed and in the high phase. This is the signature
      // hazard — steady, moderate speed is safest.
      float lateral_drift = std::sin(t * 0.022f + speed * 0.048f + memory * 2.0f) * (0.4f + 0.6f * tide) * 0.05f * c.mass * (0.5f + memory);

      // Weight force in high phase: presses the rover down, increasing normal load
      // and traction but also rolling resistance. In low phase, buoyancy lifts the
      // rover, reducing traction and making it easy to skate but hard to brake.
      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.25f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.20f;

      // Damping is lower when momentum is high (smooth gliding) and higher when speed
      // is low (the fluid grips harder, punishing stalls). Low tide also damps less.
      float damping = 0.035f + 0.085f * tide * (1.0f - memory) + 0.02f * speed_norm;

      // Sluice pulse instability: during extreme gravity spikes, a strong pitching
      // torque slams the nose down or up depending on the phase. A universal 'drive
      // carefully' policy that does not actively brake before pulses will be caught.
      float pulse_torque = tide * pulse_narrow * (0.5f + 0.7f * speed_norm) * 0.035f * c.mass * c.gravity * std::sin(pulse_phase + 0.7f);

      c.body_force->x -= lock_drag;
      c.body_force->x += release + free_assist + lateral_drift;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * tide);

      if (c.body_torque) {
        *c.body_torque += pulse_torque;
        *c.body_torque -= c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * tide);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      float tide_phase = t * 0.0042f + speed * 0.011f;
      float tide = 0.5f + 0.5f * std::sin(tide_phase);
      float pulse_phase = t * 0.025f + speed * 0.061f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Energy budget cascade: the drain is multiplicatively scaled by the tide phase.
      // High tide consumes energy roughly four times faster than low tide. On top of
      // that, a 'sluice surge' cost spikes during the rapid gravity pulses, punishing
      // high throttle at exactly the moment the torque is most dangerous.
      //
      // The successful strategy must learn to build momentum in the low phase (cheap,
      // but low traction so coasting is safer), then ride that momentum through the
      // high phase with only gentle throttle corrections, and brake carefully before
      // each pulse to avoid the pitching torque.
      float lock_cost = tide * memory * 0.020f;
      float start_cost = tide * (1.0f - memory) * (0.012f + 0.008f * speed_norm);
      float pulse_penalty = tide * pulse_narrow * (0.010f + 0.016f * speed_norm);
      float cruise_cost = speed * 0.0025f * (1.0f + 0.4f * tide);
      float free_bonus = (1.0f - tide) * (0.004f + 0.002f * speed_norm);

      *c.energy_cost += (lock_cost + start_cost + pulse_penalty + cruise_cost - free_bonus) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, near-black lagoon water with faint blue-grey sheen — visually ambiguous,
    // reveals nothing about the tide phase or the sluice pulses. Very dark to force inference.
    v.ground = {14, 20, 26};
    v.particles = {78, 108, 138};
    v.liquid = {10, 26, 40};
    v.sky = {20, 30, 40};
    v.particle_rate = 4.0f;
    v.particle_lift = 1.3f;
    v.particle_spread = 0.8f;
    v.base_particles = 2;
    v.max_particles = 20;
    v.particle_size = 2;
    v.ambient_particles = 6;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.07f; // near-total darkness: lidar is extremely expensive and nearly blind
    v.liquid_surface = true;
    return v;
  }
};

class GritShiftLode final : public Biome {
 public:
  std::string_view id() const noexcept override { return "grit_shift_lode"; }
  std::string_view display_name() const noexcept override { return "Grit Shift Lode"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate nominal grip, but the grit bed is unstable and speed-dependent.
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.012f + 0.020f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.35f + 0.45f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -50.0f + 25.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.65f + 0.25f * biome_random01(s, 4);
    // Solar is weak: recharging is a trap, forcing frugal driving and momentum.
    p.solar_charge_rate = 0.15f + 0.10f * biome_random01(s, 5);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.010f + 0.020f * biome_random01(s, 7);
    // Expensive, very short-range lidar: the bed state must be inferred from wheel response.
    p.lidar_energy_mul = 4.50f + 1.50f * biome_random01(s, 8);
    p.lidar_range_mul = 0.12f + 0.05f * biome_random01(s, 9);
    // Terrain: moderate amplitude, high roughness (loose, shifting grit), moderate craters, low steps.
    // This is not a flat sandbox; the bed is uneven and the grip shifts with speed.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.30f + 0.40f * biome_random01(s, 12);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Nominal, but the bed's grip law is imposed in body effects.
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Grit penetrates with speed but saturates; deeper penetration is sticky.
      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::tanh(speed * 0.15f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 30.0f;
      float speed = std::abs(c.wheel_speed);
      // Drag rises with depth and speed, but the critical traction law is in body effects.
      *c.wheel_force += c.contact->tangent * (-(0.25f + 0.18f * depth * speed));
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.02f + 0.08f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      *c.energy_cost += (0.010f + depth * 0.20f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Grit shift phase: a slow, position-dependent oscillation that flips between
      // 'compact' (stable, high grip) and 'shift' (loose, low grip, speed-dependent).
      // The phase is invisible: lidar is extremely short and costly, and the visual
      // is uniform dark grit. Only the rover's own slip and acceleration reveal it.
      float phase = t * 0.0052f + c.velocity.x * 0.013f;
      float shift = 0.5f + 0.5f * std::sin(phase);  // 1 = loose, 0 = compact

      // A faster sub-oscillation creates 'grit pulses': brief, sharp grip drops
      // within the loose phase. These cannot be seen ahead; the rover must learn
      // to brake or coast through them.
      float pulse_phase = t * 0.031f + c.velocity.x * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong dips

      // Core traction law: in the compact phase, grip rises gently with speed.
      // In the loose phase, grip collapses with speed (grains fluidize), with a
      // narrow 'bite' band at low speed where grains interlock. This forces a
      // completely different policy from sand or mud: slow crawling is bad, high
      // speed is deadly, but a specific moderate 'bite' speed is optimal.
      float compact_grip = (1.0f - shift) * (0.80f + 0.15f * speed_norm);
      float bite = 0.65f + 0.35f * std::exp(-(speed - 1.8f) * (speed - 1.8f) * 0.6f);
      float fluidize = 1.0f / (1.0f + 0.55f * speed * speed);
      float loose_grip = shift * bite * (0.45f + 0.35f * fluidize);
      float grip = compact_grip + loose_grip;
      // Grit pulses dip grip further.
      grip *= (1.0f - 0.55f * shift * pulse_narrow);
      grip = std::max(0.05f, grip);

      // This grip directly modulates the ground's ability to produce forward force.
      // Instead of modifying wheel_force (already applied), we add a corrective body
      // force: when grip is high, we get a forward assist; when grip is low, we get
      // a backward drag. The exact magnitude is scaled to make the difference matter.
      float grip_correction = grip - 0.85f;  // relative to baseline friction_scale
      float forward_correction = grip_correction * 0.30f * c.mass * c.gravity * (0.4f + speed_norm);

      // Lateral instability: in loose phase, especially during pulses, the bed
      // shoves sideways with a force that grows with speed. This is the flip hazard.
      float lateral_force = shift * pulse_narrow * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.1f);

      // Damping reflects grip: low grip means less damping (rover slides), high
      // grip means more solid feel.
      float damping = 0.04f + 0.14f * grip;

      // Vertical: in the loose phase, the bed 'swallows' the rover slightly,
      // increasing downward force at speed (worsening sink), but also providing
      // a small 'bite' lift at the optimal speed (rewarding the right throttle).
      float sink_force = shift * (0.05f + 0.04f * speed_norm) * c.mass * c.gravity;
      float bite_lift = shift * grip * 0.01f * c.mass * c.gravity * (1.0f - std::abs(speed - 1.8f) * 0.4f);

      c.body_force->x += forward_correction + lateral_force;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y -= sink_force - bite_lift - c.velocity.y * c.mass * (0.04f + 0.03f * shift);

      if (c.body_torque) {
        // Grit pulses cause asymmetric sink, creating a rocking torque that can
        // flip the rover if it's moving fast through a pulse. The torque is
        // strongest during loose-phase pulses and at speed.
        float torque = shift * pulse_narrow * (0.5f + 0.6f * speed_norm) * 0.030f * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * grip);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float phase = t * 0.0052f + speed * 0.013f;
      float shift = 0.5f + 0.5f * std::sin(phase);
      float pulse_phase = t * 0.031f + speed * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: compact phase is cheap and rewards steady cruising. Loose
      // phase scales cost with speed, but there's a shallow minimum near the
      // 'bite' speed. Pulses add a sharp penalty. This forces the agent to find
      // and maintain the optimal 'bite' speed in loose zones, coast through pulses,
      // and push efficiently in compact zones.
      float compact_cost = (1.0f - shift) * (0.004f + 0.004f * speed_norm) * p.energy_drain_mul;
      float loose_cost = shift * (0.008f + 0.030f * speed * speed * 0.1f) * p.energy_drain_mul;
      float bite_bonus = shift * std::exp(-(speed - 1.8f) * (speed - 1.8f) * 0.6f) * 0.008f * p.energy_drain_mul;
      float pulse_penalty = shift * pulse_narrow * (0.010f + 0.015f * speed_norm) * p.energy_drain_mul;
      float base_cost = 0.005f;

      *c.energy_cost += (base_cost + compact_cost + loose_cost - bite_bonus + pulse_penalty) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, uniform grit with faint brownish shimmer. No visual cue reveals the
    // shift phase. Screen brightness is very low and lidar is nearly blind, so the
    // hazard must be inferred from wheel slip and energy drain.
    v.ground = {30, 24, 18};        // dark grit
    v.particles = {110, 90, 60};     // dusty spray
    v.liquid = {18, 14, 10};         // dark hollows
    v.sky = {45, 38, 30};            // dim, hazy sky
    v.particle_rate = 8.0f;
    v.particle_lift = 0.6f;
    v.particle_spread = 1.0f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.10f;  // very dark: lidar is extremely expensive and short
    v.liquid_surface = false;
    return v;
  }
};

class AscendantSootBank final : public Biome {
 public:
  std::string_view id() const noexcept override { return "ascendant_soot_bank"; }
  std::string_view display_name() const noexcept override { return "Ascendant Soot Bank"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate baseline grip; the soot/ash coupling dominates.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.10f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 3);
    // Warm but not scorching; thermal transfer is low so heat buildup is slow.
    p.ambient_temperature = -10.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.35f + 0.15f * biome_random01(s, 5);
    // Solar is weak in the soot haze, so recharging is a trap.
    p.solar_charge_rate = 0.05f + 0.04f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.010f * biome_random01(s, 8);
    // Lidar is expensive and very short-range: the soot plume must be felt, not seen.
    p.lidar_energy_mul = 4.50f + 1.50f * biome_random01(s, 9);
    p.lidar_range_mul = 0.12f + 0.06f * biome_random01(s, 10);
    // Terrain: broad, gently sloped ash fields with moderate craters and low steps.
    // The hazard is the soot plume, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.70f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Baseline grip is decent; the soot plume changes effective traction via body effects.
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the loose ash bed; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.4f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the soot forces dominate in body effects.
      *c.wheel_force += c.contact->tangent * (-0.30f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Base drain plus a small ash-friction load.
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Soot plume phase: a slow, position-dependent oscillation that determines
      // whether the rover is in a 'lift' zone (rising soot, strong upward force,
      // low traction, high energy drain) or a 'settle' zone (settled ash, stable,
      // efficient cruising). The phase is completely invisible: lidar is very
      // short-range and expensive, the visual is uniformly dark ash, and only the
      // rover's own vertical motion and energy drain reveal which zone it is in.
      float lift_phase = t * 0.0052f + c.velocity.x * 0.013f;
      float lift_zone = 0.5f + 0.5f * std::sin(lift_phase);  // 1 = lift, 0 = settle

      // A faster sub-oscillation creates 'plume pulses': brief, intense upward
      // surges within the lift zone. These are the primary flip/stall hazard.
      // A naive policy that simply drives forward carefully will be caught by a
      // pulse while the rover is already light and unstable.
      float pulse_phase = t * 0.029f + c.velocity.x * 0.071f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong peaks

      // Effective upward force from the soot plume. In the lift zone, the plume
      // lifts the rover, reducing normal load and thus traction. Pulses create
      // brief, strong lift spikes that can destabilise the rover if it is moving
      // fast. In the settle zone, the ash is compact and gives a small downward
      // assist, improving traction.
      float lift_force = lift_zone * (0.16f + 0.28f * pulse_narrow) * c.mass * c.gravity;
      float settle_force = (1.0f - lift_zone) * 0.02f * c.mass * c.gravity;

      // Lateral drift: in the lift zone, the rover drifts sideways (low normal
      // load means little grip), stronger at speed and during pulses. In the
      // settle zone, the ash gives a slight forward assist (settling compaction).
      float lateral_drift = lift_zone * pulse_narrow * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.2f);
      float settle_assist = (1.0f - lift_zone) * 0.015f * c.mass * c.gravity;

      // Damping is lower in the lift zone (less grip, easier to coast but harder
      // to steer), higher in the settle zone (good grip, efficient cruising).
      float damping = 0.04f + 0.10f * (1.0f - lift_zone) + 0.02f * speed_norm;

      // Vertical: the plume lift reduces normal load, making the rover skittish;
      // the settle force adds weight, improving grip.
      c.body_force->x += lateral_drift + settle_assist - c.velocity.x * c.mass * damping;
      c.body_force->y -= lift_force - settle_force - c.velocity.y * c.mass * (0.04f + 0.02f * (1.0f - lift_zone));

      if (c.body_torque) {
        // Plume pulses induce a rocking torque, strongest during lift-zone pulses
        // and at higher speed. This is the flip hazard: a policy that maintains
        // constant speed through the lift zone will be destabilised.
        float torque = lift_zone * pulse_narrow * (0.5f + 0.7f * speed_norm) * 0.030f * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * (1.0f - lift_zone));
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Same phase but computed using speed for the spatial component.
      float lift_phase = t * 0.0052f + speed * 0.013f;
      float lift_zone = 0.5f + 0.5f * std::sin(lift_phase);
      float pulse_phase = t * 0.029f + speed * 0.071f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget cascade: the drain scales multiplicatively with the lift zone.
      // In the lift zone, fighting the plume and maintaining traction costs much
      // more than in the settle zone. Plume pulses add a sharp penalty, punishing
      // high throttle at exactly the moment the torque is most dangerous.
      //
      // The successful strategy must learn to build momentum in the settle zone
      // (cheap, high grip), then coast through the lift zone with only gentle
      // throttle corrections, and brake carefully before each pulse to avoid the
      // rocking torque. Recharging is a trap: solar is nearly zero in the dark ash.
      float lift_cost = lift_zone * (0.012f + 0.022f * speed_norm) * p.energy_drain_mul;
      float settle_cost = (1.0f - lift_zone) * (0.004f + 0.003f * speed_norm) * p.energy_drain_mul;
      float pulse_penalty = lift_zone * pulse_narrow * (0.010f + 0.018f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: the warm soot gently saps battery, slightly more in the lift zone.
      float thermal_drain = p.thermal_transfer * 0.002f * (1.0f + lift_zone * 0.6f);

      *c.energy_cost += (lift_cost + settle_cost + pulse_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, dark soot-black terrain with faint ember glow particles. No visual
    // cue reveals the lift/settle pattern; it must be inferred from vertical motion
    // and energy drain. Screen brightness is very low and lidar is nearly blind,
    // forcing the agent to learn the pattern through proprioception.
    v.ground = {18, 16, 14};        // near-black soot
    v.particles = {90, 70, 50};       // faint ember dust
    v.liquid = {10, 8, 6};           // dark melt pools
    v.sky = {25, 22, 18};            // dim, smoky sky
    v.particle_rate = 6.0f;
    v.particle_lift = 1.8f;          // rising soot particles
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 2.5f;          // drifting ember haze
    v.screen_brightness = 0.08f;     // near-total darkness: lidar is very expensive and short
    v.liquid_surface = false;
    return v;
  }
};

class SolarScavengerBreach final : public Biome {
 public:
  std::string_view id() const noexcept override { return "solar_scavenger_breach"; }
  std::string_view display_name() const noexcept override { return "Solar Scavenger Breach"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate baseline grip, but the harvest windows dominate.
    p.friction_mul = 0.38f + 0.20f * biome_random01(s);
    p.sink_rate = 0.004f + 0.010f * biome_random01(s, 1);
    p.viscosity = 0.50f + 0.80f * biome_random01(s, 2);
    p.energy_drain_mul = 1.35f + 0.35f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    // Warm, humid mud: thermal transfer is high but solar is strong in "clear" windows only.
    p.ambient_temperature = -5.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.2f + 0.8f * biome_random01(s, 5);
    p.solar_charge_rate = 2.2f + 0.8f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.008f + 0.016f * biome_random01(s, 8);
    // Lidar is moderately expensive and short: the clear window cannot be seen ahead.
    p.lidar_energy_mul = 2.0f + 0.8f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: uneven, cratered mudscape with steps. The hazard is the alternation of clear sky and stormy murk.
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.70f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 1.10f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.62f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sinking into the mud, slightly more at speed and in storms (wet ground).
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      // Viscous drag and normal resistance increase with depth.
      *c.wheel_force += c.contact->tangent * (-(0.25f + p.viscosity * 1.8f * (1.0f + depth)) * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.04f + 0.08f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      *c.energy_cost += (0.008f + depth * 0.15f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Harvest phase: alternates between 'clear' (bright sky, strong solar, dry firm ground)
      // and 'storm' (dark murk, almost no solar, wet slippery ground with strong gusts).
      float phase = t * 0.0046f + c.velocity.x * 0.011f;
      float clear = 0.5f + 0.5f * std::sin(phase);  // 1 = clear, 0 = storm
      float storm = 1.0f - clear;

      // Faster sub-oscillation creates 'breach pulses': brief, extreme storm gusts that
      // slam the rover sideways and downward, especially when moving fast.
      float pulse_phase = t * 0.033f + c.velocity.x * 0.079f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong peaks

      // Clear phase: firm, dry ground, higher grip, slight solar assist (lift from heat), low drag.
      // Storm phase: wet mud, low grip, high drag, and strong lateral gusts that grow with speed.
      float storm_lateral = storm * (0.14f + 0.22f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.2f);
      float storm_drag = storm * 0.12f * c.mass * c.gravity * speed_norm;
      float clear_assist = clear * 0.02f * c.mass * c.gravity;

      // Breach pulses: strongest in storm, cause a sudden sideways shove that can flip the rover.
      float pulse_force = storm * pulse_narrow * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.7f);

      // Damping: higher in clear (dry grip), lower in storm (slippery), leading to drift.
      float damping = 0.05f + 0.10f * clear + 0.02f * speed_norm;

      // Vertical: clear phase gives slight lift from thermal buoyancy; storm phase presses down
      // with rain weight, increasing drag and sink.
      float clear_lift = clear * 0.015f * c.mass * c.gravity;
      float storm_weight = storm * 0.03f * c.mass * c.gravity * (1.0f + 0.7f * speed_norm);

      c.body_force->x += clear_assist + storm_lateral + pulse_force;
      c.body_force->x -= storm_drag + c.velocity.x * c.mass * damping;
      c.body_force->y += clear_lift - storm_weight - c.velocity.y * c.mass * (0.04f + 0.02f * storm);

      if (c.body_torque) {
        // Storm pulses induce a rocking torque, strongest when moving fast through a storm pulse.
        float torque = storm * pulse_narrow * (0.5f + 0.7f * speed_norm) * 0.028f * c.mass * c.gravity * std::sin(pulse_phase + 1.0f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * clear);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float phase = t * 0.0046f + speed * 0.011f;
      float clear = 0.5f + 0.5f * std::sin(phase);
      float storm = 1.0f - clear;
      float pulse_phase = t * 0.033f + speed * 0.079f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: storm phase costs much more (fighting wet mud and gusts), clear phase is cheap.
      // But the clear phase is also when solar charging is strong, so stopping to deploy the panel
      // is profitable only there. The storm phase drains heavily and offers no useful charge.
      float storm_drain = storm * (0.015f + 0.025f * speed_norm) * p.energy_drain_mul;
      float clear_drain = clear * (0.005f + 0.003f * speed_norm) * p.energy_drain_mul;
      float pulse_penalty = storm * pulse_narrow * (0.012f + 0.020f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: warm mud saps battery, more in storm (wet conductive ground).
      float thermal_drain = p.thermal_transfer * 0.003f * (1.0f + storm * 0.8f);

      // Solar credit is handled by the engine using solar_charge_rate; we expose a strong clear-phase
      // signal only through the brightness and the actual charge, which the agent must learn to use.
      *c.energy_cost += (storm_drain + clear_drain + pulse_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, muddy terrain with faint warm glow; brightness oscillates but is hidden from the agent.
    v.ground = {42, 34, 26};        // dark mud
    v.particles = {120, 86, 50};     // dusty spray
    v.liquid = {30, 24, 18};         // dark puddles
    v.sky = {70, 50, 34};            // dim, stormy amber
    v.particle_rate = 8.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 1.3f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 0.12f;     // very dark: lidar is expensive and short, so the clear window must be inferred from charge rate
    v.liquid_surface = false;
    return v;
  }
};

class ChargedSinkBreach final : public Biome {
 public:
  std::string_view id() const noexcept override { return "charged_sink_breach"; }
  std::string_view display_name() const noexcept override { return "Charged Sink Breach"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Slippery, electrically active brine: moderate grip but strong coupling to energy and lift.
    p.friction_mul = 0.30f + 0.15f * biome_random01(s);
    p.sink_rate = 0.004f + 0.008f * biome_random01(s, 1);
    p.viscosity = 1.0f + 1.2f * biome_random01(s, 2);
    p.energy_drain_mul = 1.45f + 0.35f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    // Cold, conductive brine: high thermal transfer, but solar is moderate in "charge" windows only.
    p.ambient_temperature = -40.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.5f + 0.8f * biome_random01(s, 5);
    p.solar_charge_rate = 1.8f + 0.6f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Lidar is moderately expensive and short: the charge window must be inferred from lift and energy drain.
    p.lidar_energy_mul = 3.0f + 0.8f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, smooth basins with gentle craters and low steps. The hazard is the charge cycle, not the geometry.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.30f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip, but the charge cycle modulates normal force and traction.
    return p.friction_mul * 0.70f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into conductive brine, more with speed and immersion.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.8f * std::abs(c.wheel_speed) + 0.4f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      // Viscous drag plus depth-induced normal resistance.
      float drag = (0.25f + p.viscosity * 1.8f * (1.0f + depth) * c.immersion + 0.15f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      *c.energy_cost += (0.010f + depth * 0.12f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Charge cycle: a slow, position-dependent oscillation alternates between
      // 'charge' (electrically active brine, slight upward lift, strong solar,
      // but also traction loss and instability) and 'drain' (passive brine,
      // stable, efficient, but almost no solar). The cycle is invisible to lidar
      // and visually uniform, so the agent must infer it from the rover's lift
      // and energy drain.
      float cycle_phase = t * 0.0045f + c.velocity.x * 0.011f;
      float charge = 0.5f + 0.5f * std::sin(cycle_phase);  // 1 = charge, 0 = drain
      float drain = 1.0f - charge;

      // A faster sub-oscillation creates 'surge pulses': brief, intense upward
      // lifts within the charge phase, reducing normal load and traction. These
      // are the flip hazard; a policy that drives at constant speed through the
      // charge phase gets caught.
      float pulse_phase = t * 0.031f + c.velocity.x * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong peaks

      // Charge phase: the brine is charged, creating a slight upward lift that
      // reduces traction, and a lateral instability that grows with speed. The
      // drain phase is passive, with stable grip and a small forward assist.
      float charge_lift = charge * 0.06f * c.mass * c.gravity;
      float drain_lift = drain * 0.02f * c.mass * c.gravity;
      float net_lift = charge_lift - drain_lift;

      float charge_lateral = charge * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.1f);
      float drain_assist = drain * 0.02f * c.mass * c.gravity;

      // Surge pulses: strongest in charge phase, cause a brief upward shove that
      // can destabilise the rover if moving fast.
      float pulse_lift = charge * pulse_narrow * 0.14f * c.mass * c.gravity;
      float pulse_lateral = charge * pulse_narrow * (0.05f + 0.10f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);

      // Damping: higher in drain (stable), lower in charge (slippery), leading to drift.
      float damping = 0.05f + 0.08f * drain + 0.02f * speed_norm;

      // Vertical: net lift from charge phase reduces normal load; drain phase is
      // slightly heavier, improving grip.
      c.body_force->x += drain_assist + charge_lateral + pulse_lateral;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y += net_lift + pulse_lift - c.velocity.y * c.mass * (0.04f + 0.02f * drain);

      if (c.body_torque) {
        // Charge pulses induce a rocking torque, strongest at speed through a pulse.
        float torque = charge * pulse_narrow * (0.5f + 0.6f * speed_norm) * 0.026f * c.mass * c.gravity * std::sin(pulse_phase + 0.9f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * drain);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float cycle_phase = t * 0.0045f + speed * 0.011f;
      float charge = 0.5f + 0.5f * std::sin(cycle_phase);
      float drain = 1.0f - charge;
      float pulse_phase = t * 0.031f + speed * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: drain phase is cheap and efficient. Charge phase costs
      // more (fighting lift and instability), but that is exactly when solar
      // charging is strong, so stopping to deploy the panel is profitable. The
      // challenge is that the charge phase also has surge pulses that can flip
      // the rover, so recharging must be timed carefully within the charge phase.
      float drain_cost = drain * (0.004f + 0.003f * speed_norm) * p.energy_drain_mul;
      float charge_cost = charge * (0.012f + 0.020f * speed_norm) * p.energy_drain_mul;
      float pulse_penalty = charge * pulse_narrow * (0.010f + 0.016f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: cold brine saps battery, slightly more in charge phase.
      float thermal_drain = p.thermal_transfer * 0.002f * (1.0f + charge * 0.4f);

      *c.energy_cost += (drain_cost + charge_cost + pulse_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, electrically-charged brine with faint cyan glow. No visual cue
    // reveals the charge/drain pattern; it must be inferred from lift and energy.
    v.ground = {20, 28, 36};
    v.particles = {90, 180, 210};
    v.liquid = {16, 50, 70};
    v.sky = {30, 55, 70};
    v.particle_rate = 6.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 0.9f;
    v.base_particles = 2;
    v.max_particles = 20;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.15f;  // dark: lidar is expensive and short, forcing inference
    v.liquid_surface = true;
    return v;
  }
};

class ThermalSluiceLag final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_sluice_lag"; }
  std::string_view display_name() const noexcept override { return "Thermal Sluice Lag"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Slippery, warm brine: the fluid dynamics dominate over the tyre model.
    p.friction_mul = 0.30f + 0.15f * biome_random01(s);
    p.sink_rate = 0.005f + 0.010f * biome_random01(s, 1);
    p.viscosity = 1.2f + 1.6f * biome_random01(s, 2);
    p.energy_drain_mul = 1.65f + 0.50f * biome_random01(s, 3);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 4);
    // Warm but not scorching: thermal transfer high but not the primary hazard.
    p.ambient_temperature = 15.0f + 20.0f * biome_random01(s, 5);
    p.thermal_transfer = 2.0f + 0.8f * biome_random01(s, 6);
    // Moderate solar: recharging is possible but must be timed with the sluice phase.
    p.solar_charge_rate = 0.90f + 0.50f * biome_random01(s, 7);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 8);
    p.crust_deform = 0.003f + 0.007f * biome_random01(s, 9);
    // Lidar is cheap and long enough to see some terrain, but not the sluice phase —
    // the agent must infer the fluid state from its own momentum and drag.
    p.lidar_energy_mul = 0.60f + 0.30f * biome_random01(s, 10);
    p.lidar_range_mul = 0.40f + 0.20f * biome_random01(s, 11);
    // Smooth, low-amplitude terrain with shallow craters and few steps: the
    // sluice lag is the real hazard, not the geometry. This prevents a generic
    // obstacle-avoidance policy from scoring.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.50f + 0.20f * biome_random01(s, 13);
    p.terrain_crater_mul = 1.20f + 0.30f * biome_random01(s, 14);
    p.terrain_step_mul = 0.25f + 0.15f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate baseline grip, but the fluid inertia dominates through body effects.
    return p.friction_mul * 0.70f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slow sink into the warm, buoyant brine; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.5f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      // Viscous drag plus depth-induced normal resistance.
      float drag = (0.25f + p.viscosity * 1.8f * (1.0f + depth) * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      *c.energy_cost += (0.008f + depth * 0.12f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Thermal sluice phase: a slow, position-dependent oscillation alternates
      // between 'hot' (warm, buoyant brine, low viscosity but also low traction
      // and high energy drain) and 'cool' (thicker, heavier brine, high drag but
      // also high traction and lower energy drain). The phase is invisible to
      // lidar (which sees only terrain) and visually uniform, so the agent must
      // infer it from its own acceleration response and drag.
      float phase = t * 0.0044f + c.velocity.x * 0.011f;
      float hot = 0.5f + 0.5f * std::sin(phase);  // 1 = hot, 0 = cool
      float cool = 1.0f - hot;

      // A faster sub-oscillation creates 'thermals': brief, sharp buoyancy spikes
      // within the hot phase. These are the flip hazard — a policy that maintains
      // constant speed through the hot phase gets destabilised.
      float thermal_phase = t * 0.027f + c.velocity.x * 0.065f;
      float thermal = 0.5f + 0.5f * std::sin(thermal_phase);
      float thermal_narrow = thermal * thermal;  // narrow, strong peaks

      // Thermal memory: builds with speed but lags behind, creating a hysteresis
      // loop where acceleration is punished but steady momentum is rewarded.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Hot phase: low viscosity, low drag, but also low traction. The rover
      // accelerates easily but slides laterally and has poor braking. Cool phase:
      // high viscosity, high drag, but also high traction and stability.
      float hot_assist = hot * 0.03f * c.mass * c.gravity;
      float cool_drag = cool * 0.12f * c.mass * c.gravity * speed_norm;

      // Crucially, the thermal memory creates a delayed forward surge when the
      // rover decelerates from high speed through a hot zone. This is the
      // signature hazard: braking hard while hot causes a sudden forward lurch
      // that can cause a flip or a stall. The correct strategy is to ease off
      // the throttle gradually and let the momentum bleed off naturally.
      float decel = std::max(0.0f, 0.5f - speed_norm);
      float surge = hot * memory * decel * 0.16f * c.mass * c.gravity;

      // Lateral instability: hot phase causes the rover to drift sideways,
      // stronger at speed and during thermals. Cool phase is stable.
      float lateral = hot * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(thermal_phase + 1.1f);
      float thermal_lateral = hot * thermal_narrow * (0.10f + 0.16f * speed_norm) * c.mass * c.gravity * std::sin(thermal_phase + 0.8f);
      float cool_stabilize = cool * 0.02f * c.mass * c.gravity;

      // Damping: lower in hot (slippery), higher in cool (viscous).
      float damping = 0.04f + 0.10f * cool + 0.02f * speed_norm;

      // Vertical: hot phase gives slight lift from thermal buoyancy, reducing
      // traction. Cool phase presses down slightly, increasing grip.
      float buoyancy = hot * 0.05f * c.mass * c.gravity;
      float weight = cool * 0.03f * c.mass * c.gravity;

      c.body_force->x += hot_assist + surge + lateral + thermal_lateral + cool_stabilize;
      c.body_force->x -= cool_drag + c.velocity.x * c.mass * damping;
      c.body_force->y += buoyancy - weight - c.velocity.y * c.mass * (0.04f + 0.03f * cool);

      if (c.body_torque) {
        // Thermals induce a rocking torque, strongest at speed through a hot thermal.
        float torque = hot * thermal_narrow * (0.5f + 0.6f * speed_norm) * 0.028f * c.mass * c.gravity * std::sin(thermal_phase + 0.9f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * cool);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float phase = t * 0.0044f + speed * 0.011f;
      float hot = 0.5f + 0.5f * std::sin(phase);
      float cool = 1.0f - hot;
      float thermal_phase = t * 0.027f + speed * 0.065f;
      float thermal = 0.5f + 0.5f * std::sin(thermal_phase);
      float thermal_narrow = thermal * thermal;
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Energy budget: hot phase is cheaper per unit distance (low drag) but
      // punishes acceleration with a memory-dependent cost and thermals add a
      // sharp penalty. Cool phase is more expensive (high viscosity) but stable.
      // The optimal strategy is to build momentum slowly in the cool phase,
      // then coast through the hot phase, easing off gradually before thermals.
      float hot_cost = hot * (0.006f + 0.018f * memory) * p.energy_drain_mul;
      float cool_cost = cool * (0.014f + 0.010f * speed_norm) * p.energy_drain_mul;
      float thermal_penalty = hot * thermal_narrow * (0.012f + 0.016f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: warm brine saps battery, more in hot phase.
      float thermal_drain = p.thermal_transfer * 0.003f * (1.0f + hot * 0.5f);

      *c.energy_cost += (hot_cost + cool_cost + thermal_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, warm amber brine with faint shimmer. No visual cue reveals the
    // hot/cool phase or the thermals; it must be inferred from drag and energy drain.
    v.ground = {58, 42, 28};        // dark amber mud
    v.particles = {190, 150, 90};     // warm heat-haze particles
    v.liquid = {48, 62, 36};         // murky green-amber brine
    v.sky = {70, 58, 40};            // dim, hazy amber sky
    v.particle_rate = 7.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 1.1f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 1.6f;
    v.screen_brightness = 0.25f;     // moderately dark: lidar is cheap but the phase is invisible
    v.liquid_surface = true;
    return v;
  }
};

class GyreLagoon final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gyre_lagoon"; }
  std::string_view display_name() const noexcept override { return "Gyre Lagoon"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.30f + 0.15f * biome_random01(s);
    p.sink_rate = 0.004f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.60f + 0.80f * biome_random01(s, 2);
    p.energy_drain_mul = 1.50f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -55.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.2f + 0.7f * biome_random01(s, 5);
    p.solar_charge_rate = 0.08f + 0.05f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    p.lidar_energy_mul = 4.5f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.12f + 0.05f * biome_random01(s, 10);
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.15f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.6f * std::abs(c.wheel_speed) + 0.4f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      float drag = (0.20f + p.viscosity * 1.6f * (1.0f + depth) * c.immersion + 0.10f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      *c.energy_cost += (0.010f + depth * 0.12f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      float phase = t * 0.0046f + c.velocity.x * 0.012f;
      float gyre = 0.5f + 0.5f * std::sin(phase);  // 1 = forward eddy, 0 = reverse eddy
      float reverse = 1.0f - gyre;

      float pulse_phase = t * 0.027f + c.velocity.x * 0.064f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.28f) * 6.0f);

      float grav_mul = 0.70f + 0.60f * gyre + 0.40f * gyre * pulse_narrow;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      float forward_assist = gyre * (0.025f + 0.06f * memory) * c.mass * c.gravity * (0.5f + 0.5f * memory);
      float reverse_drag = reverse * memory * (0.10f + 0.06f * (1.0f - memory)) * c.mass * c.gravity * 0.55f;

      float delayed_release = gyre * memory * (0.035f + 0.055f * std::sin(phase + 1.0f)) * c.mass * c.gravity * 0.5f;
      float reverse_assist = reverse * 0.020f * c.mass * c.gravity * (0.5f + 0.5f * speed_norm);

      float lateral_drift = std::sin(t * 0.022f + speed * 0.048f + memory * 2.0f) * (0.4f + 0.6f * gyre) * 0.05f * c.mass * (0.5f + memory);

      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.25f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.20f;

      float damping = 0.035f + 0.085f * reverse * (1.0f - memory) + 0.02f * speed_norm;

      float pulse_surge = gyre * pulse_narrow * (0.10f + 0.16f * speed_norm) * c.mass * c.gravity * 0.5f;

      c.body_force->x -= reverse_drag;
      c.body_force->x += forward_assist + delayed_release + reverse_assist + lateral_drift + pulse_surge;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * gyre);

      if (c.body_torque) {
        float torque = gyre * pulse_narrow * (0.5f + 0.7f * speed_norm) * 0.030f * c.mass * c.gravity * std::sin(pulse_phase + 0.7f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * gyre);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      float phase = t * 0.0046f + speed * 0.012f;
      float gyre = 0.5f + 0.5f * std::sin(phase);
      float reverse = 1.0f - gyre;
      float pulse_phase = t * 0.027f + speed * 0.064f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.28f) * 6.0f);

      float windup_cost = reverse * memory * 0.018f;
      float start_cost = reverse * (1.0f - memory) * (0.010f + 0.007f * speed_norm);
      float pulse_penalty = gyre * pulse_narrow * (0.010f + 0.014f * speed_norm);
      float gyre_cost = gyre * (0.008f + 0.010f * speed_norm) * p.energy_drain_mul;
      float reverse_cost = reverse * (0.014f + 0.012f * speed_norm) * p.energy_drain_mul;
      float free_bonus = reverse * (0.003f + 0.002f * speed_norm);
      float thermal_drain = p.thermal_transfer * 0.003f * (1.0f + gyre * 0.3f);

      *c.energy_cost += (windup_cost + start_cost + pulse_penalty + gyre_cost + reverse_cost - free_bonus + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.ground = {12, 18, 24};
    v.particles = {64, 96, 128};
    v.liquid = {8, 22, 36};
    v.sky = {18, 28, 38};
    v.particle_rate = 3.0f;
    v.particle_lift = 1.1f;
    v.particle_spread = 0.7f;
    v.base_particles = 2;
    v.max_particles = 18;
    v.particle_size = 2;
    v.ambient_particles = 5;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = true;
    return v;
  }
};

class QuakeSwellRidge final : public Biome {
 public:
  std::string_view id() const noexcept override { return "quake_swell_ridge"; }
  std::string_view display_name() const noexcept override { return "Quake-Swell Ridge"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Decent baseline grip; the pulse changes effective traction via body effects.
    p.friction_mul = 0.60f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.5f * biome_random01(s, 3);
    p.ambient_temperature = -60.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.70f + 0.20f * biome_random01(s, 5);
    // Solar is nearly zero: recharging is a trap, forcing frugal driving.
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.008f * biome_random01(s, 8);
    // Lidar is very expensive and short: the quake phase must be inferred from motion.
    p.lidar_energy_mul = 4.0f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, undulating ridges with moderate craters and low steps.
    // The quake phase is the hazard, not the geometry.
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Decent baseline grip; the pulse changes effective traction via body effects.
    return p.friction_mul * 0.92f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the fractured rock floor; not a mud trap.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.4f * std::abs(c.wheel_speed));
    }
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the surge forces dominate in body effects.
      *c.wheel_force += c.contact->tangent * (-0.25f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Base drain plus a small rock-friction load.
      *c.energy_cost += (0.005f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Quake phase: a slow, position-dependent oscillation determines whether the
      // rock mesh is in a 'calm' state (stable, low energy drain) or a 'quake'
      // state (pulsing lateral shoves, rising energy drain, unstable torque). The
      // phase is invisible to lidar (very short, expensive) and visually uniform,
      // so the agent must infer it from the rover's own drift and energy drain.
      float quake_phase = t * 0.0048f + c.velocity.x * 0.0125f;
      float quake = 0.5f + 0.5f * std::sin(quake_phase);  // 1 = quake, 0 = calm
      float calm = 1.0f - quake;

      // Pulse sub-oscillation: brief, sharp lateral shoves within the quake state.
      // These are the principal flip hazard; a naive 'drive carefully' policy that
      // keeps a constant speed through the quake gets destabilised.
      float pulse_phase = t * 0.031f + c.velocity.x * 0.076f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong peaks

      // Quake state: strong lateral pulsing force that grows with speed, plus a
      // slight upward lift that reduces normal load and traction. Calm state:
      // steady, weak forward assist and strong damping for stable cruising.
      float quake_lateral = quake * p.wind_force * (0.20f + 0.28f * speed_norm) * c.mass * c.gravity * (0.6f + 0.4f * std::sin(pulse_phase + 1.0f));
      float calm_assist = calm * 0.025f * c.mass * c.gravity * (0.5f + 0.5f * speed_norm);
      float quake_lift = quake * 0.02f * c.mass * c.gravity * (0.5f + 0.5f * std::sin(pulse_phase + 0.5f));

      // Pulse bursts: strongest lateral surge and torque, proportional to speed.
      float pulse_lateral = quake * pulse_narrow * (0.14f + 0.24f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);

      // Damping is moderate; higher in calm (stable), lower in quake (easier to be shoved).
      float damping = 0.05f + 0.03f * calm + 0.02f * speed_norm;

      c.body_force->x += quake_lateral + calm_assist + pulse_lateral;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y += quake_lift - c.velocity.y * c.mass * (0.05f + 0.02f * calm);

      if (c.body_torque) {
        // Pulse bursts induce a yawing torque that tries to spin the rover,
        // strongest at speed through a quake pulse. Counter-steering is essential.
        float torque = (quake * pulse_narrow * (0.5f + 0.7f * speed_norm) + calm * 0.10f * (0.5f + 0.3f * speed_norm)) * 0.018f * c.mass * c.gravity * std::sin(pulse_phase + 1.2f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * calm);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float quake_phase = t * 0.0048f + speed * 0.0125f;
      float quake = 0.5f + 0.5f * std::sin(quake_phase);
      float calm = 1.0f - quake;
      float pulse_phase = t * 0.031f + speed * 0.076f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: calm state is cheap and rewards steady cruising; quake state
      // scales cost with speed, and pulses add a sharp penalty. This forces the agent
      // to learn to push in calm and coast through quake pulses.
      float calm_cost = calm * (0.004f + 0.004f * speed_norm) * p.energy_drain_mul;
      float quake_cost = quake * (0.010f + 0.022f * speed_norm) * p.energy_drain_mul;
      float pulse_penalty = quake * pulse_narrow * (0.012f + 0.016f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: cold fractured rock saps battery slightly, more in quake state.
      float thermal_drain = p.thermal_transfer * 0.002f * (1.0f + quake * 0.4f);

      *c.energy_cost += (calm_cost + quake_cost + pulse_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, pale hazy desert with faint shimmer — visually ambiguous, reveals
    // nothing about the quake phase. Very dark to force inference from drift.
    v.ground = {118, 104, 84};       // pale sand
    v.particles = {200, 185, 160};    // fine wind-blown dust
    v.liquid = {55, 48, 38};         // dark hollows
    v.sky = {215, 195, 170};         // hazy, dusty daylight
    v.particle_rate = 8.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.4f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 22;
    v.ambient_drift = 2.5f;          // strong wind-blown dust, but the actual force is hidden
    v.screen_brightness = 0.09f;     // very dark: lidar is extremely expensive and nearly blind
    v.liquid_surface = false;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const GranularThrottleTrap biome_0; out.push_back(&biome_0);
  static const CantileverGale biome_1; out.push_back(&biome_1);
  static const BarometricBrakeWells biome_2; out.push_back(&biome_2);
  static const GravityPendulumHollow biome_3; out.push_back(&biome_3);
  static const AvalancheProbe biome_4; out.push_back(&biome_4);
  static const StrobePitfall biome_5; out.push_back(&biome_5);
  static const StroboscopicCrust biome_6; out.push_back(&biome_6);
  static const AccordionGravityLadder biome_7; out.push_back(&biome_7);
  static const RuttedThrottleBasin biome_8; out.push_back(&biome_8);
  static const PendulumHollow biome_9; out.push_back(&biome_9);
  static const FlywheelBrakeLag biome_10; out.push_back(&biome_10);
  static const HystereticRotorField biome_11; out.push_back(&biome_11);
  static const RotorRhythmField biome_12; out.push_back(&biome_12);
  static const TidalBrakeVault biome_13; out.push_back(&biome_13);
  static const LullAndThermalScrub biome_14; out.push_back(&biome_14);
  static const GravityWellBrine biome_15; out.push_back(&biome_15);
  static const FrictionMirageBelt biome_16; out.push_back(&biome_16);
  static const TideSluiceReservoir biome_17; out.push_back(&biome_17);
  static const GritShiftLode biome_18; out.push_back(&biome_18);
  static const AscendantSootBank biome_19; out.push_back(&biome_19);
  static const SolarScavengerBreach biome_20; out.push_back(&biome_20);
  static const ChargedSinkBreach biome_21; out.push_back(&biome_21);
  static const ThermalSluiceLag biome_22; out.push_back(&biome_22);
  static const GyreLagoon biome_23; out.push_back(&biome_23);
  static const QuakeSwellRidge biome_24; out.push_back(&biome_24);
}
// </MARS_GENERATED_BIOMES>
}  // namespace generated_biomes

inline constexpr std::string_view kBiomeBankVersion = "sha256:39aff219154d8198bd74b37b3a17c29c552cbc05cdaf9c1b904dfd7fd9f188b8";

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
