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

class ThermalSiphonReactor final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_siphon_reactor"; }
  std::string_view display_name() const noexcept override { return "Thermal Siphon Reactor"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip but high rolling resistance; the real challenge is thermal routing.
    p.friction_mul = 0.60f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.60f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 3);
    // Hot but survivable; thermal transfer is high enough to matter.
    p.ambient_temperature = 20.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.8f + 0.8f * biome_random01(s, 5);
    // Solar is moderate but completely suppressed during thermal spikes (below).
    p.solar_charge_rate = 0.30f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Lidar is cheap but short-range; thermal spikes are invisible to it.
    p.lidar_energy_mul = 0.20f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 0.30f + 0.15f * biome_random01(s, 10);
    // Moderate terrain: rolling roughness, not extreme craters/steps.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.00f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.00f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 1.00f + 0.30f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Slightly below baseline to increase rolling resistance and energy cost.
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; thermal effects dominate in body.
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Base drain plus thermal load that scales with ambient temperature.
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.006f * p.thermal_transfer;
      *c.energy_cost += (0.008f + heat_load + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal siphon phase: a slow, position-dependent cycle that determines
      // whether the ground is "charging" (cool, higher solar) or "venting" (hot,
      // saps energy and grip). The agent must learn to push through vent phases
      // efficiently and use charge phases to recover.
      float phase = t * 0.005f + c.velocity.x * 0.013f;
      float vent = 0.5f + 0.5f * std::sin(phase);

      // Thermal spikes: brief, intense heat events that reduce grip and add a
      // constant energy drain. They are invisible but follow a deterministic
      // pattern tied to the phase. High throttle during a spike is punished
      // with extra energy loss and instability.
      float spike_phase = t * 0.037f + c.velocity.x * 0.08f;
      float spike = 0.5f + 0.5f * std::sin(spike_phase);
      spike = spike * spike * spike;  // narrow, strong peaks

      // Combined thermal intensity: baseline vent plus spikes.
      float thermal = vent * (0.6f + 0.8f * spike);

      // Grip reduction: high thermal makes the crust slippery (heat-softened).
      float grip_factor = 1.0f - 0.4f * thermal;

      // Thermal buoyancy: hot air creates slight lift, reducing normal load and
      // traction, especially during spikes.
      float lift = 0.04f * c.mass * c.gravity * thermal;

      // Lateral thermal wind: hot air pushes sideways, stronger during spikes.
      float lateral = std::sin(spike_phase + 0.8f) * 0.03f * c.mass * (0.5f + thermal);

      // Damping is lower during high thermal (less control), higher when cool.
      float damping = 0.05f + 0.05f * (1.0f - thermal);

      c.body_force->x += lateral - c.velocity.x * c.mass * damping * grip_factor;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.02f * thermal);

      if (c.body_torque) {
        // Thermal gradients induce rocking torque, strongest during spikes.
        float torque = std::sin(spike_phase + 1.1f) * thermal * 0.015f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.005f + speed * 0.013f;
      float vent = 0.5f + 0.5f * std::sin(phase);
      float spike_phase = t * 0.037f + speed * 0.08f;
      float spike = 0.5f + 0.5f * std::sin(spike_phase);
      spike = spike * spike * spike;
      float thermal = vent * (0.6f + 0.8f * spike);

      // Energy drain scales with thermal intensity and speed. High throttle
      // during spikes is punished double. Steady, moderate speed through vent
      // phases is efficient.
      float thermal_drain = thermal * 0.020f * p.thermal_transfer;
      float speed_drain = speed * 0.002f * (1.0f + thermal);
      *c.energy_cost += (0.005f + thermal_drain + speed_drain) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, heat-stressed rock with glowing veins — suggests thermal activity
    // but not the siphon/spike pattern.
    v.ground = {88, 44, 28};
    v.particles = {255, 140, 40};
    v.liquid = {40, 20, 12};
    v.sky = {160, 80, 30};
    v.particle_rate = 8.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 1.0f;
    v.base_particles = 2;
    v.max_particles = 30;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.20f;  // dark: lidar is cheap but the thermal pattern is hidden
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

class ThermalBatteryCascade final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_battery_cascade"; }
  std::string_view display_name() const noexcept override { return "Thermal Battery Cascade"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip, but the thermal cascade is the real hazard.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 3);
    // Hot but survivable; thermal transfer is high enough to matter.
    p.ambient_temperature = 30.0f + 25.0f * biome_random01(s, 4);
    p.thermal_transfer = 3.0f + 1.0f * biome_random01(s, 5);
    // Solar is moderate but completely suppressed during thermal spikes (below).
    p.solar_charge_rate = 0.25f + 0.15f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Lidar is cheap but very short-range; thermal spikes are invisible to it.
    p.lidar_energy_mul = 0.15f + 0.08f * biome_random01(s, 9);
    p.lidar_range_mul = 0.18f + 0.08f * biome_random01(s, 10);
    // Terrain: rolling moderate bumps, not extreme craters/steps,
    // so the thermal cascade is the real challenge, not geometry.
    p.terrain_amplitude_mul = 1.10f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.00f + 0.30f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.80f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 0.80f + 0.30f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Slightly below baseline to increase rolling resistance and energy cost.
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; thermal effects dominate in body.
      *c.wheel_force += c.contact->tangent * (-0.5f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Base drain plus thermal load that scales with ambient temperature.
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.006f * p.thermal_transfer;
      *c.energy_cost += (0.008f + heat_load + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      // Thermal cascade phase: a slow, position-dependent cycle that determines
      // whether the ground is "charging" (cool, higher solar) or "venting" (hot,
      // saps energy and grip). The agent must learn to push through vent phases
      // efficiently and use charge phases to recover.
      float phase = t * 0.005f + c.velocity.x * 0.013f;
      float vent = 0.5f + 0.5f * std::sin(phase);

      // Thermal spikes: brief, intense heat events that reduce grip and add a
      // constant energy drain. They are invisible but follow a deterministic
      // pattern tied to the phase. High throttle during a spike is punished
      // with extra energy loss and instability.
      float spike_phase = t * 0.031f + c.velocity.x * 0.07f;
      float spike = 0.5f + 0.5f * std::sin(spike_phase);
      spike = spike * spike * spike;  // narrow, strong peaks

      // Combined thermal intensity: baseline vent plus spikes.
      float thermal = vent * (0.6f + 0.9f * spike);

      // Grip reduction: high thermal makes the crust slippery (heat-softened).
      float grip_factor = 1.0f - 0.45f * thermal;

      // Thermal buoyancy: hot air creates slight lift, reducing normal load and
      // traction, especially during spikes.
      float lift = 0.05f * c.mass * c.gravity * thermal;

      // Lateral thermal wind: hot air pushes sideways, stronger during spikes.
      float lateral = std::sin(spike_phase + 0.8f) * 0.04f * c.mass * (0.5f + thermal);

      // Damping is lower during high thermal (less control), higher when cool.
      float damping = 0.05f + 0.06f * (1.0f - thermal);

      c.body_force->x += lateral - c.velocity.x * c.mass * damping * grip_factor;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.02f * thermal);

      if (c.body_torque) {
        // Thermal gradients induce rocking torque, strongest during spikes.
        float torque = std::sin(spike_phase + 1.1f) * thermal * 0.018f * c.mass * c.gravity;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float phase = t * 0.005f + speed * 0.013f;
      float vent = 0.5f + 0.5f * std::sin(phase);
      float spike_phase = t * 0.031f + speed * 0.07f;
      float spike = 0.5f + 0.5f * std::sin(spike_phase);
      spike = spike * spike * spike;
      float thermal = vent * (0.6f + 0.9f * spike);

      // Energy drain scales with thermal intensity and speed. High throttle
      // during spikes is punished double. Steady, moderate speed through vent
      // phases is efficient.
      float thermal_drain = thermal * 0.022f * p.thermal_transfer;
      float speed_drain = speed * 0.002f * (1.0f + thermal);
      *c.energy_cost += (0.005f + thermal_drain + speed_drain) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Pale, heat-bleached rock with faint orange cracks — suggests thermal
    // activity but not the cascade pattern.
    v.ground = {108, 68, 38};
    v.particles = {255, 160, 70};
    v.liquid = {60, 30, 14};
    v.sky = {175, 95, 35};
    v.particle_rate = 7.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 0.9f;
    v.base_particles = 2;
    v.max_particles = 26;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 1.5f;
    v.screen_brightness = 0.18f;  // very dark: lidar is cheap but the thermal pattern is hidden
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

class ThermalCrosswindLull final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_crosswind_lull"; }
  std::string_view display_name() const noexcept override { return "Thermal Crosswind Lull"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip, but the thermal crosswinds dominate.
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.50f * biome_random01(s, 2);
    p.wind_force = 2.0f + 4.0f * biome_random01(s, 3);
    // Hot but survivable; thermal transfer is high enough to matter.
    p.ambient_temperature = 25.0f + 30.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.5f + 0.8f * biome_random01(s, 5);
    // Solar is weak in the hazy dark, forcing energy discipline.
    p.solar_charge_rate = 0.10f + 0.08f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.008f * biome_random01(s, 8);
    // Lidar is expensive and very short-range: the crosswind pattern must be inferred from drift.
    p.lidar_energy_mul = 4.00f + 1.50f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 10);
    // Terrain: moderate amplitude, low roughness, sparse craters, low steps —
    // the crosswind is the hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.25f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Decent mechanical grip, but thermal softening in body effects reduces it during heat peaks.
    return p.friction_mul * 0.90f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; thermal effects dominate in body.
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Base drain plus thermal load that scales with ambient temperature.
      float heat_load = std::max(0.0f, p.ambient_temperature) * 0.006f * p.thermal_transfer;
      *c.energy_cost += (0.008f + heat_load + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Thermal load: heat from ambient temperature creates warm air currents + heat-softened ground.
      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 25.0f) * 0.03f);

      // Crosswind lull cycle: a slow, position-dependent oscillation that determines whether
      // the rover is in a 'venting' zone (strong thermal crosswind, unstable, hot) or a
      // 'lull' zone (calm, stable, efficient). The phase is invisible without lidar
      // (which is expensive and short here), so it must be inferred from drift and heat.
      float lull_phase = t * 0.005f + c.velocity.x * 0.014f;
      float lull = 0.5f + 0.5f * std::sin(lull_phase);  // 1 = lull (calm), 0 = vent (windy)

      // Thermal gusts: brief, intense heat spikes within the venting zones that create
      // strong lateral pushes and reduce grip. These are correlated with the lull phase
      // but are faster and more violent, forcing the agent to predict and brace.
      float gust_phase = t * 0.027f + c.velocity.x * 0.06f;
      float gust = 0.5f + 0.5f * std::sin(gust_phase);
      gust = gust * gust * gust;  // narrow, strong peaks

      // Combined thermal intensity: high in vents + gusts, low in lulls.
      float thermal = (1.0f - lull) * (0.45f + 0.90f * gust);

      // Thermal-driven crosswind: the dominant hazard. Strong lateral force during vents,
      // especially during gusts. In lulls, the force drops to near zero, giving the rover
      // a calm window to make progress and recover.
      float lateral_force = heat_factor * thermal * (0.15f + 0.25f * speed_norm) * c.mass * p.wind_force;

      // Thermal buoyancy: hot air creates lift, reducing normal load and traction,
      // especially during gusts. Makes the rover skittish and harder to control.
      float lift = thermal * 0.03f * c.mass * c.gravity;

      // Thermal damping: lower during hot vents (hot air less dense, less resistance),
      // higher during lulls (efficient cruising). This rewards pushing through lulls
      // and coasting through vents.
      float damping = 0.04f + 0.03f * lull + 0.02f * speed_norm;

      // Crosswind torque: the lateral force creates a yaw torque, stronger during gusts
      // and at speed. In lulls, the torque nearly vanishes, allowing straight-line driving.
      float lateral_torque = heat_factor * thermal * (0.012f + 0.02f * speed_norm) * c.mass;
      lateral_torque *= std::sin(gust_phase + 0.7f);

      c.body_force->x += lateral_force - c.velocity.x * c.mass * damping;
      c.body_force->y += lift - c.velocity.y * c.mass * (0.04f + 0.02f * thermal);

      if (c.body_torque) {
        *c.body_torque += lateral_torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float heat = std::max(0.0f, p.ambient_temperature) * p.thermal_transfer;
      float heat_factor = 0.5f + 0.5f * std::tanh((heat - 25.0f) * 0.03f);

      float lull_phase = t * 0.005f + speed * 0.014f;
      float lull = 0.5f + 0.5f * std::sin(lull_phase);

      float gust_phase = t * 0.027f + speed * 0.06f;
      float gust = 0.5f + 0.5f * std::sin(gust_phase);
      gust = gust * gust * gust;

      float thermal = (1.0f - lull) * (0.45f + 0.90f * gust);

      // Energy drain scales with thermal intensity and speed. High throttle during
      // vents is punished with extra energy loss. The efficient strategy is to
      // cruise through lulls and coast/steer through vents, using momentum already built.
      float thermal_drain = thermal * heat_factor * 0.015f * p.thermal_transfer;
      float speed_drain = speed * 0.002f * (1.0f + thermal);
      *c.energy_cost += (0.006f + thermal_drain + speed_drain) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Sun-bleached rock with shimmering heat-haze particles - suggests heat
    // but not the specific crosswind pattern.
    v.ground = {142, 118, 84};  // pale tan rock
    v.particles = {255, 180, 80};  // glowing ember dust
    v.liquid = {60, 48, 32};  // dark hollows
    v.sky = {180, 130, 60};  // hazy amber sky
    v.particle_rate = 10.0f;
    v.particle_lift = 1.5f;
    v.particle_spread = 1.3f;
    v.base_particles = 3;
    v.max_particles = 36;
    v.particle_size = 2;
    v.ambient_particles = 22;
    v.ambient_drift = 3.0f;  // heat shimmer particles, but the actual force pattern is hidden
    v.screen_brightness = 0.12f;  // very dark: lidar is cheap but very short, so the thermal pattern must be inferred from motion
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

class SwayingLidarPulse final : public Biome {
 public:
  std::string_view id() const noexcept override { return "swaying_lidar_pulse"; }
  std::string_view display_name() const noexcept override { return "Swaying Lidar Pulse"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.40f + 0.20f * biome_random01(s);
    p.sink_rate = 0.008f + 0.020f * biome_random01(s, 1);
    p.viscosity = 1.20f + 1.60f * biome_random01(s, 2);
    p.energy_drain_mul = 1.35f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -65.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.30f + 0.40f * biome_random01(s, 5);
    p.solar_charge_rate = 0.12f + 0.08f * biome_random01(s, 6);
    p.gravity_mul = 0.92f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.006f + 0.014f * biome_random01(s, 8);
    p.lidar_energy_mul = 4.20f + 1.20f * biome_random01(s, 9);
    p.lidar_range_mul = 0.16f + 0.10f * biome_random01(s, 10);
    // Reshape terrain for hidden swaying pockets: moderate roughness, dense craters, low steps.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.60f + 0.40f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.50f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul * 0.60f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sinking into soft swaying sediment, speed-dependent but saturating.
      float speed = std::abs(c.wheel_speed);
      float sat = std::tanh(speed * 0.20f);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.6f * sat);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 30.0f;
      // Viscous drag plus depth-induced normal resistance.
      *c.wheel_force += c.contact->tangent * (-(0.30f + p.viscosity * 1.5f * (1.0f + depth)) * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.04f + 0.10f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      *c.energy_cost += (0.010f + depth * 0.15f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Swaying pulse phase: a slow, position-dependent oscillation that determines
      // whether the ground is 'swaying' (unstable, strong lateral forces) or 'stable'
      // (compact, efficient). The phase is invisible without lidar (which is expensive
      // and short here), so it must be inferred from the rover's own motion and energy.
      float phase = t * 0.005f + c.velocity.x * 0.013f;
      float sway = 0.5f + 0.5f * std::sin(phase);

      // A faster sub-oscillation creates 'pockets' of stronger sway within the
      // swaying phase, forcing the rover to brace or steer through them.
      float pocket_phase = t * 0.021f + c.velocity.x * 0.05f;
      float pocket = 0.5f + 0.5f * std::sin(pocket_phase);
      pocket = pocket * pocket;  // narrow, strong pockets

      // Effective sway intensity: high during swaying phase, especially in pockets.
      float sway_intensity = sway * (0.5f + 0.8f * pocket);

      // Swaying ground: strong lateral forces that grow with speed, pushing the rover
      // sideways. During stable phases, the ground compacts and gives a small forward assist.
      float lateral_force = sway_intensity * (0.10f + 0.20f * speed_norm) * c.mass * c.gravity * 0.5f;
      float assist = (1.0f - sway) * 0.015f * c.mass * c.gravity;

      // Damping: higher during swaying phases (grippy), lower during stable phases (efficient).
      float damping = 0.05f + 0.12f * sway_intensity + 0.02f * speed_norm;

      c.body_force->x += assist - c.velocity.x * c.mass * damping;
      c.body_force->x += lateral_force;
      c.body_force->y -= sway_intensity * 0.02f * c.mass * c.gravity - c.velocity.y * c.mass * (0.04f + 0.02f * sway_intensity);

      if (c.body_torque) {
        // Asymmetric sway creates a rocking torque, stronger during swaying phases and at speed.
        float torque = std::sin(pocket_phase + 1.2f) * sway_intensity * 0.018f * c.mass * c.gravity * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float phase = t * 0.005f + speed * 0.013f;
      float sway = 0.5f + 0.5f * std::sin(phase);
      float pocket_phase = t * 0.021f + speed * 0.05f;
      float pocket = 0.5f + 0.5f * std::sin(pocket_phase);
      pocket = pocket * pocket;
      float sway_intensity = sway * (0.5f + 0.8f * pocket);

      // Energy cost: high during swaying phases (fighting lateral forces), low during stable
      // phases. But there is a small energy bonus during swaying pockets (slight forward assist),
      // making it optimal to brake and steer through them rather than accelerate.
      float sway_cost = sway_intensity * (0.010f + 0.010f * speed_norm);
      float stable_bonus = (1.0f - sway) * 0.003f;
      float speed_penalty = speed * 0.002f * (1.0f + sway_intensity);

      *c.energy_cost += (0.005f + sway_cost + speed_penalty - stable_bonus) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, muddy soil with faint greenish glow – suggests unstable ground but hides
    // the swaying pulse pattern completely.
    v.ground = {42, 50, 44};  // dark green-brown
    v.particles = {70, 130, 90};  // faint green dust
    v.liquid = {24, 30, 24};  // murky pools
    v.sky = {80, 100, 85};  // dim overcast
    v.particle_rate = 7.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 0.9f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.14f;  // very dark: lidar is expensive and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class SolarSluiceHarvest final : public Biome {
 public:
  std::string_view id() const noexcept override { return "solar_sluice_harvest"; }
  std::string_view display_name() const noexcept override { return "Solar Sluice Harvest"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip but the liquid is the secondary hazard: it forces the rover into
    // the sluices, where solar harvesting is strongest but the terrain is most dangerous.
    p.friction_mul = 0.30f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.008f * biome_random01(s, 1);
    p.viscosity = 0.70f + 1.00f * biome_random01(s, 2);
    p.energy_drain_mul = 1.20f + 0.30f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -45.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.4f + 0.6f * biome_random01(s, 5);
    // Strong solar in bright zones, but the sluices are dark so charge must be
    // harvested deliberately by entering bright zones and stopping/turning.
    p.solar_charge_rate = 1.60f + 0.60f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.001f + 0.004f * biome_random01(s, 8);
    // Cheap and long-range lidar, but the sluice pattern is dense and the rover
    // must use it to identify safe bright zones, not just the geometry.
    p.lidar_energy_mul = 0.20f + 0.10f * biome_random01(s, 9);
    p.lidar_range_mul = 1.20f + 0.40f * biome_random01(s, 10);
    // Terrain: dense, narrow sluice channels with high steps and moderate craters,
    // forcing the rover to weave between bright zones and avoid deep basins.
    p.terrain_amplitude_mul = 1.20f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.30f + 0.40f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.40f + 0.50f * biome_random01(s, 13);
    p.terrain_step_mul = 1.80f + 0.50f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Slick in water, moderate on land; the sluice channels are the real challenge.
    return p.friction_mul * 0.45f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the liquid, more in deeper water.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      float speed = std::abs(c.wheel_speed);
      float depth = c.contact->penetration * 25.0f;
      // Viscous drag plus a modest rolling resistance; the sluice forces come from body effects.
      *c.wheel_force += c.contact->tangent * (-(0.20f + p.viscosity * 1.8f * (1.0f + depth) * c.immersion + 0.15f * c.immersion) * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.10f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 25.0f;
      // Energy cost is higher in water (fighting drag) but the real cost comes from
      // the sluice timing in body effects.
      *c.energy_cost += (0.008f + c.immersion * depth * 0.12f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Sluice phase: a slow, position-dependent oscillation that alternates between
      // 'bright' zones (high solar, but unstable, with strong lateral sluice forces)
      // and 'dark' zones (stable, efficient, but almost no solar). The phase is
      // invisible without lidar, which is cheap here but the pattern is dense.
      float phase = t * 0.006f + c.velocity.x * 0.016f;
      float bright = 0.5f + 0.5f * std::sin(phase);

      // A faster sub-oscillation creates 'sluice pulses' within the bright zone:
      // brief, strong sideways pushes that can flip the rover if it's moving fast.
      float pulse_phase = t * 0.029f + c.velocity.x * 0.08f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      pulse = pulse * pulse;  // narrow, strong peaks

      // Combined sluice intensity: high in bright zones, especially during pulses.
      float sluice = bright * (0.5f + 0.9f * pulse);

      // Sluice lateral force: the bright zones are turbulent, pushing the rover
      // sideways with a force that grows with speed. This makes it dangerous to
      // cruise fast through bright zones; the rover must slow down to harvest safely.
      float lateral_force = sluice * (0.08f + 0.15f * speed_norm) * c.mass * c.gravity;

      // But there is a forward assist during the dark zones: the water recedes,
      // giving a small push that rewards efficient cruising between harvests.
      float assist = (1.0f - bright) * 0.02f * c.mass * c.gravity;

      // Damping is higher in bright zones (turbulent water), lower in dark zones (efficient).
      float damping = 0.04f + 0.08f * sluice + 0.02f * speed_norm;

      c.body_force->x += lateral_force + assist - c.velocity.x * c.mass * damping;
      // Water immersion adds buoyancy, stronger in bright zones (more water), reducing
      // traction and making control harder.
      c.body_force->y += sluice * 0.03f * c.mass * c.gravity - c.velocity.y * c.mass * (0.05f + 0.03f * sluice);

      if (c.body_torque) {
        // Sluice pulses induce a rocking torque, strongest during bright zones and at speed.
        float torque = std::sin(pulse_phase + 1.3f) * sluice * 0.020f * c.mass * c.gravity * (0.5f + speed_norm);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.03f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Sluice phase: same as above but using speed for the spatial component.
      float phase = t * 0.006f + speed * 0.016f;
      float bright = 0.5f + 0.5f * std::sin(phase);
      float pulse_phase = t * 0.029f + speed * 0.08f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      pulse = pulse * pulse;
      float sluice = bright * (0.5f + 0.9f * pulse);

      // Energy harvest: strong in bright zones (solar through the water), especially
      // during pulses, but requires enduring the lateral sluice forces. Dark zones
      // have a small net drain, so the rover must alternate between harvests and
      // efficient cruising to gain energy.
      float charge_gain = bright * p.solar_charge_rate * 0.20f;
      charge_gain += pulse * bright * 0.04f;  // extra pulse bonus

      // Drain: fighting the sluice forces and water drag, higher in bright zones.
      float sluice_drain = sluice * 0.015f * p.energy_drain_mul;
      float cruise_drain = (1.0f - bright) * (0.004f + speed_norm * 0.002f) * p.energy_drain_mul;
      float pulse_penalty = pulse * bright * 0.008f * p.energy_drain_mul;

      // Net: positive during bright zones (harvest), negative during dark zones (cruise).
      *c.energy_cost += (sluice_drain + cruise_drain + pulse_penalty - charge_gain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Watery, but with bright glinting patches on the surface that suggest solar
    // potential, hiding the sluice timing pattern. Dark enough to make lidar useful.
    v.ground = {65, 78, 72};  // dark greenish-brown mud
    v.particles = {170, 210, 180};  // bright glinting spray
    v.liquid = {90, 160, 140};  // shallow green water
    v.sky = {140, 180, 160};  // hazy daylight with bright patches
    v.particle_rate = 12.0f;
    v.particle_lift = 1.4f;
    v.particle_spread = 1.5f;
    v.base_particles = 3;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 16;
    v.ambient_drift = 3.0f;
    v.screen_brightness = 0.45f;  // moderate: lidar is cheap but the sluice pattern is dense and hidden
    v.liquid_surface = true;
    return v;
  }
};

class InverseTractionMirage final : public Biome {
 public:
  std::string_view id() const noexcept override { return "inverse_traction_mirage"; }
  std::string_view display_name() const noexcept override { return "Inverse Traction Mirage"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.35f + 0.15f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    p.ambient_temperature = -65.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 1.10f + 0.30f * biome_random01(s, 4);
    p.solar_charge_rate = 0.08f + 0.04f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.002f + 0.006f * biome_random01(s, 7);
    p.lidar_energy_mul = 4.00f + 1.50f * biome_random01(s, 8);
    p.lidar_range_mul = 0.15f + 0.10f * biome_random01(s, 9);
    // Terrain: moderate amplitude, moderate roughness, dense craters, almost no steps.
    // The mirage is in the dynamics, not the geometry.
    p.terrain_amplitude_mul = 1.20f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 1.10f + 0.30f * biome_random01(s, 11);
    p.terrain_crater_mul = 1.50f + 0.40f * biome_random01(s, 12);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Low baseline, but the inversion is the real signature.
    return p.friction_mul * 0.45f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from body effects.
      *c.wheel_force += c.contact->tangent * (-0.40f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Base drain is modest; the mirage exacts its cost in body effects.
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);

      // Mirage phase: a slow, position-dependent oscillation that alternates between
      // 'false grip' (appears grippy, but actually slides) and 'true grip' (appears
      // slick, but actually holds). The phase is invisible to lidar (expensive, very
      // short) and must be inferred from wheel slip and body drift.
      float phase = t * 0.005f + c.velocity.x * 0.013f;
      float mirage = 0.5f + 0.5f * std::sin(phase);  // 1 = false grip, 0 = true grip

      // A faster sub-oscillation creates 'mirage surges': brief windows where the
      // false grip is strongest but the actual grip is worst. These are the
      // dangerous moments — a naive 'drive carefully' policy will slide and stall.
      float surge_phase = t * 0.023f + c.velocity.x * 0.055f;
      float surge = 0.5f + 0.5f * std::sin(surge_phase);
      surge = surge * surge;  // narrow, strong surges

      // Effective grip: high during true grip phases, low during false grip phases.
      // But the surge inverts this further: during a surge in false grip, the rover
      // slides heavily; during a surge in true grip, the rover gets a brief traction
      // boost. This makes the strategy counter-intuitive: the rover must brake
      // during false grip phases (to avoid sliding) and accelerate during true grip
      // phases (to catch the boost), not the other way around.
      float grip_effective = 0.15f + 0.70f * (1.0f - mirage);
      grip_effective += surge * (mirage - 0.5f) * 0.20f;
      grip_effective = std::clamp(grip_effective, 0.10f, 0.90f);

      // False grip: strong lateral drift and a backward scrub that punishes throttle.
      // This is the part that flips a naive 'drive carefully' policy that just slows
      // down - it still slides.
      float lateral_drift = (1.0f - grip_effective) * std::sin(phase + 1.3f) * 0.09f * c.mass * (0.5f + speed_norm);
      float backward_scrub = (1.0f - grip_effective) * 0.07f * c.mass * c.gravity * (0.3f + 0.7f * speed_norm);

      // True grip surge: a forward assist that is strongest during the surge when
      // grip is high. This rewards accelerating during true grip surges, which is
      // the opposite of typical traction biomes.
      float surge_assist = surge * (1.0f - mirage) * 0.08f * c.mass * c.gravity * (0.4f + 0.6f * speed_norm);

      // False grip drag: a strong backward force that grows with speed and grip loss,
      // punishing throttle during false grip phases.
      float false_drag = mirage * (1.0f - surge * 0.5f) * 0.05f * c.mass * c.gravity * (0.5f + 0.5f * speed_norm);

      // Damping: lower during true grip (easier to slide forward), higher during false grip.
      float damping = 0.04f + 0.08f * (1.0f - grip_effective) + 0.02f * speed_norm;

      c.body_force->x += surge_assist - false_drag - backward_scrub + lateral_drift;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: false grip phases add a slight downward force (more apparent grip),
      // true grip phases add a tiny lift that reduces normal load but the actual grip
      // is better. This creates a paradox that the agent must learn to trust.
      float vertical = (mirage - 0.5f) * 0.02f * c.mass * c.gravity * std::sin(surge_phase + 0.4f);
      c.body_force->y += vertical - c.velocity.y * c.mass * (0.04f + 0.02f * (1.0f - grip_effective));

      if (c.body_torque) {
        // False grip causes yaw instability, especially at speed. True grip surges also
        // create brief stabilizing torques that reward proper throttle timing.
        float torque = (1.0f - grip_effective) * std::sin(phase + 1.8f) * 0.022f * c.mass * (0.5f + speed_norm);
        torque += surge * (1.0f - mirage) * std::sin(surge_phase * 0.7f + 0.5f) * 0.012f * c.mass * speed_norm;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);
      float phase = t * 0.005f + speed * 0.013f;
      float surge_phase = t * 0.023f + speed * 0.055f;
      float mirage = 0.5f + 0.5f * std::sin(phase);
      float surge = 0.5f + 0.5f * std::sin(surge_phase);
      surge = surge * surge;
      float grip_effective = 0.15f + 0.70f * (1.0f - mirage);
      grip_effective += surge * (mirage - 0.5f) * 0.20f;
      grip_effective = std::clamp(grip_effective, 0.10f, 0.90f);

      // Energy drain is much higher during false grip (fighting the slide) and during
      // true grip surges (accelerating). The efficient strategy is to brake during
      // false grip phases and accelerate during true grip phases.
      float false_grip_cost = mirage * (0.008f + 0.012f * speed_norm);
      float surge_cost = surge * (1.0f - mirage) * 0.008f * (1.0f + speed_norm);
      float throttle_penalty = speed * 0.002f * (1.0f + 0.5f * (1.0f - grip_effective));
      *c.energy_cost += (0.006f + false_grip_cost + surge_cost + throttle_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, shimmering rocky plain with faint heat-haze particles - suggests
    // instability but the mirage pattern is completely hidden.
    v.ground = {62, 58, 56};  // dark grey-brown rock
    v.particles = {150, 145, 140};  // shimmering dust
    v.liquid = {34, 32, 30};  // dark hollows
    v.sky = {88, 84, 80};  // dim overcast
    v.particle_rate = 4.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.2f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 10;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.08f;  // very dark: lidar is very expensive and short, forcing inference
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

class TractionTideProbe final : public Biome {
 public:
  std::string_view id() const noexcept override { return "traction_tide_probe"; }
  std::string_view display_name() const noexcept override { return "Traction Tide Probe"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Baseline friction is moderate, but the tide in body effects dominates.
    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.40f * biome_random01(s, 2);
    p.wind_force = 0.0f;
    // Cold, low thermal transfer, very low solar: the surface is dark and cold.
    p.ambient_temperature = -95.0f + 20.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 4);
    p.solar_charge_rate = 0.03f + 0.02f * biome_random01(s, 5);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 6);
    p.crust_deform = 0.001f + 0.003f * biome_random01(s, 7);
    // Lidar is expensive and very short-range: the tide must be inferred from motion.
    p.lidar_energy_mul = 4.50f + 1.50f * biome_random01(s, 8);
    p.lidar_range_mul = 0.12f + 0.08f * biome_random01(s, 9);
    // Terrain: smooth rolling plains with sparse craters and almost no steps,
    // so the tide is the real hazard, not the geometry.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 10);
    p.terrain_roughness_mul = 0.50f + 0.15f * biome_random01(s, 11);
    p.terrain_crater_mul = 0.60f + 0.25f * biome_random01(s, 12);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 13);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate baseline grip, but the tide inverts it below.
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Very slight, speed-independent sink into the cold regolith; not a trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.1f;
    }
    if (c.wheel_force && c.contact) {
      // Moderate rolling resistance; the real forces come from the body effect.
      *c.wheel_force += c.contact->tangent * (-0.40f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.02f);
    }
    if (c.energy_cost) {
      // Base drain is modest; the tide exacts its cost in body effects.
      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.08f);

      // Traction tide: a slow, position-dependent oscillation (0.005 rad/step)
      // that alternates between 'grippy' phases (tide=1) and 'slick' phases
      // (tide=0). The phase is invisible to lidar (expensive, very short) and
      // must be inferred from wheel slip and body sway.
      float phase = t * 0.005f + c.velocity.x * 0.013f;
      float tide = 0.5f + 0.5f * std::sin(phase);

      // A faster sub-oscillation creates 'reversal pulses': brief windows where the
      // effective grip flips sharply. In grippy phases these pulses give a strong
      // forward assist; in slick phases they cause a sudden loss of grip that
      // punishes throttle.
      float pulse_phase = t * 0.024f + c.velocity.x * 0.058f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      pulse = pulse * pulse;  // narrow, strong pulses

      // Effective grip: high during grippy phases (actual good grip), low during
      // slick phases (actual poor grip). But pulses invert this further: during
      // a pulse in slick phase, the rover slides heavily; during a pulse in
      // grippy phase, it gets a brief traction boost. This makes the strategy
      // counter-intuitive: the rover must ease off during slick phases (to
      // avoid the slide into the pulse) and accelerate during grippy phases
      // (to catch the boost), not the other way around.
      float grip_effective = 0.18f + 0.72f * tide;
      grip_effective += pulse * (tide - 0.5f) * 0.20f;
      grip_effective = std::clamp(grip_effective, 0.12f, 0.88f);

      // Slick phase: strong lateral drift and a backward scrub that punishes throttle.
      // This is the part that flips a naive 'drive carefully' policy that just slows down
      // - it still slides.
      float lateral_drift = (1.0f - grip_effective) * std::sin(phase + 1.4f) * 0.085f * c.mass * (0.5f + speed_norm);
      float backward_scrub = (1.0f - grip_effective) * 0.075f * c.mass * c.gravity * (0.3f + 0.7f * speed_norm);

      // Grippy phase pulse: a forward assist that is strongest during the pulse when
      // grip is high. This rewards accelerating during grippy-phase pulses.
      float surge_assist = pulse * tide * 0.075f * c.mass * c.gravity * (0.4f + 0.6f * speed_norm);

      // Slick-phase drag: a strong backward force that grows with speed and grip loss,
      // punishing throttle during slick phases.
      float slick_drag = (1.0f - tide) * (1.0f - pulse * 0.5f) * 0.055f * c.mass * c.gravity * (0.5f + 0.5f * speed_norm);

      // Damping: lower during grippy phases (easier to slide forward), higher during
      // slick phases (the substrate grips harder).
      float damping = 0.045f + 0.08f * (1.0f - grip_effective) + 0.02f * speed_norm;

      c.body_force->x += surge_assist - slick_drag - backward_scrub + lateral_drift;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: slick phases add a slight downward force (more apparent grip),
      // grippy phases add a tiny lift that reduces normal load but the actual grip
      // is better. This is the paradox the agent must learn to trust.
      float vertical = (tide - 0.5f) * 0.022f * c.mass * c.gravity * std::sin(pulse_phase + 0.4f);
      c.body_force->y += vertical - c.velocity.y * c.mass * (0.04f + 0.02f * (1.0f - grip_effective));

      if (c.body_torque) {
        // Slick phases cause yaw instability, especially at speed. Grippy-phase
        // pulses also create brief stabilizing torques that reward proper throttle timing.
        float torque = (1.0f - grip_effective) * std::sin(phase + 1.9f) * 0.024f * c.mass * (0.5f + speed_norm);
        torque += pulse * tide * std::sin(pulse_phase * 0.7f + 0.6f) * 0.013f * c.mass * speed_norm;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.02f;
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float phase = t * 0.005f + speed * 0.013f;
      float pulse_phase = t * 0.024f + speed * 0.058f;
      float tide = 0.5f + 0.5f * std::sin(phase);
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      pulse = pulse * pulse;
      float grip_effective = 0.18f + 0.72f * tide;
      grip_effective += pulse * (tide - 0.5f) * 0.20f;
      grip_effective = std::clamp(grip_effective, 0.12f, 0.88f);

      // Energy drain is much higher during slick phases (fighting the slide) and
      // during grippy-phase pulses (accelerating). The efficient strategy is to
      // ease off during slick phases and accelerate during grippy-phase pulses.
      float slick_cost = (1.0f - tide) * (0.009f + 0.012f * speed_norm);
      float surge_cost = pulse * tide * 0.009f * (1.0f + speed_norm);
      float throttle_penalty = speed * 0.002f * (1.0f + 0.5f * (1.0f - grip_effective));
      *c.energy_cost += (0.007f + slick_cost + surge_cost + throttle_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, rippled regolith with faint grey shimmer — suggests instability
    // but the reversal pattern is completely hidden.
    v.ground = {74, 62, 48};  // dark rippled sand
    v.particles = {150, 135, 110};  // fine pale dust
    v.liquid = {36, 30, 22};  // dark hollows
    v.sky = {96, 84, 66};  // dim overcast
    v.particle_rate = 8.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 1.0f;
    v.base_particles = 2;
    v.max_particles = 28;
    v.particle_size = 2;
    v.ambient_particles = 12;
    v.ambient_drift = 2.2f;
    v.screen_brightness = 0.10f;  // very dark: lidar is very expensive and short, forcing inference
    v.liquid_surface = false;
    return v;
  }
};

class MomentumDraftLocks final : public Biome {
 public:
  std::string_view id() const noexcept override { return "momentum_draft_locks"; }
  std::string_view display_name() const noexcept override { return "Momentum Draft Locks"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip but the fluid drafting and lock hysteresis dominate.
    p.friction_mul = 0.40f + 0.20f * biome_random01(s);
    p.sink_rate = 0.002f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.40f + 0.60f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    p.ambient_temperature = -55.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 3.0f + 1.0f * biome_random01(s, 5);
    p.solar_charge_rate = 0.10f + 0.08f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Expensive, short-range lidar: the lock phase must be felt, not seen.
    p.lidar_energy_mul = 4.0f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, deep basins with high amplitude and dense craters, but very few steps.
    // The hazard is the fluid momentum lock, not sharp geometry.
    p.terrain_amplitude_mul = 1.40f + 0.40f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.55f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.60f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip in fluid; the lock hysteresis creates the real struggle.
    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the fluid, more in deeper water.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      // Viscous drag plus a modest rolling resistance; the lock forces come from body effects.
      float depth = c.contact->penetration * 25.0f;
      float drag = (0.25f + p.viscosity * 2.0f * (1.0f + depth) * c.immersion + 0.15f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.10f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 25.0f;
      // Energy cost is higher in water (fighting drag) and at speed.
      *c.energy_cost += (0.008f + c.immersion * depth * 0.10f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Draft lock phase: a slow, position-dependent oscillation that determines
      // whether the fluid is in a 'locked' state (dense, resists acceleration but
      // releases a strong forward assist when momentum is already high) or an
      // 'unlocked' state (free-flowing, low resistance but no assist). The phase
      // is invisible to lidar (expensive, very short) and must be inferred from
      // the rover's acceleration response and the delayed surge that follows.
      float lock_phase = t * 0.0045f + c.velocity.x * 0.012f;
      float lock = 0.5f + 0.5f * std::sin(lock_phase);  // 1 = locked, 0 = unlocked

      // Hysteresis memory: builds with speed but lags behind, so accelerating
      // into a locked zone is punished more than steady cruising through it.
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Locked-phase drag: resists acceleration strongly, but only when the rover
      // is already moving (the fluid 'grabs' moving objects). At rest, the fluid
      // does not resist as much, giving the rover a chance to start moving.
      float lock_drag = lock * memory * (0.14f + 0.10f * speed_norm) * c.mass * c.gravity;

      // Delayed elastic release: once the rover builds enough momentum, the locked
      // fluid suddenly releases as a forward assist, carrying the rover efficiently
      // through the rest of the locked zone. This rewards building momentum in the
      // unlocked phase and sustaining it through the locked phase.
      float release = lock * memory * (0.05f + 0.08f * std::sin(lock_phase + 0.9f)) * c.mass * c.gravity;

      // Unlocked-phase assist: the free-flowing fluid gives a small, constant
      // forward push that makes it easy to build momentum here.
      float free_assist = (1.0f - lock) * 0.025f * c.mass * c.gravity;

      // Lateral 'draft': the fluid swirls with a delayed, memory-dependent oscillation.
      // In locked phases this is strong and can shove the rover off course; in
      // unlocked phases it is gentle.
      float draft = lock * std::sin(t * 0.024f + speed * 0.05f + memory * 2.0f) * 0.05f * c.mass * (0.4f + memory);

      // Damping is lower when momentum is high (momentum carries through) and
      // higher when speed is low (the fluid grips harder).
      float damping = 0.04f + 0.10f * lock * (1.0f - memory) + 0.015f * speed_norm;

      c.body_force->x -= lock_drag;
      c.body_force->x += release + free_assist + draft;
      c.body_force->x -= c.velocity.x * c.mass * damping;

      // Vertical: locked fluid provides slight buoyancy (reduces normal load and
      // traction), making control harder at speed. Unlocked fluid is denser and
      // gives better grip.
      c.body_force->y += lock * memory * 0.03f * c.mass * c.gravity * std::sin(t * 0.021f + 0.4f);
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.03f * (1.0f - lock));

      if (c.body_torque) {
        // Draft asymmetry induces a rocking torque that grows with memory and lock,
        // strongest during deceleration when the release surges.
        float decel = std::max(0.0f, 0.5f - speed_norm);
        float torque = lock * memory * (0.014f + 0.012f * decel) * std::sin(t * 0.033f + speed * 0.04f) * c.mass;
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (1.0f + 0.3f * lock);
      }
    }
    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);
      float lock_phase = t * 0.0045f + speed * 0.012f;
      float lock = 0.5f + 0.5f * std::sin(lock_phase);
      float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

      // Energy cost: peaks when fighting the lock (high memory, high lock) and
      // when accelerating from low speed in a locked zone. The efficient strategy
      // is to build momentum in unlocked zones and carry it through locked zones
      // without excessive throttle.
      float lock_cost = lock * memory * 0.014f;
      float start_cost = lock * (1.0f - memory) * (0.010f + 0.006f * speed_norm);
      float cruise_cost = speed * 0.0015f * (1.0f + 0.3f * lock);
      *c.energy_cost += (0.006f + lock_cost + start_cost + cruise_cost) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, deep liquid with swirling blue-grey particles - suggests fluid
    // but not the lock pattern itself.
    v.ground = {58, 66, 78};  // dark blue-grey mud
    v.particles = {130, 190, 220};  // swirling light-blue spray
    v.liquid = {40, 90, 130};  // deep blue liquid
    v.sky = {90, 120, 150};  // dim overcast
    v.particle_rate = 11.0f;
    v.particle_lift = 1.8f;
    v.particle_spread = 1.6f;
    v.base_particles = 3;
    v.max_particles = 40;
    v.particle_size = 2;
    v.ambient_particles = 14;
    v.ambient_drift = 2.5f;
    v.screen_brightness = 0.10f;  // very dark: lidar is expensive and short, so the lock phase must be inferred from motion
    v.liquid_surface = true;
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

class QuasarSiphonVein final : public Biome {
 public:
  std::string_view id() const noexcept override { return "quasar_siphon_vein"; }
  std::string_view display_name() const noexcept override { return "Quasar Siphon Vein"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate grip but the gravity siphons dominate the challenge.
    p.friction_mul = 0.50f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.60f + 0.50f * biome_random01(s, 2);
    p.wind_force = 0.5f + 1.0f * biome_random01(s, 3);
    // Cold and nearly dark: solar is negligible, forcing strict energy budgeting.
    p.ambient_temperature = -80.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.0f + 0.6f * biome_random01(s, 5);
    p.solar_charge_rate = 0.02f + 0.02f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.005f * biome_random01(s, 8);
    // Expensive, very short-range lidar: the siphon veins are invisible ahead.
    p.lidar_energy_mul = 5.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain reshaped into rough, vein-like ridges with dense small craters and steps,
    // so a generic "smooth low-gravity" policy cannot coast through.
    p.terrain_amplitude_mul = 1.30f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.60f + 0.40f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.70f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 1.30f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Baseline grip is decent; the gravity siphons alter effective load and traction.
    return p.friction_mul * 0.85f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Tiny sink into the fractured vein-rock; not a traction trap.
      c.contact->penetration += p.sink_rate * c.dt * 0.1f;
    }
    if (c.wheel_force && c.contact) {
      // Light rolling resistance; the siphon forces dominate in body effects.
      *c.wheel_force += c.contact->tangent * (-0.35f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.015f);
    }
    if (c.energy_cost) {
      // Base drain plus a small cold-soak thermal load.
      float cold_load = std::max(0.0f, -p.ambient_temperature) * 0.002f * p.thermal_transfer;
      *c.energy_cost += (0.006f + cold_load + std::abs(c.wheel_speed) * 0.002f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.09f);

      // Siphon phase: a slow, position-dependent oscillation that alternates
      // between 'light' (gravity ~0.4x, buoyant, hard to keep traction) and
      // 'heavy' (gravity ~1.6x, crushing, high traction but high energy).
      // The phase is invisible to lidar (very short, expensive) and must be
      // inferred from suspension compression, body acceleration, and energy drain.
      float siphon_phase = t * 0.0045f + c.velocity.x * 0.012f;
      float siphon = 0.5f + 0.5f * std::sin(siphon_phase); // 0 = light, 1 = heavy

      // A faster sub-oscillation creates 'siphon pulses': brief, extreme gravity
      // spikes within the heavy phase that are the primary flip/stall hazard. A naive
      // policy that merely drives forward carefully will hit a pulse while heavy and
      // either pitch over on the rough terrain or drain its battery catastrophically.
      float pulse_phase = t * 0.023f + c.velocity.x * 0.058f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse; // narrow, strong peaks

      // Effective gravity multiplier: light ~0.35, heavy ~1.45, pulses push to ~1.9.
      float grav_mul = 0.35f + 1.10f * siphon + 0.45f * siphon * pulse_narrow;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      // Weight force in heavy phase: presses the rover down, increasing normal load
      // and traction but also rolling resistance. In light phase, buoyancy lifts the
      // rover, reducing traction and making it easy to skate but hard to brake.
      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.22f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.18f;

      // Lateral 'siphon drift': in the light phase the rover drifts sideways (very
      // low normal load -> little grip), while in the heavy phase it is stable but
      // sluggish. Stronger at speed.
      float lateral_drift = std::sin(siphon_phase + 1.2f) * (1.0f - siphon) * 0.07f * c.mass * (0.5f + speed_norm);

      // Damping: heavy phase has higher damping (more grip), light phase lower
      // (easy to coast but hard to steer). Rewards timing acceleration with the
      // heavy phase and coasting through the light phase.
      float damping = 0.04f + 0.10f * siphon + 0.02f * speed_norm;

      // Siphon pulse instability: during the narrow gravity spikes, a strong
      // pitching torque slams the nose down or up depending on the phase. If the
      // rover is moving fast through a pulse on the rough terrain, this can flip it.
      // A universal 'drive carefully' policy that does not actively brake before
      // pulses will be caught.
      float pulse_torque = siphon * pulse_narrow * (0.5f + 0.5f * speed_norm) * 0.030f * c.mass * c.gravity * std::sin(pulse_phase + 0.7f);

      c.body_force->x += lateral_drift - c.velocity.x * c.mass * damping;
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.04f + 0.03f * siphon);

      if (c.body_torque) {
        *c.body_torque += pulse_torque - c.angular_velocity * c.mass * 0.025f;
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);
      float siphon_phase = t * 0.0045f + speed * 0.012f;
      float siphon = 0.5f + 0.5f * std::sin(siphon_phase);
      float pulse_phase = t * 0.023f + speed * 0.058f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;
      float grav_mul = 0.35f + 1.10f * siphon + 0.45f * siphon * pulse_narrow;

      // Energy budget cascade: the drain is multiplicatively scaled by the current
      // gravity phase. Driving through the heavy phase consumes energy roughly six
      // times faster than in the light phase. On top of that, a 'siphon surge' cost
      // spikes during the rapid gravity pulses, punishing high throttle at exactly
      // the moment the torque is most dangerous.
      //
      // The successful strategy must learn to build momentum in the light phase
      // (cheap, but low traction so coasting is safer), then ride that momentum
      // through the heavy phase with only gentle throttle corrections, and brake
      // carefully before each pulse to avoid the pitching torque.
      float phase_cost = 0.008f + 0.045f * siphon;
      float pulse_delta = pulse_narrow * siphon;
      float surge_cost = pulse_delta * (0.006f + 0.014f * speed_norm);

      // Throttle penalty: accelerating while in the heavy phase or during a pulse
      // is extremely wasteful (fighting the extra weight), so the policy must learn
      // to conserve throttle there.
      float throttle_penalty = grav_mul * speed_norm * 0.006f;

      // Light-phase coasting bonus: energy cost is lower when moving steadily in
      // the light phase, rewarding a smooth, momentum-based traversal.
      float coast_bonus = (1.0f - siphon) * speed_norm * 0.004f;

      *c.energy_cost += (phase_cost + surge_cost + throttle_penalty - coast_bonus) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dull, uniform dark grey rock with faint violet tint — visually suggests a
    // low-gravity environment but reveals nothing about the siphon pattern. The sky
    // is dark and hazy; screen_brightness is very low so that lidar is prohibitively
    // expensive (and useless at this range) — the agent must rely entirely on
    // proprioception and trial-and-error to identify the phase.
    v.ground = {62, 58, 70};
    v.particles = {145, 135, 160};
    v.liquid = {38, 34, 44};
    v.sky = {85, 80, 95};
    v.particle_rate = 4.0f;
    v.particle_lift = 1.6f;
    v.particle_spread = 0.8f;
    v.base_particles = 0;
    v.max_particles = 16;
    v.particle_size = 2;
    v.ambient_particles = 6;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.08f; // near-total darkness: lidar is very expensive and short-range, so the siphon must be inferred from motion
    v.liquid_surface = false;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const GranularThrottleTrap biome_0; out.push_back(&biome_0);
  static const CantileverGale biome_1; out.push_back(&biome_1);
  static const BarometricBrakeWells biome_2; out.push_back(&biome_2);
  static const GravityPendulumHollow biome_3; out.push_back(&biome_3);
  static const ThermalSiphonReactor biome_4; out.push_back(&biome_4);
  static const AvalancheProbe biome_5; out.push_back(&biome_5);
  static const StrobePitfall biome_6; out.push_back(&biome_6);
  static const StroboscopicCrust biome_7; out.push_back(&biome_7);
  static const AccordionGravityLadder biome_8; out.push_back(&biome_8);
  static const RuttedThrottleBasin biome_9; out.push_back(&biome_9);
  static const PendulumHollow biome_10; out.push_back(&biome_10);
  static const FlywheelBrakeLag biome_11; out.push_back(&biome_11);
  static const ThermalBatteryCascade biome_12; out.push_back(&biome_12);
  static const HystereticRotorField biome_13; out.push_back(&biome_13);
  static const RotorRhythmField biome_14; out.push_back(&biome_14);
  static const ThermalCrosswindLull biome_15; out.push_back(&biome_15);
  static const TidalBrakeVault biome_16; out.push_back(&biome_16);
  static const SwayingLidarPulse biome_17; out.push_back(&biome_17);
  static const SolarSluiceHarvest biome_18; out.push_back(&biome_18);
  static const InverseTractionMirage biome_19; out.push_back(&biome_19);
  static const LullAndThermalScrub biome_20; out.push_back(&biome_20);
  static const TractionTideProbe biome_21; out.push_back(&biome_21);
  static const MomentumDraftLocks biome_22; out.push_back(&biome_22);
  static const GravityWellBrine biome_23; out.push_back(&biome_23);
  static const QuasarSiphonVein biome_24; out.push_back(&biome_24);
}
// </MARS_GENERATED_BIOMES>
}  // namespace generated_biomes

inline constexpr std::string_view kBiomeBankVersion = "sha256:4d0c88455b87deeef0b0664d4fcb48c6622828ce6a9be6e680f18e9e9ab8e141";

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
