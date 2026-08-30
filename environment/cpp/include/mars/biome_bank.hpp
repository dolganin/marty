#pragma once
















#include <cmath>
#include <array>
#include <chrono>
#include <ctime>
#include <cstdint>
#include <string_view>
#include <vector>

#include "mars/mechanics.hpp"

namespace mars {

enum class BiomeSplit : int { Builtin = 0, Train = 1, Test = 2 };

// Frozen, reviewed combinations.  The split denotes the combination, rather
// than hiding individual mechanisms: held-out evaluation sees familiar pieces
// in an unseen ordered stack.
struct FrozenMechanismStack {
  std::array<MechanicType, 4> types{};
  int count = 0;
  BiomeSplit split = BiomeSplit::Builtin;
};

inline constexpr std::array<FrozenMechanismStack, 6> kFrozenMechanismStacks{{
    {{MechanicType::Normal, MechanicType::Wind, MechanicType::Normal, MechanicType::Normal}, 2, BiomeSplit::Builtin},
    {{MechanicType::Sand, MechanicType::Wind, MechanicType::Normal, MechanicType::Normal}, 2, BiomeSplit::Train},
    {{MechanicType::Mud, MechanicType::Crust, MechanicType::Wind, MechanicType::Normal}, 3, BiomeSplit::Train},
    {{MechanicType::Ice, MechanicType::LowGravity, MechanicType::Normal, MechanicType::Normal}, 2, BiomeSplit::Train},
    {{MechanicType::Liquid, MechanicType::Wind, MechanicType::Normal, MechanicType::Normal}, 2, BiomeSplit::Test},
    {{MechanicType::Sand, MechanicType::Mud, MechanicType::Wind, MechanicType::Crust}, 4, BiomeSplit::Test},
}};

enum class CouplingInput : int { Moisture, Viscosity, Heat, TirePressure, Slip, Speed };
enum class CouplingTarget : int { Traction, Heat, TirePressure };

struct FrozenCouplingRule {
  std::array<CouplingInput, 2> inputs{};
  std::array<float, 2> coefficients{};
  int input_count = 0;
  CouplingTarget target = CouplingTarget::Traction;
  float bias = 0.0f;
  float lower = 0.0f;
  float upper = 1.0f;
};

// Mirror of coupling_rules in biome_bank.json.  Bounds are part of the frozen
// binary contract, so a manifest cannot introduce an unbounded formula.
inline constexpr std::array<FrozenCouplingRule, 3> kFrozenCouplingRules{{
    {{CouplingInput::Moisture, CouplingInput::Viscosity}, {-0.62f, -0.16f}, 2,
      CouplingTarget::Traction, 1.18f, 0.12f, 1.45f},
    {{CouplingInput::Slip, CouplingInput::Speed}, {0.14f, 0.02f}, 2,
      CouplingTarget::Heat, 0.18f, 0.0f, 1.0f},
    {{CouplingInput::Moisture, CouplingInput::Heat}, {-0.035f, 0.02f}, 2,
      CouplingTarget::TirePressure, 1.0f, 0.65f, 1.25f},
}};

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



  virtual float terrain_height_delta(float local_x, uint64_t seed) const noexcept {
    (void)local_x;
    (void)seed;
    return 0.0f;
  }





  virtual int hazard_at(int step) const noexcept { (void)step; return 0; }
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


















namespace handcrafted_biomes {




class MoltenWindowBiome : public Biome {
 public:
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }

  virtual int cycle_steps() const noexcept = 0;
  virtual int melt_steps() const noexcept = 0;
  virtual int phase_offset() const noexcept { return 0; }

  int hazard_at(int step) const noexcept override { return molten(step) ? 1 : 0; }

  bool molten(int step) const noexcept {
    const int cycle = cycle_steps();
    int phase = (step + phase_offset()) % cycle;
    if (phase < 0) phase += cycle;
    return phase < melt_steps();
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.95f + 0.10f * biome_random01(s);
    p.crust_deform = 0.010f + 0.006f * biome_random01(s, 1);
    p.ambient_temperature = 5.0f + 12.0f * biome_random01(s, 2);
    p.thermal_transfer = 1.1f + 0.3f * biome_random01(s, 3);
    p.solar_charge_rate = 1.4f + 0.4f * biome_random01(s, 4);

    p.lidar_range_mul = 0.35f;
    p.lidar_energy_mul = 2.0f;
    p.terrain_amplitude_mul = 0.8f;
    p.terrain_step_mul = 0.6f;
    return p;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active) return;
    if (!molten(c.step_index)) return;
    const float speed = std::abs(c.wheel_speed);
    const float excess = speed - kCreepSpeed;
    if (excess <= 0.0f) return;


    const float severity = clamp(excess * 1.6f, 0.0f, 6.0f);
    c.contact->penetration += (0.30f + 0.55f * severity) * c.dt;
    if (c.wheel_force) {
      *c.wheel_force += c.contact->tangent * (-9.0f * c.wheel_speed * (1.0f + severity));
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.9f + 0.5f * severity));
    }
    if (c.energy_cost)
      *c.energy_cost += (0.10f + 0.16f * severity) * p.energy_drain_mul * c.dt;
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!molten(c.step_index)) return;
    const float speed = std::abs(c.velocity.x);
    if (speed <= kCreepSpeed) return;
    const float severity = clamp((speed - kCreepSpeed) * 1.4f, 0.0f, 5.0f);
    if (c.body_force) c.body_force->y -= c.mass * std::abs(c.gravity) * (0.55f + 0.30f * severity);






  }

 protected:
  static constexpr float kCreepSpeed = 0.55f;
};


class CollapseWindowFlats final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_flats"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Flats"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  int cycle_steps() const noexcept override { return 480; }
  int melt_steps() const noexcept override { return 110; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {196, 126, 88};
    v.ground = {150, 84, 52};
    v.particles = {224, 146, 92};
    v.particle_rate = 9.0f;
    v.screen_brightness = 0.95f;
    return v;
  }
};



class CollapseWindowTerrace final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_terrace"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Terrace"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  int cycle_steps() const noexcept override { return 700; }
  int melt_steps() const noexcept override { return 140; }
  int phase_offset() const noexcept override { return 350; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {172, 138, 108};
    v.ground = {118, 96, 74};
    v.particles = {206, 178, 138};
    v.particle_rate = 7.0f;
    v.screen_brightness = 0.90f;
    return v;
  }
};



class SetpointRimeShelf final : public Biome {
 public:
  std::string_view id() const noexcept override { return "setpoint_rime_shelf"; }
  std::string_view display_name() const noexcept override { return "Setpoint Rime Shelf"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }

  int hazard_at(int) const noexcept override { return 2; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.85f + 0.15f * biome_random01(s);
    p.ambient_temperature = -70.0f + 10.0f * biome_random01(s, 1);
    p.thermal_transfer = 1.6f + 0.4f * biome_random01(s, 2);
    p.solar_charge_rate = 0.9f + 0.3f * biome_random01(s, 3);
    p.energy_drain_mul = 1.05f + 0.15f * biome_random01(s, 4);
    p.lidar_range_mul = 0.40f;
    p.lidar_energy_mul = 1.8f;
    p.terrain_amplitude_mul = 0.9f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {148, 176, 200};
    v.ground = {166, 186, 198};
    v.particles = {226, 238, 246};
    v.particle_rate = 6.0f;
    v.ambient_particles = 40;
    v.ambient_drift = 2.0f;
    v.screen_brightness = 1.0f;
    return v;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active) return;
    const float speed = std::abs(c.wheel_speed);
    if (speed >= kReleaseSpeed) return;


    const float bite = clamp((kReleaseSpeed - speed) / kReleaseSpeed, 0.0f, 1.0f);
    c.contact->penetration += 0.16f * bite * c.dt;
    if (c.wheel_force) *c.wheel_force += c.contact->tangent * (-14.0f * bite * c.wheel_speed);
    if (c.energy_cost) *c.energy_cost += 0.09f * bite * p.energy_drain_mul * c.dt;
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    const float speed = std::abs(c.velocity.x);
    if (speed >= kReleaseSpeed) return;
    const float bite = clamp((kReleaseSpeed - speed) / kReleaseSpeed, 0.0f, 1.0f);
    if (c.body_force) c.body_force->x -= c.mass * 2.2f * bite * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
    if (c.energy_cost) *c.energy_cost += 0.05f * bite * c.dt;
  }

 private:
  static constexpr float kReleaseSpeed = 0.85f;
};



class CollapseWindowGulch final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_gulch"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Gulch"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  int cycle_steps() const noexcept override { return 600; }
  int melt_steps() const noexcept override { return 130; }
  int phase_offset() const noexcept override { return 200; }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {182, 150, 96};
    v.ground = {126, 104, 60};
    v.particles = {214, 190, 130};
    v.particle_rate = 8.0f;
    v.screen_brightness = 0.92f;
    return v;
  }
};










class CollapseWindowScarp final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_scarp"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Scarp"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  int cycle_steps() const noexcept override { return 560; }
  int melt_steps() const noexcept override { return 125; }





  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p = MoltenWindowBiome::sample_params(s);
    p.friction_mul = 0.72f + 0.10f * biome_random01(s, 11);
    p.gravity_mul = 1.42f + 0.12f * biome_random01(s, 12);
    p.ambient_temperature = -34.0f + 10.0f * biome_random01(s, 13);
    p.energy_drain_mul = 1.30f + 0.20f * biome_random01(s, 14);
    p.solar_charge_rate = 0.55f + 0.20f * biome_random01(s, 15);
    p.terrain_amplitude_mul = 1.62f;
    p.terrain_step_mul = 2.10f;
    p.terrain_roughness_mul = 1.35f;
    return p;
  }
  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    MoltenWindowBiome::apply_body_effects(p, c);

    if (c.body_force) c.body_force->x -= c.mass * 0.85f * clamp(c.velocity.x, 0.0f, 3.0f);
  }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {186, 118, 96};
    v.ground = {138, 78, 64};
    v.particles = {218, 152, 118};
    v.particle_rate = 9.0f;
    v.screen_brightness = 0.93f;
    return v;
  }
};

class CollapseWindowPlaya final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_playa"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Playa"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  int cycle_steps() const noexcept override { return 820; }
  int melt_steps() const noexcept override { return 175; }
  int phase_offset() const noexcept override { return 410; }
  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p = MoltenWindowBiome::sample_params(s);
    p.friction_mul = 1.22f + 0.14f * biome_random01(s, 11);
    p.gravity_mul = 0.74f + 0.08f * biome_random01(s, 12);
    p.viscosity = 0.45f + 0.20f * biome_random01(s, 13);
    p.ambient_temperature = 26.0f + 12.0f * biome_random01(s, 14);
    p.thermal_transfer = 2.4f + 0.5f * biome_random01(s, 15);
    p.solar_charge_rate = 2.3f + 0.4f * biome_random01(s, 16);
    p.terrain_amplitude_mul = 0.62f;
    p.terrain_crater_mul = 2.10f;
    p.terrain_step_mul = 0.45f;
    return p;
  }
  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    MoltenWindowBiome::apply_effects(p, c);

    if (c.wheel_force && c.contact && c.contact->active)
      *c.wheel_force += c.contact->tangent * (-p.viscosity * 2.8f * c.wheel_speed);
  }
  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {166, 146, 122};
    v.ground = {132, 118, 96};
    v.particles = {212, 196, 164};
    v.particle_rate = 6.0f;
    v.screen_brightness = 0.90f;
    return v;
  }
};



class SpeedBandTalus final : public Biome {
 public:
  std::string_view id() const noexcept override { return "speed_band_talus"; }
  std::string_view display_name() const noexcept override { return "Speed Band Talus"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.90f + 0.12f * biome_random01(s);
    p.crust_deform = 0.012f + 0.006f * biome_random01(s, 1);
    p.ambient_temperature = -20.0f + 15.0f * biome_random01(s, 2);
    p.thermal_transfer = 1.0f + 0.3f * biome_random01(s, 3);
    p.solar_charge_rate = 1.2f + 0.4f * biome_random01(s, 4);
    p.lidar_range_mul = 0.40f;
    p.lidar_energy_mul = 2.2f;
    p.terrain_roughness_mul = 1.3f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {176, 132, 104};
    v.ground = {122, 96, 84};
    v.particles = {198, 168, 148};
    v.particle_rate = 8.0f;
    v.screen_brightness = 0.95f;
    return v;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active) return;
    const float speed = std::abs(c.wheel_speed);
    if (speed > kCeiling) {

      const float over = clamp((speed - kCeiling) * 1.5f, 0.0f, 5.0f);
      c.contact->penetration += (0.22f + 0.40f * over) * c.dt;
      if (c.wheel_force) {
        *c.wheel_force += c.contact->tangent * (-7.5f * c.wheel_speed * (1.0f + over));
        *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.6f + 0.4f * over));
      }
      if (c.energy_cost) *c.energy_cost += (0.08f + 0.12f * over) * p.energy_drain_mul * c.dt;
    } else if (speed < kFloor) {

      const float bite = clamp((kFloor - speed) / kFloor, 0.0f, 1.0f);
      c.contact->penetration += 0.14f * bite * c.dt;
      if (c.wheel_force) *c.wheel_force += c.contact->tangent * (-11.0f * bite * c.wheel_speed);
      if (c.energy_cost) *c.energy_cost += 0.07f * bite * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    const float speed = std::abs(c.velocity.x);
    if (speed <= kCeiling) return;
    const float over = clamp((speed - kCeiling) * 1.3f, 0.0f, 4.0f);
    if (c.body_force) c.body_force->y -= c.mass * std::abs(c.gravity) * (0.35f + 0.25f * over);
    if (c.body_torque) *c.body_torque -= c.mass * (0.6f + 0.5f * over);
  }

 private:
  static constexpr float kFloor = 0.75f;
  static constexpr float kCeiling = 2.30f;
};








class SlipPhasePan final : public Biome {
 public:
  std::string_view id() const noexcept override { return "slip_phase_pan"; }
  std::string_view display_name() const noexcept override { return "Slip Phase Pan"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Crust; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

  bool slipping(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSlipSteps;
  }
  int hazard_at(int step) const noexcept override { return slipping(step) ? 1 : 0; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 1.15f + 0.20f * biome_random01(s);
    p.ambient_temperature = -8.0f + 16.0f * biome_random01(s, 1);
    p.thermal_transfer = 0.65f + 0.25f * biome_random01(s, 2);
    p.solar_charge_rate = 1.7f + 0.5f * biome_random01(s, 3);
    p.energy_drain_mul = 1.05f + 0.15f * biome_random01(s, 4);
    p.lidar_range_mul = 0.36f;
    p.lidar_energy_mul = 2.1f;
    p.terrain_amplitude_mul = 1.15f;
    p.terrain_roughness_mul = 1.4f;
    p.crust_deform = 0.014f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {170, 150, 130};
    v.ground = {146, 134, 118};
    v.particles = {214, 202, 182};
    v.particle_rate = 6.0f;
    v.ambient_particles = 28;
    v.ambient_drift = 1.8f;
    v.screen_brightness = 0.98f;
    return v;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    return p.friction_mul;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active || !slipping(c.step_index)) return;
    const float effort = std::abs(c.drive_force);
    if (effort <= 1.0f) return;


    const float waste = clamp(effort / 60.0f, 0.0f, 4.0f);
    c.contact->slip = clamp(c.contact->slip + 0.55f + 0.15f * waste, 0.0f, 1.0f);
    if (c.wheel_force) {
      *c.wheel_force += c.contact->tangent * (-c.drive_force * 0.92f);
      *c.wheel_force += c.contact->tangent * (-2.6f * waste);
    }
    if (c.energy_cost) *c.energy_cost += (0.06f + 0.11f * waste) * p.energy_drain_mul * c.dt;
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!slipping(c.step_index)) return;
    const float speed = std::abs(c.velocity.x);
    if (speed <= 0.35f) return;
    const float slide = clamp(speed * 0.8f, 0.0f, 3.0f);
    if (c.body_force) c.body_force->x -= c.mass * 1.6f * slide;
    if (c.energy_cost) *c.energy_cost += 0.04f * slide * c.dt;
  }

 private:
  static constexpr int kCycle = 640;
  static constexpr int kSlipSteps = 150;
  static constexpr int kPhase = 90;
};














class SolarWindowPan final : public Biome {
 public:
  std::string_view id() const noexcept override { return "solar_window_pan"; }
  std::string_view display_name() const noexcept override { return "Solar Window Pan"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

  bool clear_sky(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kWindowSteps;
  }




  int hazard_at(int step) const noexcept override { return clear_sky(step) ? 3 : 2; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 1.02f + 0.12f * biome_random01(s);
    p.gravity_mul = 0.92f + 0.08f * biome_random01(s, 1);
    p.ambient_temperature = 16.0f + 12.0f * biome_random01(s, 2);
    p.thermal_transfer = 1.45f + 0.35f * biome_random01(s, 3);

    p.solar_charge_rate = 0.18f + 0.10f * biome_random01(s, 4);
    p.energy_drain_mul = 1.35f + 0.20f * biome_random01(s, 5);
    p.lidar_range_mul = 0.32f;
    p.lidar_energy_mul = 2.4f;
    p.terrain_amplitude_mul = 0.72f;
    p.terrain_crater_mul = 1.65f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {204, 158, 98};
    v.ground = {152, 118, 70};
    v.particles = {230, 192, 128};
    v.particle_rate = 7.0f;
    v.ambient_particles = 24;
    v.ambient_drift = 2.2f;
    v.screen_brightness = 0.96f;
    return v;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.energy_cost) return;
    const float speed = std::abs(c.wheel_speed);
    if (clear_sky(c.step_index) && speed <= kStillThreshold) {

      *c.energy_cost -= kChargePerSecond * c.dt;
      return;
    }
    if (speed > kStillThreshold)
      *c.energy_cost += kDriveTax * p.energy_drain_mul * c.dt;
  }

 private:
  static constexpr int kCycle = 740;
  static constexpr int kWindowSteps = 210;
  static constexpr int kPhase = 300;
  static constexpr float kStillThreshold = 0.18f;
  static constexpr float kChargePerSecond = 2.40f;
  static constexpr float kDriveTax = 0.30f;
};





class PulseGravityReef final : public Biome {
 public:
  std::string_view id() const noexcept override { return "pulse_gravity_reef"; }
  std::string_view display_name() const noexcept override { return "Pulse Gravity Reef"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }

  bool pulsing(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kPulseSteps;
  }
  int hazard_at(int step) const noexcept override { return pulsing(step) ? 1 : 0; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.68f + 0.12f * biome_random01(s);
    p.gravity_mul = 0.55f + 0.10f * biome_random01(s, 1);
    p.ambient_temperature = -52.0f + 12.0f * biome_random01(s, 2);
    p.thermal_transfer = 0.75f + 0.20f * biome_random01(s, 3);
    p.solar_charge_rate = 1.35f + 0.35f * biome_random01(s, 4);
    p.energy_drain_mul = 0.95f + 0.15f * biome_random01(s, 5);
    p.lidar_range_mul = 0.30f;
    p.lidar_energy_mul = 2.6f;
    p.terrain_amplitude_mul = 1.30f;
    p.terrain_crater_mul = 1.80f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {132, 128, 176};
    v.ground = {104, 100, 138};
    v.particles = {186, 182, 222};
    v.particle_rate = 5.0f;
    v.particle_lift = 2.6f;
    v.screen_brightness = 0.94f;
    return v;
  }

  void apply_effects(const MechanicParams&, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active || !pulsing(c.step_index)) return;

    if (c.wheel_force)
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.72f);
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!pulsing(c.step_index)) return;
    const float speed = std::abs(c.velocity.x);
    if (speed <= kSafeSpeed) return;
    const float over = clamp((speed - kSafeSpeed) * 1.4f, 0.0f, 4.0f);
    if (c.body_force) c.body_force->y += c.mass * std::abs(c.gravity) * (0.30f + 0.22f * over);
    if (c.body_torque) *c.body_torque += c.mass * (1.1f + 0.9f * over);
  }

 private:
  static constexpr int kCycle = 690;
  static constexpr int kPulseSteps = 165;
  static constexpr int kPhase = 120;
  static constexpr float kSafeSpeed = 0.70f;
};





class CadenceDuneBelt final : public Biome {
 public:
  std::string_view id() const noexcept override { return "cadence_dune_belt"; }
  std::string_view display_name() const noexcept override { return "Cadence Dune Belt"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Sand; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

  int phase_of(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    return phase < 0 ? phase + kCycle : phase;
  }
  bool must_hold(int step) const noexcept { return phase_of(step) < kHoldSteps; }
  bool must_move(int step) const noexcept {
    const int phase = phase_of(step);
    return phase >= kHoldSteps + kFreeSteps && phase < kHoldSteps + kFreeSteps + kMoveSteps;
  }
  int hazard_at(int step) const noexcept override {
    return must_hold(step) ? 1 : (must_move(step) ? 2 : 0);
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 0.58f + 0.12f * biome_random01(s);
    p.sink_rate = 0.042f + 0.014f * biome_random01(s, 1);
    p.ambient_temperature = 34.0f + 16.0f * biome_random01(s, 2);
    p.thermal_transfer = 2.10f + 0.40f * biome_random01(s, 3);
    p.solar_charge_rate = 1.05f + 0.35f * biome_random01(s, 4);
    p.energy_drain_mul = 1.20f + 0.20f * biome_random01(s, 5);
    p.lidar_range_mul = 0.34f;
    p.lidar_energy_mul = 2.3f;
    p.gravity_mul = 1.18f + 0.10f * biome_random01(s, 6);
    p.wind_force = 2.6f + 0.8f * biome_random01(s, 7);
    p.terrain_amplitude_mul = 1.20f;
    p.terrain_roughness_mul = 1.90f;
    p.terrain_crater_mul = 0.45f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {206, 164, 112};
    v.ground = {176, 140, 88};
    v.particles = {232, 200, 146};
    v.particle_rate = 11.0f;
    v.ambient_particles = 48;
    v.ambient_drift = 2.8f;
    v.screen_brightness = 0.97f;
    return v;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.contact || !c.contact->active) return;
    const float speed = std::abs(c.wheel_speed);
    if (must_hold(c.step_index)) {
      if (speed <= kCreep) return;
      const float over = clamp((speed - kCreep) * 1.5f, 0.0f, 5.0f);
      c.contact->penetration += (0.24f + 0.42f * over) * c.dt;
      if (c.wheel_force) {
        *c.wheel_force += c.contact->tangent * (-8.0f * c.wheel_speed * (1.0f + over));
        *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.7f + 0.4f * over));
      }
      if (c.energy_cost) *c.energy_cost += (0.09f + 0.13f * over) * p.energy_drain_mul * c.dt;
    } else if (must_move(c.step_index)) {
      if (speed >= kRelease) return;
      const float bite = clamp((kRelease - speed) / kRelease, 0.0f, 1.0f);
      c.contact->penetration += 0.15f * bite * c.dt;
      if (c.wheel_force) *c.wheel_force += c.contact->tangent * (-12.0f * bite * c.wheel_speed);
      if (c.energy_cost) *c.energy_cost += 0.08f * bite * p.energy_drain_mul * c.dt;
    }
  }

 private:
  static constexpr int kCycle = 900;
  static constexpr int kPhase = 310;
  static constexpr int kHoldSteps = 170;
  static constexpr int kFreeSteps = 260;
  static constexpr int kMoveSteps = 200;
  static constexpr float kCreep = 0.55f;
  static constexpr float kRelease = 0.85f;
};






class CommitmentLedgeField final : public Biome {
 public:
  std::string_view id() const noexcept override { return "commitment_ledge_field"; }
  std::string_view display_name() const noexcept override { return "Commitment Ledge Field"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }


  int hazard_at(int) const noexcept override { return 2; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 1.15f + 0.10f * biome_random01(s);
    p.gravity_mul = 0.92f + 0.06f * biome_random01(s, 1);
    p.energy_drain_mul = 0.85f + 0.10f * biome_random01(s, 2);
    p.ambient_temperature = -42.0f + 8.0f * biome_random01(s, 3);
    p.thermal_transfer = 0.9f + 0.2f * biome_random01(s, 4);
    p.solar_charge_rate = 1.2f + 0.2f * biome_random01(s, 5);
    p.lidar_energy_mul = 0.30f;
    p.lidar_range_mul = 1.0f;
    p.terrain_amplitude_mul = 0.55f;
    p.terrain_roughness_mul = 0.55f;
    p.terrain_crater_mul = 0.30f;
    p.terrain_step_mul = 0.30f;
    p.ledge_start_x = 18.0f;
    p.ledge_spacing = 25.0f;
    p.ledge_gap_width = 3.2f + 0.4f * biome_random01(s, 6);
    p.ledge_ramp_length = 2.8f;
    p.ledge_ramp_height = 0.72f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {152, 112, 92};
    v.ground = {102, 82, 72};
    v.particles = {188, 154, 128};
    v.particle_rate = 5.0f;
    v.screen_brightness = 0.95f;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const CollapseWindowFlats collapse_flats; out.push_back(&collapse_flats);
  static const CollapseWindowGulch collapse_gulch; out.push_back(&collapse_gulch);
  static const SetpointRimeShelf rime_shelf; out.push_back(&rime_shelf);
  static const CollapseWindowScarp collapse_scarp; out.push_back(&collapse_scarp);
  static const CollapseWindowPlaya collapse_playa; out.push_back(&collapse_playa);
  static const SpeedBandTalus speed_talus; out.push_back(&speed_talus);
  static const SlipPhasePan slip_pan; out.push_back(&slip_pan);
  static const SolarWindowPan solar_window; out.push_back(&solar_window);
  static const PulseGravityReef pulse_reef; out.push_back(&pulse_reef);
  static const CadenceDuneBelt cadence_belt; out.push_back(&cadence_belt);
  static const CommitmentLedgeField commitment_ledge; out.push_back(&commitment_ledge);
}

}

namespace generated_biomes {

inline constexpr int kGeneratedBiomeBankStart = 0;


class LateralShearBelt final : public Biome {
 public:
  std::string_view id() const noexcept override { return "lateral_shear_belt"; }
  std::string_view display_name() const noexcept override { return "Lateral Shear Belt"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  int hazard_at(int) const noexcept override { return 2; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.78f + 0.15f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.35f + 0.25f * biome_random01(s, 2);
    p.wind_force = 4.0f + 2.0f * biome_random01(s, 3);

    p.ambient_temperature = -55.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.60f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.10f + 0.06f * biome_random01(s, 6);
    p.gravity_mul = 1.05f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.006f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.08f + 0.03f * biome_random01(s, 10);


    p.terrain_amplitude_mul = 0.80f + 0.15f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.40f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.50f + 0.15f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.15f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.12f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);






    const float shear_phase = t * 0.0044f + c.velocity.x * 0.011f;
    const float cross = 0.5f + 0.5f * std::sin(shear_phase);
    const float straight = 1.0f - cross;




    const float pulse_phase = t * 0.031f + c.velocity.x * 0.073f;
    const float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
    const float pulse_narrow = pulse * pulse;

    if (c.body_force) {





      const float cross_lateral = cross * std::sin(shear_phase + 1.2f) *
          (0.08f + 0.30f * speed_norm * speed_norm) * c.mass * c.gravity * 0.35f;


      const float pulse_lateral = cross * pulse_narrow *
          (0.05f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.9f);


      const float straight_assist = straight * 0.020f * c.mass * c.gravity;



      const float precession = std::sin(shear_phase * 0.5f + 0.6f) *
          (0.03f + 0.10f * speed_norm) * c.mass * c.gravity;


      const float damping = 0.04f + 0.02f * cross + 0.02f * speed_norm;

      c.body_force->x += straight_assist + cross_lateral + pulse_lateral + precession;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.02f * cross);

      if (c.body_torque) {



        const float yaw_cross = cross * std::cos(shear_phase + 1.5f) *
            (0.05f + 0.18f * speed_norm) * c.mass * c.gravity * 0.22f;


        const float pulse_torque = cross * pulse_narrow *
            (0.03f + 0.10f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.6f);


        const float pitch_damping = 0.025f + 0.02f * straight + 0.01f * speed_norm;
        *c.body_torque += yaw_cross + pulse_torque;
        *c.body_torque -= c.angular_velocity * c.mass * pitch_damping;
      }
    }

    if (c.energy_cost) {






      const bool in_cross = cross > 0.5f;
      if (in_cross) {
        if (speed > 0.25f) {

          *c.energy_cost += (0.014f + 0.030f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += 0.018f * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.005f + 0.003f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;




    v.ground = {28, 30, 36};
    v.particles = {96, 106, 124};
    v.liquid = {16, 18, 24};
    v.sky = {48, 54, 68};
    v.particle_rate = 3.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 4;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = false;
    return v;
  }
};

class HysteresisSurgeBog final : public Biome {
 public:
  std::string_view id() const noexcept override { return "hysteresis_surge_bog"; }
  std::string_view display_name() const noexcept override { return "Hysteresis Surge Bog"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }



  int hazard_at(int) const noexcept override { return 2; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.42f + 0.18f * biome_random01(s);
    p.sink_rate = 0.014f + 0.020f * biome_random01(s, 1);
    p.viscosity = 1.4f + 1.6f * biome_random01(s, 2);
    p.energy_drain_mul = 1.60f + 0.45f * biome_random01(s, 3);
    p.wind_force = 0.0f;

    p.ambient_temperature = -25.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.2f + 0.5f * biome_random01(s, 5);
    p.solar_charge_rate = 0.05f + 0.04f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.012f + 0.016f * biome_random01(s, 8);

    p.lidar_energy_mul = 5.8f + 1.2f * biome_random01(s, 9);
    p.lidar_range_mul = 0.09f + 0.04f * biome_random01(s, 10);


    p.terrain_amplitude_mul = 1.35f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.60f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.40f + 0.15f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 0.55f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      float speed = std::abs(c.wheel_speed);

      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.8f * std::tanh(speed * 0.20f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 28.0f;
      float speed = std::abs(c.wheel_speed);

      float drag = (0.28f + p.viscosity * 1.6f * (1.0f + depth) + 0.10f * depth) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.09f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 28.0f;

      *c.energy_cost += (0.015f + depth * 0.20f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);





    const float surge_phase = t * 0.0045f + c.velocity.x * 0.012f;
    const float surge = 0.5f + 0.5f * std::sin(surge_phase);
    const float release = 1.0f - surge;




    const float pulse_phase = t * 0.035f + c.velocity.x * 0.085f;
    const float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
    const float pulse_narrow = pulse * pulse;



    const float memory = 0.5f + 0.5f * std::tanh((speed_norm - 0.25f) * 5.0f);
    const float hysteresis = surge * memory * (0.5f + 0.5f * pulse_narrow);

    if (c.body_force) {


      const float suck_drag = surge * (0.08f + 0.16f * memory) * c.mass * speed_norm *
          (c.velocity.x >= 0.0f ? 1.0f : -1.0f);




      const float release_assist = release * (0.02f + 0.04f * memory) * c.mass * c.gravity;



      const float pulse_drag = hysteresis * (0.10f + 0.14f * speed_norm) * c.mass *
          (c.velocity.x >= 0.0f ? 1.0f : -1.0f);



      const float lateral = std::sin(t * 0.021f + c.velocity.x * 0.049f) *
          (0.04f + 0.12f * std::abs(surge - 0.5f) * 2.0f + 0.08f * speed_norm) * c.mass * c.gravity;


      const float damping = 0.05f + 0.04f * surge + 0.02f * speed_norm;

      c.body_force->x += release_assist + lateral - suck_drag - pulse_drag;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.02f * surge);

      if (c.body_torque) {



        const float suck_torque = surge * (0.2f + 0.4f * speed_norm) * c.mass;
        const float release_torque = release * (0.1f + 0.3f * speed_norm) * c.mass;
        const float pulse_torque = hysteresis * (0.3f + 0.35f * speed_norm) * c.mass *
            std::sin(pulse_phase + 0.8f);
        *c.body_torque -= suck_torque;
        *c.body_torque += release_torque + pulse_torque;
        *c.body_torque -= c.angular_velocity * c.mass * (0.03f + 0.02f * surge);
      }
    }

    if (c.energy_cost) {





      const float suck_cost = surge * (0.012f + 0.028f * memory) * p.energy_drain_mul;
      const float pulse_penalty = hysteresis * (0.014f + 0.020f * speed_norm) * p.energy_drain_mul;
      const float release_cost = release * (0.006f + 0.004f * speed_norm) * p.energy_drain_mul;
      const float low_speed_cost = std::max(0.0f, 0.25f - speed_norm) * 0.012f * p.energy_drain_mul;


      const float base_cost = 0.008f;

      *c.energy_cost += (base_cost + suck_cost + pulse_penalty + release_cost + low_speed_cost) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {34, 36, 28};
    v.particles = {120, 124, 88};
    v.liquid = {18, 20, 14};
    v.sky = {42, 44, 34};
    v.particle_rate = 8.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 1.0f;
    v.base_particles = 2;
    v.max_particles = 22;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.07f;
    v.liquid_surface = false;
    return v;
  }
};

class GravityShelfLug final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gravity_shelf_lug"; }
  std::string_view display_name() const noexcept override { return "Gravity Shelf Lug"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }


  int hazard_at(int step) const noexcept override { return shelving(step) ? 1 : 0; }

  bool shelving(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kShelfSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.95f + 0.10f * biome_random01(s);
    p.sink_rate = 0.001f + 0.002f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.10f + 0.15f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = -70.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.50f + 0.15f * biome_random01(s, 5);
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.85f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.08f + 0.03f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.65f + 0.15f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.40f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.80f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 0.40f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool shelf = shelving(c.step_index);

    if (c.body_force) {
      if (shelf) {







        const float heavy = 3.8f * c.mass * c.gravity;
        c.body_force->y += heavy;




        if (speed > kShelfCreep) {
          const float over = clamp((speed - kShelfCreep) * 1.5f, 0.0f, 5.0f);
          const float drag = (2.5f + 2.0f * over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= drag;


          const float lateral = speed_norm * 0.6f * c.mass * std::sin(t * 0.013f + 0.6f);
          c.body_force->x += lateral;


          c.body_force->x -= c.velocity.x * c.mass * 2.0f;
        } else {

          c.body_force->x -= c.velocity.x * c.mass * 0.8f;
        }


        c.body_force->y -= c.velocity.y * c.mass * 1.2f;
      } else {


        const float assist = 0.018f * c.mass * c.gravity;
        const float damping = 0.03f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (shelf) {


          const float pitch = (0.05f + 0.35f * speed_norm) * c.mass;
          *c.body_torque += pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);


          *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
        } else {

          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (shelf) {
        if (speed <= kShelfCreep) {

          *c.energy_cost += 0.012f * p.energy_drain_mul * c.dt;
        } else {


          const float over = clamp((speed - kShelfCreep) * 1.5f, 0.0f, 5.0f);
          *c.energy_cost += (0.30f + 0.90f * over * over) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.005f + 0.003f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;




    v.ground = {104, 112, 124};
    v.particles = {178, 186, 198};
    v.liquid = {48, 56, 66};
    v.sky = {64, 74, 88};
    v.particle_rate = 4.0f;
    v.particle_lift = 1.1f;
    v.particle_spread = 0.6f;
    v.base_particles = 1;
    v.max_particles = 14;
    v.particle_size = 1;
    v.ambient_particles = 6;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 720;
  static constexpr int kShelfSteps = 150;
  static constexpr int kPhase = 170;
  static constexpr float kShelfCreep = 0.30f;
};

class ThermalSurgeRelay final : public Biome {
 public:
  std::string_view id() const noexcept override { return "thermal_surge_relay"; }
  std::string_view display_name() const noexcept override { return "Thermal Surge Relay"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }


  int hazard_at(int step) const noexcept override { return hot_surge(step) ? 1 : 0; }

  bool hot_surge(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kHotSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.55f + 0.20f * biome_random01(s);
    p.sink_rate = 0.006f + 0.010f * biome_random01(s, 1);
    p.viscosity = 0.60f + 0.80f * biome_random01(s, 2);
    p.energy_drain_mul = 1.25f + 0.25f * biome_random01(s, 3);
    p.wind_force = 0.2f + 0.5f * biome_random01(s, 4);

    p.ambient_temperature = 48.0f + 12.0f * biome_random01(s, 5);
    p.thermal_transfer = 3.4f + 0.6f * biome_random01(s, 6);
    p.solar_charge_rate = 0.12f + 0.06f * biome_random01(s, 7);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 8);
    p.crust_deform = 0.004f + 0.006f * biome_random01(s, 9);

    p.lidar_energy_mul = 5.8f + 1.0f * biome_random01(s, 10);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 11);

    p.terrain_amplitude_mul = 0.60f + 0.15f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.35f + 0.10f * biome_random01(s, 13);
    p.terrain_crater_mul = 0.40f + 0.10f * biome_random01(s, 14);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 0.72f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      float drag = (0.20f + p.viscosity * 1.4f * (1.0f + depth) * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.02f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.008f + std::abs(c.wheel_speed) * 0.004f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool hot = hot_surge(c.step_index);

    if (c.body_force) {
      if (hot) {




        const float churn = 0.12f + 0.25f * speed_norm;
        c.body_force->x -= c.velocity.x * c.mass * (0.5f + 1.2f * speed_norm);
        c.body_force->y += churn * c.mass * c.gravity * std::sin(t * 0.017f);

        c.body_force->x += speed_norm * 0.22f * c.mass * c.gravity * std::sin(t * 0.013f + 0.7f);

        c.body_force->x -= c.velocity.x * c.mass * 1.4f;
        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {

        const float assist = 0.014f * c.mass * c.gravity;
        const float damping = 0.03f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (hot) {

          const float pitch = (0.05f + 0.35f * speed_norm) * c.mass * std::sin(t * 0.015f + 0.4f);
          *c.body_torque += pitch;

          *c.body_torque -= c.angular_velocity * c.mass * 1.2f;
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (hot) {
        if (speed <= 0.14f) {

          *c.energy_cost -= 2.4f * c.dt;

          *c.energy_cost += 0.04f * c.dt;
        } else {

          *c.energy_cost += (0.35f + 0.85f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.006f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;


    v.ground = {20, 28, 34};
    v.particles = {120, 160, 170};
    v.liquid = {14, 40, 50};
    v.sky = {32, 42, 52};
    v.particle_rate = 4.0f;
    v.particle_lift = 1.6f;
    v.particle_spread = 0.8f;
    v.base_particles = 2;
    v.max_particles = 18;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.3f;
    v.screen_brightness = 0.07f;
    v.liquid_surface = true;
    return v;
  }

 private:
  static constexpr int kCycle = 700;
  static constexpr int kHotSteps = 180;
  static constexpr int kPhase = 220;
};

class RimeQuarryDawn final : public Biome {
 public:
  std::string_view id() const noexcept override { return "rime_quarry_dawn"; }
  std::string_view display_name() const noexcept override { return "Rime Quarry Dawn"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }

  bool rime_phase(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kRimeSteps;
  }

  int hazard_at(int step) const noexcept override { return rime_phase(step) ? 2 : 1; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.85f + 0.15f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.20f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = -95.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.4f + 0.3f * biome_random01(s, 5);
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.55f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.004f + 0.006f * biome_random01(s, 8);

    p.lidar_energy_mul = 5.5f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.60f + 0.15f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.55f + 0.15f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.90f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 0.70f + 0.20f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.20f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool rime = rime_phase(c.step_index);

    if (c.body_force) {
      if (rime) {




        const float release = 1.10f;
        if (speed < release) {
          const float bite = clamp((release - speed) / release, 0.0f, 1.0f);
          const float bite_force = (0.6f + 2.8f * bite) * c.mass *
                                   (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= bite_force;


          c.body_force->x += (0.2f + 0.5f * bite) * c.mass * std::sin(t * 0.017f + 0.6f);


          c.body_force->x -= c.velocity.x * c.mass * (0.8f + 1.5f * bite);
        } else {


          c.body_force->x += 0.025f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.03f;
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float thaw_damping = 0.04f + 0.02f * speed_norm;
        c.body_force->x -= c.velocity.x * c.mass * thaw_damping;
        if (speed <= 0.2f) {

          c.body_force->x += 0.012f * c.mass * c.gravity;
        }
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (rime) {
          if (speed < 1.10f) {



            const float bite = clamp((1.10f - speed) / 1.10f, 0.0f, 1.0f);
            const float pitch = (0.02f + 0.30f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (rime) {
        if (speed < 1.10f) {


          const float bite = clamp((1.10f - speed) / 1.10f, 0.0f, 1.0f);
          *c.energy_cost += (0.15f + 0.75f * bite * bite) * p.energy_drain_mul * c.dt;
        } else {


          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.2f) {


          *c.energy_cost -= 0.8f * c.dt;
        } else {

          *c.energy_cost += (0.006f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;


    v.ground = {102, 110, 122};
    v.particles = {196, 204, 214};
    v.liquid = {44, 52, 62};
    v.sky = {56, 66, 80};
    v.particle_rate = 4.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 14;
    v.particle_size = 1;
    v.ambient_particles = 6;
    v.ambient_drift = 0.7f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 760;
  static constexpr int kRimeSteps = 210;
  static constexpr int kPhase = 240;
};

class GravityShearEscarpment final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gravity_shear_escarpment"; }
  std::string_view display_name() const noexcept override { return "Gravity Shear Escarpment"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }


  int hazard_at(int step) const noexcept override { return shear_phase(step) ? 1 : 0; }

  bool shear_phase(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    return phase < 0 ? phase + kCycle : phase < kShearSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.85f + 0.15f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.15f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.8f + 0.6f * biome_random01(s, 3);

    p.ambient_temperature = -80.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.55f + 0.15f * biome_random01(s, 5);
    p.solar_charge_rate = 0.05f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.60f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.09f + 0.03f * biome_random01(s, 10);


    p.terrain_amplitude_mul = 1.50f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.80f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.20f * biome_random01(s, 13);
    p.terrain_step_mul = 1.80f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.10f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.09f);
    const bool shear = shear_phase(c.step_index);

    if (c.body_force) {
      if (shear) {







        const float creep_limit = 0.60f;
        if (speed > creep_limit) {
          const float over = clamp((speed - creep_limit) * 1.8f, 0.0f, 5.0f);

          const float shove = (2.2f + 2.8f * over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= shove;


          c.body_force->x += (0.3f + 0.7f * over) * c.mass * std::sin(t * 0.013f + 0.6f);


          c.body_force->x -= c.velocity.x * c.mass * (0.8f + 1.2f * over);


          c.body_force->y += 0.15f * over * c.mass * c.gravity;
        } else {


          c.body_force->x += 0.020f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
        }


        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float assist = 0.030f * c.mass * c.gravity;
        const float damping = 0.025f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (shear) {
          if (speed > 0.60f) {



            const float over = clamp((speed - 0.60f) * 1.8f, 0.0f, 5.0f);
            const float pitch = (0.04f + 0.28f * over) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);



            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (shear) {
        if (speed > 0.60f) {


          const float over = clamp((speed - 0.60f) * 1.8f, 0.0f, 5.0f);
          *c.energy_cost += (0.20f + 0.70f * over * over) * p.energy_drain_mul * c.dt;
        } else {


          *c.energy_cost += (0.005f + 0.003f * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.004f + 0.003f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;




    v.ground = {118, 122, 128};
    v.particles = {188, 192, 200};
    v.liquid = {46, 52, 62};
    v.sky = {72, 78, 90};
    v.particle_rate = 4.0f;
    v.particle_lift = 1.0f;
    v.particle_spread = 0.7f;
    v.base_particles = 1;
    v.max_particles = 14;
    v.particle_size = 1;
    v.ambient_particles = 6;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 800;
  static constexpr int kShearSteps = 220;
  static constexpr int kPhase = 190;
};

class EbbTractionDark final : public Biome {
 public:
  std::string_view id() const noexcept override { return "ebb_traction_dark"; }
  std::string_view display_name() const noexcept override { return "Ebb-Traction Dark"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }


  int hazard_at(int step) const noexcept override { return slick(step) ? 2 : 1; }

  bool slick(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSlickSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.90f + 0.10f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = -90.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.2f + 0.3f * biome_random01(s, 5);
    p.solar_charge_rate = 0.03f + 0.02f * biome_random01(s, 6);
    p.gravity_mul = 1.00f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.09f + 0.03f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.55f + 0.10f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.30f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.40f + 0.10f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool wet = slick(c.step_index);

    if (c.body_force) {
      if (wet) {




        const float release_speed = 1.05f;
        if (speed < release_speed) {
          const float bite = clamp((release_speed - speed) / release_speed, 0.0f, 1.0f);
          const float lock_force = (0.8f + 3.4f * bite) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= lock_force;
          c.body_force->x += (0.25f + 0.7f * bite) * c.mass * std::sin(t * 0.017f + 0.7f);
          c.body_force->x -= c.velocity.x * c.mass * (1.2f + 2.2f * bite);
        } else {


          c.body_force->x += 0.020f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {


        const float stall_speed = 0.45f;
        if (speed > stall_speed) {
          const float over = clamp((speed - stall_speed) * 1.4f, 0.0f, 4.0f);
          const float drag = (0.6f + 2.2f * over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= drag;
          c.body_force->x += (0.2f + 0.5f * over) * c.mass * std::sin(t * 0.021f + 0.9f);
          c.body_force->x -= c.velocity.x * c.mass * (0.5f + 1.1f * over);
        } else {


          c.body_force->x += 0.012f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.03f;
        }
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (wet) {
          if (speed < 1.05f) {

            const float bite = clamp((1.05f - speed) / 1.05f, 0.0f, 1.0f);
            const float pitch = (0.03f + 0.34f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 1.0f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          if (speed > 0.45f) {

            const float over = clamp((speed - 0.45f) * 1.4f, 0.0f, 4.0f);
            const float pitch = (0.02f + 0.26f * over) * c.mass;
            *c.body_torque += pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.8f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        }
      }
    }

    if (c.energy_cost) {
      if (wet) {
        if (speed < 1.05f) {

          const float bite = clamp((1.05f - speed) / 1.05f, 0.0f, 1.0f);
          *c.energy_cost += (0.16f + 0.90f * bite * bite) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += (0.006f + 0.012f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.18f) {

          *c.energy_cost -= 0.7f * c.dt;
        } else if (speed <= 0.45f) {

          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        } else {

          const float over = clamp((speed - 0.45f) * 1.4f, 0.0f, 4.0f);
          *c.energy_cost += (0.08f + 0.70f * over * over) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;

    v.ground = {24, 30, 40};
    v.particles = {96, 116, 140};
    v.liquid = {14, 18, 26};
    v.sky = {30, 38, 52};
    v.particle_rate = 3.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 4;
    v.ambient_drift = 0.6f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 520;
  static constexpr int kSlickSteps = 190;
  static constexpr int kPhase = 70;
};

class BakeCycleBasaltFan final : public Biome {
 public:
  std::string_view id() const noexcept override { return "bake_cycle_basalt_fan"; }
  std::string_view display_name() const noexcept override { return "Bake-Cycle Basalt Fan"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }



  int hazard_at(int step) const noexcept override { return blast(step) ? 3 : 0; }

  bool blast(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kBlastSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.95f + 0.10f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.15f + 0.20f * biome_random01(s, 2);
    p.wind_force = 2.5f + 1.0f * biome_random01(s, 3);

    p.ambient_temperature = 52.0f + 18.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.9f + 0.6f * biome_random01(s, 5);
    p.solar_charge_rate = 0.06f + 0.04f * biome_random01(s, 6);
    p.gravity_mul = 1.02f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.50f + 0.10f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.30f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.30f + 0.10f * biome_random01(s, 13);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool hot = blast(c.step_index);

    if (c.body_force) {
      if (hot) {





        const float jolt_amp = 0.08f + 0.14f * speed_norm;
        c.body_force->y += jolt_amp * c.mass * c.gravity * std::sin(t * 0.014f);


        const float lateral = (0.15f + 0.35f * speed_norm) * c.mass * c.gravity * std::sin(t * 0.017f + 0.7f);
        c.body_force->x += lateral;


        c.body_force->x -= c.velocity.x * c.mass * (0.8f + 1.2f * speed_norm);
        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float assist = 0.020f * c.mass * c.gravity;
        const float damping = 0.03f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (hot) {

          const float pitch = (0.04f + 0.30f * speed_norm) * c.mass * std::sin(t * 0.015f + 0.4f);
          *c.body_torque += pitch;

          *c.body_torque -= c.angular_velocity * c.mass * 1.1f;
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (hot) {
        if (speed <= 0.14f) {

          *c.energy_cost -= 2.2f * c.dt;

          *c.energy_cost += 0.04f * c.dt;
        } else {

          *c.energy_cost += (0.30f + 0.90f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {28, 26, 30};
    v.particles = {150, 110, 80};
    v.liquid = {16, 14, 20};
    v.sky = {56, 44, 48};
    v.particle_rate = 3.0f;
    v.particle_lift = 1.1f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 5;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 780;
  static constexpr int kBlastSteps = 200;
  static constexpr int kPhase = 240;
};

class ReversibleMomentumLagoon final : public Biome {
 public:
  std::string_view id() const noexcept override { return "reversible_momentum_lagoon"; }
  std::string_view display_name() const noexcept override { return "Reversible-Momentum Lagoon"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }


  int hazard_at(int step) const noexcept override { return surging(step) ? 2 : 1; }

  bool surging(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSurgeSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.45f + 0.15f * biome_random01(s);
    p.sink_rate = 0.004f + 0.006f * biome_random01(s, 1);
    p.viscosity = 0.35f + 0.40f * biome_random01(s, 2);
    p.energy_drain_mul = 1.30f + 0.25f * biome_random01(s, 3);
    p.wind_force = 0.2f + 0.3f * biome_random01(s, 4);

    p.ambient_temperature = 12.0f + 14.0f * biome_random01(s, 5);
    p.thermal_transfer = 1.4f + 0.4f * biome_random01(s, 6);
    p.solar_charge_rate = 0.05f + 0.03f * biome_random01(s, 7);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 8);
    p.crust_deform = 0.002f + 0.003f * biome_random01(s, 9);

    p.lidar_energy_mul = 5.5f + 1.0f * biome_random01(s, 10);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 11);

    p.terrain_amplitude_mul = 0.55f + 0.10f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.30f + 0.10f * biome_random01(s, 13);
    p.terrain_crater_mul = 0.45f + 0.10f * biome_random01(s, 14);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 0.75f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {

      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.2f * std::tanh(speed * 0.16f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 22.0f;

      float drag = (0.18f + p.viscosity * 1.3f * (1.0f + depth) * c.immersion + 0.06f * depth) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.02f + 0.07f * depth * c.immersion));
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.006f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool surge = surging(c.step_index);



    const float momentum = 0.5f + 0.5f * std::tanh((speed_norm - 0.35f) * 5.0f);

    if (c.body_force) {
      if (surge) {



        if (speed <= 0.75f) {
          const float bite = clamp((0.75f - speed) / 0.75f, 0.0f, 1.0f);
          const float lock_force = (0.5f + 2.6f * bite) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= lock_force;
          c.body_force->x += (0.2f + 0.6f * bite) * c.mass * std::sin(t * 0.019f + 0.8f);
          c.body_force->x -= c.velocity.x * c.mass * (0.9f + 1.8f * bite);
        } else {

          c.body_force->x += 0.035f * c.mass * c.gravity * speed_norm;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float ebb_drag = (0.4f + 1.8f * momentum) * c.mass * speed_norm * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
        const float ebb_settle = 0.02f * c.mass * c.gravity;
        c.body_force->x -= ebb_drag;
        c.body_force->x += ebb_settle;
        if (speed <= 0.25f) {

          c.body_force->x += 0.015f * c.mass * c.gravity;
        }
        c.body_force->x -= c.velocity.x * c.mass * (0.04f + 0.03f * momentum);
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (surge) {
          if (speed <= 0.75f) {

            const float bite = clamp((0.75f - speed) / 0.75f, 0.0f, 1.0f);
            const float pitch = (0.03f + 0.30f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {

            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          if (speed > 0.45f) {

            const float over = clamp((speed - 0.45f) * 1.5f, 0.0f, 4.0f);
            const float pitch = (0.02f + 0.24f * over) * c.mass;
            *c.body_torque += pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.8f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        }
      }
    }

    if (c.energy_cost) {
      if (surge) {
        if (speed <= 0.75f) {

          const float bite = clamp((0.75f - speed) / 0.75f, 0.0f, 1.0f);
          *c.energy_cost += (0.18f + 0.85f * bite * bite) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += (0.006f + 0.010f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.18f) {

          *c.energy_cost -= 0.6f * c.dt;
        } else if (speed <= 0.45f) {

          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        } else {

          const float over = clamp((speed - 0.45f) * 1.5f, 0.0f, 4.0f);
          *c.energy_cost += (0.08f + 0.65f * over * over) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {42, 60, 58};
    v.particles = {150, 190, 180};
    v.liquid = {36, 80, 76};
    v.sky = {48, 66, 62};
    v.particle_rate = 5.0f;
    v.particle_lift = 0.8f;
    v.particle_spread = 0.7f;
    v.base_particles = 1;
    v.max_particles = 16;
    v.particle_size = 1;
    v.ambient_particles = 5;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = true;
    return v;
  }

 private:
  static constexpr int kCycle = 620;
  static constexpr int kSurgeSteps = 210;
  static constexpr int kPhase = 140;
};

class MassSwingFetch final : public Biome {
 public:
  std::string_view id() const noexcept override { return "mass_swing_fetch"; }
  std::string_view display_name() const noexcept override { return "Mass-Swing Fetch"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }



  int hazard_at(int step) const noexcept override { return window(step) ? 1 : 0; }

  bool window(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kWindowSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.90f + 0.10f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.10f + 0.15f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = -76.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.55f + 0.15f * biome_random01(s, 5);
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.55f + 0.15f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.07f + 0.03f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.55f + 0.10f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.35f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.40f + 0.10f * biome_random01(s, 13);
    p.terrain_step_mul = 0.25f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool win = window(c.step_index);

    if (c.body_force) {
      if (win) {





        const float swing_amp = 0.08f + 0.22f * speed_norm;
        const float swing = swing_amp * c.mass * c.gravity * std::sin(t * 0.019f);
        c.body_force->x += swing;


        c.body_force->x -= c.velocity.x * c.mass * (1.2f + 1.5f * speed_norm);
        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float assist = 0.028f * c.mass * c.gravity;
        const float damping = 0.025f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (win) {



          const float pitch = (0.05f + 0.38f * speed_norm) * c.mass * std::sin(t * 0.021f + 0.5f);
          *c.body_torque += pitch;


          *c.body_torque -= c.angular_velocity * c.mass * 1.2f;
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (win) {
        if (speed <= 0.16f) {


          *c.energy_cost -= 2.6f * c.dt;

          *c.energy_cost += 0.04f * c.dt;
        } else {


          *c.energy_cost += (0.30f + 0.95f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {74, 84, 82};
    v.particles = {168, 180, 172};
    v.liquid = {36, 46, 44};
    v.sky = {58, 68, 66};
    v.particle_rate = 3.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 5;
    v.ambient_drift = 0.7f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 800;
  static constexpr int kWindowSteps = 180;
  static constexpr int kPhase = 260;
};

class BatteryBayTycho final : public Biome {
 public:
  std::string_view id() const noexcept override { return "battery_bay_tycho"; }
  std::string_view display_name() const noexcept override { return "Battery Bay Tycho"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }



  int hazard_at(int step) const noexcept override { return glimmer(step) ? 3 : 0; }

  bool glimmer(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kWindowSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.95f + 0.10f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = 24.0f + 14.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.3f + 0.3f * biome_random01(s, 5);
    p.solar_charge_rate = 0.10f + 0.06f * biome_random01(s, 6);
    p.gravity_mul = 1.00f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 10);

    p.terrain_amplitude_mul = 0.50f + 0.10f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.30f + 0.10f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.30f + 0.10f * biome_random01(s, 13);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.05f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool glimmer = this->glimmer(c.step_index);

    if (c.body_force) {
      if (glimmer) {




        const float jolt_amp = 0.06f + 0.12f * speed_norm;
        c.body_force->y += jolt_amp * c.mass * c.gravity * std::sin(t * 0.015f);


        const float lateral = (0.12f + 0.30f * speed_norm) * c.mass * c.gravity * std::sin(t * 0.018f + 0.7f);
        c.body_force->x += lateral;


        c.body_force->x -= c.velocity.x * c.mass * (0.8f + 1.2f * speed_norm);
        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {


        const float assist = 0.020f * c.mass * c.gravity;
        const float damping = 0.03f + 0.02f * speed_norm;
        c.body_force->x += assist - c.velocity.x * c.mass * damping;
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (glimmer) {

          const float pitch = (0.04f + 0.28f * speed_norm) * c.mass * std::sin(t * 0.016f + 0.4f);
          *c.body_torque += pitch;

          *c.body_torque -= c.angular_velocity * c.mass * 1.1f;
        } else {
          *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
        }
      }
    }

    if (c.energy_cost) {
      if (glimmer) {
        if (speed <= 0.14f) {

          *c.energy_cost -= 2.5f * c.dt;

          *c.energy_cost += 0.04f * c.dt;
        } else {

          *c.energy_cost += (0.32f + 0.85f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {

        *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;

    v.ground = {24, 26, 32};
    v.particles = {110, 130, 150};
    v.liquid = {14, 16, 22};
    v.sky = {40, 48, 60};
    v.particle_rate = 2.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 10;
    v.particle_size = 1;
    v.ambient_particles = 3;
    v.ambient_drift = 0.5f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 860;
  static constexpr int kWindowSteps = 180;
  static constexpr int kPhase = 280;
};

class ShatterStepBerm final : public Biome {
 public:
  std::string_view id() const noexcept override { return "shatter_step_berm"; }
  std::string_view display_name() const noexcept override { return "Shatter-Step Berm"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }


  int hazard_at(int step) const noexcept override { return brittle(step) ? 2 : 1; }

  bool brittle(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kBrittleSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.90f + 0.12f * biome_random01(s);
    p.sink_rate = 0.003f + 0.004f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.18f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.6f * biome_random01(s, 3);

    p.ambient_temperature = -70.0f + 14.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.65f + 0.20f * biome_random01(s, 5);
    p.solar_charge_rate = 0.05f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.98f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.012f + 0.008f * biome_random01(s, 8);

    p.lidar_energy_mul = 5.8f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.09f + 0.03f * biome_random01(s, 10);


    p.terrain_amplitude_mul = 1.15f + 0.25f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.90f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.60f + 0.15f * biome_random01(s, 13);
    p.terrain_step_mul = 2.40f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.10f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool brittle = this->brittle(c.step_index);



    const bool on_berm = speed <= 0.22f;

    if (c.body_force) {
      if (brittle) {





        if (speed > 0.65f) {
          const float over = clamp((speed - 0.65f) * 1.6f, 0.0f, 5.0f);

          c.body_force->x -= (1.2f + 2.4f * over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->y -= (0.5f + 0.8f * over) * c.mass * std::abs(c.gravity);

          c.body_force->x += (0.3f + 0.6f * over) * c.mass * std::sin(t * 0.017f + 0.8f);

          c.body_force->x -= c.velocity.x * c.mass * (0.9f + 1.3f * over);
        } else if (speed > 0.22f) {

          const float creep_over = clamp((speed - 0.22f) * 2.0f, 0.0f, 2.0f);
          c.body_force->x -= (0.6f + 0.8f * creep_over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->y -= 0.20f * creep_over * c.mass * std::abs(c.gravity);
          c.body_force->x -= c.velocity.x * c.mass * (0.6f + 0.5f * creep_over);
        } else {


          c.body_force->x += 0.010f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.03f;
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        if (on_berm) {

          const float bite = clamp((0.30f - speed) / 0.30f, 0.0f, 1.0f);
          const float lock = (0.8f + 2.4f * bite) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= lock;
          c.body_force->x += (0.2f + 0.5f * bite) * c.mass * std::sin(t * 0.019f + 0.6f);
          c.body_force->x -= c.velocity.x * c.mass * (0.9f + 1.6f * bite);
        } else {

          c.body_force->x += 0.025f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
        }
        c.body_force->y -= c.velocity.y * c.mass * 0.04f;
      }

      if (c.body_torque) {
        if (brittle) {
          if (speed > 0.65f) {

            const float over = clamp((speed - 0.65f) * 1.6f, 0.0f, 5.0f);
            const float pitch = (0.05f + 0.32f * over) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 1.0f;
          } else if (speed > 0.22f) {

            const float creep_over = clamp((speed - 0.22f) * 2.0f, 0.0f, 2.0f);
            const float pitch = (0.02f + 0.14f * creep_over) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {

            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          if (on_berm) {

            const float bite = clamp((0.30f - speed) / 0.30f, 0.0f, 1.0f);
            const float pitch = (0.03f + 0.28f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {

            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        }
      }
    }

    if (c.energy_cost) {
      if (brittle) {
        if (speed <= 0.18f) {

          *c.energy_cost -= 0.9f * c.dt;
        } else if (speed <= 0.65f) {

          const float creep_over = clamp((speed - 0.18f) * 2.0f, 0.0f, 2.0f);
          *c.energy_cost += (0.10f + 0.25f * creep_over) * p.energy_drain_mul * c.dt;
        } else {

          const float over = clamp((speed - 0.65f) * 1.6f, 0.0f, 5.0f);
          *c.energy_cost += (0.25f + 0.80f * over * over) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.18f) {

          *c.energy_cost += (0.15f + 0.55f * clamp((0.30f - speed) / 0.30f, 0.0f, 1.0f)) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {112, 100, 88};
    v.particles = {194, 178, 156};
    v.liquid = {48, 42, 38};
    v.sky = {66, 58, 52};
    v.particle_rate = 4.0f;
    v.particle_lift = 0.6f;
    v.particle_spread = 0.6f;
    v.base_particles = 1;
    v.max_particles = 14;
    v.particle_size = 1;
    v.ambient_particles = 5;
    v.ambient_drift = 0.7f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 720;
  static constexpr int kBrittleSteps = 170;
  static constexpr int kPhase = 210;
};

class FathomDraftFlux final : public Biome {
 public:
  std::string_view id() const noexcept override { return "fathom_draft_flux"; }
  std::string_view display_name() const noexcept override { return "Fathom Draft Flux"; }
  std::string_view skill_stratum() const noexcept override { return "inertia_hysteresis"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }


  int hazard_at(int step) const noexcept override { return suction(step) ? 2 : 1; }

  bool suction(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSuctionSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.32f + 0.10f * biome_random01(s);
    p.sink_rate = 0.010f + 0.008f * biome_random01(s, 1);
    p.viscosity = 1.2f + 1.0f * biome_random01(s, 2);
    p.energy_drain_mul = 1.50f + 0.30f * biome_random01(s, 3);
    p.wind_force = 0.2f + 0.3f * biome_random01(s, 4);

    p.ambient_temperature = -30.0f + 15.0f * biome_random01(s, 5);
    p.thermal_transfer = 2.6f + 0.5f * biome_random01(s, 6);
    p.solar_charge_rate = 0.04f + 0.03f * biome_random01(s, 7);
    p.gravity_mul = 0.92f + 0.08f * biome_random01(s, 8);
    p.crust_deform = 0.004f + 0.005f * biome_random01(s, 9);

    p.lidar_energy_mul = 5.5f + 1.0f * biome_random01(s, 10);
    p.lidar_range_mul = 0.09f + 0.03f * biome_random01(s, 11);

    p.terrain_amplitude_mul = 0.55f + 0.10f * biome_random01(s, 12);
    p.terrain_roughness_mul = 0.35f + 0.10f * biome_random01(s, 13);
    p.terrain_crater_mul = 0.40f + 0.10f * biome_random01(s, 14);
    p.terrain_step_mul = 0.20f + 0.10f * biome_random01(s, 15);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {

      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 1.4f * std::tanh(speed * 0.18f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 24.0f;

      float drag = (0.22f + p.viscosity * 1.6f * (1.0f + depth) * c.immersion + 0.08f * depth) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.007f + std::abs(c.wheel_speed) * 0.003f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool suck = suction(c.step_index);



    const float momentum = 0.5f + 0.5f * std::tanh((speed_norm - 0.30f) * 6.0f);

    if (c.body_force) {
      if (suck) {



        if (speed <= 0.85f) {
          const float bite = clamp((0.85f - speed) / 0.85f, 0.0f, 1.0f);
          const float lock_force = (0.6f + 2.6f * bite) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= lock_force;
          c.body_force->x += (0.2f + 0.6f * bite) * c.mass * std::sin(t * 0.019f + 0.8f);
          c.body_force->x -= c.velocity.x * c.mass * (1.0f + 1.8f * bite);
          c.body_force->y -= (0.4f + 0.9f * bite) * c.mass * std::abs(c.gravity);
        } else {

          c.body_force->x += 0.030f * c.mass * c.gravity * speed_norm;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
          c.body_force->y -= 0.10f * c.mass * std::abs(c.gravity);
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {



        const float draft_drag = (0.3f + 1.6f * momentum) * c.mass * speed_norm * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
        const float draft_settle = 0.02f * c.mass * c.gravity;
        c.body_force->x -= draft_drag;
        c.body_force->x += draft_settle;
        if (speed <= 0.25f) {

          c.body_force->x += 0.015f * c.mass * c.gravity;
        }
        c.body_force->x -= c.velocity.x * c.mass * (0.04f + 0.03f * momentum);
        c.body_force->y -= c.velocity.y * c.mass * 0.03f;
      }

      if (c.body_torque) {
        if (suck) {
          if (speed <= 0.85f) {

            const float bite = clamp((0.85f - speed) / 0.85f, 0.0f, 1.0f);
            const float pitch = (0.03f + 0.32f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.9f;
          } else {

            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          if (speed > 0.45f) {

            const float over = clamp((speed - 0.45f) * 1.5f, 0.0f, 4.0f);
            const float pitch = (0.02f + 0.24f * over) * c.mass;
            *c.body_torque += pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.8f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        }
      }
    }

    if (c.energy_cost) {
      if (suck) {
        if (speed <= 0.85f) {

          const float bite = clamp((0.85f - speed) / 0.85f, 0.0f, 1.0f);
          *c.energy_cost += (0.16f + 0.85f * bite * bite) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += (0.006f + 0.010f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.18f) {

          *c.energy_cost -= 0.7f * c.dt;
        } else if (speed <= 0.45f) {

          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        } else {

          const float over = clamp((speed - 0.45f) * 1.5f, 0.0f, 4.0f);
          *c.energy_cost += (0.08f + 0.65f * over * over) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;



    v.ground = {30, 46, 40};
    v.particles = {120, 156, 140};
    v.liquid = {20, 60, 52};
    v.sky = {36, 52, 46};
    v.particle_rate = 5.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 0.6f;
    v.base_particles = 1;
    v.max_particles = 16;
    v.particle_size = 1;
    v.ambient_particles = 5;
    v.ambient_drift = 0.7f;
    v.screen_brightness = 0.06f;
    v.liquid_surface = true;
    return v;
  }

 private:
  static constexpr int kCycle = 660;
  static constexpr int kSuctionSteps = 220;
  static constexpr int kPhase = 160;
};

class GripInversionScree final : public Biome {
 public:
  std::string_view id() const noexcept override { return "grip_inversion_scree"; }
  std::string_view display_name() const noexcept override { return "Grip Inversion Scree"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Ice; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }



  int hazard_at(int step) const noexcept override { return slick(step) ? 2 : 1; }

  bool slick(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSlickSteps;
  }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;

    p.friction_mul = 0.92f + 0.08f * biome_random01(s);
    p.sink_rate = 0.002f + 0.003f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.40f + 0.20f * biome_random01(s, 2);
    p.wind_force = 0.5f + 0.5f * biome_random01(s, 3);

    p.ambient_temperature = -85.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.1f + 0.3f * biome_random01(s, 5);
    p.solar_charge_rate = 0.03f + 0.02f * biome_random01(s, 6);
    p.gravity_mul = 1.02f + 0.06f * biome_random01(s, 7);
    p.crust_deform = 0.003f + 0.004f * biome_random01(s, 8);

    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.08f + 0.03f * biome_random01(s, 10);


    p.terrain_amplitude_mul = 1.40f + 0.25f * biome_random01(s, 11);
    p.terrain_roughness_mul = 1.10f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.35f + 0.10f * biome_random01(s, 13);
    p.terrain_step_mul = 2.60f + 0.40f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {

    return p.friction_mul * 1.08f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {

      *c.wheel_force += c.contact->tangent * (-0.10f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {

      *c.energy_cost += (0.004f + 0.002f * std::abs(c.wheel_speed)) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    const float t = static_cast<float>(c.step_index);
    const float speed = std::abs(c.velocity.x);
    const float speed_norm = std::tanh(speed * 0.10f);
    const bool wet = slick(c.step_index);

    if (c.body_force) {
      if (wet) {




        const float release_speed = 1.05f;
        if (speed < release_speed) {
          const float bite = clamp((release_speed - speed) / release_speed, 0.0f, 1.0f);
          const float lock_force = (0.7f + 3.2f * bite) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= lock_force;
          c.body_force->x += (0.25f + 0.7f * bite) * c.mass * std::sin(t * 0.017f + 0.7f);
          c.body_force->x -= c.velocity.x * c.mass * (1.2f + 2.2f * bite);
        } else {


          c.body_force->x += 0.020f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.04f;
        }

        c.body_force->y -= c.velocity.y * c.mass * 0.9f;
      } else {




        const float stall_speed = 0.30f;
        if (speed > stall_speed) {
          const float over = clamp((speed - stall_speed) * 1.6f, 0.0f, 4.0f);
          const float drag = (0.8f + 2.0f * over) * c.mass * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
          c.body_force->x -= drag;
          c.body_force->x += (0.2f + 0.5f * over) * c.mass * std::sin(t * 0.021f + 0.9f);
          c.body_force->x -= c.velocity.x * c.mass * (0.6f + 1.0f * over);
        } else {


          c.body_force->x += 0.015f * c.mass * c.gravity;
          c.body_force->x -= c.velocity.x * c.mass * 0.03f;
        }
        c.body_force->y -= c.velocity.y * c.mass * 0.04f;
      }

      if (c.body_torque) {
        if (wet) {
          if (speed < 1.05f) {

            const float bite = clamp((1.05f - speed) / 1.05f, 0.0f, 1.0f);
            const float pitch = (0.03f + 0.30f * bite) * c.mass;
            *c.body_torque -= pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 1.0f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        } else {
          if (speed > 0.30f) {

            const float over = clamp((speed - 0.30f) * 1.6f, 0.0f, 4.0f);
            const float pitch = (0.02f + 0.24f * over) * c.mass;
            *c.body_torque += pitch * (c.velocity.x >= 0.0f ? 1.0f : -1.0f);
            *c.body_torque -= c.angular_velocity * c.mass * 0.8f;
          } else {
            *c.body_torque -= c.angular_velocity * c.mass * 0.03f;
          }
        }
      }
    }

    if (c.energy_cost) {
      if (wet) {
        if (speed < 1.05f) {

          const float bite = clamp((1.05f - speed) / 1.05f, 0.0f, 1.0f);
          *c.energy_cost += (0.16f + 0.90f * bite * bite) * p.energy_drain_mul * c.dt;
        } else {

          *c.energy_cost += (0.006f + 0.012f * speed_norm * speed_norm) * p.energy_drain_mul * c.dt;
        }
      } else {
        if (speed <= 0.15f) {

          *c.energy_cost -= 0.8f * c.dt;
        } else if (speed <= 0.30f) {

          *c.energy_cost += (0.005f + 0.004f * speed_norm) * p.energy_drain_mul * c.dt;
        } else {

          const float over = clamp((speed - 0.30f) * 1.6f, 0.0f, 4.0f);
          *c.energy_cost += (0.08f + 0.70f * over * over) * p.energy_drain_mul * c.dt;
        }
      }
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;

    v.ground = {66, 72, 78};
    v.particles = {168, 176, 184};
    v.liquid = {24, 28, 34};
    v.sky = {72, 80, 92};
    v.particle_rate = 3.0f;
    v.particle_lift = 0.5f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 4;
    v.ambient_drift = 0.6f;
    v.screen_brightness = 0.05f;
    v.liquid_surface = false;
    return v;
  }

 private:
  static constexpr int kCycle = 640;
  static constexpr int kSlickSteps = 190;
  static constexpr int kPhase = 90;
};

inline void append(std::vector<const Biome*>& out) {
  static const LateralShearBelt biome_0; out.push_back(&biome_0);
  static const HysteresisSurgeBog biome_1; out.push_back(&biome_1);
  static const GravityShelfLug biome_2; out.push_back(&biome_2);
  static const ThermalSurgeRelay biome_3; out.push_back(&biome_3);
  static const RimeQuarryDawn biome_4; out.push_back(&biome_4);
  static const GravityShearEscarpment biome_5; out.push_back(&biome_5);
  static const EbbTractionDark biome_6; out.push_back(&biome_6);
  static const BakeCycleBasaltFan biome_7; out.push_back(&biome_7);
  static const ReversibleMomentumLagoon biome_8; out.push_back(&biome_8);
  static const MassSwingFetch biome_9; out.push_back(&biome_9);
  static const BatteryBayTycho biome_10; out.push_back(&biome_10);
  static const ShatterStepBerm biome_11; out.push_back(&biome_11);
  static const FathomDraftFlux biome_12; out.push_back(&biome_12);
  static const GripInversionScree biome_13; out.push_back(&biome_13);
}

inline constexpr int kGeneratedBiomeBankEnd = 0;


}

inline constexpr std::string_view kBiomeBankVersion = "sha256:6108f82bc055f011540c0219838ddb557a2f632bf0c0b97eba1f3b0ab77a57b7";

inline const std::vector<const Biome*>& biome_registry() {
  static const NormalBiome normal; static const SandBiome sand; static const IceBiome ice;
  static const MudBiome mud; static const WindBiome wind; static const LowGravityBiome low_gravity;
  static const CrustBiome crust; static const LiquidBiome liquid;
  static const std::vector<const Biome*> registry = [] {
    std::vector<const Biome*> out{&normal, &sand, &ice, &mud, &wind, &low_gravity, &crust, &liquid};
    handcrafted_biomes::append(out);
    generated_biomes::append(out); return out;
  }();
  return registry;
}

inline const Biome& biome_by_id(int id) noexcept {
  const auto& bank = biome_registry();
  return *bank[static_cast<size_t>(id >= 0 && id < static_cast<int>(bank.size()) ? id : 0)];
}

}
