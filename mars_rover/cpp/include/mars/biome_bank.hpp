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
  // Privileged solvability channel, exposed ONLY through debug_info and used ONLY by the
  // gate's oracle. A biome whose hazard is scheduled rather than continuous cannot be
  // crossed by any always-driving policy, so the oracle would report it unsolvable and the
  // gate would reject exactly the biomes the benchmark needs. The RL observation is built
  // separately and never sees this. 0 = none, 1 = must hold still, 2 = must keep moving.
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

// Hand-built biomes that demand a stop-and-go rhythm.
//
// Six generations of LLM-written biomes never produced one, and the measurement says why:
// every generated mechanic punishes going too FAST in some continuous way, so one cautious
// reflex (settle in gear 2 and grind) survives all of them and collects 89% of what knowing
// the biome collects. A benchmark for adaptation needs mechanics where the right behaviour
// is not a compromise but a schedule the agent has to discover: drive here, stop there.
//
// These are written by hand rather than generated because the generator kept being asked to
// invent something the control surface did not obviously express, and a worked example both
// proves it is expressible and gives the prompt something concrete to imitate.
//
// The three form a deliberate set: two demand stopping on OPPOSITE phases, and the third
// punishes stopping at all. No fixed policy can serve all three, so bank-wide adaptation
// headroom is positive by construction rather than by hope. The phase is a property of the
// biome, identical in every episode of a trial, so one episode of probing identifies it and
// the rest can exploit it — which is exactly the ability the benchmark exists to measure.
namespace handcrafted_biomes {

// Shared machinery: a crust that is solid for part of a cycle and molten for the rest.
// While molten, anything moving faster than a crawl breaks through, sinks and is dragged
// down; while solid, the ground carries the rover normally.
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
    // The schedule must be inferred from consequences, never read off a scan.
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
    // Breaking through is meant to end the run, not to slow it down: the wheel loses the
    // surface, sinks, and the drag scales with how fast it was going when the crust gave.
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
    // A nose-down pitch as the front wheel drops through, so the failure looks like a
    // collapse rather than like mud.
    if (c.body_torque) *c.body_torque -= c.mass * (0.9f + 0.7f * severity);
  }

 protected:
  static constexpr float kCreepSpeed = 0.55f;
};

// Long solid stretches broken by short collapses: drive hard, then hold still.
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

// The same mechanic on the opposite phase and a slower rhythm. A rover that learned the
// timing of the biome above and replays it here stops exactly when it should be driving.
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

// The inverse trap, so that "when in doubt, stop" is not a safe universal reflex either:
// here the ground sets around a stationary wheel and the rover has to keep rolling.
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
    // Rime sets around a wheel that is barely turning and grips it in place. The slower it
    // is, the harder it is to break out, so hesitating is worse than committing.
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

// Fills the inertia_hysteresis stratum on the train split with a third rhythm, so an agent
// cannot succeed by learning just two periods and guessing between them.
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

// ---------------------------------------------------------------------------------------
// HELD-OUT COUNTERPARTS.
//
// The held-out split is where the readiness gate is decided, and it consisted entirely of
// continuous-modifier biomes that one cautious reflex crossed. These are the same three
// ideas at periods and phases the agent has never seen, so a policy can only do well by
// inferring the schedule during the trial rather than by recalling a specific one.
// ---------------------------------------------------------------------------------------

class CollapseWindowScarp final : public MoltenWindowBiome {
 public:
  std::string_view id() const noexcept override { return "collapse_window_scarp"; }
  std::string_view display_name() const noexcept override { return "Collapse Window Scarp"; }
  std::string_view skill_stratum() const noexcept override { return "dynamic_obstacle"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
  int cycle_steps() const noexcept override { return 560; }
  int melt_steps() const noexcept override { return 125; }
  // Differ from the train windows in PHYSICS, not only in the schedule. The de-duplication
  // guard compares behavioural fingerprints, and it was right to flag a held-out biome that
  // shared every coefficient with a train biome and changed only a period: an agent could
  // then transfer everything except one number. A steeper, stonier, colder scarp under
  // stronger gravity is a different place that happens to also collapse on a timer.
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
    // Loose scree drags the body back on any climb, whatever the crust is doing.
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
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }
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
    // A wet playa: viscous drag on the wheel at all times, independent of the window.
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

// A third distinct answer: neither fast nor stopped, but a narrow band. Present so the
// held-out split cannot be crossed by choosing between "flat out" and "stop on a rhythm".
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
      // Loose talus shears away above the band and the wheel digs itself under.
      const float over = clamp((speed - kCeiling) * 1.5f, 0.0f, 5.0f);
      c.contact->penetration += (0.22f + 0.40f * over) * c.dt;
      if (c.wheel_force) {
        *c.wheel_force += c.contact->tangent * (-7.5f * c.wheel_speed * (1.0f + over));
        *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.6f + 0.4f * over));
      }
      if (c.energy_cost) *c.energy_cost += (0.08f + 0.12f * over) * p.energy_drain_mul * c.dt;
    } else if (speed < kFloor) {
      // Below the band the slope packs around the wheel and holds it.
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

// Held-out traction_loss. Its predecessor, setpoint_glaze_pan, was rejected by the gate on
// its own merits: memoryless robust scored 0.809 there because "drive fast" was the whole
// answer, which is the opposite of what the split is for. This keeps the stratum but changes
// the failure mode: nothing sinks and nothing grips, the surface simply stops accepting drive
// on a schedule, so throttling through the window spins the wheels, burns the battery and
// slides backwards. The right answer is to stop asking for traction and coast until it
// returns - a third distinct discipline, not another collapse window.
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
    // Drive goes into spin instead of motion: the harder it is asked, the more it costs and
    // the further back it slides.
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

// Held-out energy_mode. The two generated biomes that carried this stratum were dropped for
// duplicating train biomes, and replacing them with another continuous drain would recreate
// the problem the whole bank was rebuilt to escape. So the tariff runs on a schedule: for
// part of the cycle every metre costs several times its usual energy, and on a 6-unit budget
// a rover that drives straight through the surges is flat before the course ends. Waiting
// them out is not caution, it is the only way to finish - a battery discipline rather than a
// traction one, so it fails differently from the collapse windows.
class SurgeTariffBasin final : public Biome {
 public:
  std::string_view id() const noexcept override { return "surge_tariff_basin"; }
  std::string_view display_name() const noexcept override { return "Surge Tariff Basin"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  MechanicType visual_type() const noexcept override { return MechanicType::Normal; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

  bool surging(int step) const noexcept {
    int phase = (step + kPhase) % kCycle;
    if (phase < 0) phase += kCycle;
    return phase < kSurgeSteps;
  }
  int hazard_at(int step) const noexcept override { return surging(step) ? 1 : 0; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    p.friction_mul = 1.05f + 0.12f * biome_random01(s);
    p.gravity_mul = 0.88f + 0.08f * biome_random01(s, 1);
    p.ambient_temperature = 12.0f + 14.0f * biome_random01(s, 2);
    p.thermal_transfer = 1.5f + 0.4f * biome_random01(s, 3);
    p.solar_charge_rate = 1.9f + 0.5f * biome_random01(s, 4);
    p.energy_drain_mul = 1.15f + 0.15f * biome_random01(s, 5);
    p.lidar_range_mul = 0.32f;
    p.lidar_energy_mul = 2.4f;
    p.terrain_amplitude_mul = 0.78f;
    p.terrain_crater_mul = 1.55f;
    return p;
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    v.sky = {198, 150, 104};
    v.ground = {148, 112, 74};
    v.particles = {226, 186, 132};
    v.particle_rate = 7.0f;
    v.ambient_particles = 20;
    v.ambient_drift = 2.4f;
    v.screen_brightness = 0.96f;
    return v;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (!c.energy_cost || !surging(c.step_index)) return;
    // The tariff is charged per unit TIME in motion, not per unit distance. The first
    // version scaled with speed, which let a slow crawl dodge it entirely - so the trap
    // rewarded the exact reflex it was built to punish (robust scored 0.32 against a
    // witness of 0.20, and gear-2 grinding came within 5% of the best style). Charging by
    // the second makes creeping through a surge strictly worse than waiting it out: same
    // cost, less ground.
    if (std::abs(c.wheel_speed) <= kMotionThreshold) return;
    *c.energy_cost += kTariffPerSecond * p.energy_drain_mul * c.dt;
  }

  void apply_body_effects(const MechanicParams&, MechanicBodyContext& c) const noexcept override {
    if (!surging(c.step_index) || !c.energy_cost) return;
    if (std::abs(c.velocity.x) <= kMotionThreshold) return;
    *c.energy_cost += 0.22f * c.dt;
  }

 private:
  static constexpr int kCycle = 740;
  static constexpr int kSurgeSteps = 190;
  static constexpr int kPhase = 260;
  static constexpr float kMotionThreshold = 0.18f;
  static constexpr float kTariffPerSecond = 0.85f;
};

// Held-out gravity_change on a schedule. During the pulse the ground barely holds the rover
// down, so anything carrying speed goes light over the next rise and lands on its back; the
// answer is to shed speed for the duration rather than to drive more carefully. Fails by
// flipping, which is a different death from sinking or from a flat battery.
class PulseGravityReef final : public Biome {
 public:
  std::string_view id() const noexcept override { return "pulse_gravity_reef"; }
  std::string_view display_name() const noexcept override { return "Pulse Gravity Reef"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Test; }

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
    // The wheels barely load, so traction all but disappears at speed.
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

// Held-out dynamic_obstacle carrying BOTH traps in one place on different phases: for part
// of the cycle the surface will not bear a moving wheel, and for another part it grips a
// stationary one. Neither "wait when unsure" nor "keep going" survives it, so the schedule
// itself has to be identified rather than a single safe habit.
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

inline void append(std::vector<const Biome*>& out) {
  static const CollapseWindowFlats collapse_flats; out.push_back(&collapse_flats);
  static const CollapseWindowTerrace collapse_terrace; out.push_back(&collapse_terrace);
  static const CollapseWindowGulch collapse_gulch; out.push_back(&collapse_gulch);
  static const SetpointRimeShelf rime_shelf; out.push_back(&rime_shelf);
  static const CollapseWindowScarp collapse_scarp; out.push_back(&collapse_scarp);
  static const CollapseWindowPlaya collapse_playa; out.push_back(&collapse_playa);
  static const SpeedBandTalus speed_talus; out.push_back(&speed_talus);
  static const SlipPhasePan slip_pan; out.push_back(&slip_pan);
  static const SurgeTariffBasin surge_basin; out.push_back(&surge_basin);
  static const PulseGravityReef pulse_reef; out.push_back(&pulse_reef);
  static const CadenceDuneBelt cadence_belt; out.push_back(&cadence_belt);
}

}  // namespace handcrafted_biomes

namespace generated_biomes {
// <MARS_GENERATED_BIOMES>
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

class ClayMantleBog final : public Biome {
 public:
  std::string_view id() const noexcept override { return "clay_mantle_bog"; }
  std::string_view display_name() const noexcept override { return "Clay Mantle Bog"; }
  std::string_view skill_stratum() const noexcept override { return "traction_loss"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Mud; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate nominal grip, but the clay mantle's 'stiction' and 'melt' phases dominate.
    p.friction_mul = 0.42f + 0.18f * biome_random01(s);
    p.sink_rate = 0.016f + 0.024f * biome_random01(s, 1);
    p.viscosity = 1.2f + 1.4f * biome_random01(s, 2);
    p.energy_drain_mul = 1.55f + 0.45f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    // Mild but present thermal load; solar is weak, so recharging is a trap.
    p.ambient_temperature = -15.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 1.6f + 0.6f * biome_random01(s, 5);
    p.solar_charge_rate = 0.06f + 0.04f * biome_random01(s, 6);
    p.gravity_mul = 0.92f + 0.08f * biome_random01(s, 7);
    p.crust_deform = 0.012f + 0.018f * biome_random01(s, 8);
    // Expensive, short-range lidar: the clay phase must be inferred from wheel slip.
    p.lidar_energy_mul = 5.2f + 1.2f * biome_random01(s, 9);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 10);
    // Terrain: broad, undulating clay flats with moderate craters and low steps.
    // The hidden mechanic is the stiction/melt cycle, not the geometry.
    p.terrain_amplitude_mul = 1.25f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.70f + 0.25f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.30f + 0.40f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Nominal baseline; the clay mantle law is imposed in body effects.
    return p.friction_mul * 0.52f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Heavy penetration into the sticky clay, strongly speed-dependent but saturating.
      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 2.2f * std::tanh(speed * 0.22f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 30.0f;
      float speed = std::abs(c.wheel_speed);
      // Strong viscous drag from the clay; deeper = more resistant.
      *c.wheel_force += c.contact->tangent * (-(0.25f + 0.28f * depth * speed + p.viscosity * 1.5f) * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.10f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      // High base energy cost from the viscous clay, plus depth and speed.
      *c.energy_cost += (0.016f + depth * 0.22f + std::abs(c.wheel_speed) * 0.006f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      // Clay mantle phase: a slow, position-dependent oscillation flips between
      // 'stiction' (clay is cold and stiff, providing high grip but only up to a
      // modest speed and only if the wheels are not slipping; beyond that it
      // shatters) and 'melt' (clay has warmed and fluidized, providing a narrow
      // low-speed 'bite' window but collapsing with speed). The phase is completely
      // invisible: lidar is nearly blind, the visual is uniform dark clay, so the
      // agent must infer the state from wheel slip and body sway.
      float phase = t * 0.0050f + c.velocity.x * 0.012f;
      float melt = 0.5f + 0.5f * std::sin(phase);  // 1 = melt, 0 = stiction
      float stiction = 1.0f - melt;

      // Fast sub-oscillation: 'clay pulses' -- brief, sharp traction dips within
      // the melt phase, or stiction breaks within the stiction phase when the
      // clutch churns. They cannot be seen ahead; the rover must learn to feather
      // throttle through them.
      float pulse_phase = t * 0.035f + c.velocity.x * 0.085f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Core traction law: stiction phase gives high grip but only if the rover
      // is moving slowly and the wheels are not slipping (i.e., wheel speed ~ body
      // speed). If the rover accelerates too hard from low speed, the clay shatters
      // and grip collapses. Melt phase gives a narrow 'bite' near 1.2 m/s, but
      // collapses at higher speed as the clay fluidizes.
      // This forces a completely different policy from sand/mud: slow crawling in
      // stiction phase is best, but a moderate 'bite' speed in melt phase is optimal.
      float slip = std::max(0.0f, speed - 0.3f);  // proxy for wheel slip
      float stiction_grip = stiction * (0.82f - 0.40f * std::min(1.0f, slip * 2.2f));
      // Stiction break: sudden grip drop if slip is too high.
      stiction_grip *= (1.0f - 0.55f * stiction * std::min(1.0f, slip * 1.8f) * pulse_narrow);

      float bite = 0.58f + 0.42f * std::exp(-(speed - 1.2f) * (speed - 1.2f) * 1.2f);
      float fluidize = 1.0f / (1.0f + 0.70f * speed * speed);
      float melt_grip = melt * bite * (0.32f + 0.28f * fluidize);

      float grip = stiction_grip + melt_grip;
      grip *= (1.0f - 0.50f * melt * pulse_narrow);
      grip = std::max(0.03f, grip);

      // IMPORTANT: clutch churn penalty. When the wheels are slipping (body speed
      // not matched by wheel speed), the clutch churns the sticky clay, creating a
      // strong backward drag and destabilizing torque. The rover must learn to
      // match wheel speed to body speed (no slip) and avoid clutch engagement in
      // both phases.
      float clutch_churn = slip * (0.16f + 0.10f * speed_norm) * c.mass * c.gravity;

      // This grip directly affects the forward force: a positive correction when
      // grip exceeds the nominal, negative when below.
      float grip_correction = grip - 0.72f;
      float forward_correction = grip_correction * 0.34f * c.mass * c.gravity * (0.4f + speed_norm);

      // Lateral instability: strongest during stiction breaks or melt pulses,
      // and when the clutch churns. This is the flip hazard.
      float stiction_lateral = stiction * std::min(1.0f, slip * 1.8f) * pulse_narrow * 0.20f * c.mass * c.gravity * std::sin(pulse_phase + 1.1f);
      float melt_lateral = melt * pulse_narrow * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.9f);
      float churn_lateral = clutch_churn * 0.14f * std::sin(t * 0.043f + 0.8f);

      // Damping reflects effective grip: low grip means more sliding.
      float damping = 0.04f + 0.13f * grip;

      // Vertical: melt phase 'swallows' the rover slightly at speed; stiction
      // breaks cause a brief upward lurch.
      float melt_sink = melt * (0.04f + 0.05f * speed_norm) * c.mass * c.gravity;
      float break_lift = stiction * std::min(1.0f, slip * 1.8f) * 0.04f * c.mass * c.gravity;

      c.body_force->x += forward_correction + stiction_lateral + melt_lateral + churn_lateral - clutch_churn;
      c.body_force->x -= c.velocity.x * c.mass * damping;
      c.body_force->y -= melt_sink - break_lift - c.velocity.y * c.mass * (0.05f + 0.03f * melt);

      if (c.body_torque) {
        // Stiction breaks and melt pulses both induce a pitching torque that can
        // flip the rover if moving fast or slipping too much.
        float stiction_torque = stiction * std::min(1.0f, slip * 1.8f) * pulse_narrow * (0.5f + 0.7f * speed_norm) * 0.040f * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);
        float melt_torque = melt * pulse_narrow * (0.5f + 0.6f * speed_norm) * 0.028f * c.mass * c.gravity * std::sin(pulse_phase + 1.0f);
        float churn_torque = clutch_churn * 0.022f * std::sin(t * 0.047f + 1.2f);
        *c.body_torque += stiction_torque + melt_torque + churn_torque;
        *c.body_torque -= c.angular_velocity * c.mass * 0.026f * (0.5f + 0.4f * grip);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      float phase = t * 0.0050f + speed * 0.012f;
      float melt = 0.5f + 0.5f * std::sin(phase);
      float stiction = 1.0f - melt;
      float pulse_phase = t * 0.035f + speed * 0.085f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;
      float slip = std::max(0.0f, speed - 0.3f);

      // Energy budget: stiction phase is cheap if the rover maintains low slip
      // (smooth throttle). But if the clutch churns (high slip), the cost spikes.
      // Melt phase punishes both high speed and very low speed, but has a shallow
      // minimum near the 'bite' speed. Clay pulses add a sharp penalty.
      float stiction_cost = stiction * (0.004f + 0.004f * speed_norm + 0.028f * std::min(1.0f, slip * 2.0f)) * 1.3f;
      float melt_cost = melt * (0.010f + 0.038f * speed * speed * 0.12f);
      float bite_bonus = melt * std::exp(-(speed - 1.2f) * (speed - 1.2f) * 1.2f) * 0.010f;
      float pulse_penalty = (stiction * std::min(1.0f, slip * 1.8f) + melt) * pulse_narrow * (0.014f + 0.022f * speed_norm);
      float base_cost = 0.006f;

      *c.energy_cost += (base_cost + stiction_cost + melt_cost - bite_bonus + pulse_penalty) * 1.7f * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, uniform clay with faint grey-green sheen. No visual cue reveals the
    // stiction/melt phase or the pulses. Screen brightness is very low and lidar
    // is nearly blind, forcing the agent to infer the clay state from wheel slip.
    v.ground = {32, 34, 28};         // dark olive clay
    v.particles = {128, 132, 96};     // pale dusty spray
    v.liquid = {20, 22, 16};          // dark hollows
    v.sky = {48, 50, 40};             // dim, hazy sky
    v.particle_rate = 9.0f;
    v.particle_lift = 0.7f;
    v.particle_spread = 1.1f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.4f;
    v.screen_brightness = 0.07f;      // very dark: lidar is extremely expensive and short
    v.liquid_surface = false;
    return v;
  }
};

class CantileverShearVane final : public Biome {
 public:
  std::string_view id() const noexcept override { return "cantilever_shear_vane"; }
  std::string_view display_name() const noexcept override { return "Cantilever Shear Vane"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Excellent nominal grip: the vane torque is the only real hazard.
    p.friction_mul = 0.75f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.002f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.35f * biome_random01(s, 2);
    p.wind_force = 2.5f + 1.5f * biome_random01(s, 3);
    // Cold, dry rock: low thermal transfer, moderate solar.
    p.ambient_temperature = -70.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.50f + 0.15f * biome_random01(s, 5);
    // Solar is weak: recharging is a trap, forcing frugal momentum driving.
    p.solar_charge_rate = 0.10f + 0.06f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.004f * biome_random01(s, 8);
    // Lidar is very expensive and extremely short: the vane angle cannot be seen ahead.
    p.lidar_energy_mul = 6.0f + 1.0f * biome_random01(s, 9);
    p.lidar_range_mul = 0.08f + 0.03f * biome_random01(s, 10);
    // Terrain: broad, smooth, flat plain with negligible craters and steps.
    // The hazard is purely the hidden vane torque, not the geometry.
    p.terrain_amplitude_mul = 0.8f + 0.2f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.4f + 0.1f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.3f + 0.1f * biome_random01(s, 13);
    p.terrain_step_mul = 0.2f + 0.1f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Excellent baseline grip — the torque is the only real challenge.
    return p.friction_mul * 1.12f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Clean, low rolling resistance; the vane torque lives in body effects.
      *c.wheel_force += c.contact->tangent * (-0.15f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.008f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.004f + std::abs(c.wheel_speed) * 0.001f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Vane angle: a slow, position-dependent oscillation that controls the
      // orientation of a buried, hidden vane. The vane is completely invisible —
      // lidar reveals only flat, featureless terrain, and the visual is uniform
      // dark rock. The rover feels its effect only as a lateral force and a yaw
      // torque that grow with speed. The angle cannot be read off; the agent must
      // infer it from body drift and yaw and respond with active steering.
      float vane_phase = t * 0.0052f + c.velocity.x * 0.013f;
      float vane_angle = vane_phase + 0.35f * std::sin(t * 0.019f + c.velocity.x * 0.046f);

      // The vane generates a lateral force that oscillates with its angle, but
      // with a strong component that grows quadratically with speed. At low speed
      // the force is modest and easily corrected; at high speed it becomes
      // overwhelming, shoving the rover sideways off course. There is no safe
      // 'fast cruise'—the rover must find and hold a moderate, steady speed.
      float lateral_vane = std::sin(vane_angle) * (0.10f + 0.30f * speed_norm * speed_norm) * c.mass * c.gravity * 0.35f;

      // A slower 'precession' slowly walks the vane's effective direction around,
      // so holding a single counter-steer angle is never sufficient; the rover
      // must continuously adjust its steering.
      float precession = std::sin(vane_phase * 0.5f + 0.6f) * (0.04f + 0.12f * speed_norm) * c.mass * c.gravity;

      // Damping is moderate — the rover should not be twitchy, but must remain
      // controllable with steering inputs.
      float damping = 0.05f + 0.03f * speed_norm;

      c.body_force->x += lateral_vane + precession - c.velocity.x * c.mass * damping;
      c.body_force->y -= c.velocity.y * c.mass * (0.04f + 0.02f * speed_norm);

      if (c.body_torque) {
        // The vane also applies a yaw torque that tries to rotate the rover
        // toward the lateral force direction. This is the core hazard: a policy
        // that only adjusts throttle, without actively steering against the
        // torque, will be slowly rotated off course and eventually flip when
        // the body yaws far enough on the flat plain.
        float yaw_vane = std::cos(vane_angle) * (0.06f + 0.20f * speed_norm) * c.mass * c.gravity * 0.22f;
        // Pitch damping keeps the rover from rolling immediately; the yaw
        // torque is the primary destabilizer.
        float pitch_damping = 0.03f + 0.02f * speed_norm;
        *c.body_torque += yaw_vane - c.angular_velocity * c.mass * pitch_damping;
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float vane_phase = t * 0.0052f + speed * 0.013f;
      float vane_angle = vane_phase + 0.35f * std::sin(t * 0.019f + speed * 0.046f);

      // Energy cost is low and mostly constant, but there is a clear penalty
      // for high speed: the vane's lateral force grows quadratically with speed,
      // so the rover must spend more energy fighting it. There is also a small
      // penalty for being at very low speed, because the rover must constantly
      // correct its heading while barely moving. The sweet spot is a moderate,
      // steady cruise with gentle steering corrections.
      float speed_penalty = 0.022f * speed_norm * speed_norm;
      float low_speed_penalty = std::max(0.0f, 0.3f - speed_norm) * 0.006f;
      float base_cost = 0.005f;

      *c.energy_cost += (base_cost + speed_penalty + low_speed_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, dark basalt-like rock with faint mineral streaks. The vane angle
    // is completely hidden; the scene is homogeneous and deliberately ambiguous.
    // Low brightness and extremely short lidar force inference from body drift.
    v.ground = {24, 26, 32};         // dark slate
    v.particles = {90, 100, 118};     // fine pale dust
    v.liquid = {12, 14, 20};          // dark hollows
    v.sky = {42, 48, 62};             // dim, hazy sky
    v.particle_rate = 3.0f;
    v.particle_lift = 0.6f;
    v.particle_spread = 0.5f;
    v.base_particles = 1;
    v.max_particles = 12;
    v.particle_size = 1;
    v.ambient_particles = 4;
    v.ambient_drift = 0.8f;
    v.screen_brightness = 0.06f;      // near-total darkness: lidar is extremely expensive and nearly blind
    v.liquid_surface = false;
    return v;
  }
};

class GaleBankDrift final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gale_bank_drift"; }
  std::string_view display_name() const noexcept override { return "Gale Bank Drift"; }
  std::string_view skill_stratum() const noexcept override { return "lateral_force"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Wind; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Firm, dry ground with excellent nominal grip: the hazard is purely the
    // banked, slowly meandering crosswind that grows with speed.
    p.friction_mul = 0.70f + 0.20f * biome_random01(s);
    p.sink_rate = 0.001f + 0.002f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.30f + 0.35f * biome_random01(s, 2);
    p.wind_force = 5.0f + 3.0f * biome_random01(s, 3);
    // Cold, dry: low thermal transfer, but solar is weak so recharging is a trap.
    p.ambient_temperature = -70.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.50f + 0.15f * biome_random01(s, 5);
    p.solar_charge_rate = 0.05f + 0.03f * biome_random01(s, 6);
    p.gravity_mul = 0.95f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.002f + 0.004f * biome_random01(s, 8);
    // Lidar is very expensive and short: the wind bank cannot be seen ahead.
    p.lidar_energy_mul = 5.5f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.10f + 0.04f * biome_random01(s, 10);
    // Terrain: broad, smooth, almost flat plain with negligible craters and steps.
    // A generic terrain-avoidance policy has nothing to avoid and will be caught.
    p.terrain_amplitude_mul = 0.7f + 0.15f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.3f + 0.1f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.3f + 0.1f * biome_random01(s, 13);
    p.terrain_step_mul = 0.2f + 0.1f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Excellent baseline grip — the wind bank is the only real challenge.
    return p.friction_mul * 1.10f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.wheel_force && c.contact) {
      // Clean, low rolling resistance; the wind bank lives in body effects.
      *c.wheel_force += c.contact->tangent * (-0.12f * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * 0.006f);
    }
    if (c.energy_cost) {
      *c.energy_cost += (0.003f + std::abs(c.wheel_speed) * 0.001f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Wind bank phase: a slow, position-dependent oscillation that controls the
      // direction and magnitude of a persistent crosswind. The wind is completely
      // invisible to lidar and the visual is uniform dark rock, so the agent must
      // infer the bank from its own body drift and yaw. The bank slowly meanders,
      // becoming stronger at speed, and is only navigable by actively and
      // continuously counter-steering at a moderate speed.
      float bank_phase = t * 0.0052f + c.velocity.x * 0.013f;
      float bank_angle = bank_phase + 0.35f * std::sin(t * 0.019f + c.velocity.x * 0.046f);

      // The lateral wind force grows QUADRATICALLY with speed. At low speed it
      // is modest and easily corrected; at high speed it becomes overwhelming,
      // shoving the rover sideways off course. A slow, high-torque policy in a
      // low gear will never build enough speed to stay controllable and will
      // drift off-course, while a reckless high-gear sprint gets shoved sideways
      // and flips. The only safe strategy is a moderate cruise with active
      // steering corrections.
      float wind_lateral = std::sin(bank_angle) * (0.10f + 0.35f * speed_norm * speed_norm) * p.wind_force * c.mass * c.gravity * 0.12f;

      // A slower precession walks the bank direction around, so a single constant
      // counter-steer is never sufficient: the rover must continuously adjust.
      float precession = std::sin(bank_phase * 0.5f + 0.6f) * (0.04f + 0.12f * speed_norm) * p.wind_force * c.mass * c.gravity * 0.08f;

      // Damping is moderate — the rover should not be twitchy, but must remain
      // controllable with steering inputs.
      float damping = 0.06f + 0.03f * speed_norm;

      c.body_force->x += wind_lateral + precession - c.velocity.x * c.mass * damping;
      c.body_force->y -= c.velocity.y * c.mass * (0.05f + 0.02f * speed_norm);

      if (c.body_torque) {
        // The wind bank also applies a yaw torque that tries to rotate the rover
        // downwind. This is the core hazard: a policy that only adjusts throttle,
        // without actively steering against the torque, will be slowly rotated off
        // course and eventually flip.
        float yaw_torque = std::cos(bank_angle) * (0.06f + 0.22f * speed_norm) * p.wind_force * c.mass * c.gravity * 0.10f;
        // Pitch damping keeps the rover from rolling immediately; yaw torque is
        // the primary destabilizer.
        float pitch_damping = 0.04f + 0.02f * speed_norm;
        *c.body_torque += yaw_torque - c.angular_velocity * c.mass * pitch_damping;
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float bank_phase = t * 0.0052f + speed * 0.013f;
      float bank_angle = bank_phase + 0.35f * std::sin(t * 0.019f + speed * 0.046f);

      // Energy cost is low and mostly constant. There is a penalty for high speed
      // (fighting the quadratic wind) and a small penalty for very low speed
      // (constantly correcting heading while barely moving). The sweet spot is a
      // moderate, steady cruise with gentle steering corrections — not a low-gear
      // crawl, which wastes energy fighting the wind for little progress, and not
      // a high-gear sprint, which is unstable and drains through steering.
      float speed_penalty = 0.024f * speed_norm * speed_norm;
      float low_speed_penalty = std::max(0.0f, 0.3f - speed_norm) * 0.008f;
      float base_cost = 0.004f;

      *c.energy_cost += (base_cost + speed_penalty + low_speed_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Uniform, dark basalt-like rock with faint wind-streaked lines. The wind
    // bank direction is completely hidden; the scene is homogeneous and
    // deliberately ambiguous. Low brightness and extremely short lidar force
    // inference from body drift.
    v.ground = {24, 26, 32};
    v.particles = {90, 100, 118};
    v.liquid = {12, 14, 20};
    v.sky = {42, 48, 62};
    v.particle_rate = 3.0f;
    v.particle_lift = 0.6f;
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

class SubsidenceThermalSink final : public Biome {
 public:
  std::string_view id() const noexcept override { return "subsidence_thermal_sink"; }
  std::string_view display_name() const noexcept override { return "Subsidence Thermal Sink"; }
  std::string_view skill_stratum() const noexcept override { return "energy_mode"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::Liquid; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate nominal grip, but the subsidence cycle dominates.
    p.friction_mul = 0.35f + 0.15f * biome_random01(s);
    p.sink_rate = 0.008f + 0.010f * biome_random01(s, 1);
    p.viscosity = 1.0f + 1.0f * biome_random01(s, 2);
    p.energy_drain_mul = 1.50f + 0.40f * biome_random01(s, 3);
    p.wind_force = 0.0f;
    // Warm, conductive brine: high thermal transfer but solar is strong only in "stable" windows.
    p.ambient_temperature = 15.0f + 15.0f * biome_random01(s, 4);
    p.thermal_transfer = 2.5f + 0.8f * biome_random01(s, 5);
    p.solar_charge_rate = 2.0f + 0.8f * biome_random01(s, 6);
    p.gravity_mul = 0.90f + 0.10f * biome_random01(s, 7);
    p.crust_deform = 0.010f + 0.014f * biome_random01(s, 8);
    // Lidar is expensive and short: the subsidence cycle must be inferred from energy drain.
    p.lidar_energy_mul = 2.5f + 0.8f * biome_random01(s, 9);
    p.lidar_range_mul = 0.15f + 0.08f * biome_random01(s, 10);
    // Terrain: broad, shallow basins with moderate craters and low steps. The hazard is the subsidence cycle, not the geometry.
    p.terrain_amplitude_mul = 1.4f + 0.30f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.50f + 0.20f * biome_random01(s, 12);
    p.terrain_crater_mul = 1.20f + 0.30f * biome_random01(s, 13);
    p.terrain_step_mul = 0.30f + 0.15f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Moderate grip; the subsidence cycle modulates normal load and traction.
    return p.friction_mul * 0.70f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Slight sink into the warm brine, more with speed and immersion.
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 0.8f * std::abs(c.wheel_speed) + 0.4f * c.immersion);
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 20.0f;
      // Viscous drag plus depth-induced normal resistance.
      float drag = (0.25f + p.viscosity * 1.5f * (1.0f + depth) * c.immersion + 0.12f * c.immersion) * c.wheel_speed;
      *c.wheel_force += c.contact->tangent * (-drag);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.08f * depth * c.immersion));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 20.0f;
      // Base energy cost, higher at speed and depth.
      *c.energy_cost += (0.008f + depth * 0.12f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      // Subsidence cycle: a slow, position-dependent oscillation alternates between
      // 'stable' (firm substrate, low sink, high grip, strong solar) and 'sinking'
      // (soft, collapsing substrate, high sink, low grip, almost no solar).
      // The cycle is invisible to lidar and visually uniform, so the agent must
      // infer it from the rover's energy drain and sink rate.
      float cycle_phase = t * 0.0044f + c.velocity.x * 0.011f;
      float sinking = 0.5f + 0.5f * std::sin(cycle_phase);  // 1 = sinking, 0 = stable
      float stable = 1.0f - sinking;

      // A faster sub-oscillation creates 'collapse pulses': brief, intense sink
      // events within the sinking phase, reducing normal load and traction.
      // These are the flip hazard; a policy that drives at constant speed through
      // the sinking phase gets caught.
      float pulse_phase = t * 0.031f + c.velocity.x * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;  // narrow, strong peaks

      // Sinking phase: the substrate collapses, creating a downward pull that
      // increases sink and reduces traction, and a lateral instability that grows
      // with speed. The stable phase is firm, with stable grip and a small forward assist.
      float sinking_pull = sinking * 0.05f * c.mass * c.gravity;
      float stable_assist = stable * 0.025f * c.mass * c.gravity;

      float sinking_lateral = sinking * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.1f);
      float stable_drag = stable * 0.01f * c.mass * c.gravity * speed_norm;

      // Collapse pulses: strongest in sinking phase, cause a brief downward shove
      // that can destabilise the rover if moving fast.
      float pulse_pull = sinking * pulse_narrow * (0.14f + 0.20f * speed_norm) * c.mass * c.gravity;
      float pulse_lateral = sinking * pulse_narrow * (0.05f + 0.10f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);

      // Damping: higher in stable (firm), lower in sinking (slippery), leading to drift.
      float damping = 0.05f + 0.08f * stable + 0.02f * speed_norm;

      // Vertical: sinking phase has a downward pull that reduces normal load;
      // stable phase is firm, improving grip.
      c.body_force->x += stable_assist + sinking_lateral + pulse_lateral;
      c.body_force->x -= (stable_drag + c.velocity.x * c.mass * damping);
      c.body_force->y -= (sinking_pull + pulse_pull + c.velocity.y * c.mass * (0.04f + 0.02f * stable));

      if (c.body_torque) {
        // Collapse pulses induce a rocking torque, strongest at speed through a pulse.
        float torque = sinking * pulse_narrow * (0.5f + 0.6f * speed_norm) * 0.028f * c.mass * c.gravity * std::sin(pulse_phase + 0.9f);
        *c.body_torque += torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * stable);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.10f);

      float cycle_phase = t * 0.0044f + speed * 0.011f;
      float sinking = 0.5f + 0.5f * std::sin(cycle_phase);
      float stable = 1.0f - sinking;
      float pulse_phase = t * 0.031f + speed * 0.073f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: stable phase is cheap and rewards steady cruising. Sinking phase
      // costs more per unit TIME, not distance, because the rover is fighting the
      // collapsing substrate. Collapse pulses add a sharp penalty.
      //
      // Critically, the thermal drain is strongly coupled: the warm brine accelerates
      // battery drain during the sinking phase, so a policy that tries to push through
      // the sinking phase (rather than stopping to charge) will run out of energy.
      // A policy that stops to charge during the stable phase will get strong solar
      // and survive; a policy that stops during the sinking phase gets almost no solar
      // and sinks deeper, wasting energy.
      float stable_cost = stable * (0.004f + 0.003f * speed_norm) * 1.2f;
      float sinking_time_cost = sinking * 0.022f * p.energy_drain_mul;
      float sinking_speed_cost = sinking * speed * 0.0012f * (1.0f + 0.5f * speed_norm);
      float pulse_penalty = sinking * pulse_narrow * (0.012f + 0.016f * speed_norm) * p.energy_drain_mul;

      // Thermal drain: warm brine saps battery, strongly coupled to sinking phase.
      float thermal_drain = p.thermal_transfer * 0.003f * (1.0f + sinking * 1.2f);

      // Base cost.
      float base_cost = 0.005f;

      *c.energy_cost += (base_cost + stable_cost + sinking_time_cost + sinking_speed_cost + pulse_penalty + thermal_drain) * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, warm brine with faint amber glow. No visual cue reveals the
    // stable/sinking pattern; it must be inferred from sink and energy drain.
    v.ground = {36, 28, 20};
    v.particles = {190, 150, 90};
    v.liquid = {48, 62, 36};
    v.sky = {70, 56, 38};
    v.particle_rate = 7.0f;
    v.particle_lift = 1.2f;
    v.particle_spread = 1.0f;
    v.base_particles = 2;
    v.max_particles = 24;
    v.particle_size = 2;
    v.ambient_particles = 8;
    v.ambient_drift = 1.0f;
    v.screen_brightness = 0.15f;  // dark: lidar is expensive and short, forcing inference
    v.liquid_surface = true;
    return v;
  }
};

class GravitySinkCrawl final : public Biome {
 public:
  std::string_view id() const noexcept override { return "gravity_sink_crawl"; }
  std::string_view display_name() const noexcept override { return "Gravity Sink Crawl"; }
  std::string_view skill_stratum() const noexcept override { return "gravity_change"; }
  BiomeSplit split() const noexcept override { return BiomeSplit::Train; }
  MechanicType visual_type() const noexcept override { return MechanicType::LowGravity; }

  MechanicParams sample_params(uint64_t s) const noexcept override {
    MechanicParams p;
    // Moderate nominal grip but the gravity oscillation dominates.
    p.friction_mul = 0.45f + 0.20f * biome_random01(s);
    p.sink_rate = 0.010f + 0.016f * biome_random01(s, 1);
    p.viscosity = 0.0f;
    p.energy_drain_mul = 1.25f + 0.30f * biome_random01(s, 2);
    p.wind_force = 0.3f + 0.6f * biome_random01(s, 3);
    // Cold, dry with weak solar: recharging is a trap.
    p.ambient_temperature = -70.0f + 20.0f * biome_random01(s, 4);
    p.thermal_transfer = 0.50f + 0.15f * biome_random01(s, 5);
    p.solar_charge_rate = 0.03f + 0.02f * biome_random01(s, 6);
    p.gravity_mul = 0.50f + 0.20f * biome_random01(s, 7);
    p.crust_deform = 0.008f + 0.012f * biome_random01(s, 8);
    // Lidar is very expensive and nearly blind: the gravity phase is invisible.
    p.lidar_energy_mul = 6.0f + 1.5f * biome_random01(s, 9);
    p.lidar_range_mul = 0.08f + 0.03f * biome_random01(s, 10);
    // Terrain: smooth, low-amplitude basins with almost no craters or steps.
    // The only real obstacle is the gravity-driven sink, not the geometry.
    p.terrain_amplitude_mul = 0.7f + 0.15f * biome_random01(s, 11);
    p.terrain_roughness_mul = 0.4f + 0.1f * biome_random01(s, 12);
    p.terrain_crater_mul = 0.4f + 0.1f * biome_random01(s, 13);
    p.terrain_step_mul = 0.2f + 0.1f * biome_random01(s, 14);
    return p;
  }

  float friction_scale(const MechanicParams& p) const noexcept override {
    // Nominal baseline; the sink law is imposed in body effects.
    return p.friction_mul * 0.65f;
  }

  void apply_effects(const MechanicParams& p, MechanicContext& c) const noexcept override {
    if (c.contact) {
      // Sink into the loose regolith, strongly speed-dependent but saturating.
      float speed = std::abs(c.wheel_speed);
      c.contact->penetration += p.sink_rate * c.dt * (1.0f + 2.0f * std::tanh(speed * 0.2f));
    }
    if (c.wheel_force && c.contact) {
      float depth = c.contact->penetration * 30.0f;
      float speed = std::abs(c.wheel_speed);
      // Strong drag from the deep regolith; deeper = more resistant.
      *c.wheel_force += c.contact->tangent * (-(0.25f + 0.30f * depth * speed) * c.wheel_speed);
      *c.wheel_force -= c.contact->normal * (c.contact->normal_force * (0.03f + 0.12f * depth));
    }
    if (c.energy_cost) {
      float depth = c.contact->penetration * 30.0f;
      // High base cost from the loose regolith, plus depth and speed.
      *c.energy_cost += (0.015f + depth * 0.20f + std::abs(c.wheel_speed) * 0.005f) * p.energy_drain_mul * c.dt;
    }
  }

  void apply_body_effects(const MechanicParams& p, MechanicBodyContext& c) const noexcept override {
    if (c.body_force) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      // Gravity phase: a slow, position-dependent oscillation flips between
      // 'light' (effective gravity ~0.35x, buoyant, cheap to move, but low traction
      // and high sink) and 'heavy' (effective gravity ~1.7x, heavy, high traction,
      // but expensive to climb and dangerous to descend). The phase is completely
      // invisible: lidar is nearly blind and the visual is uniform dark rock, forcing
      // the agent to infer the gravity state from suspension compression, sink rate,
      // and body pitch.
      float gravity_phase = t * 0.0048f + c.velocity.x * 0.012f;
      float heavy = 0.5f + 0.5f * std::sin(gravity_phase);  // 1 = heavy, 0 = light
      float light = 1.0f - heavy;

      // Fast sub-oscillation: 'sink pulses' -- brief, sharp gravity swings that
      // increase sink and reduce traction. These are the primary flip hazard:
      // a policy that simply maintains throttle through a pulse gets destabilised.
      float pulse_phase = t * 0.034f + c.velocity.x * 0.083f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Effective gravity multiplier: light ~0.35x, heavy ~1.7x, with pulses pushing heavy to ~2.0x.
      float grav_mul = 1.70f * heavy + 0.35f * light + 0.30f * heavy * pulse_narrow;
      float effective_g = p.gravity_mul * grav_mul;
      float base_g = p.gravity_mul;

      // Weight force: heavy phase presses down, increasing traction but also rolling
      // resistance and sink. Light phase lifts the rover, reducing traction and
      // making the rover more prone to sinking into the loose regolith.
      float weight = (effective_g - base_g) * c.mass * c.gravity * 0.25f;
      float buoyancy = (base_g - effective_g) * c.mass * c.gravity * 0.20f;

      // Core sink law: light phase has low normal load, so the wheels sink deeper
      // into the regolith. This creates a strong drag that grows with speed.
      // Heavy phase has high normal load, so the wheels stay on top, but climbing
      // is expensive and descending fast is dangerous.
      // This forces a completely different policy: light phase rewards slow,
      // deliberate crawling with gentle throttle to avoid sinking; heavy phase
      // rewards a brisker cruise with careful braking on descents.
      float light_sink_drag = light * (0.06f + 0.18f * speed_norm) * c.mass * c.gravity;
      float heavy_grip = heavy * (0.55f + 0.25f * std::max(0.0f, 1.0f - speed_norm * 0.5f));
      float light_grip = light * (0.20f + 0.25f * (1.0f - speed_norm * 0.7f));
      float traction = heavy_grip + light_grip;
      traction = std::max(0.03f, traction);
      float grip_correction = traction - 0.75f;
      float forward_correction = grip_correction * 0.30f * c.mass * c.gravity * (0.4f + speed_norm);

      // Slope-dependent heavy torque: when the rover is on a slope (represented by
      // body pitch, approximated from angular velocity), heavy gravity amplifies the
      // torque that tries to tip the rover over the crest. This is the key hazard:
      // a policy that simply maintains speed through a heavy crest will pitch over.
      float pitch_proxy = std::max(0.0f, std::abs(c.angular_velocity) * 0.1f);
      float slope_torque = heavy * pitch_proxy * (0.5f + 0.8f * speed_norm) * 0.055f * c.mass * c.gravity;

      // Lateral instability: light phase causes drift, stronger at speed and during
      // gravity pulses. This is what punishes full throttle in light phase.
      float lateral_drift = light * (0.10f + 0.18f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 1.1f);
      float pulse_lateral = light * pulse_narrow * (0.12f + 0.16f * speed_norm) * c.mass * c.gravity * std::sin(pulse_phase + 0.8f);

      // Damping: higher in heavy (heavy, stable), lower in light (light, easy to sway).
      float damping = 0.05f + 0.09f * heavy + 0.02f * speed_norm;

      c.body_force->x += forward_correction + lateral_drift + pulse_lateral;
      c.body_force->x -= (light_sink_drag + c.velocity.x * c.mass * damping);
      c.body_force->y += weight + buoyancy - c.velocity.y * c.mass * (0.05f + 0.02f * light);

      if (c.body_torque) {
        // Heavy pulses induce a strong pitching torque, especially on slopes
        // (pitch_proxy). Light-phase pulses cause a rocking torque that can tip the
        // rover over crests.
        float heavy_torque = heavy * pulse_narrow * (0.5f + 0.7f * speed_norm + slope_torque) * 0.045f * c.mass * c.gravity * std::sin(pulse_phase + 0.7f);
        float light_torque = light * pulse_narrow * (0.5f + 0.6f * speed_norm) * 0.030f * c.mass * c.gravity * std::sin(pulse_phase + 0.9f);
        *c.body_torque += heavy_torque + light_torque - c.angular_velocity * c.mass * 0.025f * (0.5f + 0.4f * heavy);
      }
    }

    if (c.energy_cost) {
      float t = static_cast<float>(c.step_index);
      float speed = std::abs(c.velocity.x);
      float speed_norm = std::tanh(speed * 0.12f);

      float gravity_phase = t * 0.0048f + speed * 0.012f;
      float heavy = 0.5f + 0.5f * std::sin(gravity_phase);
      float light = 1.0f - heavy;
      float pulse_phase = t * 0.034f + speed * 0.083f;
      float pulse = 0.5f + 0.5f * std::sin(pulse_phase);
      float pulse_narrow = pulse * pulse;

      // Energy budget: heavy phase is expensive per unit distance (heavy weight,
      // climbing costs), and pulses add a sharp penalty. Light phase is cheap
      // per unit distance but punishes high throttle through sink drag and drift.
      // The optimal strategy is to crawl slowly through heavy (especially on climbs),
      // then cruise at moderate speed with feathered throttle in light.
      // Solar is extremely weak throughout, so recharging is a trap: the agent must
      // conserve energy and finish on the starting charge plus minimal harvest.
      float heavy_cost = heavy * (0.014f + 0.032f * speed_norm * speed_norm);
      float light_cost = light * (0.005f + 0.018f * speed_norm * speed_norm);
      float pulse_penalty = heavy * pulse_narrow * (0.018f + 0.024f * speed_norm);
      float base_cost = 0.005f;

      *c.energy_cost += (base_cost + heavy_cost + light_cost + pulse_penalty) * p.energy_drain_mul * c.dt;
    }
  }

  BiomeVisuals visuals() const noexcept override {
    BiomeVisuals v;
    // Dark, uniform regolith with faint grey-blue shimmer. No visual cue
    // reveals the gravity phase. Screen brightness is very low and lidar is nearly
    // blind, forcing the agent to infer the gravity state from wheel load and body pitch.
    v.ground = {28, 32, 40};         // dark slate-blue regolith
    v.particles = {108, 120, 138};    // fine pale dust
    v.liquid = {14, 18, 26};          // dark hollows
    v.sky = {50, 56, 72};             // dim, hazy sky
    v.particle_rate = 5.0f;
    v.particle_lift = 1.7f;           // low-gravity floating dust
    v.particle_spread = 0.9f;
    v.base_particles = 1;
    v.max_particles = 16;
    v.particle_size = 2;
    v.ambient_particles = 6;
    v.ambient_drift = 1.2f;
    v.screen_brightness = 0.07f;      // very dark: lidar is extremely expensive and short
    v.liquid_surface = false;
    return v;
  }
};

inline void append(std::vector<const Biome*>& out) {
  static const TidalBrakeVault biome_0; out.push_back(&biome_0);
  static const GravityWellBrine biome_1; out.push_back(&biome_1);
  static const TideSluiceReservoir biome_2; out.push_back(&biome_2);
  static const ThermalSluiceLag biome_3; out.push_back(&biome_3);
  static const ClayMantleBog biome_4; out.push_back(&biome_4);
  static const CantileverShearVane biome_5; out.push_back(&biome_5);
  static const GaleBankDrift biome_6; out.push_back(&biome_6);
  static const SubsidenceThermalSink biome_7; out.push_back(&biome_7);
  static const GravitySinkCrawl biome_8; out.push_back(&biome_8);
}
// </MARS_GENERATED_BIOMES>
}  // namespace generated_biomes

inline constexpr std::string_view kBiomeBankVersion = "sha256:7e5ffcf8b67ca543fa0d9913ff206ccc29b7a38e1558b6ff2d5311a05a78464d";

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

}  // namespace mars
