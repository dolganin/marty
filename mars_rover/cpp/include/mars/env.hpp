#pragma once

#include <array>
#include <cstdint>
#include <random>
#include <string_view>

#include "mars/observation.hpp"
#include "mars/physics.hpp"
#include "mars/reward.hpp"

namespace mars {

inline constexpr std::string_view kEnvironmentVersion =
    "mars-env-v7-ballistic-survival-progress-loss-macro-shift";

struct EnvConfig {
  TerrainConfig terrain{};
  PhysicsConfig physics{};
  RewardConfig reward{};
  TerminationConfig termination{};
  RoverRig rig = RoverRig::default_two_wheel();
  int episodes_per_trial = 4;
  // 0: all biomes (debug), 1: train only, 2: test only, 3: anchors only.
  int biome_split = 1;
  int fixed_biome_id = -1;  // >= 0 selects one registry entry for deterministic debugging.
  bool debug = false;
  // A trial is not one fixed biome: it is a chain of zones concatenated end to
  // end so the rover keeps encountering unfamiliar mechanics as it advances,
  // and "how far it got" (state().body.position.x / high_water_x) becomes the
  // signal, not completion of one fixed course. Anchors are always included as
  // a stable backbone; the rest is drawn from the active split so the specific
  // chain differs trial to trial while the anchor set stays comparable across
  // runs. Ignored when fixed_biome_id >= 0 (single-zone debugging mode).
  // Off by default: single-biome-per-episode tooling (evaluate_biomes.py,
  // policy_divergence.py, anchor_eval.py) samples biome_split directly without
  // fixed_biome_id and asserts an exact, anchor-free set per split; chaining
  // would silently mix the anchor backbone into those measurements. Training
  // and manual play configs opt in explicitly.
  bool chain_biomes = false;
  int chain_zone_count = 14;
  float chain_segment_min_length = 30.0f;
  float chain_segment_max_length = 70.0f;
};

struct StepOutput {
  float reward = 0.0f;
  bool terminated = false;
  bool truncated = false;
};

class Env {
 public:
  explicit Env(EnvConfig config = {});

  void reset(uint64_t seed, bool trial_start, float* obs_out);
  StepOutput step(int action, float* obs_out);
  void build_observation(float* obs_out) const;

  const RoverState& state() const { return state_; }
  const Terrain& terrain() const { return terrain_; }
  const EnvConfig& config() const { return config_; }
  MechanicType mechanic_type() const { return mechanic_type_; }
  const MechanicParams& mechanic_params() const { return mechanic_params_; }
  const MechanicZone& mechanic_at(float x) const { return mechanic_layout_.at(x); }

  int obs_dim() const { return kObservationDim; }
  int action_dim() const { return 4096; }

 private:
  void select_mechanic_layout(uint64_t seed);
  void finalize_mechanic_layout();
  bool is_flipped() const;
  bool is_stuck() const;

  EnvConfig config_{};
  Terrain terrain_{};
  PhysicsEngine physics_{};
  RoverState state_{};
  std::mt19937_64 rng_{1};
  MechanicType mechanic_type_ = MechanicType::Normal;
  MechanicParams mechanic_params_{};
  MechanicLayout mechanic_layout_{};
  uint64_t trial_mechanic_seed_ = 0;
  bool has_trial_mechanic_seed_ = false;
  int stuck_counter_ = 0;
  float best_progress_x_ = 0.0f;
  std::array<float, kMaxMechanicZones> pending_basin_depth_{};
};

}  // namespace mars
