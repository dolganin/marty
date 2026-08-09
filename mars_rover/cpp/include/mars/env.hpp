#pragma once

#include <cstdint>
#include <random>
#include <string_view>

#include "mars/observation.hpp"
#include "mars/physics.hpp"
#include "mars/reward.hpp"

namespace mars {

inline constexpr std::string_view kEnvironmentVersion =
    "mars-env-v4-disjoint-train-test-anchor-splits";

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
  float pending_basin_depth_ = -1.0f;
};

}  // namespace mars
