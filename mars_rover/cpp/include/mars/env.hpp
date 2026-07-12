#pragma once

#include <cstdint>
#include <random>

#include "mars/observation.hpp"
#include "mars/physics.hpp"
#include "mars/reward.hpp"

namespace mars {

struct EnvConfig {
  TerrainConfig terrain{};
  PhysicsConfig physics{};
  RewardConfig reward{};
  TerminationConfig termination{};
  RoverRig rig = RoverRig::default_two_wheel();
  int episodes_per_trial = 4;
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

  int obs_dim() const { return kObservationDim; }
  int action_dim() const { return 6; }

 private:
  void sample_hidden_mechanic();
  bool is_flipped() const;
  bool is_stuck() const;

  EnvConfig config_{};
  Terrain terrain_{};
  PhysicsEngine physics_{};
  RoverState state_{};
  std::mt19937_64 rng_{1};
  MechanicType mechanic_type_ = MechanicType::Normal;
  MechanicParams mechanic_params_{};
  int stuck_counter_ = 0;
};

}  // namespace mars
