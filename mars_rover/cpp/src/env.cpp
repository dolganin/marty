#include "mars/env.hpp"

#include <algorithm>
#include <cmath>

namespace mars {

Env::Env(EnvConfig config) : config_(std::move(config)), physics_(config_.physics) {
  terrain_.configure(config_.terrain);
}

void Env::reset(uint64_t seed, bool trial_start, float* obs_out) {
  rng_.seed(seed);
  const int next_episode_in_trial = trial_start ? 0 : state_.episode_in_trial + 1;
  if (trial_start) {
    sample_hidden_mechanic();
  }
  terrain_.generate(seed ^ 0x9e3779b97f4a7c15ULL);
  const float spawn_y = terrain_.query(1.0f).height + 1.0f;
  physics_.initialize_state(config_.rig, state_, {1.0f, spawn_y});
  state_.trial_start = trial_start;
  state_.episode_in_trial = next_episode_in_trial;
  stuck_counter_ = 0;
  build_observation(obs_out);
}

StepOutput Env::step(int action, float* obs_out) {
  state_.previous_x = state_.body.position.x;
  const auto stats =
      physics_.step(config_.rig, terrain_, state_, action, mechanic_type_, mechanic_params_);

  if (mechanic_type_ == MechanicType::Crust || mechanic_type_ == MechanicType::Sand ||
      mechanic_type_ == MechanicType::Mud) {
    float deform_scale = 0.0f;
    if (mechanic_type_ == MechanicType::Crust) {
      deform_scale = mechanic_params_.crust_deform;
    } else if (mechanic_type_ == MechanicType::Sand) {
      deform_scale = mechanic_params_.sink_rate * 0.25f;
    } else if (mechanic_type_ == MechanicType::Mud) {
      deform_scale = 0.0025f + mechanic_params_.viscosity * 0.0008f;
    }
    for (int i = 0; i < state_.wheel_count; ++i) {
      const auto& c = stats.contacts[static_cast<size_t>(i)];
      if (c.active) {
        terrain_.deform(c.point.x, state_.wheels[static_cast<size_t>(i)].radius * 1.5f,
                        deform_scale * stats.contacts[static_cast<size_t>(i)].penetration);
      }
    }
  }
  state_.damage += stats.hard_contact > 1500.0f ? (stats.hard_contact - 1500.0f) * 0.000001f : 0.0f;

  const bool finished = state_.body.position.x >= config_.termination.finish_x;
  const bool flipped = is_flipped();
  if (state_.step_index > 120 && std::abs(state_.body.velocity.x) < 0.02f) {
    stuck_counter_ += 1;
  } else {
    stuck_counter_ = 0;
  }
  const bool stuck = is_stuck();
  StepOutput out{};
  out.terminated = finished || flipped || state_.energy <= config_.termination.min_energy || stuck;
  out.truncated = state_.step_index + 1 >= config_.termination.max_steps;
  out.reward = compute_reward(config_.reward, state_, stats.energy_cost, finished, flipped, stuck);
  state_.last_reward = out.reward;
  state_.previous_action = action;
  state_.step_index += 1;
  build_observation(obs_out);
  return out;
}

void Env::build_observation(float* obs_out) const {
  if (!obs_out) {
    return;
  }
  int k = 0;
  obs_out[k++] = state_.body.position.x;
  obs_out[k++] = state_.body.position.y;
  obs_out[k++] = state_.body.velocity.x;
  obs_out[k++] = state_.body.velocity.y;
  obs_out[k++] = state_.body.angle;
  obs_out[k++] = state_.body.angular_velocity;
  obs_out[k++] = state_.energy;
  obs_out[k++] = state_.trial_start ? 1.0f : 0.0f;
  for (int i = 0; i < kMaxWheels; ++i) {
    const auto& w = state_.wheels[static_cast<size_t>(i)];
    obs_out[k++] = (i < state_.wheel_count && w.in_contact) ? 1.0f : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.slip : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.normal_force : 0.0f;
  }
  constexpr float sample_dx = 0.5f;
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {
    const auto s = terrain_.query(state_.body.position.x + sample_dx * static_cast<float>(i + 1));
    obs_out[k++] = s.height - state_.body.position.y;
  }
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {
    const auto s = terrain_.query(state_.body.position.x + sample_dx * static_cast<float>(i + 1));
    obs_out[k++] = s.slope;
  }
  obs_out[k++] = static_cast<float>(state_.previous_action);
  obs_out[k++] = state_.last_reward;
  obs_out[k++] = static_cast<float>(state_.episode_in_trial);
}

void Env::sample_hidden_mechanic() {
  std::uniform_int_distribution<int> type_dist(0, 6);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  mechanic_type_ = static_cast<MechanicType>(type_dist(rng_));
  mechanic_params_ = MechanicParams{};
  switch (mechanic_type_) {
    case MechanicType::Sand:
      mechanic_params_.friction_mul = 0.50f + 0.28f * u(rng_);
      mechanic_params_.sink_rate = 0.018f + 0.035f * u(rng_);
      mechanic_params_.energy_drain_mul = 1.25f + 0.65f * u(rng_);
      break;
    case MechanicType::Ice:
      mechanic_params_.friction_mul = 0.10f + 0.22f * u(rng_);
      break;
    case MechanicType::Mud:
      mechanic_params_.friction_mul = 0.38f + 0.28f * u(rng_);
      mechanic_params_.viscosity = 1.5f + 4.0f * u(rng_);
      mechanic_params_.energy_drain_mul = 1.5f + 1.0f * u(rng_);
      break;
    case MechanicType::Wind:
      mechanic_params_.wind_force = (u(rng_) < 0.5f ? -1.0f : 1.0f) * (2.0f + 5.0f * u(rng_));
      break;
    case MechanicType::LowGravity:
      mechanic_params_.gravity_mul = 0.32f + 0.35f * u(rng_);
      break;
    case MechanicType::Crust:
      mechanic_params_.friction_mul = 0.75f + 0.25f * u(rng_);
      mechanic_params_.crust_deform = 0.006f + 0.02f * u(rng_);
      break;
    case MechanicType::Normal:
    default:
      break;
  }
}

bool Env::is_flipped() const {
  return std::abs(state_.body.angle) > config_.termination.flip_angle;
}

bool Env::is_stuck() const {
  const float speed = std::abs(state_.body.velocity.x);
  if (state_.step_index > 120 && speed < 0.02f) {
    return stuck_counter_ > config_.termination.stuck_steps;
  }
  return false;
}

}  // namespace mars
