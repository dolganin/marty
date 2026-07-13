#include "mars/env.hpp"
#include "mars/biome_bank.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mars {

Env::Env(EnvConfig config) : config_(std::move(config)), physics_(config_.physics) {
  terrain_.configure(config_.terrain);
}

void Env::reset(uint64_t seed, bool trial_start, float* obs_out) {
  rng_.seed(seed);
  const int next_episode_in_trial = trial_start ? 0 : state_.episode_in_trial + 1;
  terrain_.generate(seed ^ 0x9e3779b97f4a7c15ULL);
  if (trial_start || !has_trial_mechanic_seed_) {
    trial_mechanic_seed_ = seed;
    has_trial_mechanic_seed_ = true;
  }
  generate_mechanic_layout(trial_mechanic_seed_);
  const float spawn_y = terrain_.query(1.0f).height + 1.0f;
  physics_.initialize_state(config_.rig, state_, {1.0f, spawn_y});
  state_.trial_start = trial_start;
  state_.episode_in_trial = next_episode_in_trial;
  stuck_counter_ = 0;
  build_observation(obs_out);
}

StepOutput Env::step(int action, float* obs_out) {
  state_.previous_x = state_.body.position.x;
  const auto stats = physics_.step(config_.rig, terrain_, state_, action, mechanic_layout_);

  const auto& current_zone = mechanic_layout_.at(state_.body.position.x);
  mechanic_type_ = current_zone.type;
  mechanic_params_ = current_zone.params;

  for (int i = 0; i < state_.wheel_count; ++i) {
    const auto& c = stats.deformation_contacts[static_cast<size_t>(i)];
    if (c.active) {
      const auto& zone = mechanic_layout_.at(c.x);
      float deform_scale = 0.0f;
      if (zone.type == MechanicType::Crust) {
        deform_scale = c.penetration > 0.008f ? 0.025f + zone.params.crust_deform * 2.0f : 0.0f;
      } else if (zone.type == MechanicType::Sand) {
        deform_scale = zone.params.sink_rate * 1.5f;
      } else if (zone.type == MechanicType::Mud) {
        deform_scale = 0.012f + zone.params.viscosity * 0.003f;
      }
      if (deform_scale > 0.0f) {
        terrain_.deform(c.x, state_.wheels[static_cast<size_t>(i)].radius * 1.5f,
                        deform_scale * c.penetration);
      }
    }
  }
  state_.damage += stats.hard_contact > 1500.0f ? (stats.hard_contact - 1500.0f) * 0.000001f : 0.0f;

  const bool finished = state_.body.position.x >= config_.termination.finish_x;
  const bool flipped = is_flipped();
  if (!state_.solar_panel_requested && state_.solar_panel_deployment <= 0.001f &&
      state_.step_index > 120 && std::abs(state_.body.velocity.x) < 0.02f) {
    stuck_counter_ += 1;
  } else {
    stuck_counter_ = 0;
  }
  const bool stuck = is_stuck();
  StepOutput out{};
  out.terminated = finished || flipped || state_.energy <= config_.termination.min_energy || stuck;
  out.truncated = config_.termination.max_steps > 0 &&
                  state_.step_index + 1 >= config_.termination.max_steps;
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
  const bool lidar_active = state_.lidar_active_steps > 0;
  const int height_base = k;
  const int slope_base = k + kTerrainSamplesAhead;
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {
    float height, slope;
    terrain_.query_height_slope(state_.body.position.x + sample_dx * static_cast<float>(i + 1),
                                height, slope);
    const float distance = sample_dx * static_cast<float>(i + 1);
    const bool visible = lidar_active && distance <= state_.lidar_range;
    obs_out[height_base + i] = visible ? height - state_.body.position.y : 0.0f;
    obs_out[slope_base + i] = visible ? slope : 0.0f;
  }
  k += kTerrainSamplesAhead * 2;
  for (int i = 0; i < kBiomeSamplesAhead; ++i) {
    const float sample_x = state_.body.position.x + 1.5f * static_cast<float>(i + 1);
    const auto& zone = mechanic_layout_.at(sample_x);
    const bool visible = lidar_active && 1.5f * static_cast<float>(i + 1) <= state_.lidar_range;
    obs_out[k++] = visible ? static_cast<float>(static_cast<int>(zone.type)) / 7.0f : 0.0f;
    obs_out[k++] = visible && zone.type == MechanicType::Liquid
                       ? std::max(0.0f, zone.liquid_level - terrain_.query(sample_x).height) * 0.5f
                       : 0.0f;
    obs_out[k++] = visible
                       ? mechanic_layout_.thermal_at(sample_x, zone).ambient_temperature / 100.0f
                       : 0.0f;
  }
  obs_out[k++] = static_cast<float>(state_.gear_index + 1) / 8.0f;
  obs_out[k++] = state_.driveline_load_factor;
  obs_out[k++] = state_.gear_energy_multiplier / 3.1f;
  obs_out[k++] = state_.engine_rpm / 9000.0f;
  obs_out[k++] = static_cast<float>(state_.drive_mode) * 0.5f;
  obs_out[k++] = state_.engine_stalled ? 1.0f : 0.0f;
  obs_out[k++] = state_.clutch_engagement;
  obs_out[k++] = state_.engine_temperature / 120.0f;
  obs_out[k++] = state_.ambient_temperature / 100.0f;
  obs_out[k++] = state_.engine_overheated ? 1.0f : 0.0f;
  obs_out[k++] = state_.engine_cold_locked ? 1.0f : 0.0f;
  obs_out[k++] = state_.solar_panel_deployment;
  obs_out[k++] = state_.charging_active ? 1.0f : 0.0f;
  obs_out[k++] = state_.solar_charge_rate / 3.0f;
  obs_out[k++] = lidar_active ? 1.0f : 0.0f;
  obs_out[k++] = state_.lidar_cooldown_steps > 0
                     ? static_cast<float>(state_.lidar_cooldown_steps) * config_.physics.dt
                     : 0.0f;
  obs_out[k++] = state_.lidar_range / std::max(1.0f, config_.physics.lidar_base_range);
  obs_out[k++] = static_cast<float>(state_.previous_action);
  obs_out[k++] = state_.last_reward;
  obs_out[k++] = static_cast<float>(state_.episode_in_trial);
}

void Env::generate_mechanic_layout(uint64_t seed) {
  rng_.seed(seed ^ 0xd1b54a32d192ed03ULL);
  const auto& bank = biome_registry();
  std::vector<int> eligible_biomes;
  eligible_biomes.reserve(bank.size());
  if (config_.fixed_biome_id >= 0 && config_.fixed_biome_id < static_cast<int>(bank.size())) {
    eligible_biomes.push_back(config_.fixed_biome_id);
  } else {
    for (int i = 0; i < std::min(8, static_cast<int>(bank.size())); ++i)
      eligible_biomes.push_back(i);
    for (int i = 8; i < static_cast<int>(bank.size()); ++i) {
      const auto split = bank[static_cast<size_t>(i)]->split();
      if (config_.biome_split == 0 || static_cast<int>(split) == config_.biome_split)
        eligible_biomes.push_back(i);
    }
  }
  std::vector<int> biome_bag = eligible_biomes;
  size_t bag_cursor = biome_bag.size();
  auto draw_balanced_biome = [&](int previous) {
    if (bag_cursor >= biome_bag.size()) {
      std::shuffle(biome_bag.begin(), biome_bag.end(), rng_);
      bag_cursor = 0;
      if (biome_bag.size() > 1 && biome_bag.front() == previous)
        std::swap(biome_bag.front(), biome_bag[1]);
    }
    return biome_bag[bag_cursor++];
  };
  std::uniform_real_distribution<float> length_dist(8.0f, 26.0f);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);

  mechanic_layout_.count = 0;
  float begin = -1000000.0f;
  float visible_begin = 0.0f;
  int previous_type = -1;
  int pending_liquid_id = -1;
  bool pending_exit_beach = false;
  const float world_end = std::max(terrain_.length(), config_.terrain.length);
  while (visible_begin < world_end && mechanic_layout_.count < kMaxMechanicZones) {
    bool beach_zone = false;
    int type_index = 0;
    if (pending_liquid_id >= 0) {
      type_index = pending_liquid_id;
      pending_liquid_id = -1;
    } else if (pending_exit_beach) {
      type_index = builtin_biome_id(MechanicType::Sand);
      beach_zone = true;
      pending_exit_beach = false;
    } else {
      type_index = draw_balanced_biome(previous_type);
      const bool previous_is_sand = previous_type >= 0 &&
          bank[static_cast<size_t>(previous_type)]->visual_type() == MechanicType::Sand;
      if (bank[static_cast<size_t>(type_index)]->visual_type() == MechanicType::Liquid &&
          !previous_is_sand) {
        pending_liquid_id = type_index;
        type_index = builtin_biome_id(MechanicType::Sand);
        beach_zone = true;
      }
    }
    const Biome& biome = *bank[static_cast<size_t>(type_index)];
    const bool is_liquid = biome.visual_type() == MechanicType::Liquid;
    const float zone_length = beach_zone ? 5.0f + u(rng_) * 5.0f
                                         : (is_liquid ? 18.0f + u(rng_) * 24.0f
                                                      : length_dist(rng_));
    const float end = visible_begin + zone_length;
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(mechanic_layout_.count++)];
    zone.begin_x = begin;
    zone.end_x = end;
    zone.type = biome.visual_type();
    zone.biome_id = type_index;
    zone.params = biome.sample_params(rng_());
    if (beach_zone) {
      zone.params.friction_mul = 0.78f + u(rng_) * 0.12f;
      zone.params.sink_rate = 0.006f + u(rng_) * 0.008f;
      zone.params.energy_drain_mul = 1.08f;
      zone.params.ambient_temperature = -30.0f + u(rng_) * 18.0f;
      zone.params.thermal_transfer = 0.9f + u(rng_) * 0.25f;
      zone.params.solar_charge_rate = 0.45f + u(rng_) * 0.25f;
    }
    zone.liquid_level = -1.0e9f;
    if (is_liquid) {
      const float basin_depth = 0.65f + u(rng_) * 0.85f;
      zone.liquid_level = terrain_.carve_basin(visible_begin, std::min(end, world_end), basin_depth);
      pending_exit_beach = true;
    }
    previous_type = type_index;
    begin = end;
    visible_begin = end;
  }
  mechanic_layout_.zones[static_cast<size_t>(mechanic_layout_.count - 1)].end_x = 1000000.0f;
  const auto& spawn_zone = mechanic_layout_.at(1.0f);
  mechanic_type_ = spawn_zone.type;
  mechanic_params_ = spawn_zone.params;
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
