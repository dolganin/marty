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
  if (trial_start || !has_trial_mechanic_seed_) {
    trial_mechanic_seed_ = seed;
    has_trial_mechanic_seed_ = true;
  }



  select_mechanic_layout(trial_mechanic_seed_);
  const auto& picked_params = mechanic_layout_.zones[0].params;
  TerrainConfig scaled_terrain = config_.terrain;




  scaled_terrain.amplitude *= clamp(picked_params.terrain_amplitude_mul, 0.5f, 1.8f);
  scaled_terrain.roughness *= clamp(picked_params.terrain_roughness_mul, 0.5f, 2.5f);
  scaled_terrain.crater_count = static_cast<int>(std::lround(
      static_cast<float>(scaled_terrain.crater_count) *
      clamp(picked_params.terrain_crater_mul, 0.3f, 3.0f)));
  scaled_terrain.step_count = static_cast<int>(std::lround(
      static_cast<float>(scaled_terrain.step_count) *
      clamp(picked_params.terrain_step_mul, 0.3f, 3.0f)));
  terrain_.configure(scaled_terrain);
  terrain_.generate(seed ^ 0x9e3779b97f4a7c15ULL);
  finalize_mechanic_layout();
  const float spawn_y = terrain_.query(1.0f).height + 1.0f;
  physics_.initialize_state(config_.rig, state_, {1.0f, spawn_y});
  const auto& spawn_zone = mechanic_layout_.at(1.0f);
  const auto spawn_thermal = mechanic_layout_.thermal_at(1.0f, spawn_zone);
  state_.ambient_temperature = spawn_thermal.ambient_temperature;
  state_.solar_irradiance = std::max(0.0f, spawn_zone.params.solar_charge_rate);
  state_.trial_start = trial_start;
  state_.episode_in_trial = next_episode_in_trial;
  stuck_counter_ = 0;
  best_progress_x_ = state_.body.position.x;
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
  const bool fallen = state_.body.position.y <= config_.termination.fatal_fall_y;
  if (fallen) state_.fatal_error = true;
  const bool flipped = is_flipped();
  const bool fatal = flipped || state_.fatal_error;





  if (state_.body.position.x >= best_progress_x_ + 0.5f) {
    best_progress_x_ = state_.body.position.x;
    stuck_counter_ = 0;
  } else {
    stuck_counter_ += 1;
  }
  const bool stuck = is_stuck();
  StepOutput out{};
  out.terminated = finished || fatal || state_.energy <= config_.termination.min_energy || stuck;
  out.truncated = config_.termination.max_steps > 0 &&
                  state_.step_index + 1 >= config_.termination.max_steps;
  state_.termination_reason = 0;
  if (finished) state_.termination_reason = 1;
  else if (state_.landing_fatal) state_.termination_reason = 2;
  else if (flipped) state_.termination_reason = 3;
  else if (fallen) state_.termination_reason = 4;
  else if (state_.energy <= config_.termination.min_energy) state_.termination_reason = 5;
  else if (stuck) state_.termination_reason = 6;
  else if (out.truncated) state_.termination_reason = 7;
  out.reward = compute_reward(config_.reward, state_, stats.energy_cost, finished, fatal, stuck);
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
  const float finish_scale = std::max(1.0f, config_.termination.finish_x);
  const float energy_scale = std::max(1.0f, config_.physics.energy_capacity);
  obs_out[k++] = state_.body.position.x / finish_scale;
  obs_out[k++] = state_.body.position.y / 10.0f;
  obs_out[k++] = state_.body.velocity.x / 20.0f;
  obs_out[k++] = state_.body.velocity.y / 20.0f;
  obs_out[k++] = state_.body.angle;
  obs_out[k++] = state_.body.angular_velocity / 10.0f;
  obs_out[k++] = state_.energy / energy_scale;






  obs_out[k++] = 0.0f;
  for (int i = 0; i < kMaxWheels; ++i) {
    const auto& w = state_.wheels[static_cast<size_t>(i)];
    obs_out[k++] = (i < state_.wheel_count && w.in_contact) ? 1.0f : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.slip : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.normal_force / 200.0f : 0.0f;
  }
  const bool contact_scanners_available = !state_.airborne;
  const bool lidar_active = state_.lidar_active_steps > 0 && contact_scanners_available;
  const int height_base = k;
  const int slope_base = k + kTerrainSamplesAhead;
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {

    const float distance = i < 12 ? 0.5f * static_cast<float>(i + 1)
                                  : 6.0f + 1.5f * static_cast<float>(i - 11);
    const auto sample = terrain_.query(state_.body.position.x + distance);
    const bool visible = lidar_active && distance <= state_.lidar_range;
    obs_out[height_base + i] = visible && sample.solid
                                   ? sample.height - state_.body.position.y
                                   : 0.0f;
    obs_out[slope_base + i] = visible && sample.solid ? sample.slope : 0.0f;
  }
  k += kTerrainSamplesAhead * 2;
  for (int i = 0; i < kBiomeSamplesAhead; ++i) {
    const float distance = 3.0f * static_cast<float>(i + 1);
    const float sample_x = state_.body.position.x + distance;
    const bool visible = lidar_active && distance <= state_.lidar_range;
    const auto sample = terrain_.query(sample_x);
    const auto behind = terrain_.query(sample_x - 0.25f);
    const auto ahead = terrain_.query(sample_x + 0.25f);


    obs_out[k++] = visible && sample.solid
                       ? 1.0f / (1.0f + std::abs(sample.slope))
                       : 0.0f;
    obs_out[k++] = visible
                       ? (sample.solid ? clamp((sample.height - state_.body.position.y) / 10.0f,
                                               -1.0f, 1.0f)
                                       : -1.0f)
                       : 0.0f;
    obs_out[k++] = visible && sample.solid && behind.solid && ahead.solid
                       ? clamp(std::abs(ahead.slope - behind.slope), 0.0f, 1.0f)
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









  obs_out[k++] = state_.ambient_temperature / 120.0f;
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
  obs_out[k++] = static_cast<float>(state_.previous_action) / 4095.0f;
  obs_out[k++] = state_.last_reward / std::max(1.0f, config_.reward.finish_bonus);

  obs_out[k++] = state_.solar_irradiance / 3.0f;


  obs_out[k++] = clamp(state_.imu_acceleration.x / 20.0f, -4.0f, 4.0f);
  obs_out[k++] = clamp(state_.imu_acceleration.y / 20.0f, -4.0f, 4.0f);
  obs_out[k++] = clamp(state_.imu_angular_acceleration / 20.0f, -4.0f, 4.0f);
  obs_out[k++] = clamp(state_.imu_impact / 10.0f, 0.0f, 4.0f);


  obs_out[k++] = state_.body_contact_front ? 1.0f : 0.0f;
  obs_out[k++] = state_.body_contact_belly ? 1.0f : 0.0f;
  obs_out[k++] = state_.body_contact_rear ? 1.0f : 0.0f;


  for (int i = 0; i < kMaxWheels; ++i) {
    obs_out[k++] = i < state_.wheel_count
                       ? clamp(state_.wheels[static_cast<size_t>(i)].angular_velocity / 60.0f,
                               -4.0f, 4.0f)
                       : 0.0f;
  }
  for (int i = 0; i < kMaxWheels; ++i) {
    if (i >= state_.wheel_count || i >= static_cast<int>(config_.rig.wheels.size())) {
      obs_out[k++] = 0.0f;
      continue;
    }
    const auto& wheel = state_.wheels[static_cast<size_t>(i)];
    const auto& wheel_rig = config_.rig.wheels[static_cast<size_t>(i)];
    const Vec2 anchor = state_.body.position + rotate(wheel_rig.local_anchor, state_.body.angle);
    const Vec2 axis = rotate({0.0f, -1.0f}, state_.body.angle);
    const float travel = dot(wheel.position - anchor, axis);
    const float span = std::max(0.01f, wheel_rig.suspension.max_length -
                                           wheel_rig.suspension.min_length);
    obs_out[k++] = clamp((wheel_rig.suspension.max_length - travel) / span, 0.0f, 1.0f);
  }
}

void Env::select_mechanic_layout(uint64_t seed) {
  rng_.seed(seed ^ 0xd1b54a32d192ed03ULL);
  const auto& bank = biome_registry();
  std::vector<int> eligible_biomes;
  eligible_biomes.reserve(bank.size());
  if (config_.fixed_biome_id >= 0 && config_.fixed_biome_id < static_cast<int>(bank.size())) {
    eligible_biomes.push_back(config_.fixed_biome_id);
  } else {
    for (int i = 0; i < static_cast<int>(bank.size()); ++i) {
      const Biome& biome = *bank[static_cast<size_t>(i)];
      const bool eligible =
          config_.biome_split == 0 ||
          (config_.biome_split == 1 && biome.split() == BiomeSplit::Train) ||
          (config_.biome_split == 2 && biome.split() == BiomeSplit::Test) ||
          (config_.biome_split == 3 && biome.is_anchor());
      if (eligible) eligible_biomes.push_back(i);
    }
  }
  if (eligible_biomes.empty()) {


    eligible_biomes.push_back(builtin_biome_id(MechanicType::Normal));
  }
  std::uniform_real_distribution<float> u(0.0f, 1.0f);

  if (config_.fixed_biome_id >= 0 || !config_.chain_biomes) {



    std::uniform_int_distribution<size_t> biome_dist(0, eligible_biomes.size() - 1);
    const int biome_id = eligible_biomes[biome_dist(rng_)];
    const Biome& biome = *bank[static_cast<size_t>(biome_id)];
    mechanic_layout_.count = 1;
    auto& zone = mechanic_layout_.zones[0];
    zone.begin_x = -1000000.0f;
    zone.end_x = 1000000.0f;
    zone.type = biome.visual_type();
    zone.biome_id = biome_id;
    zone.params = biome.sample_params(rng_());
    zone.terrain_seed = rng_();
    zone.terrain_surprise_mode = 0;
    zone.terrain_surprise_strength = 0.0f;
    if (config_.fixed_biome_id < 0 &&
        u(rng_) < clamp(config_.terrain_surprise_probability, 0.0f, 1.0f)) {
      zone.terrain_surprise_mode = 1 + static_cast<int>(u(rng_) * 4.0f) % 4;
      zone.terrain_surprise_strength =
          std::max(0.0f, config_.terrain_surprise_strength) * (0.75f + 0.5f * u(rng_));
    }
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[0] =
        biome.visual_type() == MechanicType::Liquid ? 0.65f + u(rng_) * 0.85f : -1.0f;
    return;
  }














  std::vector<int> anchors;
  std::vector<int> rest;
  rest.reserve(eligible_biomes.size());
  for (int id : eligible_biomes) {
    if (!bank[static_cast<size_t>(id)]->is_anchor()) rest.push_back(id);
  }
  for (int i = 0; i < static_cast<int>(bank.size()); ++i) {
    if (bank[static_cast<size_t>(i)]->is_anchor()) anchors.push_back(i);
  }
  std::shuffle(anchors.begin(), anchors.end(), rng_);
  std::shuffle(rest.begin(), rest.end(), rng_);
  const std::vector<int>& pool = !rest.empty() ? rest : (!anchors.empty() ? anchors : eligible_biomes);

  const int zone_count = std::clamp(config_.chain_zone_count, 1, kMaxMechanicZones);
  std::vector<int> order;
  order.reserve(static_cast<size_t>(zone_count));
  size_t anchor_cursor = 0;
  size_t pool_cursor = 0;
  int previous_id = -1;
  for (int slot = 0; slot < zone_count; ++slot) {


    const bool take_anchor = !anchors.empty() && (slot % 3 == 0);
    int candidate;
    if (take_anchor) {
      candidate = anchors[anchor_cursor % anchors.size()];
      ++anchor_cursor;
    } else {
      candidate = pool[pool_cursor % pool.size()];
      ++pool_cursor;
      if (candidate == previous_id && pool.size() > 1) {
        pool_cursor += 1;
        candidate = pool[pool_cursor % pool.size()];
      }
    }
    order.push_back(candidate);
    previous_id = candidate;
  }

  mechanic_layout_.count = static_cast<int>(order.size());
  float cursor = -8.0f;
  std::uniform_real_distribution<float> length_dist(config_.chain_segment_min_length,
                                                      config_.chain_segment_max_length);
  for (int slot = 0; slot < mechanic_layout_.count; ++slot) {
    const int biome_id = order[static_cast<size_t>(slot)];
    const Biome& biome = *bank[static_cast<size_t>(biome_id)];
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(slot)];
    zone.begin_x = cursor;
    const bool last = slot + 1 == mechanic_layout_.count;



    cursor += last ? 1000000.0f : length_dist(rng_);
    zone.end_x = cursor;
    zone.type = biome.visual_type();
    zone.biome_id = biome_id;
    zone.params = biome.sample_params(rng_());
    zone.terrain_seed = rng_();
    zone.terrain_surprise_mode = 0;
    zone.terrain_surprise_strength = 0.0f;
    if (config_.fixed_biome_id < 0 &&
        u(rng_) < clamp(config_.terrain_surprise_probability, 0.0f, 1.0f)) {
      zone.terrain_surprise_mode = 1 + static_cast<int>(u(rng_) * 4.0f) % 4;
      zone.terrain_surprise_strength =
          std::max(0.0f, config_.terrain_surprise_strength) * (0.75f + 0.5f * u(rng_));
    }
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[static_cast<size_t>(slot)] =
        biome.visual_type() == MechanicType::Liquid ? 0.65f + u(rng_) * 0.85f : -1.0f;
  }
}

void Env::finalize_mechanic_layout() {





  for (int slot = 0; slot < mechanic_layout_.count; ++slot) {
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(slot)];
    const Biome& biome = biome_by_id(zone.biome_id);
    const int first_sample = std::max(
        0, static_cast<int>(std::ceil(std::max(0.0f, zone.begin_x) / terrain_.dx())));
    const int last_sample = std::min(
        terrain_.sample_count() - 1,
        static_cast<int>(std::floor(std::min(terrain_.length(), zone.end_x) / terrain_.dx())));
    for (int sample = first_sample; sample <= last_sample; ++sample) {
      const float world_x = static_cast<float>(sample) * terrain_.dx();
      const float local_x = world_x - zone.begin_x;
      const float delta = biome.terrain_height_delta(local_x, zone.terrain_seed);
      if (std::isfinite(delta)) {
        terrain_.add_height_at_index(sample, clamp(delta, -2.5f, 2.5f));
      }
      if (zone.terrain_surprise_mode > 0) {
        const float phase = 6.2831853f * biome_random01(zone.terrain_seed, 91);
        const float entrance = clamp(local_x / 3.0f, 0.0f, 1.0f);
        const float spawn_safe = clamp((world_x - 5.0f) / 4.0f, 0.0f, 1.0f);
        const float remaining = zone.end_x - world_x;
        const float exit = zone.end_x < 999999.0f
                               ? clamp(remaining / 3.0f, 0.0f, 1.0f)
                               : 1.0f;
        const float envelope = entrance * entrance * (3.0f - 2.0f * entrance) *
                               exit * exit * (3.0f - 2.0f * exit) * spawn_safe;
        float surprise = 0.0f;
        switch (zone.terrain_surprise_mode) {
          case 1:
            surprise = 0.42f * std::sin(0.12f * local_x + 0.005f * local_x * local_x + phase);
            break;
          case 2:
            surprise = 0.34f * std::tanh(3.2f * std::sin(0.19f * local_x + phase)) +
                       0.10f * std::sin(0.83f * local_x + phase * 0.37f);
            break;
          case 3: {
            const float ridge = std::max(0.0f, std::sin(0.31f * local_x + phase));
            surprise = 0.62f * ridge * ridge * ridge * ridge - 0.10f;
            break;
          }
          case 4:
            surprise = -0.38f * std::abs(std::sin(0.105f * local_x + phase)) +
                       0.13f * std::sin(1.17f * local_x + phase);
            break;
          default:
            break;
        }
        terrain_.add_height_at_index(
            sample, clamp(surprise * zone.terrain_surprise_strength * envelope, -1.25f, 1.25f));
      }
    }
    const float depth = pending_basin_depth_[static_cast<size_t>(slot)];
    if (depth >= 0.0f) {
      const float world_end =
          std::min(std::max(terrain_.length(), config_.terrain.length), zone.end_x);
      zone.liquid_level = terrain_.carve_basin(std::max(0.0f, zone.begin_x), world_end, depth);
    }
    if (zone.params.ledge_gap_width > 0.0f && zone.params.ledge_spacing > 0.0f) {
      const float zone_start = std::max(zone.begin_x, zone.params.ledge_start_x);
      const float last_ledge = std::min({config_.termination.finish_x - 4.0f,
                                         terrain_.length() - 4.0f, zone.end_x - 4.0f});
      for (float begin = zone_start; begin < last_ledge; begin += zone.params.ledge_spacing) {
        terrain_.carve_ledge(begin, begin + zone.params.ledge_gap_width,
                             zone.params.ledge_ramp_length,
                             zone.params.ledge_ramp_height);
      }
    }
  }
  const auto& first = mechanic_layout_.zones[0];
  mechanic_type_ = first.type;
  mechanic_params_ = first.params;
}

bool Env::is_flipped() const {
  return std::abs(state_.body.angle) > config_.termination.flip_angle;
}

bool Env::is_stuck() const {
  return state_.step_index > 120 &&
         stuck_counter_ > config_.termination.stuck_steps;
}

}
