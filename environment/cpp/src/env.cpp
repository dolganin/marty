#include "mars/env.hpp"
#include "mars/biome_bank.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

namespace mars {

Env::Env(EnvConfig config) : config_(std::move(config)), physics_(config_.physics) {
  terrain_.configure(config_.terrain);
}

void Env::reset(uint64_t seed, bool trial_start, float* obs_out) {
  rng_.seed(seed);
  endgame_test_world_ = config_.biome_split == 2;
  const int next_episode_in_trial = trial_start ? 0 : state_.episode_in_trial + 1;
  if (trial_start || !has_trial_mechanic_seed_) {
    trial_mechanic_seed_ = seed;
    has_trial_mechanic_seed_ = true;
    trial_steps_used_ = 0;
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
  scaled_terrain.safe_start_fraction = difficulty_safe_fraction_;
  scaled_terrain.difficulty_exponent = config_.difficulty_exponent;
  if (endgame_test_world_) {
    // Test is a held-out endgame probe, not another progressive training
    // course. Generate every metre from the far-distance distribution.
    scaled_terrain.safe_start_fraction = 0.0f;
    const float physical_length = std::max(
        scaled_terrain.dx, scaled_terrain.dx * static_cast<float>(scaled_terrain.sample_count - 1));
    scaled_terrain.difficulty_distance_offset = physical_length * 32.0f;
    scaled_terrain.preserve_spawn_safety = false;
  }
  terrain_.configure(scaled_terrain);
  terrain_.generate(seed ^ 0x9e3779b97f4a7c15ULL);
  finalize_mechanic_layout();
  const float spawn_y = terrain_.query(1.0f).height + 1.0f;
  physics_.initialize_state(config_.rig, state_, {1.0f, spawn_y});
  state_.world_seed = seed;
  update_world_latents();
  state_.ambient_temperature = state_.latent_ambient_temperature;
  state_.solar_irradiance = state_.latent_solar_rate;
  state_.trial_start = trial_start;
  state_.episode_in_trial = next_episode_in_trial;
  stuck_counter_ = 0;
  best_progress_x_ = state_.body.position.x;
  build_observation(obs_out);
}

StepOutput Env::step(int action, float* obs_out) {
  state_.previous_x = state_.body.position.x;
  update_world_latents();
  const auto stats = physics_.step(config_.rig, terrain_, state_, action, mechanic_layout_);

  const auto& current_zone = mechanic_layout_.at(state_.body.position.x);
  mechanic_type_ = current_zone.type;
  mechanic_params_ = current_zone.params;
  state_.route_branch = 0;
  if (current_zone.type == MechanicType::Liquid) {
    const float lower = terrain_.query(state_.body.position.x).height;
    const float contacted = terrain_.query_near(state_.body.position.x, state_.body.position.y).height;
    state_.route_branch = contacted > lower + 0.30f ? 2 : 1;
  }

  for (int i = 0; i < state_.wheel_count; ++i) {
    const auto& c = stats.deformation_contacts[static_cast<size_t>(i)];
    if (c.active) {
      const auto& zone = mechanic_layout_.at(c.x);
      float deform_scale = 0.0f;
      if (zone.type == MechanicType::Crust) {
        deform_scale = c.penetration > 0.008f ? 0.025f + zone.params.crust_deform * 2.0f : 0.0f;
      } else if (zone.type == MechanicType::Sand) {
        const auto& wheel = state_.wheels[static_cast<size_t>(i)];
        const float travel_speed = std::abs(state_.body.velocity.x);
        const float tread_speed = std::abs(wheel.angular_velocity) * wheel.radius;
        const float speed_relief = 1.0f / (1.0f + 0.42f * travel_speed);
        const float digging = 1.0f + 1.8f * clamp(wheel.slip, 0.0f, 2.0f) +
                              0.08f * tread_speed;
        const float difficulty = terrain_.difficulty_at(c.x);
        deform_scale = zone.params.sink_rate * (0.45f + 1.35f * difficulty) *
                       speed_relief * digging;
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

  // A run ends only on the fixed two-minute timer.  Falls, flips and empty
  // batteries are costly states to recover from, never terminal shortcuts.
  const bool finished = false;
  const bool flipped = is_flipped();
  const bool fatal = false;





  if (state_.body.position.x >= best_progress_x_ + 0.5f) {
    best_progress_x_ = state_.body.position.x;
    stuck_counter_ = 0;
  } else {
    stuck_counter_ += 1;
  }
  const bool stuck = false;
  trial_steps_used_ += 1;
  StepOutput out{};
  out.terminated = false;
  out.truncated = trial_exhausted();
  state_.termination_reason = 0;
  if (out.truncated) state_.termination_reason = 7;
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
  const float finish_scale = 1000.0f;
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
  const bool lidar_active = state_.lidar_active_steps > 0 && !state_.airborne;
  const float near_range = std::max(0.0f, config_.physics.near_sense_range);
  const int height_base = k;
  const int slope_base = k + kTerrainSamplesAhead;
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {

    const float distance = i < 12 ? 0.5f * static_cast<float>(i + 1)
                                  : 6.0f + 1.5f * static_cast<float>(i - 11);
    const auto sample = terrain_.query_near(state_.body.position.x + distance, 1.0e6f);
    const bool near_visible = distance <= near_range;
    const bool far_visible = lidar_active && distance > near_range &&
                             distance <= state_.lidar_range;
    const bool visible = near_visible || far_visible;
    obs_out[height_base + i] = visible && sample.solid
                                   ? sample.height - state_.body.position.y
                                   : 0.0f;
    obs_out[slope_base + i] = visible && sample.solid ? sample.slope : 0.0f;
  }
  k += kTerrainSamplesAhead * 2;
  for (int i = 0; i < kBiomeSamplesAhead; ++i) {
    const float distance = 3.0f * static_cast<float>(i + 1);
    const float sample_x = state_.body.position.x + distance;
    const bool near_visible = distance <= near_range;
    const bool far_visible = lidar_active && distance > near_range &&
                             distance <= state_.lidar_range;
    const bool visible = near_visible || far_visible;
    const auto sample = terrain_.query_near(sample_x, 1.0e6f);
    const auto behind = terrain_.query_near(sample_x - 0.25f, 1.0e6f);
    const auto ahead = terrain_.query_near(sample_x + 0.25f, 1.0e6f);


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
  obs_out[k++] = state_.heater_active ? 1.0f : 0.0f;









  obs_out[k++] = state_.engine_overheated ? 1.0f : 0.0f;
  obs_out[k++] = state_.engine_cold_locked ? 1.0f : 0.0f;
  obs_out[k++] = state_.solar_panel_deployment;
  obs_out[k++] = state_.charging_active ? 1.0f : 0.0f;
  obs_out[k++] = state_.solar_charge_rate / 3.0f;
  obs_out[k++] = lidar_active ? 1.0f : 0.0f;
  obs_out[k++] = state_.lidar_cooldown_steps > 0
                     ? static_cast<float>(state_.lidar_cooldown_steps) * config_.physics.dt
                     : 0.0f;
  obs_out[k++] = static_cast<float>(state_.previous_action) / 8388607.0f;
  obs_out[k++] = state_.last_reward;


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
  // Observable organs and contact history; hidden layer IDs/order and latent
  // values intentionally do not appear here.
  obs_out[k++] = state_.climb_mode ? 1.0f : 0.0f;
  obs_out[k++] = state_.propeller_mode ? 1.0f : 0.0f;
  obs_out[k++] = state_.jump_cooldown_steps * config_.physics.dt;
  obs_out[k++] = state_.recovery_state;
  obs_out[k++] = state_.energy / energy_scale;
  obs_out[k++] = clamp(state_.drive_fuel_rate, 0.0f, 4.0f) / 4.0f;
  obs_out[k++] = state_.airborne ? 1.0f : 0.0f;
  obs_out[k++] = state_.body_contact_front ? 1.0f : 0.0f;
  obs_out[k++] = state_.body_contact_belly ? 1.0f : 0.0f;
  obs_out[k++] = state_.body_contact_rear ? 1.0f : 0.0f;
  obs_out[k++] = state_.propeller_deployment;
  obs_out[k++] = state_.suspension_jump_charge;
  obs_out[k++] = state_.roof_piston_extension;
  obs_out[k++] = clamp(state_.energy_cost_rate, 0.0f, 32.0f) / 32.0f;
  obs_out[k++] = state_.ballast_air;
  assert(k == kObservationDim);
}

void Env::update_world_latents() {
  std::array<const MechanicZone*, kMaxActiveMechanisms> active{};
  const int n = mechanic_layout_.active_layers(state_.body.position.x, active);
  state_.active_layer_count = n;
  float weight_sum = 0.0f;
  // Blend sampled sources and uncoupled target bases first. The influence graph
  // is then evaluated once, so overlapping layers contribute to one summed,
  // logarithmically saturated adjustment rather than being coupled separately.
  MechanicParams mixed{};
  mixed.moisture = 0.22f;
  mixed.ambient_temperature = -21.0f;
  mixed.base_friction_mul = 1.0f;
  mixed.base_viscosity = 0.0f;
  mixed.base_energy_drain_mul = 1.0f;
  mixed.base_thermal_transfer = 1.0f;
  for (int i = 0; i < n; ++i) {
    const auto& z = *active[static_cast<size_t>(i)];
    const float feather = std::min(25.0f, std::max(15.0f, (z.end_x - z.begin_x) * 0.16f));
    const float enter = clamp((state_.body.position.x - z.begin_x) / feather, 0.0f, 1.0f);
    const float leave = clamp((z.end_x - state_.body.position.x) / feather, 0.0f, 1.0f);
    const float w = enter * enter * (3.0f - 2.0f * enter) * leave * leave * (3.0f - 2.0f * leave);
    weight_sum += w;
    const auto& p = z.params;
    mixed.moisture += w * (p.moisture - 0.22f);
    mixed.sink_rate += w * p.sink_rate;
    mixed.ambient_temperature += w * (p.ambient_temperature + 21.0f);
    mixed.base_friction_mul += w * (p.base_friction_mul - 1.0f);
    mixed.base_viscosity += w * p.base_viscosity;
    mixed.base_energy_drain_mul += w * (p.base_energy_drain_mul - 1.0f);
    mixed.base_thermal_transfer += w * (p.base_thermal_transfer - 1.0f);
    mixed.gravity_mul += w * (p.gravity_mul - 1.0f);
    mixed.wind_force += w * p.wind_force;
    mixed.solar_charge_rate += w * (p.solar_charge_rate - 1.0f);
    mixed.lidar_energy_mul += w * (p.lidar_energy_mul - 1.0f);
    mixed.lidar_range_mul += w * (p.lidar_range_mul - 1.0f);
  }
  apply_generation_influences(mixed);
  state_.latent_moisture = mixed.moisture;
  state_.latent_heat = clamp((mixed.ambient_temperature + 58.0f) / 130.0f, 0.0f, 1.0f);
  state_.latent_viscosity = mixed.viscosity;
  state_.latent_sink = mixed.sink_rate;
  state_.latent_charge_reserve = clamp(state_.energy / std::max(1.0f, config_.physics.energy_capacity), 0.0f, 1.0f);
  state_.latent_suspension = clamp(1.0f - mixed.sink_rate * 0.25f +
                                           (state_.climb_mode ? 0.12f : 0.0f),
                                   0.65f, 1.25f);
  state_.latent_traction = mixed.friction_mul;
  state_.latent_energy_resistance = mixed.energy_drain_mul;
  state_.latent_gravity_multiplier = mixed.gravity_mul;
  state_.latent_wind_force = mixed.wind_force;
  state_.latent_ambient_temperature = mixed.ambient_temperature;
  state_.latent_thermal_transfer = mixed.thermal_transfer;
  state_.latent_solar_rate = mixed.solar_charge_rate;
  state_.latent_lidar_energy_multiplier = mixed.lidar_energy_mul;
  state_.latent_lidar_range_multiplier = mixed.lidar_range_mul;
  state_.active_layer_weight = weight_sum;
}

float Env::course_difficulty(float x) const {
  if (endgame_test_world_) return 1.0f;
  const float course_length = std::max(1.0f, config_.terrain.length);
  const float progress = clamp(x / course_length, 0.0f, 1.0f);
  if (progress <= difficulty_safe_fraction_) return 0.0f;
  const float t = clamp((progress - difficulty_safe_fraction_) /
                            std::max(0.01f, 1.0f - difficulty_safe_fraction_),
                        0.0f, 1.0f);
  const float exponent = clamp(config_.difficulty_exponent, 0.5f, 4.0f);
  return clamp(0.68f * std::pow(t, exponent) +
                   0.32f * std::pow(t, exponent * 2.35f),
               0.0f, 1.0f);
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
  const float safe_min = clamp(std::min(config_.difficulty_safe_fraction_min,
                                        config_.difficulty_safe_fraction_max), 0.0f, 0.40f);
  const float safe_max = clamp(std::max(config_.difficulty_safe_fraction_min,
                                        config_.difficulty_safe_fraction_max), safe_min, 0.45f);
  difficulty_safe_fraction_ = endgame_test_world_ ? 0.0f : safe_min + (safe_max - safe_min) * u(rng_);
  const auto build_layers = [&]() {
    mechanic_layout_.layer_count = 0;
    // Physical regions span 55–120m, with 1–4 rules simultaneously
    // active.  They are independent from the contiguous terrain carrier zones.
    std::uniform_real_distribution<float> region_length(55.0f, 120.0f);
    float begin = -20.0f;
    const auto id_for_type = [&](MechanicType type) {
      for (int id : eligible_biomes) {
        if (bank[static_cast<size_t>(id)]->visual_type() == type) return id;
      }
      for (int id = 0; id < static_cast<int>(bank.size()); ++id) {
        if (bank[static_cast<size_t>(id)]->is_anchor() &&
            bank[static_cast<size_t>(id)]->visual_type() == type) return id;
      }
      return eligible_biomes[static_cast<size_t>(rng_() % eligible_biomes.size())];
    };
    std::array<int, kFrozenMechanismStacks.size()> allowed{};
    int allowed_count = 0;
    for (int i = 0; i < static_cast<int>(kFrozenMechanismStacks.size()); ++i) {
      const auto split = kFrozenMechanismStacks[static_cast<size_t>(i)].split;
      const bool matches = config_.biome_split == 0 ||
          (config_.biome_split == 1 && split == BiomeSplit::Train) ||
          (config_.biome_split == 2 && split == BiomeSplit::Test) ||
          (config_.biome_split == 3 && split == BiomeSplit::Builtin);
      if (matches) allowed[static_cast<size_t>(allowed_count++)] = i;
    }
    if (allowed_count == 0) {
      for (int i = 0; i < static_cast<int>(kFrozenMechanismStacks.size()); ++i) {
        allowed[static_cast<size_t>(allowed_count++)] = i;
      }
    }
    int region = 0;
    while (begin < terrain_.length() + 250.0f &&
           mechanic_layout_.layer_count < kMaxMechanicZones - kMaxActiveMechanisms) {
      const float end = begin + region_length(rng_);
      const float difficulty = course_difficulty((begin + end) * 0.5f);
      const int stack_id = allowed[static_cast<size_t>(rng_() % static_cast<uint64_t>(allowed_count))];
      const auto& approved = kFrozenMechanismStacks[static_cast<size_t>(stack_id)];
      const bool calm_region = u(rng_) > difficulty;
      const int stack = calm_region ? 1 : approved.count;
      for (int j = 0; j < stack && mechanic_layout_.layer_count < kMaxMechanicZones; ++j) {
        const MechanicType type = calm_region ? MechanicType::Normal
                                               : approved.types[static_cast<size_t>(j)];
        const int id = id_for_type(type);
        const Biome& b = *bank[static_cast<size_t>(id)];
        auto& layer = mechanic_layout_.layers[static_cast<size_t>(mechanic_layout_.layer_count++)];
        layer.begin_x = begin; layer.end_x = end; layer.type = b.visual_type();
        const uint64_t parameter_seed = rng_();
        layer.biome_id = id; layer.params = b.sample_params(parameter_seed);
        prepare_generation_params(layer.params, layer.type, parameter_seed);
        layer.terrain_seed = rng_(); layer.liquid_level = -1.0e9f;
      }
      begin = end; ++region;
    }
  };

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
    const uint64_t parameter_seed = rng_();
    zone.params = biome.sample_params(parameter_seed);
    prepare_generation_params(zone.params, zone.type, parameter_seed);
    zone.terrain_seed = rng_();
    zone.terrain_surprise_mode = 0;
    zone.terrain_surprise_strength = 0.0f;
    const float zone_difficulty = course_difficulty(config_.terrain.length * 0.5f);
    if (config_.fixed_biome_id < 0 &&
        u(rng_) < clamp(config_.terrain_surprise_probability, 0.0f, 1.0f) *
                      (0.03f + 0.97f * zone_difficulty)) {
      zone.terrain_surprise_mode = 1 + static_cast<int>(u(rng_) * 4.0f) % 4;
      zone.terrain_surprise_strength =
          std::max(0.0f, config_.terrain_surprise_strength) *
          (0.08f + 0.92f * zone_difficulty) * (0.75f + 0.5f * u(rng_));
    }
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[0] =
        biome.visual_type() == MechanicType::Liquid
            ? 2.5f + zone_difficulty * 4.5f + u(rng_) * (1.0f + 2.0f * zone_difficulty)
            : -1.0f;
    build_layers();
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


    const float slot_progress = (static_cast<float>(slot) + 0.5f) /
                                static_cast<float>(std::max(1, zone_count));
    const float slot_x = slot_progress * config_.terrain.length;
    const float difficulty = course_difficulty(slot_x);
    const bool take_anchor = !anchors.empty() &&
                             (!endgame_test_world_ && (slot == 0 || u(rng_) > difficulty));
    int candidate;
    if (take_anchor) {
      if (difficulty < 0.18f) {
        candidate = builtin_biome_id(MechanicType::Normal);
      } else {
        candidate = anchors[anchor_cursor % anchors.size()];
        ++anchor_cursor;
      }
    } else {
      candidate = pool[pool_cursor % pool.size()];
      ++pool_cursor;
      if (candidate == previous_id && pool.size() > 1) {
        pool_cursor += 1;
        candidate = pool[pool_cursor % pool.size()];
      }
    }
    // Water is a progressive mixture component rather than merely one of the
    // shuffled anchors. Keep the opening readable and avoid adjacent basins.
    const int liquid_id = builtin_biome_id(MechanicType::Liquid);
    const bool previous_was_liquid =
        previous_id >= 0 && bank[static_cast<size_t>(previous_id)]->visual_type() ==
                                MechanicType::Liquid;
    const float liquid_probability = 0.10f + 0.16f * difficulty;
    if (slot > 0 && difficulty >= 0.10f && !previous_was_liquid &&
        u(rng_) < liquid_probability) {
      candidate = liquid_id;
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
    const uint64_t parameter_seed = rng_();
    zone.params = biome.sample_params(parameter_seed);
    prepare_generation_params(zone.params, zone.type, parameter_seed);
    zone.terrain_seed = rng_();
    zone.terrain_surprise_mode = 0;
    zone.terrain_surprise_strength = 0.0f;
    const float zone_probe = last ? std::min(config_.terrain.length, zone.begin_x + 35.0f)
                                  : (zone.begin_x + zone.end_x) * 0.5f;
    const float zone_difficulty = course_difficulty(zone_probe);
    if (config_.fixed_biome_id < 0 &&
        u(rng_) < clamp(config_.terrain_surprise_probability, 0.0f, 1.0f) *
                      (0.03f + 0.97f * zone_difficulty)) {
      zone.terrain_surprise_mode = 1 + static_cast<int>(u(rng_) * 4.0f) % 4;
      zone.terrain_surprise_strength =
          std::max(0.0f, config_.terrain_surprise_strength) *
          (0.08f + 0.92f * zone_difficulty) * (0.75f + 0.5f * u(rng_));
    }
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[static_cast<size_t>(slot)] =
        biome.visual_type() == MechanicType::Liquid
            ? 2.5f + zone_difficulty * 4.5f + u(rng_) * (1.0f + 2.0f * zone_difficulty)
            : -1.0f;
  }
  build_layers();
}

void Env::finalize_mechanic_layout() {





  for (int slot = 0; slot < mechanic_layout_.count; ++slot) {
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(slot)];
    const Biome& biome = biome_by_id(zone.biome_id);
    const float finite_end = std::min(terrain_.length(), zone.end_x);
    const float profile_x = clamp((std::max(0.0f, zone.begin_x) + finite_end) * 0.5f,
                                  0.0f, terrain_.length());
    const float zone_difficulty = terrain_.difficulty_at(profile_x);
    const float simple_weight = 1.0f - zone_difficulty;
    constexpr float medium_weight = 0.4f;
    const float complex_weight = zone_difficulty;
    const float profile_draw = biome_random01(zone.terrain_seed, 18) *
                               (simple_weight + medium_weight + complex_weight);
    int profile = 1;
    if (profile_draw < simple_weight) {
      profile = biome_random01(zone.terrain_seed, 19) < 0.5f ? 0 : 4;
    } else if (profile_draw >= simple_weight + medium_weight) {
      profile = biome_random01(zone.terrain_seed, 19) < 0.5f ? 2 : 3;
    }
    const float frequency_scale =
        1.0f + std::max(0.0f, config_.terrain_profile_frequency_growth) * zone_difficulty;
    zone.terrain_profile = profile;
    zone.terrain_frequency_scale = frequency_scale;
    const int first_sample = std::max(
        0, static_cast<int>(std::ceil(std::max(0.0f, zone.begin_x) / terrain_.dx())));
    const int last_sample = std::min(
        terrain_.sample_count() - 1,
        static_cast<int>(std::floor(std::min(terrain_.length(), zone.end_x) / terrain_.dx())));
    for (int sample = first_sample; sample <= last_sample; ++sample) {
      const float world_x = static_cast<float>(sample) * terrain_.dx();
      const float local_x = world_x - zone.begin_x;
      // Profile category is sampled once per zone. Distance shifts mass from
      // predictable waves toward terraces/ridges; frequency increases
      // independently, while amplitude retains the central difficulty scale.
      const float difficulty = terrain_.difficulty_at(world_x);
      const float phase = 6.2831853f * biome_random01(zone.terrain_seed, 17);
      float palette_delta = 0.0f;
      switch (profile) {
        case 0:
          palette_delta = 0.22f * std::sin(local_x * 0.07f * frequency_scale + phase) +
                          0.08f * std::sin(local_x * 0.71f * frequency_scale + phase * 0.37f);
          break;
        case 1: { // deterministic value-noise interpolation, no global period
          const float cell = 3.8f / frequency_scale;
          const int a = static_cast<int>(std::floor(local_x / cell));
          const float t = local_x / cell - static_cast<float>(a);
          const float va = biome_random01(zone.terrain_seed, 300 + a) * 2.0f - 1.0f;
          const float vb = biome_random01(zone.terrain_seed, 301 + a) * 2.0f - 1.0f;
          const float smooth = t * t * (3.0f - 2.0f * t);
          palette_delta = (va + (vb - va) * smooth) * 0.28f;
          break;
        }
        case 2: {
          const float step = std::floor(local_x / (3.0f / frequency_scale));
          palette_delta = (biome_random01(zone.terrain_seed, 500 + static_cast<uint64_t>(step)) - 0.5f) *
                          0.48f;
          break;
        }
        case 3: {
          const float period = 7.0f / frequency_scale;
          const float f = local_x / period - std::floor(local_x / period);
          palette_delta = (1.0f - 4.0f * std::abs(f - 0.5f)) * 0.38f;
          break;
        }
        default:
          palette_delta = 0.32f * std::sin(local_x * 0.045f * frequency_scale + phase) +
                          (biome_random01(
                               zone.terrain_seed,
                               800 + static_cast<uint64_t>(local_x / (2.0f / frequency_scale))) -
                           0.5f) * 0.16f;
          break;
      }
      const float region_edge = clamp(local_x / 18.0f, 0.0f, 1.0f) *
                                clamp((zone.end_x - world_x) / 18.0f, 0.0f, 1.0f);
      terrain_.add_height_at_index(sample, palette_delta * region_edge *
                                           (0.18f + 0.82f * difficulty));
      const float delta = biome.terrain_height_delta(local_x, zone.terrain_seed);
      if (std::isfinite(delta)) {
        terrain_.add_height_at_index(
            sample, clamp(delta, -2.5f, 2.5f) * (0.10f + 0.90f * difficulty));
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
      zone.liquid_level = terrain_.carve_basin(
          std::max(0.0f, zone.begin_x), world_end, depth, zone.terrain_seed);
      // A raised, dry crossing is a separate collision surface; missing the
      // ramp leaves the rover on the lower water route, where the propeller is
      // useful but expensive.
      const float left = std::max(0.0f, zone.begin_x) + 9.0f;
      const float right = world_end - 9.0f;
      if (right - left > 24.0f) {
        const float h0 = terrain_.query(left).height + 0.85f;
        const float h1 = terrain_.query(right).height + 0.85f;
        terrain_.add_surface(left, right, h0, h1);
      }
    }
    if (zone.params.ledge_gap_width > 0.0f && zone.params.ledge_spacing > 0.0f) {
      const float zone_start = std::max(zone.begin_x, zone.params.ledge_start_x);
      const float last_ledge = std::min({config_.termination.finish_x - 4.0f,
                                         terrain_.length() - 4.0f, zone.end_x - 4.0f});
      for (float begin = zone_start; begin < last_ledge; begin += zone.params.ledge_spacing) {
        const float difficulty = terrain_.difficulty_at(begin);
        if (difficulty < 0.08f) continue;
        const uint64_t ledge_index = static_cast<uint64_t>(std::max(0.0f, std::floor(begin)));
        const bool heavy_ledge = biome_random01(zone.terrain_seed, 700 + ledge_index) > 0.66f;
        const float gap_scale = heavy_ledge
                                    ? 1.65f
                                    : 0.90f + 0.35f * biome_random01(zone.terrain_seed, 900 + ledge_index);
        const float ramp_scale = heavy_ledge
                                     ? 0.70f
                                     : 1.0f + 0.20f * biome_random01(zone.terrain_seed, 1100 + ledge_index);
        const float gap_width = zone.params.ledge_gap_width * gap_scale *
                                (0.25f + 0.75f * difficulty);
        terrain_.carve_ledge(begin, begin + gap_width,
                             zone.params.ledge_ramp_length * ramp_scale,
                             zone.params.ledge_ramp_height * (heavy_ledge ? 1.20f : 1.0f) *
                                 (0.35f + 0.65f * difficulty));
      }
    }
  }
  const auto& first = mechanic_layout_.zones[0];
  mechanic_type_ = first.type;
  mechanic_params_ = first.params;
}

int Env::trial_step_budget() const {
  if (config_.termination.trial_time_limit <= 0.0f) {
    return 0;
  }
  const float dt = std::max(0.0001f, config_.physics.dt);
  return std::max(1, static_cast<int>(std::lround(config_.termination.trial_time_limit / dt)));
}

bool Env::trial_exhausted() const {
  const int budget = trial_step_budget();
  return budget > 0 && trial_steps_used_ >= budget;
}

float Env::trial_time_left() const {
  const int budget = trial_step_budget();
  if (budget <= 0) {
    return -1.0f;
  }
  return static_cast<float>(std::max(0, budget - trial_steps_used_)) * config_.physics.dt;
}

bool Env::is_flipped() const {
  return std::abs(state_.body.angle) > config_.termination.flip_angle;
}

bool Env::is_stuck() const {
  return state_.step_index > 120 &&
         stuck_counter_ > config_.termination.stuck_steps;
}

}
