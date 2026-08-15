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
  // The biome must be picked before the terrain is shaped: a biome may reshape
  // the obstacle-generating noise law itself, not just physics coefficients
  // layered on top of a fixed course.
  select_mechanic_layout(trial_mechanic_seed_);
  const auto& picked_params = mechanic_layout_.zones[0].params;
  TerrainConfig scaled_terrain = config_.terrain;
  // Ceiling lowered from 3.0: at the old range seven of ten gen2 biomes ended with the rover
  // jammed against terrain at 48-73 m holding a full battery, and raising the step cap to
  // 6000 or 12000 bought no extra distance at all. Terrain that stops everyone equally ranks
  // no one, and it hid the energy trade-off this bank is now built around.
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
  // Track irreversible forward progress, not instantaneous speed. The old
  // detector exempted a requested solar panel, so a policy could deploy it and
  // remain almost stationary forever while being scored as perfectly safe.
  // A half-metre advance resets the mission-loss clock; reversing or oscillating
  // around the same point cannot game it.
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
  // Trial/episode position is deliberately NOT exposed. A memoryless policy was
  // reading it as a proxy for "how far into the trial am I" and scoring +20.1 higher
  // on IDENTICAL courses merely by being told it was episode 4 rather than episode 1,
  // which manufactured an adaptation delta out of nothing. Agents that legitimately
  // need the boundary get it out of band: the trial reset flag is passed to
  // Agent.reset(trial_start) and the episode boundary appears as previous_done.
  obs_out[k++] = 0.0f;
  for (int i = 0; i < kMaxWheels; ++i) {
    const auto& w = state_.wheels[static_cast<size_t>(i)];
    obs_out[k++] = (i < state_.wheel_count && w.in_contact) ? 1.0f : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.slip : 0.0f;
    obs_out[k++] = i < state_.wheel_count ? w.normal_force / 200.0f : 0.0f;
  }
  constexpr float sample_dx = 0.5f;
  const bool contact_scanners_available = !state_.airborne;
  const bool lidar_active = state_.lidar_active_steps > 0 && contact_scanners_available;
  const int height_base = k;
  const int slope_base = k + kTerrainSamplesAhead;
  for (int i = 0; i < kTerrainSamplesAhead; ++i) {
    const auto sample = terrain_.query(
        state_.body.position.x + sample_dx * static_cast<float>(i + 1));
    const float distance = sample_dx * static_cast<float>(i + 1);
    const bool visible = lidar_active && distance <= state_.lidar_range;
    obs_out[height_base + i] = visible && sample.solid
                                   ? sample.height - state_.body.position.y
                                   : 0.0f;
    obs_out[slope_base + i] = visible && sample.solid ? sample.slope : 0.0f;
  }
  k += kTerrainSamplesAhead * 2;
  for (int i = 0; i < kBiomeSamplesAhead; ++i) {
    const float sample_x = state_.body.position.x + 1.5f * static_cast<float>(i + 1);
    const bool visible = lidar_active && 1.5f * static_cast<float>(i + 1) <= state_.lidar_range;
    const auto sample = terrain_.query(sample_x);
    const auto behind = terrain_.query(sample_x - 0.25f);
    const auto ahead = terrain_.query(sample_x + 0.25f);
    // Raw scanner correlates only: echo strength, surface break and local
    // vibration/curvature. Never expose MechanicType or biome parameters.
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
  // Withheld: ambient temperature comes straight from the biome's MechanicParams and ranged
  // from -70 to +26 across the test bank, so a single scalar identified the world on step one.
  // Measured consequence: observations from different biomes were already 0.33 apart after ONE
  // step of identical actions and no further apart after forty — the mechanic was fully visible
  // immediately, so a memoryless reflex policy could match anything memory could do. That is why
  // separating biomes, hiding lidar, raising crash costs and starving the battery all failed to
  // dislodge the memoryless ceiling. The rover still feels temperature through its consequences
  // (engine_temperature, cold-start lockout, overheating, solar output), which is what it must
  // now infer the world from.
  obs_out[k++] = 0.0f;
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
  // Same leak, other end of the observation: see the note on trial_start above.
  obs_out[k++] = 0.0f;
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
    // Keep invalid/empty-bank configurations safe and deterministic. Python tooling
    // treats this fallback as a validation failure rather than a scored episode.
    eligible_biomes.push_back(builtin_biome_id(MechanicType::Normal));
  }
  std::uniform_real_distribution<float> u(0.0f, 1.0f);

  if (config_.fixed_biome_id >= 0 || !config_.chain_biomes) {
    // Single-zone mode: deterministic debugging / evaluation tools that score one
    // named biome (evaluate_biomes.py, policy_divergence.py, anchor_eval.py) rely
    // on a lone zone spanning the whole world.
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
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[0] =
        biome.visual_type() == MechanicType::Liquid ? 0.65f + u(rng_) * 0.85f : -1.0f;
    return;
  }

  // Chained mode: a trial is a random walk through a concatenation of zones, so
  // the specific course differs trial to trial even though the mechanic
  // vocabulary (anchors + the frozen generated bank) does not. Anchors are
  // always present as a stable backbone; the remaining slots are filled from
  // whatever the active split allows, shuffled and, if the pool is smaller than
  // the requested chain length, revisited without ever repeating twice in a row.
  // Anchors default split() to Builtin (see biome_bank.hpp), so the split filter
  // above never includes them for biome_split 1/2 - by design, single-zone train/
  // test scoring (evaluate_biomes.py, the exact-set test in test_biome_bank.py)
  // must never see an anchor when it asked for "train" or "test". The chain's
  // backbone is a DIFFERENT requirement (every trial gets a stable, comparable
  // opening regardless of which split it draws generated zones from), so it
  // pulls anchors straight from the bank rather than from eligible_biomes.
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
    // Interleave one anchor for every two generated/handwritten picks so the
    // stable backbone is spread across the whole chain, not front-loaded.
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
  float cursor = -8.0f;  // a short lead-in before the first zone boundary
  std::uniform_real_distribution<float> length_dist(config_.chain_segment_min_length,
                                                      config_.chain_segment_max_length);
  for (int slot = 0; slot < mechanic_layout_.count; ++slot) {
    const int biome_id = order[static_cast<size_t>(slot)];
    const Biome& biome = *bank[static_cast<size_t>(biome_id)];
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(slot)];
    zone.begin_x = cursor;
    const bool last = slot + 1 == mechanic_layout_.count;
    // The last zone runs out far enough that no policy can outrun the chain
    // within one episode; earlier zones get a randomized, bounded length so the
    // rover actually reaches several distinct mechanics per trial.
    cursor += last ? 1000000.0f : length_dist(rng_);
    zone.end_x = cursor;
    zone.type = biome.visual_type();
    zone.biome_id = biome_id;
    zone.params = biome.sample_params(rng_());
    zone.liquid_level = -1.0e9f;
    pending_basin_depth_[static_cast<size_t>(slot)] =
        biome.visual_type() == MechanicType::Liquid ? 0.65f + u(rng_) * 0.85f : -1.0f;
  }
}

void Env::finalize_mechanic_layout() {
  // Runs after the (possibly biome-reshaped) terrain has been generated, so
  // basin carving reads real heights instead of stale/unshaped ones. Every
  // zone in the chain gets its own pass: a liquid zone in the middle of the
  // course still needs its basin carved, and a ledge zone still needs its
  // gaps/ramps cut in its own span, not only at the head of the world.
  for (int slot = 0; slot < mechanic_layout_.count; ++slot) {
    auto& zone = mechanic_layout_.zones[static_cast<size_t>(slot)];
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

}  // namespace mars
