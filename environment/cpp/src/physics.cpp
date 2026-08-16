#include "mars/physics.hpp"

#include <cmath>

namespace mars {
namespace {

constexpr float kGearRatios[] = {4.20f, 3.10f, 2.35f, 1.80f, 1.40f, 1.10f, 0.86f, 0.68f};
constexpr float kGearMaxSpeeds[] = {2.0f, 3.0f, 4.2f, 5.6f, 7.2f, 9.0f, 11.0f, 13.5f};
constexpr float kMinimumLoadedRpm[] = {800.0f, 1400.0f, 1900.0f, 2600.0f,
                                       3000.0f, 3300.0f, 3500.0f, 3700.0f};
constexpr float kGearEnergyMul[] = {1.00f, 1.12f, 1.28f, 1.48f,
                                    1.75f, 2.10f, 2.55f, 3.10f};



constexpr float kBasePostShiftRpm[] = {3000.0f, 3050.0f, 3150.0f, 3250.0f,
                                       3350.0f, 3450.0f, 3550.0f};
constexpr int kGearCount = static_cast<int>(sizeof(kGearRatios) / sizeof(kGearRatios[0]));
constexpr float kIdleRpm = 1100.0f;
constexpr float kRedlineRpm = 9000.0f;
constexpr float kSafeDownshiftRpm = 8500.0f;

}  

PhysicsEngine::PhysicsEngine(PhysicsConfig config) : config_(config) {}

void PhysicsEngine::initialize_state(const RoverRig& rig, RoverState& state, Vec2 spawn) const {
  state = RoverState{};
  state.body.position = spawn;
  state.body.mass = rig.body.mass;
  state.body.inv_mass = rig.body.mass > 0.0f ? 1.0f / rig.body.mass : 0.0f;
  state.body.inertia = rig.body.inertia;
  state.body.inv_inertia = rig.body.inertia > 0.0f ? 1.0f / rig.body.inertia : 0.0f;
  state.wheel_count = static_cast<int>(rig.wheels.size());
  for (int i = 0; i < state.wheel_count; ++i) {
    const auto& wr = rig.wheels[static_cast<size_t>(i)];
    auto& w = state.wheels[static_cast<size_t>(i)];
    w.radius = wr.radius;
    w.mass = wr.mass;
    w.inv_mass = wr.mass > 0.0f ? 1.0f / wr.mass : 0.0f;
    w.position = spawn + wr.local_anchor + Vec2{0.0f, -wr.suspension.rest_length};
  }
  state.energy = clamp(config_.initial_energy, 0.0f, config_.energy_capacity);
  state.previous_x = state.body.position.x;
  state.engine_temperature = config_.initial_engine_temperature;
  state.engine_cold_locked =
      state.engine_temperature < config_.cold_start_temperature;
  state.engine_overheated =
      state.engine_temperature >= config_.overheat_temperature;
  state.engine_running = !state.engine_cold_locked && !state.engine_overheated;
  state.engine_stalled = !state.engine_running;
}

PhysicsStepStats PhysicsEngine::step(const RoverRig& rig, const Terrain& terrain, RoverState& state,
                                     int discrete_action, const MechanicLayout& mechanics) const {
  PhysicsStepStats stats{};
  const float dt = config_.dt;
  state.landing_event = false;
  state.landing_fatal = false;
  const bool was_airborne = state.airborne;
  const int previous_airborne_steps = state.airborne_steps;
  const float pre_contact_vertical_speed = state.body.velocity.y;
  ControlInput control = decode_discrete_action(discrete_action, config_.body_tilt_torque);
  
  
  
  if (was_airborne) {
    state.lidar_active_steps = 0;
    state.lidar_range = 0.0f;
    control.lidar = false;
  }
  if (state.lidar_active_steps > 0) --state.lidar_active_steps;
  if (state.lidar_cooldown_steps > 0) --state.lidar_cooldown_steps;
  if (state.lidar_active_steps == 0) state.lidar_range = 0.0f;
  state.lidar_last_energy_cost = 0.0f;
  float lidar_energy_cost = 0.0f;
  const bool lidar_pressed =
      control.lidar && (state.previous_action & ControlLidar) == 0;
  if (lidar_pressed && state.lidar_cooldown_steps == 0) {
    const auto& lidar_zone = mechanics.at(state.body.position.x);
    const float requested_cost =
        config_.lidar_energy_cost * std::max(0.0f, lidar_zone.params.lidar_energy_mul);
    if (state.energy >= requested_cost) {
      lidar_energy_cost = requested_cost;
      state.lidar_last_energy_cost = requested_cost;
      state.lidar_active_steps =
          std::max(1, static_cast<int>(config_.lidar_scan_duration / std::max(0.0001f, dt)));
      state.lidar_cooldown_steps =
          std::max(state.lidar_active_steps,
                   static_cast<int>(config_.lidar_cooldown / std::max(0.0001f, dt)));
      state.lidar_range = config_.lidar_base_range *
                          std::max(0.0f, lidar_zone.params.lidar_range_mul);
    }
  }
  const bool toggle_charge_pressed =
      control.toggle_charge && (state.previous_action & ControlToggleCharge) == 0;
  if (toggle_charge_pressed) {
    state.solar_panel_requested = !state.solar_panel_requested;
  }
  const bool parked = std::abs(state.body.velocity.x) < 0.08f &&
                      std::abs(state.body.velocity.y) < 0.08f &&
                      std::abs(state.body.angular_velocity) < 0.12f;
  if (state.solar_panel_requested) {
    if (parked) {
      state.solar_panel_deployment =
          std::min(1.0f, state.solar_panel_deployment +
                             dt / std::max(0.1f, config_.panel_deploy_time));
    }
  } else {
    state.solar_panel_deployment =
        std::max(0.0f, state.solar_panel_deployment -
                           dt / std::max(0.1f, config_.panel_retract_time));
  }
  state.charging_active =
      state.solar_panel_requested && state.solar_panel_deployment >= 0.999f && parked;
  const bool charging_lockout =
      state.solar_panel_requested || state.solar_panel_deployment > 0.001f;
  if (charging_lockout) {
    state.engine_running = false;
    state.engine_stalled = true;
    control.throttle = 0.0f;
    control.brake = 1.0f;
    control.clutch = 0.0f;
    control.body_torque = 0.0f;
    control.shift_up = false;
    control.shift_down = false;
    control.toggle_drive = false;
    control.ignition = false;
  }
  const bool shift_up_pressed = control.shift_up && (state.previous_action & ControlShiftUp) == 0;
  const bool shift_down_pressed =
      control.shift_down && (state.previous_action & ControlShiftDown) == 0;
  const bool toggle_drive_pressed =
      control.toggle_drive && (state.previous_action & ControlToggleDrive) == 0;
  if (toggle_drive_pressed) {
    state.drive_mode = (state.drive_mode + 1) % 3;
  }
  if (state.shift_up_buffer_steps > 0) --state.shift_up_buffer_steps;
  if (state.shift_down_buffer_steps > 0) --state.shift_down_buffer_steps;
  if (shift_up_pressed) {
    state.shift_up_buffer_steps = 60;
    state.shift_down_buffer_steps = 0;
  }
  if (shift_down_pressed) {
    state.shift_down_buffer_steps = 60;
    state.shift_up_buffer_steps = 0;
  }
  const float shift_speed = std::abs(state.body.velocity.x);
  const bool clutch_pedal_down = control.clutch < 0.5f;
  
  
  
  
  const bool powered_upshift_request =
      control.shift_up && std::abs(control.throttle) > 0.5f && !clutch_pedal_down;
  
  
  
  
  
  
  
  if ((control.shift_up && clutch_pedal_down && !control.shift_down) ||
      powered_upshift_request) {
    state.shift_up_buffer_steps = std::max(state.shift_up_buffer_steps, 60);
    state.shift_down_buffer_steps = 0;
  }
  if (control.shift_down && clutch_pedal_down && !control.shift_up) {
    state.shift_down_buffer_steps = std::max(state.shift_down_buffer_steps, 60);
    state.shift_up_buffer_steps = 0;
  }
  const float clutch_target = control.clutch;
  const float clutch_rate = clutch_target > state.clutch_engagement ? 3.5f : 10.0f;
  state.clutch_engagement +=
      clamp(clutch_target - state.clutch_engagement, -clutch_rate * dt, clutch_rate * dt);
  state.clutch_engagement = clamp(state.clutch_engagement, 0.0f, 1.0f);
  if (state.shift_cooldown_steps > 0) {
    --state.shift_cooldown_steps;
  }
  
  
  
  const float drivetrain_load =
      clamp(std::abs(std::sin(state.body.angle)) * 2.2f, 0.0f, 1.0f);
  const float desired_post_shift_rpm =
      state.gear_index >= 0 && state.gear_index < kGearCount - 1
          ? kBasePostShiftRpm[state.gear_index] +
                std::abs(control.throttle) * 1000.0f + drivetrain_load * 1200.0f
          : 2500.0f;
  const float minimum_post_shift_rpm = 1700.0f + drivetrain_load * 800.0f;
  float next_gear_rpm = kIdleRpm;
  float recommended_upshift_rpm = 0.0f;
  float minimum_upshift_rpm = 0.0f;
  if (state.gear_index >= 0 && state.gear_index < kGearCount - 1) {
    next_gear_rpm = shift_speed / kGearMaxSpeeds[state.gear_index + 1] * kRedlineRpm;
    
    
    
    const float speed_step =
        kGearMaxSpeeds[state.gear_index + 1] / kGearMaxSpeeds[state.gear_index];
    recommended_upshift_rpm =
        clamp(desired_post_shift_rpm * speed_step, 2800.0f, 8500.0f);
    minimum_upshift_rpm =
        clamp(minimum_post_shift_rpm * speed_step, 1800.0f, 8500.0f);
  }
  const bool upshift_speed_ok =
      state.gear_index < 0 || next_gear_rpm >= minimum_post_shift_rpm * 0.98f;
  const bool upshift_recommended =
      state.gear_index >= 0 && state.engine_rpm >= recommended_upshift_rpm &&
      next_gear_rpm >= desired_post_shift_rpm * 0.92f;
  const bool can_shift_up =
      (clutch_pedal_down || powered_upshift_request) && state.shift_cooldown_steps == 0 &&
      state.gear_index < kGearCount - 1 &&
      upshift_speed_ok;
  const float lower_gear_rpm =
      state.gear_index > 0
          ? shift_speed / kGearMaxSpeeds[state.gear_index - 1] * kRedlineRpm
          : kIdleRpm;
  const bool can_shift_down = clutch_pedal_down && state.shift_cooldown_steps == 0 &&
                              state.gear_index >= 0 &&
                              (state.gear_index == 0 || lower_gear_rpm <= kSafeDownshiftRpm);
  const float minimum_loaded_rpm = 1500.0f + drivetrain_load * 1800.0f;
  state.recommended_upshift_rpm = recommended_upshift_rpm;
  state.minimum_upshift_rpm = minimum_upshift_rpm;
  state.projected_upshift_rpm = next_gear_rpm;
  state.upshift_speed_ok = upshift_speed_ok;
  state.upshift_recommended = upshift_recommended;
  state.can_shift_up = can_shift_up;
  state.can_shift_down = can_shift_down;
  state.should_shift_down = state.gear_index > 0 && state.engine_rpm < minimum_loaded_rpm;
  if (state.shift_up_buffer_steps > 0 && can_shift_up) {
    state.gear_index = std::min(state.gear_index + 1, kGearCount - 1);
    state.shift_cooldown_steps = 45;
    state.shift_up_buffer_steps = 0;
    state.shift_down_buffer_steps = 0;
    state.can_shift_up = false;
    state.can_shift_down = false;
  }
  if (state.shift_down_buffer_steps > 0 && can_shift_down) {
    state.gear_index = std::max(state.gear_index - 1, -1);
    state.shift_cooldown_steps = 45;
    state.shift_up_buffer_steps = 0;
    state.shift_down_buffer_steps = 0;
    state.can_shift_up = false;
    state.can_shift_down = false;
  }
  const bool in_neutral = state.gear_index < 0;
  const float gear_ratio = in_neutral ? 0.0f : kGearRatios[state.gear_index];
  const float gear_max_speed = in_neutral ? 1.0e9f : kGearMaxSpeeds[state.gear_index];
  const float road_coupled_rpm =
      in_neutral ? state.engine_rpm
                 : shift_speed / std::max(0.1f, gear_max_speed) * kRedlineRpm;
  const float driveline_load_factor =
      in_neutral ? 0.0f
                 : (state.gear_index == 0
                        ? 1.0f
                        : clamp((road_coupled_rpm -
                                 kMinimumLoadedRpm[state.gear_index] * 0.85f) /
                                    (kMinimumLoadedRpm[state.gear_index] * 0.15f),
                                0.0f, 1.0f));
  state.driveline_load_factor = driveline_load_factor;
  state.gear_energy_multiplier =
      in_neutral ? 0.0f : kGearEnergyMul[state.gear_index];
  state.should_shift_down =
      state.should_shift_down || (state.gear_index > 0 && driveline_load_factor < 0.35f);
  int front_wheel_index = 0;
  int rear_wheel_index = 0;
  for (int i = 1; i < state.wheel_count; ++i) {
    if (rig.wheels[static_cast<size_t>(i)].local_anchor.x >
        rig.wheels[static_cast<size_t>(front_wheel_index)].local_anchor.x) {
      front_wheel_index = i;
    }
    if (rig.wheels[static_cast<size_t>(i)].local_anchor.x <
        rig.wheels[static_cast<size_t>(rear_wheel_index)].local_anchor.x) {
      rear_wheel_index = i;
    }
  }
  const auto is_driven_wheel = [&](int i) {
    return state.drive_mode == 2 || (state.drive_mode == 1 && i == front_wheel_index) ||
           (state.drive_mode == 0 && i == rear_wheel_index);
  };
  const float tyre_coupling = state.drivetrain_grounded
                                  ? 1.0f - clamp(state.drivetrain_slip * 0.95f, 0.0f, 0.95f)
                                  : 0.05f;
  const float drivetrain_coupling = state.clutch_engagement * tyre_coupling;

  if (state.engine_overheated &&
      state.engine_temperature <= config_.overheat_restart_temperature) {
    state.engine_overheated = false;
  }
  if (state.engine_running &&
      state.engine_temperature <= config_.minimum_operating_temperature) {
    state.engine_running = false;
  }
  state.engine_cold_locked =
      state.engine_temperature < config_.cold_start_temperature;
  if (!state.engine_running && control.ignition && (in_neutral || clutch_pedal_down) &&
      !state.engine_overheated &&
      !state.engine_cold_locked) {
    state.engine_running = true;
    state.engine_rpm = std::max(state.engine_rpm, 700.0f);
  }
  state.cold_power_factor =
      clamp((state.engine_temperature - config_.minimum_operating_temperature) /
                std::max(1.0f, config_.full_power_temperature -
                                   config_.minimum_operating_temperature),
            0.0f, 1.0f);

  
  
  
  
  
  const float preliminary_rpm01 =
      clamp((state.engine_rpm - kIdleRpm) / (kRedlineRpm - kIdleRpm), 0.0f, 1.0f);
  const float preliminary_torque_curve =
      0.62f + 0.58f * (1.0f - std::abs(preliminary_rpm01 - 0.55f) / 0.55f);
  const float throttle_torque =
      state.engine_running
          ? std::abs(control.throttle) * config_.motor_torque * preliminary_torque_curve *
                (0.35f + state.cold_power_factor * 0.65f)
          : 0.0f;
  const float drag_torque =
      config_.engine_drag_torque * (0.35f + 0.65f * state.engine_rpm / kRedlineRpm);
  const float idle_governor =
      state.engine_running && (in_neutral || clutch_pedal_down) &&
              state.engine_rpm < kIdleRpm
          ? clamp((kIdleRpm - state.engine_rpm) * 0.08f, 0.0f,
                  config_.motor_torque * 0.45f)
          : 0.0f;
  constexpr float kRadPerSecToRpm = 60.0f / 6.28318530718f;
  const float free_rpm_acceleration =
      (throttle_torque + idle_governor - drag_torque) /
      std::max(0.05f, config_.engine_inertia) * kRadPerSecToRpm;
  float rpm_acceleration = free_rpm_acceleration;
  if (!in_neutral && state.clutch_engagement > 0.001f) {
    const float coupled_rpm = state.gear_index == 0 && std::abs(control.throttle) > 0.0f
                                  ? std::max(kIdleRpm, road_coupled_rpm)
                                  : road_coupled_rpm;
    // The clutch transfers torque over a finite synchronization interval.  A
    // first-order speed match avoids the previous near-instant RPM snap when
    // the pedal was released or a gear was selected under load.
    const float clutch_sync_gain = 2.4f + 1.2f * state.clutch_engagement;
    const float clutch_acceleration =
        clamp((coupled_rpm - state.engine_rpm) * clutch_sync_gain,
              -config_.clutch_sync_rate * 1.6f, config_.clutch_sync_rate);
    
    
    rpm_acceleration = clutch_acceleration * drivetrain_coupling +
                       free_rpm_acceleration * (1.0f - drivetrain_coupling * 0.88f);
  }
  state.engine_rpm =
      clamp(state.engine_rpm + rpm_acceleration * dt, 0.0f, kRedlineRpm);
  if (state.gear_index > 0 && state.clutch_engagement > 0.85f &&
      driveline_load_factor < 0.12f) {
    ++state.engine_lug_steps;
  } else {
    state.engine_lug_steps = 0;
  }
  if (state.engine_running && !in_neutral && !clutch_pedal_down &&
      state.gear_index > 0 &&
      (state.engine_rpm < 650.0f || state.engine_lug_steps >= 45)) {
    state.engine_running = false;
  }
  const float rpm01 = clamp((state.engine_rpm - kIdleRpm) / (kRedlineRpm - kIdleRpm), 0.0f, 1.0f);
  const float torque_curve = 0.62f + 0.58f * (1.0f - std::abs(rpm01 - 0.55f) / 0.55f);
  const float lug_factor = state.gear_index <= 0
                               ? 1.0f
                               : clamp((state.engine_rpm - 650.0f) / 1550.0f, 0.0f, 1.0f);

  const auto& body_zone = mechanics.at(state.body.position.x);
  state.solar_charge_rate =
      state.charging_active ? body_zone.params.solar_charge_rate : 0.0f;
  stats.energy_gain = state.solar_charge_rate * dt;
  const auto thermal = mechanics.thermal_at(state.body.position.x, body_zone);
  float ambient_temperature = thermal.ambient_temperature;
  float thermal_transfer = std::max(0.0f, thermal.thermal_transfer);
  if (body_zone.type == MechanicType::Liquid) {
    const float engine_immersion =
        clamp((body_zone.liquid_level -
               (state.body.position.y - rig.body.size.y * 0.25f)) /
                  std::max(0.1f, rig.body.size.y),
              0.0f, 1.0f);
    constexpr float kMarsAirTemperature = -50.0f;
    ambient_temperature = kMarsAirTemperature +
                          (ambient_temperature - kMarsAirTemperature) * engine_immersion;
    thermal_transfer = 1.0f + (thermal_transfer - 1.0f) * engine_immersion;
  }
  state.ambient_temperature = ambient_temperature;
  state.thermal_transfer = thermal_transfer;
  const float speed_cooling = 1.0f + std::abs(state.body.velocity.x) * 0.12f;
  const float cooling_rate =
      (state.engine_temperature - ambient_temperature) * config_.engine_cooling_rate *
      thermal_transfer * speed_cooling;
  const float combustion_heat =
      state.engine_running
          ? 0.08f + config_.engine_heat_rate * (0.15f + rpm01 * 0.85f) *
                         (0.12f + std::abs(control.throttle) * 0.88f) *
                         (0.80f + drivetrain_load * 0.35f)
          : 0.0f;
  state.engine_temperature += (combustion_heat - cooling_rate) * dt;
  if (state.engine_temperature >= config_.overheat_temperature) {
    state.engine_temperature = std::min(state.engine_temperature,
                                        config_.overheat_temperature + 15.0f);
    state.engine_overheated = true;
    state.engine_running = false;
  }
  if (state.engine_temperature <= config_.minimum_operating_temperature) {
    state.engine_running = false;
  }
  state.cold_power_factor =
      clamp((state.engine_temperature - config_.minimum_operating_temperature) /
                std::max(1.0f, config_.full_power_temperature -
                                   config_.minimum_operating_temperature),
            0.0f, 1.0f);
  state.engine_cold_locked =
      !state.engine_running &&
      state.engine_temperature < config_.cold_start_temperature;
  state.engine_stalled = !state.engine_running;
  
  
  const float gravity = config_.gravity * body_zone.params.gravity_mul;
  Vec2 body_force{0.0f, state.body.mass * gravity};
  body_force.x += body_zone.params.wind_force;
  float body_torque = control.body_torque;
  MechanicBodyContext body_context{};
  body_context.body_force = &body_force;
  body_context.body_torque = &body_torque;
  body_context.energy_cost = &stats.energy_cost;
  body_context.velocity = state.body.velocity;
  body_context.angular_velocity = state.body.angular_velocity;
  body_context.mass = state.body.mass;
  body_context.gravity = gravity;
  body_context.dt = dt;
  body_context.step_index = state.step_index;
  apply_body_mechanic(body_zone.biome_id, body_zone.params, body_context);
  const float body_cos = std::cos(state.body.angle);
  const float body_sin = std::sin(state.body.angle);
  const auto rotate_body = [body_cos, body_sin](Vec2 v) {
    return Vec2{body_cos * v.x - body_sin * v.y, body_sin * v.x + body_cos * v.y};
  };

  if (rig.body.size.y > 0.0f) {
    const float half_width = rig.body.size.x * 0.5f;
    const float half_height = rig.body.size.y * 0.5f;
    constexpr float sample_x[] = {-0.72f, 0.0f, 0.72f};
    const float sample_mass = state.body.mass / 3.0f;
    for (float nx : sample_x) {
      const Vec2 r = rotate_body({nx * half_width, -half_height});
      const Vec2 point = state.body.position + r;
      const auto& water_zone = mechanics.at(point.x);
      if (water_zone.type != MechanicType::Liquid) continue;
      const float submerged =
          clamp((water_zone.liquid_level - point.y) / rig.body.size.y, 0.0f, 1.0f);
      if (submerged <= 0.0f) continue;
      const Vec2 point_velocity = state.body.velocity + perp(r) * state.body.angular_velocity;
      const float point_speed = length(point_velocity);
      const Vec2 buoyancy{0.0f, sample_mass * -gravity * 0.74f * submerged};
      const Vec2 water_drag =
          point_velocity * (-sample_mass * submerged * (0.32f + point_speed * 0.52f));
      const Vec2 water_force = buoyancy + water_drag;
      body_force += water_force;
      body_torque += cross(r, water_force);
      stats.energy_cost += submerged * point_speed * point_speed * 0.006f * dt;
    }
  }

  float next_drivetrain_slip = 0.0f;
  bool next_drivetrain_grounded = false;
  bool any_wheel_grounded = false;
  for (int i = 0; i < state.wheel_count; ++i) {
    auto& wheel = state.wheels[static_cast<size_t>(i)];
    const auto& wr = rig.wheels[static_cast<size_t>(i)];
    const auto& wheel_zone = mechanics.at(wheel.position.x);
    const auto wheel_ground_sample = terrain.query(wheel.position.x);
    const float liquid_depth =
        wheel_zone.type == MechanicType::Liquid && wheel_ground_sample.solid
            ? std::max(0.0f, wheel_zone.liquid_level - wheel_ground_sample.height)
            : 0.0f;
    const float liquid_immersion = wheel_zone.type == MechanicType::Liquid
                                       ? clamp((wheel_zone.liquid_level -
                                                (wheel.position.y - wheel.radius)) /
                                                   std::max(0.05f, wheel.radius * 2.0f),
                                               0.0f, 1.0f)
                                       : 0.0f;
    const MechanicType wheel_mechanic =
        wheel_zone.type == MechanicType::Liquid && liquid_depth < 0.08f
            ? MechanicType::Sand
            : wheel_zone.type;
    const int wheel_biome_id =
        wheel_zone.type == MechanicType::Liquid && liquid_depth < 0.08f
            ? builtin_biome_id(MechanicType::Sand)
            : wheel_zone.biome_id;
    MechanicParams wheel_params = wheel_zone.params;
    if (wheel_mechanic == MechanicType::Sand && wheel_zone.type == MechanicType::Liquid) {
      wheel_params = MechanicParams{};
    }
    Vec2 wheel_force{0.0f, wheel.mass * gravity};
    if (liquid_immersion > 0.0f) {
      const float water_speed = length(wheel.velocity);
      wheel_force += wheel.velocity *
                     (-wheel.mass * liquid_immersion * (0.55f + water_speed * 0.85f));
      wheel_force.y += wheel.mass * -gravity * 0.48f * liquid_immersion;
    }
    const bool driven_wheel = is_driven_wheel(i);
    const int driven_wheel_count = state.drive_mode == 2 ? std::max(1, state.wheel_count) : 1;
    const float torque_split = driven_wheel ? 1.0f / static_cast<float>(driven_wheel_count) : 0.0f;

    const Vec2 anchor_offset = rotate_body(wr.local_anchor);
    const Vec2 anchor_world = state.body.position + anchor_offset;
    const Vec2 anchor_velocity = state.body.velocity + perp(anchor_offset) * state.body.angular_velocity;
    const Vec2 suspension_axis = rotate_body({0.0f, -1.0f});
    const float min_len = std::max(0.02f, std::min(wr.suspension.min_length, wr.suspension.rest_length));
    const float max_len = std::max(min_len + 0.01f,
                                   std::max(wr.suspension.max_length, wr.suspension.rest_length));
    auto constrain_to_suspension = [&]() {
      const Vec2 offset = wheel.position - anchor_world;
      float travel = dot(offset, suspension_axis);
      const Vec2 lateral = offset - suspension_axis * travel;
      if (dot(lateral, lateral) > 1.0e-10f) {
        wheel.position -= lateral;
      }

      const Vec2 rel_velocity = wheel.velocity - anchor_velocity;
      const Vec2 lateral_velocity =
          rel_velocity - suspension_axis * dot(rel_velocity, suspension_axis);
      if (dot(lateral_velocity, lateral_velocity) > 1.0e-10f) {
        const Vec2 delta_v = -lateral_velocity;
        wheel.velocity += delta_v;
        if (wheel.inv_mass > 0.0f) {
          const Vec2 impulse_on_wheel = delta_v / wheel.inv_mass;
          apply_body_impulse(state.body, -impulse_on_wheel, anchor_world);
        }
      }

      travel = dot(wheel.position - anchor_world, suspension_axis);
      const float clamped_travel = clamp(travel, min_len, max_len);
      if (clamped_travel != travel) {
        wheel.position = anchor_world + suspension_axis * clamped_travel;
        const float travel_speed = dot(wheel.velocity - anchor_velocity, suspension_axis);
        const bool moving_past_min = travel < min_len && travel_speed < 0.0f;
        const bool moving_past_max = travel > max_len && travel_speed > 0.0f;
        if (moving_past_min || moving_past_max) {
          const Vec2 delta_v = suspension_axis * (-travel_speed);
          wheel.velocity += delta_v;
          if (wheel.inv_mass > 0.0f) {
            const Vec2 impulse_on_wheel = delta_v / wheel.inv_mass;
            apply_body_impulse(state.body, -impulse_on_wheel, anchor_world);
          }
        }
      }
    };

    constrain_to_suspension();

    const float suspension_travel = dot(wheel.position - anchor_world, suspension_axis);
    const float spring_vel = dot(wheel.velocity - anchor_velocity, suspension_axis);
    const float spring_force_mag =
        -wr.suspension.stiffness * (suspension_travel - wr.suspension.rest_length) -
        wr.suspension.damping * spring_vel;
    // Progressive bump/rebound stops keep a landing compliant through most of the
    // stroke but resist bottoming-out much more strongly near either travel limit.
    // This makes the wheel assembly absorb a touchdown before the chassis receives
    // the corresponding impulse, rather than behaving like a rigid strut.
    constexpr float kStopTravel = 0.045f;
    float stop_force_mag = 0.0f;
    const float compression_stop = (min_len + kStopTravel) - suspension_travel;
    if (compression_stop > 0.0f) {
      stop_force_mag += 2400.0f * compression_stop * compression_stop;
      if (spring_vel < 0.0f) stop_force_mag += -spring_vel * 70.0f;
    }
    const float rebound_stop = suspension_travel - (max_len - kStopTravel);
    if (rebound_stop > 0.0f) {
      stop_force_mag -= 1200.0f * rebound_stop * rebound_stop;
      if (spring_vel > 0.0f) stop_force_mag -= spring_vel * 40.0f;
    }
    const Vec2 spring_force = suspension_axis * (spring_force_mag + stop_force_mag);
    wheel_force += spring_force;
    body_force -= spring_force;
    body_torque += cross(anchor_world - state.body.position, -spring_force);

    auto terrain_sample = terrain.query(wheel.position.x);
    float contact_x = wheel.position.x;
    float penetration = terrain_sample.height + wheel.radius - wheel.position.y;

    
    
    
    
    const float motion_sign = control.throttle != 0.0f
                                  ? (control.throttle > 0.0f ? 1.0f : -1.0f)
                                  : (anchor_velocity.x >= 0.0f ? 1.0f : -1.0f);
    const float probe_dx = motion_sign * wheel.radius * 0.72f;
    const auto leading_sample = terrain.query(wheel.position.x + probe_dx);
    const float tyre_half_height = wheel.radius * 0.693974f;  
    const float leading_penetration =
        leading_sample.height + tyre_half_height - wheel.position.y;
    if (leading_penetration > penetration) {
      terrain_sample = leading_sample;
      contact_x = wheel.position.x + probe_dx;
      penetration = leading_penetration;
    }
    WheelContact contact{};
    contact.wheel_index = i;
    
    
    
    constexpr float kContactSlop = 0.018f;
    if (terrain_sample.solid && penetration > -kContactSlop) {
      contact.active = true;
      contact.point = {contact_x, terrain_sample.height};
      contact.normal = terrain_sample.normal;
      contact.tangent = normalized({contact.normal.y, -contact.normal.x});
      contact.penetration = penetration;
      contact.ground_height = terrain_sample.height;
      contact.slope = terrain_sample.slope;

      const float normal_vel = dot(wheel.velocity, contact.normal);
      const float normal_force = penetration * 1800.0f - normal_vel * 80.0f;
      contact.normal_force = std::max(0.0f, normal_force);
      wheel_force += contact.normal * contact.normal_force;

      const float axle_speed = dot(anchor_velocity, contact.tangent);
      const float driven_speed = axle_speed * control.throttle;
      const float speed_fade = driven_speed > 0.0f
                                   ? clamp(1.0f - driven_speed / std::max(0.1f, gear_max_speed),
                                           0.0f, 1.0f)
                                   : 1.0f;
      
      
      
      const float drive_torque =
          (state.engine_running ? config_.motor_torque : 0.0f) * state.cold_power_factor *
          torque_curve * driveline_load_factor *
          lug_factor * gear_ratio *
          config_.final_drive_ratio *
          torque_split * speed_fade * control.throttle * state.clutch_engagement;
      const float drive_force = drive_torque / std::max(0.05f, wheel.radius);
      Vec2 traction_force{0.0f, 0.0f};
      MechanicContext ctx{};
      
      
      
      const float supported_mass =
          (state.body.mass + static_cast<float>(state.wheel_count) * wheel.mass) /
          static_cast<float>(std::max(1, state.wheel_count));
      contact.normal_force = std::max(contact.normal_force, supported_mass * -gravity);
      ctx.contact = &contact;
      ctx.wheel_force = &traction_force;
      ctx.body_force = &body_force;
      ctx.energy_cost = &stats.energy_cost;
      ctx.dt = dt;
      ctx.wheel_radius = wheel.radius;
      ctx.base_friction = config_.wheel_friction;
      ctx.drive_force = drive_force;
      ctx.immersion = liquid_immersion;
      
      
      
      float crawl_grip = 1.8f;
      switch (wheel_mechanic) {
        case MechanicType::Ice: crawl_grip = 0.38f; break;
        case MechanicType::Liquid: crawl_grip = 0.42f; break;
        case MechanicType::Mud: crawl_grip = 0.72f; break;
        case MechanicType::Sand: crawl_grip = 1.05f; break;
        case MechanicType::Crust: crawl_grip = 2.0f; break;
        default: break;
      }
      ctx.minimum_drive_limit =
          state.engine_running && driven_wheel && !in_neutral && state.gear_index == 0 &&
                  shift_speed < 1.5f &&
                  state.clutch_engagement > 0.5f
              ? supported_mass * -gravity * crawl_grip
              : 0.0f;
      ctx.wheel_speed = dot(wheel.velocity, contact.tangent);
      ctx.step_index = state.step_index;
      apply_mechanic(wheel_biome_id, wheel_params, ctx);
      body_force += traction_force;
      body_torque += cross(anchor_world - state.body.position, traction_force);

      if (control.brake > 0.0f) {
        float brake_surface_scale = mechanic_friction_scale(wheel_biome_id, wheel_params);
        if (wheel_mechanic == MechanicType::Liquid) {
          brake_surface_scale = 1.0f + (brake_surface_scale - 1.0f) * liquid_immersion;
        }
        const float max_brake = contact.normal_force * config_.wheel_friction *
                                brake_surface_scale * 1.5f;
        const float brake_force = -clamp(axle_speed * config_.brake_strength * control.brake,
                                        -max_brake, max_brake);
        const Vec2 brake = contact.tangent * brake_force;
        body_force += brake;
        body_torque += cross(anchor_world - state.body.position, brake);
      }
    }

    wheel.in_contact = contact.active;
    wheel.normal_force = contact.normal_force;
    wheel.slip = contact.slip;
    auto& deformation_contact = stats.deformation_contacts[static_cast<size_t>(i)];
    deformation_contact.active = contact.active;
    deformation_contact.x = contact.point.x;
    deformation_contact.penetration = contact.penetration;

    wheel.velocity += wheel_force * wheel.inv_mass * dt;
    wheel.velocity *= (1.0f - config_.linear_damping);
    wheel.position += wheel.velocity * dt;
    constrain_to_suspension();

    auto post_ground = terrain.query(wheel.position.x);
    const float post_penetration = post_ground.height + wheel.radius - wheel.position.y;
    if (post_ground.solid && post_penetration > 0.0f) {
      wheel.position += post_ground.normal * post_penetration;
      const float normal_speed = dot(wheel.velocity, post_ground.normal);
      if (normal_speed < 0.0f) {
        wheel.velocity -= post_ground.normal * normal_speed;
      }
      wheel.in_contact = true;
      contact.active = true;
      contact.point = {wheel.position.x, post_ground.height};
      contact.penetration = std::max(contact.penetration, post_penetration);
      deformation_contact.active = true;
      deformation_contact.x = contact.point.x;
      deformation_contact.penetration = contact.penetration;
    }
    constrain_to_suspension();

    const float wheel_brake_torque =
        -clamp(wheel.angular_velocity * config_.brake_strength * control.brake,
               -config_.brake_strength, config_.brake_strength);
    if (wheel.in_contact) {
      const auto rolling_ground = terrain.query(wheel.position.x);
      const Vec2 rolling_tangent = normalized({rolling_ground.normal.y, -rolling_ground.normal.x});
      const float rolling_speed = dot(state.body.velocity, rolling_tangent);
      
      
      const float target_angular_velocity = -rolling_speed / std::max(0.05f, wheel.radius);
      if (control.brake > 0.0f) {
        wheel.angular_velocity = 0.0f;
      } else if (driven_wheel && state.engine_running && !in_neutral &&
                 std::abs(control.throttle) > 0.0f) {
        const float powered_speed =
            control.throttle * (state.engine_rpm / kRedlineRpm) * gear_max_speed *
            driveline_load_factor;
        const float powered_angular_velocity =
            -powered_speed / std::max(0.05f, wheel.radius);
        const float visual_slip = clamp(wheel.slip, 0.0f, 1.0f);
        wheel.angular_velocity = target_angular_velocity +
                                 (powered_angular_velocity - target_angular_velocity) * visual_slip;
      } else {
        wheel.angular_velocity =
            std::abs(rolling_speed) <= 0.12f ? 0.0f : target_angular_velocity;
      }
    } else {
      if (driven_wheel && state.engine_running && !in_neutral &&
          std::abs(control.throttle) > 0.0f) {
        const float powered_speed =
            control.throttle * (state.engine_rpm / kRedlineRpm) * gear_max_speed *
            driveline_load_factor;
        const float powered_angular_velocity =
            -powered_speed / std::max(0.05f, wheel.radius);
        wheel.angular_velocity +=
            (powered_angular_velocity - wheel.angular_velocity) * 0.18f;
      } else {
        wheel.angular_velocity += wheel_brake_torque * wheel.inv_mass * dt;
        wheel.angular_velocity *= 0.985f;
      }
    }
    wheel.angle += wheel.angular_velocity * dt;
    if (driven_wheel) {
      next_drivetrain_slip = std::max(next_drivetrain_slip, wheel.slip);
      next_drivetrain_grounded = next_drivetrain_grounded || wheel.in_contact;
    }
    any_wheel_grounded = any_wheel_grounded || wheel.in_contact;
    stats.energy_cost += driven_wheel && state.engine_running && !in_neutral
                             ? std::abs(config_.motor_torque * control.throttle) *
                                   torque_split * state.clutch_engagement *
                                   kGearEnergyMul[state.gear_index] * 0.00012f * dt
                             : 0.0f;
  }
  state.drivetrain_slip = next_drivetrain_slip;
  state.drivetrain_grounded = next_drivetrain_grounded;

  bool body_ground_contact = false;
  if (rig.body.collision.type == CollisionType::Box) {
    const float hw = rig.body.collision.size.x * 0.5f;
    const float hh = rig.body.collision.size.y * 0.5f;
    const Vec2 samples[3] = {{-hw, -hh}, {0.0f, -hh}, {hw, -hh}};
    for (const Vec2 local : samples) {
      const Vec2 point = state.body.position + rotate_body(local);
      const auto ground = terrain.query(point.x);
      if (!ground.solid) continue;
      const float penetration = ground.height - point.y;
      if (penetration > 0.0f) {
        body_ground_contact = true;
        const Vec2 ground_normal = ground.normal;
        const Vec2 r = point - state.body.position;
        const Vec2 point_velocity = state.body.velocity + perp(r) * state.body.angular_velocity;
        const float normal_vel = dot(point_velocity, ground_normal);
        const float force_mag = std::max(0.0f, penetration * 3000.0f - normal_vel * 120.0f);
        const Vec2 force = ground_normal * force_mag;
        body_force += force;
        body_torque += cross(r, force);
        stats.hard_contact = std::max(stats.hard_contact, force_mag);
      }
    }
  }

  state.body.velocity += body_force * state.body.inv_mass * dt;
  state.body.velocity *= (1.0f - config_.linear_damping);
  if (charging_lockout) {
    state.body.velocity *= 0.84f;
    state.body.angular_velocity *= 0.80f;
  }
  state.body.position += state.body.velocity * dt;
  state.body.angular_velocity += body_torque * state.body.inv_inertia * dt;
  state.body.angular_velocity *= (1.0f - config_.angular_damping);
  state.body.angle += state.body.angular_velocity * dt;

  const bool grounded_now = any_wheel_grounded || body_ground_contact;
  if (!grounded_now) {
    
    
    state.airborne = state.has_grounded;
    state.airborne_steps = state.airborne
                               ? (was_airborne ? previous_airborne_steps + 1 : 1)
                               : 0;
  } else {
    state.has_grounded = true;
    state.airborne = false;
    state.airborne_steps = 0;
    if (was_airborne && previous_airborne_steps >= config_.ballistic_min_air_steps) {
      state.landing_event = true;
      state.last_impact_speed = std::max(0.0f, -pre_contact_vertical_speed);
      state.last_landing_angle = std::abs(state.body.angle);
      const bool bad_speed = state.last_impact_speed > config_.safe_landing_speed;
      const bool bad_angle = state.last_landing_angle > config_.safe_landing_angle;
      if (bad_speed || bad_angle || body_ground_contact) {
        state.landing_fatal = true;
        state.fatal_error = true;
        const float sign = state.body.angle < 0.0f ? -1.0f : 1.0f;
        state.body.angle = sign * std::max(std::abs(state.body.angle),
                                           config_.fatal_landing_flip_angle);
        state.body.angular_velocity += sign * (1.5f + state.last_impact_speed * 0.25f);
      }
    }
  }
  if (charging_lockout) {
    stats.energy_cost = 0.0f;
  }
  stats.energy_cost += lidar_energy_cost;
  state.energy = clamp(state.energy - stats.energy_cost + stats.energy_gain,
                       0.0f, config_.energy_capacity);
  return stats;
}

void PhysicsEngine::apply_body_impulse(RigidBodyState& body, Vec2 impulse, Vec2 point) const {
  body.velocity += impulse * body.inv_mass;
  body.angular_velocity += cross(point - body.position, impulse) * body.inv_inertia;
}

}  
