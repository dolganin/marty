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
  state.render_camera_position = spawn;
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
  const Vec2 previous_body_velocity = state.body.velocity;
  const float previous_body_angular_velocity = state.body.angular_velocity;
  state.landing_event = false;
  state.landing_fatal = false;
  state.body_contact_front = false;
  state.body_contact_belly = false;
  state.body_contact_rear = false;
  const bool was_airborne = state.airborne;
  const int previous_airborne_steps = state.airborne_steps;
  const float pre_contact_vertical_speed = state.body.velocity.y;
  ControlInput control = decode_discrete_action(discrete_action, config_.body_tilt_torque);
  const int jump_action_mask = ControlJump | ControlJumpFront | ControlJumpRear;
  const bool jump_active = control.jump || control.jump_front || control.jump_rear;
  const bool piston_active = control.roof_piston || control.roof_piston_front ||
                             control.roof_piston_rear;
  const bool jump_released = !jump_active && (state.previous_action & jump_action_mask) != 0;
  const bool climb_pressed = control.toggle_climb &&
      (state.previous_action & ControlToggleClimb) == 0;
  const bool propeller_pressed = control.toggle_propeller &&
      (state.previous_action & ControlTogglePropeller) == 0;
  if (climb_pressed) state.climb_mode = !state.climb_mode;
  if (propeller_pressed) state.propeller_mode = !state.propeller_mode;
  if (state.propeller_mode) {
    state.propeller_phase = std::fmod(
        state.propeller_phase + dt * (11.0f + 21.0f * std::abs(control.throttle)), 6.2831853f);
  }
  if (state.jump_cooldown_steps > 0) --state.jump_cooldown_steps;
  if (state.suspension_jump_phase_steps > 0) --state.suspension_jump_phase_steps;
  if (state.suspension_jump_phase == 2 && state.suspension_jump_phase_steps == 0) {
    state.suspension_jump_phase = 0;
    state.suspension_jump_charge = 0.0f;
  }
  if (piston_active) {
    state.roof_piston_mask = control.roof_piston ? 3 :
        (control.roof_piston_front ? 1 : 2);
  }
  const float piston_speed = piston_active ? 1.0f / 0.18f : -1.0f / 0.10f;
  state.roof_piston_extension = clamp(state.roof_piston_extension + piston_speed * dt, 0.0f, 1.0f);
  state.roof_piston_contact = false;



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
  const bool lidar_any = control.lidar || control.lidar_front || control.lidar_rear ||
                         control.lidar_left || control.lidar_right;
  const int lidar_action_mask = ControlLidar | ControlLidarFront | ControlLidarRear |
                                ControlLidarLeft | ControlLidarRight;
  const bool lidar_pressed = lidar_any && (state.previous_action & lidar_action_mask) == 0;
  if (lidar_pressed && state.lidar_cooldown_steps == 0) {
    state.lidar_direction = control.lidar || control.lidar_front ? 0 :
        (control.lidar_rear ? 1 : (control.lidar_left ? 2 : 3));
    const float requested_cost =
        config_.lidar_energy_cost * state.latent_lidar_energy_multiplier;
    if (state.energy >= requested_cost) {
      lidar_energy_cost = requested_cost;
      state.lidar_last_energy_cost = requested_cost;
      state.lidar_active_steps =
          std::max(1, static_cast<int>(config_.lidar_scan_duration / std::max(0.0001f, dt)));
      state.lidar_cooldown_steps =
          std::max(state.lidar_active_steps,
                   static_cast<int>(config_.lidar_cooldown / std::max(0.0001f, dt)));
      state.lidar_range = config_.lidar_base_range * state.latent_lidar_range_multiplier;
    }
  }
  const bool toggle_charge_pressed = control.toggle_charge &&
      (state.previous_action & ControlToggleCharge) == 0;
  if (toggle_charge_pressed) state.solar_panel_requested = !state.solar_panel_requested;
  if (state.solar_panel_requested) {
    state.solar_panel_deployment = std::min(
        1.0f, state.solar_panel_deployment + dt / std::max(0.1f, config_.panel_deploy_time));
  } else {
    state.solar_panel_deployment = std::max(
        0.0f, state.solar_panel_deployment - dt / std::max(0.1f, config_.panel_retract_time));
  }
  const bool rover_stationary = length(state.body.velocity) < 0.15f;
  state.charging_active = state.solar_panel_requested &&
                          state.solar_panel_deployment >= 0.999f && rover_stationary;
  const bool charging_lockout = false;
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
  // A flat battery leaves only limp mechanical creep. The motor is never
  // allowed to retain a meaningful fraction of nominal torque at zero charge.
  const float reserve_power = 0.025f + 0.975f * state.latent_charge_reserve;
  const float climb_torque = state.climb_mode ? 1.75f : 1.0f;

  const auto& body_zone = mechanics.at(state.body.position.x);
  state.solar_charge_rate = state.charging_active ? state.latent_solar_rate * 10.0f : 0.0f;
  state.solar_irradiance = state.latent_solar_rate;
  stats.energy_gain = state.solar_charge_rate * dt;
  float ambient_temperature = state.latent_ambient_temperature;
  float thermal_transfer = state.latent_thermal_transfer;
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
  const float speed = std::abs(state.body.velocity.x);
  const float airflow_factor = speed / (speed + 6.0f);
  const float conductance =
      (config_.engine_cooling_conductance +
       config_.engine_cooling_airflow * airflow_factor) * thermal_transfer;
  const float heat_removed = conductance * (state.engine_temperature - ambient_temperature);
  const float fuel_power = state.drive_fuel_rate;
  const float rpm_heat_factor =
      0.55f + 0.45f * clamp(state.engine_rpm / kRedlineRpm, 0.0f, 1.0f);
  const float combustion_heat =
      state.engine_running
          ? config_.engine_idle_heat * (0.6f + 0.4f * rpm01) +
                config_.engine_heat_per_fuel * fuel_power * rpm_heat_factor +
                config_.engine_idle_heat * 1.6f * std::abs(control.throttle) * rpm01
          : 0.0f;
  const float thermal_mass = std::max(0.01f, config_.engine_thermal_mass);
  const float heater_energy =
      std::max(0.0f, config_.engine_heater_energy_rate) * dt;
  state.heater_active = control.heater && !charging_lockout &&
                        !state.engine_overheated && state.energy >= heater_energy;
  if (state.heater_active) {
    stats.energy_cost += heater_energy;
  }
  const float total_heat = combustion_heat +
                           (state.heater_active
                                ? std::max(0.0f, config_.engine_heater_heat)
                                : 0.0f);
  state.engine_temperature += (total_heat - heat_removed) / thermal_mass * dt;
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


  const float gravity = config_.gravity * state.latent_gravity_multiplier;
  Vec2 body_force{0.0f, state.body.mass * gravity};
  body_force.x += state.latent_wind_force;
  float body_torque = control.body_torque;
  if (state.solar_panel_deployment > 0.01f && std::abs(state.body.velocity.x) > 0.3f) {
    const Vec2 panel_axis = rotate({0.0f, 1.0f}, state.body.angle);
    const float panel_lift = state.solar_panel_deployment *
        std::min(22.0f, std::abs(state.body.velocity.x) * std::abs(state.body.velocity.x) * 0.55f);
    const Vec2 panel_force = panel_axis * panel_lift;
    body_force += panel_force;
    body_torque += cross(rotate({0.0f, rig.body.size.y * 0.5f + 0.25f}, state.body.angle), panel_force);
  }
  if (jump_active && state.jump_cooldown_steps == 0 && !state.airborne) {
    // Holding K progressively compresses both struts. Releasing K converts the
    // stored compression into a rebound, so press duration directly controls
    // launch strength instead of creating a fixed impulse.
    if (state.suspension_jump_phase != 1) {
      state.suspension_jump_mask = control.jump ? 3 : (control.jump_front ? 1 : 2);
    }
    state.suspension_jump_phase = 1;
    state.suspension_jump_charge = clamp(state.suspension_jump_charge + dt / 0.75f, 0.0f, 1.0f);
    stats.energy_cost += 2.50f * dt;
  } else if (jump_released && state.suspension_jump_phase == 1 &&
             state.suspension_jump_charge >= 0.06f) {
    state.suspension_jump_phase = 2;
    state.suspension_jump_phase_steps = std::max(
        1, static_cast<int>((0.06f + 0.15f * state.suspension_jump_charge) / dt));
    state.jump_cooldown_steps = std::max(1, static_cast<int>(0.45f / dt));
  }
  if (std::abs(state.body.angle) > 1.35f) {
    state.recovery_state = clamp((std::abs(state.body.angle) - 1.35f) / 1.6f, 0.0f, 1.0f);
    // A suspension kick is a physical recovery aid, not a teleport: it only
    // creates a bounded moment and still needs ground contact/traction.
    if (jump_active) body_torque += state.body.angle > 0.0f ? -16.0f : 16.0f;
  } else {
    state.recovery_state = 0.0f;
  }
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
  // Layer effects have already been folded into WorldLatents by Env.  Keep the
  // old biome hook out of runtime force accumulation: otherwise overlapping
  // mechanics would secretly add independent forces instead of composing.
  (void)body_context;
  const float body_cos = std::cos(state.body.angle);
  const float body_sin = std::sin(state.body.angle);
  const auto rotate_body = [body_cos, body_sin](Vec2 v) {
    return Vec2{body_cos * v.x - body_sin * v.y, body_sin * v.x + body_cos * v.y};
  };
  const auto register_body_contact = [&](Vec2 local) {
    const float half_width = std::max(0.01f, rig.body.collision.size.x * 0.5f);
    if (local.x > half_width * 0.55f) {
      state.body_contact_front = true;
    } else if (local.x < -half_width * 0.55f) {
      state.body_contact_rear = true;
    } else {
      state.body_contact_belly = true;
    }
  };

  // The roof actuator can only push when its visible rod has found solid
  // terrain. This makes it a genuine contact force (useful after a rollover),
  // rather than a second hidden mid-air jump.
  if (piston_active && state.roof_piston_extension > 0.02f &&
      std::abs(state.body.angle) > 1.55f && rig.body.size.y > 0.0f) {
    const Vec2 roof_axis = rotate_body({0.0f, 1.0f});
    for (int piston = 0; piston < 2; ++piston) {
      const int piston_bit = 1 << piston;
      if ((state.roof_piston_mask & piston_bit) == 0) continue;
      const float local_x = piston == 0 ? rig.body.size.x * 0.34f : -rig.body.size.x * 0.34f;
      const Vec2 base = state.body.position + rotate_body({local_x, rig.body.size.y * 0.5f});
      const float rod_length = 0.14f + 0.36f * state.roof_piston_extension;
      const Vec2 tip = base + roof_axis * rod_length;
      const auto surface = terrain.query_near(tip.x, tip.y);
      if (!surface.solid || surface.height < tip.y - 0.015f) continue;
      const float penetration = std::min(0.12f, surface.height - tip.y + 0.015f);
      const float tip_speed = dot(state.body.velocity, roof_axis);
      // A compact recovery piston, deliberately far weaker than the former
      // kick and unavailable while upright. It cannot serve as a low-gravity
      // flight engine because contact is lost as soon as it lifts the rover.
      const float force = state.roof_piston_extension *
          std::max(0.0f, 70.0f * penetration - 8.0f * tip_speed);
      const Vec2 reaction = -roof_axis * force;
      body_force += reaction;
      body_torque += cross(base - state.body.position, reaction);
      // The off-centre rod already supplies a physical moment. A small
      // actuator bias chooses the rotation toward wheels-down rather than
      // launching straight away from the contact point.
      body_torque += state.body.angle > 0.0f ? -force * 0.75f : force * 0.75f;
      stats.hard_contact = std::max(stats.hard_contact, force);
      stats.energy_cost += 8.0f * state.roof_piston_extension * dt;
      state.roof_piston_contact = true;
    }
  }

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
  if (state.propeller_mode && std::abs(control.throttle) > 0.0f) {
    // The propeller works in every medium. Dense mud/water reduce efficiency
    // through the world viscosity instead of turning the system fully off.
    const float medium_efficiency = 1.0f / (1.0f + state.latent_viscosity * 0.65f);
    const bool in_liquid = body_zone.type == MechanicType::Liquid;
    const float rpm_threshold = in_liquid ? 1800.0f : 7600.0f;
    const float propeller_ready = clamp((state.engine_rpm - rpm_threshold) /
                                            std::max(1.0f, kRedlineRpm - rpm_threshold),
                                        0.0f, 1.0f);
    const float thrust = 26.0f * medium_efficiency * control.throttle * reserve_power *
                         propeller_ready;
    body_force += rotate_body({thrust, 0.0f});
    stats.energy_cost += (8.0f + 35.0f * std::abs(control.throttle)) *
                         medium_efficiency * dt;
  }

  float next_drivetrain_slip = 0.0f;
  bool next_drivetrain_grounded = false;
  bool any_wheel_grounded = false;
  for (int i = 0; i < state.wheel_count; ++i) {
    auto& wheel = state.wheels[static_cast<size_t>(i)];
    const auto& wr = rig.wheels[static_cast<size_t>(i)];
    const auto& wheel_zone = mechanics.at(wheel.position.x);
    const auto wheel_ground_sample = terrain.query_near(wheel.position.x, wheel.position.y);
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
    Vec2 wheel_force{0.0f, wheel.mass * gravity};
    if (liquid_immersion > 0.0f) {
      const float water_speed = length(wheel.velocity);
      wheel_force += wheel.velocity *
                     (-wheel.mass * liquid_immersion * (0.55f + water_speed * 0.85f));
      wheel_force.y += wheel.mass * -gravity * 0.48f * liquid_immersion;
    }
    const bool driven_wheel = is_driven_wheel(i);
    const bool jump_selected = state.suspension_jump_mask == 3 ||
        (state.suspension_jump_mask == 1 && i == front_wheel_index) ||
        (state.suspension_jump_mask == 2 && i == rear_wheel_index);
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
    float jump_rest_length = wr.suspension.rest_length;
    float jump_stiffness = wr.suspension.stiffness;
    if (jump_selected && state.suspension_jump_phase == 1) {
      jump_rest_length -= 0.11f * state.suspension_jump_charge;
      jump_stiffness *= 1.0f + 2.5f * state.suspension_jump_charge;
    } else if (jump_selected && state.suspension_jump_phase == 2) {
      jump_rest_length += 0.16f * state.suspension_jump_charge;
      jump_stiffness *= 1.0f + 4.5f * state.suspension_jump_charge;
    }
    const float spring_force_mag =
        -jump_stiffness * (suspension_travel - jump_rest_length) -
        wr.suspension.damping * spring_vel;




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

    auto terrain_sample = terrain.query_near(wheel.position.x, wheel.position.y);
    float contact_x = wheel.position.x;
    float penetration = terrain_sample.height + wheel.radius - wheel.position.y;





    const float motion_sign = control.throttle != 0.0f
                                  ? (control.throttle > 0.0f ? 1.0f : -1.0f)
                                  : (anchor_velocity.x >= 0.0f ? 1.0f : -1.0f);
    const float probe_dx = motion_sign * wheel.radius * 0.72f;
    const auto leading_sample = terrain.query_near(wheel.position.x + probe_dx, wheel.position.y);
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
      const float stable_penetration = clamp(penetration, 0.0f, 0.14f);
      const float impact_speed = clamp(-normal_vel, 0.0f, 4.0f);
      const float normal_force = stable_penetration * 1400.0f + impact_speed * 55.0f;
      contact.normal_force = clamp(normal_force, 0.0f, 700.0f);
      wheel_force += contact.normal * contact.normal_force;

      const float axle_speed = dot(anchor_velocity, contact.tangent);
      const float driven_speed = axle_speed * control.throttle;
      const float speed_fade = 1.0f / (1.0f + std::max(0.0f, driven_speed) /
                                               std::max(3.0f, gear_max_speed * 3.5f));



      const float drive_torque =
          (state.engine_running ? config_.motor_torque : 0.0f) * state.cold_power_factor *
          torque_curve * driveline_load_factor *
          lug_factor * reserve_power * climb_torque * gear_ratio *
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
      ctx.base_friction = config_.wheel_friction * state.latent_traction *
                          (state.climb_mode ? 1.18f : 1.0f);
      ctx.drive_force = drive_force;
      ctx.immersion = liquid_immersion;






      ctx.minimum_drive_limit =
          state.engine_running && driven_wheel && !in_neutral && state.gear_index == 0 &&
                  shift_speed < 1.5f &&
                  state.clutch_engagement > 0.5f
              ? supported_mass * -gravity * state.latent_traction
              : 0.0f;
      ctx.wheel_speed = dot(wheel.velocity, contact.tangent);
      ctx.step_index = state.step_index;
      const float limit = std::max(ctx.minimum_drive_limit,
                                   contact.normal_force * ctx.base_friction);
      const float applied_drive = clamp(ctx.drive_force, -limit, limit);
      traction_force += contact.tangent * applied_drive;
      contact.slip = std::abs(ctx.drive_force - applied_drive) /
                     (std::abs(ctx.drive_force) + 1.0f);
      // Viscosity and sink are outputs of the entire latent chain, not a
      // contribution from this zone alone.
      traction_force += contact.tangent *
                        (-state.latent_viscosity * 2.2f * ctx.wheel_speed);
      stats.energy_cost += (0.10f + state.latent_sink * 0.06f +
                            state.latent_viscosity * 0.02f * std::abs(ctx.wheel_speed)) * dt;
      body_force += traction_force;
      body_torque += cross(anchor_world - state.body.position, traction_force);

      if (control.brake > 0.0f) {
        float brake_surface_scale = state.latent_traction;
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

    auto post_ground = terrain.query_near(wheel.position.x, wheel.position.y);
    const float post_penetration = post_ground.height + wheel.radius - wheel.position.y;
    if (post_ground.solid && post_penetration > 0.0f) {
      const float correction = std::min(post_penetration, 0.08f);
      wheel.position += post_ground.normal * correction;
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
      const auto rolling_ground = terrain.query_near(wheel.position.x, wheel.position.y);
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
    if (state.climb_mode && driven_wheel && std::abs(control.throttle) > 0.0f) {
      stats.energy_cost += 5.0f * std::abs(control.throttle) * dt;
    }
  }
  state.drivetrain_slip = next_drivetrain_slip;
  state.drivetrain_grounded = next_drivetrain_grounded;

  bool body_ground_contact = false;
  if (rig.body.collision.type == CollisionType::Box) {
    const float hw = rig.body.collision.size.x * 0.5f;
    const float hh = rig.body.collision.size.y * 0.5f;



    const Vec2 samples[8] = {
        {-hw, -hh}, {0.0f, -hh}, {hw, -hh}, {hw, 0.0f},
        {hw, hh}, {0.0f, hh}, {-hw, hh}, {-hw, 0.0f},
    };
    for (const Vec2 local : samples) {
      const Vec2 point = state.body.position + rotate_body(local);
      const auto ground = terrain.query_near(point.x, point.y);
      if (!ground.solid) continue;
      const float penetration = ground.height - point.y;
      if (penetration > 0.0f) {
        body_ground_contact = true;
        register_body_contact(local);
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
  const float camera_follow = clamp(dt * 7.0f, 0.0f, 1.0f);
  state.render_camera_position +=
      (state.body.position - state.render_camera_position) * camera_follow;





  float body_contact_impulse = 0.0f;
  if (rig.body.collision.type == CollisionType::Box) {
    const float hw = rig.body.collision.size.x * 0.5f;
    const float hh = rig.body.collision.size.y * 0.5f;
    const Vec2 samples[8] = {
        {-hw, -hh}, {0.0f, -hh}, {hw, -hh}, {hw, 0.0f},
        {hw, hh}, {0.0f, hh}, {-hw, hh}, {-hw, 0.0f},
    };
    constexpr float kBodyFriction = 0.72f;
    constexpr float kRestitution = 0.03f;
    constexpr float kContactSlop = 0.001f;
    for (int iteration = 0; iteration < 4; ++iteration) {
      float deepest = 0.0f;
      Vec2 deepest_local{};
      TerrainSample deepest_ground{};
      bool found = false;
      for (const Vec2 local : samples) {
        const Vec2 point = state.body.position + rotate(local, state.body.angle);
        const auto ground = terrain.query_near(point.x, point.y);
        if (!ground.solid) continue;
        const float penetration = ground.height - point.y;
        if (penetration > deepest) {
          deepest = penetration;
          deepest_local = local;
          deepest_ground = ground;
          found = true;
        }
      }
      if (!found) break;

      body_ground_contact = true;
      register_body_contact(deepest_local);
      const Vec2 normal = deepest_ground.normal;
      state.body.position += normal * (deepest + kContactSlop);
      const Vec2 r = rotate(deepest_local, state.body.angle);
      Vec2 point_velocity = state.body.velocity + perp(r) * state.body.angular_velocity;
      const float normal_speed = dot(point_velocity, normal);
      float normal_impulse = 0.0f;
      if (normal_speed < 0.0f) {
        const float lever = cross(r, normal);
        const float effective_inv_mass =
            state.body.inv_mass + lever * lever * state.body.inv_inertia;
        if (effective_inv_mass > 1.0e-8f) {
          normal_impulse = -(1.0f + kRestitution) * normal_speed / effective_inv_mass;
          body_contact_impulse = std::max(body_contact_impulse, normal_impulse);
          apply_body_impulse(state.body, normal * normal_impulse,
                             state.body.position + r);
        }
      }

      point_velocity = state.body.velocity + perp(r) * state.body.angular_velocity;
      const Vec2 tangent = normalized({normal.y, -normal.x});
      const float tangent_speed = dot(point_velocity, tangent);
      const float tangent_lever = cross(r, tangent);
      const float tangent_inv_mass =
          state.body.inv_mass + tangent_lever * tangent_lever * state.body.inv_inertia;
      if (tangent_inv_mass > 1.0e-8f) {
        const float desired_impulse = -tangent_speed / tangent_inv_mass;
        const float gravity_support = state.body.mass * std::abs(gravity) * dt;
        const float friction_limit =
            kBodyFriction * std::max(normal_impulse, gravity_support);
        const float friction_impulse =
            clamp(desired_impulse, -friction_limit, friction_limit);
        apply_body_impulse(state.body, tangent * friction_impulse,
                           state.body.position + r);
      }
    }
  }

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
      }
    }
  }
  if (charging_lockout) {
    stats.energy_cost = 0.0f;
  }
  const Vec2 world_acceleration =
      (state.body.velocity - previous_body_velocity) / std::max(0.0001f, dt);
  state.imu_acceleration = rotate(world_acceleration, -state.body.angle);
  state.imu_angular_acceleration =
      (state.body.angular_velocity - previous_body_angular_velocity) /
      std::max(0.0001f, dt);
  state.imu_impact = std::max(stats.hard_contact * dt, body_contact_impulse) /
                     std::max(0.1f, state.body.mass);
  stats.energy_cost += lidar_energy_cost;
  state.passive_charge_rate = 0.0f;
  state.drive_fuel_rate = stats.drive_energy_cost / std::max(0.0001f, dt);
  state.energy = clamp(state.energy - stats.energy_cost + stats.energy_gain,
                       0.0f, config_.energy_capacity);
  return stats;
}

void PhysicsEngine::apply_body_impulse(RigidBodyState& body, Vec2 impulse, Vec2 point) const {
  body.velocity += impulse * body.inv_mass;
  body.angular_velocity += cross(point - body.position, impulse) * body.inv_inertia;
}

}
