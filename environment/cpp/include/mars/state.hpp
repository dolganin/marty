#pragma once

#include <array>
#include <cstdint>

#include "mars/math.hpp"

namespace mars {

constexpr int kMaxWheels = 8;
constexpr int kDefaultWheelCount = 2;

struct RigidBodyState {
  Vec2 position{};
  Vec2 velocity{};
  float angle = 0.0f;
  float angular_velocity = 0.0f;
  float mass = 1.0f;
  float inv_mass = 1.0f;
  float inertia = 1.0f;
  float inv_inertia = 1.0f;
};

struct WheelState {
  Vec2 position{};
  Vec2 velocity{};
  float angle = 0.0f;
  float angular_velocity = 0.0f;
  float radius = 0.24f;
  float mass = 1.0f;
  float inv_mass = 1.0f;
  bool in_contact = false;
  float slip = 0.0f;
  float normal_force = 0.0f;
};

struct RoverState {
  RigidBodyState body{};
  std::array<WheelState, kMaxWheels> wheels{};
  int wheel_count = kDefaultWheelCount;
  float energy = 100.0f;
  float damage = 0.0f;
  float previous_x = 0.0f;
  float last_reward = 0.0f;
  int previous_action = 0;
  int gear_index = -1;
  float engine_rpm = 900.0f;
  float engine_temperature = 20.0f;
  float ambient_temperature = -45.0f;
  float thermal_transfer = 1.0f;
  float cold_power_factor = 1.0f;
  float clutch_engagement = 1.0f;
  float drivetrain_slip = 0.0f;
  bool drivetrain_grounded = false;
  bool solar_panel_requested = false;
  bool charging_active = false;
  float solar_panel_deployment = 0.0f;
  float solar_charge_rate = 0.0f;
  float solar_irradiance = 0.0f;
  int lidar_active_steps = 0;
  int lidar_cooldown_steps = 0;
  float lidar_last_energy_cost = 0.0f;
  float lidar_range = 0.0f;
  Vec2 imu_acceleration{};
  float imu_angular_acceleration = 0.0f;
  float imu_impact = 0.0f;
  bool body_contact_front = false;
  bool body_contact_belly = false;
  bool body_contact_rear = false;
  int drive_mode = 0;
  bool engine_running = true;
  bool engine_stalled = false;
  bool engine_overheated = false;
  bool engine_cold_locked = false;
  float recommended_upshift_rpm = 0.0f;
  float minimum_upshift_rpm = 0.0f;
  float projected_upshift_rpm = 0.0f;
  bool upshift_speed_ok = false;
  bool upshift_recommended = false;
  bool can_shift_up = false;
  bool can_shift_down = false;
  bool should_shift_down = false;
  int shift_cooldown_steps = 0;
  int shift_up_buffer_steps = 0;
  int shift_down_buffer_steps = 0;
  int engine_lug_steps = 0;
  float driveline_load_factor = 0.0f;
  float gear_energy_multiplier = 1.0f;
  int step_index = 0;
  int episode_in_trial = 0;
  bool trial_start = true;



  bool airborne = false;
  bool has_grounded = false;
  int airborne_steps = 0;
  bool landing_event = false;
  bool landing_fatal = false;
  bool fatal_error = false;


  int termination_reason = 0;
  float last_impact_speed = 0.0f;
  float last_landing_angle = 0.0f;
};

struct EpisodeResult {
  float reward = 0.0f;
  bool terminated = false;
  bool truncated = false;
};

}
