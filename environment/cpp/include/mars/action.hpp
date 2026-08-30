#pragma once

namespace mars {

enum ControlBits : int {
  ControlGas = 1 << 0,
  ControlBrake = 1 << 1,
  ControlReverse = 1 << 2,
  ControlClutchPedal = 1 << 3,
  ControlTiltLeft = 1 << 4,
  ControlTiltRight = 1 << 5,
  ControlShiftUp = 1 << 6,
  ControlShiftDown = 1 << 7,
  ControlToggleDrive = 1 << 8,
  ControlIgnition = 1 << 9,
  ControlToggleCharge = 1 << 10,
  ControlLidar = 1 << 11,
  ControlHeater = 1 << 12,
  // Momentary suspension kick, roof piston, and persistent low-range / propeller modes.
  ControlJump = 1 << 13,
  ControlToggleClimb = 1 << 14,
  ControlTogglePropeller = 1 << 15,
  ControlRoofPiston = 1 << 16,
  ControlJumpFront = 1 << 17,
  ControlJumpRear = 1 << 18,
  ControlRoofPistonFront = 1 << 19,
  ControlRoofPistonRear = 1 << 20,
  ControlLidarFront = 1 << 21,
  ControlLidarRear = 1 << 22,
  ControlLidarLeft = 1 << 23,
  ControlLidarRight = 1 << 24,
};

struct ControlInput {
  float throttle = 0.0f;
  float brake = 0.0f;
  float clutch = 1.0f;
  float body_torque = 0.0f;
  bool shift_up = false;
  bool shift_down = false;
  bool toggle_drive = false;
  bool ignition = false;
  bool toggle_charge = false;
  bool lidar = false;
  bool heater = false;
  bool jump = false;
  bool roof_piston = false;
  bool jump_front = false;
  bool jump_rear = false;
  bool roof_piston_front = false;
  bool roof_piston_rear = false;
  bool lidar_front = false;
  bool lidar_rear = false;
  bool lidar_left = false;
  bool lidar_right = false;
  bool toggle_climb = false;
  bool toggle_propeller = false;
};

inline ControlInput decode_discrete_action(int action, float tilt_torque) {
  ControlInput out{};
  if ((action & ControlGas) != 0) {
    out.throttle = (action & ControlReverse) != 0 ? -1.0f : 1.0f;
  }
  out.brake = (action & ControlBrake) != 0 ? 1.0f : 0.0f;
  out.clutch = (action & ControlClutchPedal) != 0 ? 0.0f : 1.0f;
  if ((action & ControlTiltLeft) != 0) {
    out.body_torque += tilt_torque;
  }
  if ((action & ControlTiltRight) != 0) {
    out.body_torque -= tilt_torque;
  }
  out.shift_up = (action & ControlShiftUp) != 0;
  out.shift_down = (action & ControlShiftDown) != 0;
  out.toggle_drive = (action & ControlToggleDrive) != 0;
  out.ignition = (action & ControlIgnition) != 0;
  out.toggle_charge = (action & ControlToggleCharge) != 0;
  out.lidar = (action & ControlLidar) != 0;
  out.heater = (action & ControlHeater) != 0;
  out.jump = (action & ControlJump) != 0;
  out.roof_piston = (action & ControlRoofPiston) != 0;
  out.jump_front = (action & ControlJumpFront) != 0;
  out.jump_rear = (action & ControlJumpRear) != 0;
  out.roof_piston_front = (action & ControlRoofPistonFront) != 0;
  out.roof_piston_rear = (action & ControlRoofPistonRear) != 0;
  out.lidar_front = (action & ControlLidarFront) != 0;
  out.lidar_rear = (action & ControlLidarRear) != 0;
  out.lidar_left = (action & ControlLidarLeft) != 0;
  out.lidar_right = (action & ControlLidarRight) != 0;
  out.toggle_climb = (action & ControlToggleClimb) != 0;
  out.toggle_propeller = (action & ControlTogglePropeller) != 0;
  return out;
}

}
