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
  return out;
}

}  
