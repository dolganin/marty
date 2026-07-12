#pragma once

namespace mars {

enum class DiscreteAction : int {
  Nothing = 0,
  Gas = 1,
  Brake = 2,
  Reverse = 3,
  GasTiltLeft = 4,
  GasTiltRight = 5,
};

struct ControlInput {
  float motor_torque = 0.0f;
  float brake = 0.0f;
  float body_torque = 0.0f;
};

inline ControlInput decode_discrete_action(int action, float motor_torque, float tilt_torque) {
  ControlInput out{};
  switch (static_cast<DiscreteAction>(action)) {
    case DiscreteAction::Gas:
      out.motor_torque = motor_torque;
      break;
    case DiscreteAction::Brake:
      out.brake = 1.0f;
      break;
    case DiscreteAction::Reverse:
      out.motor_torque = -motor_torque;
      break;
    case DiscreteAction::GasTiltLeft:
      out.motor_torque = motor_torque;
      out.body_torque = tilt_torque;
      break;
    case DiscreteAction::GasTiltRight:
      out.motor_torque = motor_torque;
      out.body_torque = -tilt_torque;
      break;
    case DiscreteAction::Nothing:
    default:
      break;
  }
  return out;
}

}  // namespace mars
