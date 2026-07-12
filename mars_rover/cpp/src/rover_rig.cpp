#include "mars/rover_rig.hpp"

namespace mars {

RoverRig RoverRig::default_two_wheel() {
  RoverRig rig{};
  rig.body.sprite = "rover_body";
  rig.body.mass = 12.0f;
  rig.body.inertia = 4.0f;
  rig.body.size = {1.8f, 0.5f};
  rig.body.collision = {CollisionType::Box, {1.8f, 0.5f}, 0.0f};
  rig.wheels = {
      {"wheel_front", "rover_wheel", 0.24f, 1.0f, {0.65f, -0.35f}, {0.28f, 120.0f, 12.0f}},
      {"wheel_rear", "rover_wheel", 0.24f, 1.0f, {-0.65f, -0.35f}, {0.28f, 120.0f, 12.0f}},
  };
  rig.visual_parts = {
      {"camera_mast", "rover_camera_mast", "body", {0.15f, 0.45f}, 0.0f, {1.0f, 1.0f}, {}},
      {"antenna", "rover_antenna", "body", {-0.35f, 0.43f}, 0.0f, {1.0f, 1.0f}, {}},
      {"arm", "rover_arm", "body", {0.75f, 0.05f}, 0.0f, {1.0f, 1.0f}, {}},
  };
  return rig;
}

}  // namespace mars
