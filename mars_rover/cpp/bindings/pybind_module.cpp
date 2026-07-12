#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mars/batch_env.hpp"
#include "mars/renderer.hpp"

namespace py = pybind11;

namespace {

template <typename T>
T* checked_ptr(py::array_t<T, py::array::c_style>& arr, py::ssize_t expected) {
  if (arr.size() < expected) {
    throw std::runtime_error("array is smaller than expected");
  }
  return static_cast<T*>(arr.mutable_data());
}

const char* mechanic_name(mars::MechanicType type) {
  switch (type) {
    case mars::MechanicType::Normal:
      return "Normal";
    case mars::MechanicType::Sand:
      return "Sand";
    case mars::MechanicType::Ice:
      return "Ice";
    case mars::MechanicType::Mud:
      return "Mud";
    case mars::MechanicType::Wind:
      return "Wind";
    case mars::MechanicType::LowGravity:
      return "LowGravity";
    case mars::MechanicType::Crust:
      return "Crust";
  }
  return "Unknown";
}

}  // namespace

PYBIND11_MODULE(_mars_rover_cpp, m) {
  py::enum_<mars::CollisionType>(m, "CollisionType")
      .value("None_", mars::CollisionType::None)
      .value("Box", mars::CollisionType::Box)
      .value("Circle", mars::CollisionType::Circle);

  py::enum_<mars::MechanicType>(m, "MechanicType")
      .value("Normal", mars::MechanicType::Normal)
      .value("Sand", mars::MechanicType::Sand)
      .value("Ice", mars::MechanicType::Ice)
      .value("Mud", mars::MechanicType::Mud)
      .value("Wind", mars::MechanicType::Wind)
      .value("LowGravity", mars::MechanicType::LowGravity)
      .value("Crust", mars::MechanicType::Crust);

  py::class_<mars::Vec2>(m, "Vec2")
      .def(py::init<>())
      .def(py::init<float, float>())
      .def_readwrite("x", &mars::Vec2::x)
      .def_readwrite("y", &mars::Vec2::y);

  py::class_<mars::TerrainConfig>(m, "TerrainConfig")
      .def(py::init<>())
      .def_readwrite("sample_count", &mars::TerrainConfig::sample_count)
      .def_readwrite("dx", &mars::TerrainConfig::dx)
      .def_readwrite("base_height", &mars::TerrainConfig::base_height)
      .def_readwrite("amplitude", &mars::TerrainConfig::amplitude)
      .def_readwrite("roughness", &mars::TerrainConfig::roughness)
      .def_readwrite("crater_count", &mars::TerrainConfig::crater_count)
      .def_readwrite("step_count", &mars::TerrainConfig::step_count)
      .def_readwrite("length", &mars::TerrainConfig::length);

  py::class_<mars::PhysicsConfig>(m, "PhysicsConfig")
      .def(py::init<>())
      .def_readwrite("dt", &mars::PhysicsConfig::dt)
      .def_readwrite("gravity", &mars::PhysicsConfig::gravity)
      .def_readwrite("wheel_friction", &mars::PhysicsConfig::wheel_friction)
      .def_readwrite("motor_torque", &mars::PhysicsConfig::motor_torque)
      .def_readwrite("brake_strength", &mars::PhysicsConfig::brake_strength)
      .def_readwrite("body_tilt_torque", &mars::PhysicsConfig::body_tilt_torque)
      .def_readwrite("linear_damping", &mars::PhysicsConfig::linear_damping)
      .def_readwrite("angular_damping", &mars::PhysicsConfig::angular_damping);

  py::class_<mars::RewardConfig>(m, "RewardConfig")
      .def(py::init<>())
      .def_readwrite("progress_scale", &mars::RewardConfig::progress_scale)
      .def_readwrite("energy_cost_scale", &mars::RewardConfig::energy_cost_scale)
      .def_readwrite("flip_penalty", &mars::RewardConfig::flip_penalty)
      .def_readwrite("stuck_penalty", &mars::RewardConfig::stuck_penalty)
      .def_readwrite("hard_contact_penalty", &mars::RewardConfig::hard_contact_penalty)
      .def_readwrite("finish_bonus", &mars::RewardConfig::finish_bonus);

  py::class_<mars::TerminationConfig>(m, "TerminationConfig")
      .def(py::init<>())
      .def_readwrite("finish_x", &mars::TerminationConfig::finish_x)
      .def_readwrite("min_energy", &mars::TerminationConfig::min_energy)
      .def_readwrite("flip_angle", &mars::TerminationConfig::flip_angle)
      .def_readwrite("stuck_steps", &mars::TerminationConfig::stuck_steps)
      .def_readwrite("max_steps", &mars::TerminationConfig::max_steps);

  py::class_<mars::CollisionShapeConfig>(m, "CollisionShapeConfig")
      .def(py::init<>())
      .def_readwrite("type", &mars::CollisionShapeConfig::type)
      .def_readwrite("size", &mars::CollisionShapeConfig::size)
      .def_readwrite("radius", &mars::CollisionShapeConfig::radius);

  py::class_<mars::BodyRigConfig>(m, "BodyRigConfig")
      .def(py::init<>())
      .def_readwrite("sprite", &mars::BodyRigConfig::sprite)
      .def_readwrite("mass", &mars::BodyRigConfig::mass)
      .def_readwrite("inertia", &mars::BodyRigConfig::inertia)
      .def_readwrite("size", &mars::BodyRigConfig::size)
      .def_readwrite("collision", &mars::BodyRigConfig::collision)
      .def_readwrite("local_position", &mars::BodyRigConfig::local_position);

  py::class_<mars::SuspensionRigConfig>(m, "SuspensionRigConfig")
      .def(py::init<>())
      .def_readwrite("rest_length", &mars::SuspensionRigConfig::rest_length)
      .def_readwrite("stiffness", &mars::SuspensionRigConfig::stiffness)
      .def_readwrite("damping", &mars::SuspensionRigConfig::damping);

  py::class_<mars::WheelRigConfig>(m, "WheelRigConfig")
      .def(py::init<>())
      .def_readwrite("name", &mars::WheelRigConfig::name)
      .def_readwrite("sprite", &mars::WheelRigConfig::sprite)
      .def_readwrite("radius", &mars::WheelRigConfig::radius)
      .def_readwrite("mass", &mars::WheelRigConfig::mass)
      .def_readwrite("local_anchor", &mars::WheelRigConfig::local_anchor)
      .def_readwrite("suspension", &mars::WheelRigConfig::suspension);

  py::class_<mars::VisualPartRigConfig>(m, "VisualPartRigConfig")
      .def(py::init<>())
      .def_readwrite("name", &mars::VisualPartRigConfig::name)
      .def_readwrite("sprite", &mars::VisualPartRigConfig::sprite)
      .def_readwrite("parent", &mars::VisualPartRigConfig::parent)
      .def_readwrite("local_position", &mars::VisualPartRigConfig::local_position)
      .def_readwrite("local_rotation", &mars::VisualPartRigConfig::local_rotation)
      .def_readwrite("scale", &mars::VisualPartRigConfig::scale)
      .def_readwrite("collision", &mars::VisualPartRigConfig::collision);

  py::class_<mars::RoverRig>(m, "RoverRig")
      .def(py::init<>())
      .def_static("default_two_wheel", &mars::RoverRig::default_two_wheel)
      .def_readwrite("body", &mars::RoverRig::body)
      .def_readwrite("wheels", &mars::RoverRig::wheels)
      .def_readwrite("visual_parts", &mars::RoverRig::visual_parts);

  py::class_<mars::EnvConfig>(m, "EnvConfig")
      .def(py::init<>())
      .def_readwrite("terrain", &mars::EnvConfig::terrain)
      .def_readwrite("physics", &mars::EnvConfig::physics)
      .def_readwrite("reward", &mars::EnvConfig::reward)
      .def_readwrite("termination", &mars::EnvConfig::termination)
      .def_readwrite("rig", &mars::EnvConfig::rig)
      .def_readwrite("episodes_per_trial", &mars::EnvConfig::episodes_per_trial)
      .def_readwrite("debug", &mars::EnvConfig::debug);

  py::class_<mars::BatchEnv>(m, "MarsRoverBatchEnv")
      .def(py::init<int, mars::EnvConfig>(), py::arg("num_envs"), py::arg("config") = mars::EnvConfig{})
      .def_property_readonly("num_envs", &mars::BatchEnv::num_envs)
      .def_property_readonly("obs_dim", &mars::BatchEnv::obs_dim)
      .def_property_readonly("action_dim", &mars::BatchEnv::action_dim)
      .def("reset_all",
           [](mars::BatchEnv& self, uint64_t seed, py::array_t<float, py::array::c_style> obs) {
             auto* obs_ptr = checked_ptr(obs, self.num_envs() * self.obs_dim());
             py::gil_scoped_release release;
             self.reset_all(seed, obs_ptr);
           })
      .def("reset_at",
           [](mars::BatchEnv& self, int env_id, uint64_t seed, bool trial_start,
              py::array_t<float, py::array::c_style> obs) {
             auto* obs_ptr = checked_ptr(obs, self.obs_dim());
             py::gil_scoped_release release;
             self.reset_at(env_id, seed, trial_start, obs_ptr);
           })
      .def("step",
           [](mars::BatchEnv& self, py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<float, py::array::c_style> rewards,
              py::array_t<uint8_t, py::array::c_style> terminated,
              py::array_t<uint8_t, py::array::c_style> truncated) {
             if (actions.size() < self.num_envs()) {
               throw std::runtime_error("actions array is smaller than num_envs");
             }
             auto* obs_ptr = checked_ptr(obs, self.num_envs() * self.obs_dim());
             auto* rewards_ptr = checked_ptr(rewards, self.num_envs());
             auto* term_ptr = checked_ptr(terminated, self.num_envs());
             auto* trunc_ptr = checked_ptr(truncated, self.num_envs());
             const int* actions_ptr = static_cast<const int*>(actions.data());
             py::gil_scoped_release release;
             self.step_batch(actions_ptr, obs_ptr, rewards_ptr, term_ptr, trunc_ptr);
           })
      .def("render_rgb",
           [](mars::BatchEnv& self, int env_id, py::array_t<uint8_t, py::array::c_style> rgb,
              int width, int height, bool debug_overlay) {
             auto* rgb_ptr = checked_ptr(rgb, width * height * 3);
             self.render_rgb(env_id, rgb_ptr, width, height, debug_overlay);
           })
      .def("debug_info",
           [](mars::BatchEnv& self, int env_id) {
             const auto& env = self.env_at(env_id);
             const auto& state = env.state();
             const auto& params = env.mechanic_params();
             py::dict d;
             d["mechanic"] = mechanic_name(env.mechanic_type());
             d["x"] = state.body.position.x;
             d["y"] = state.body.position.y;
             d["vx"] = state.body.velocity.x;
             d["vy"] = state.body.velocity.y;
             d["angle"] = state.body.angle;
             d["energy"] = state.energy;
             d["damage"] = state.damage;
             d["episode_in_trial"] = state.episode_in_trial;
             d["friction_mul"] = params.friction_mul;
             d["sink_rate"] = params.sink_rate;
             d["viscosity"] = params.viscosity;
             d["wind_force"] = params.wind_force;
             d["gravity_mul"] = params.gravity_mul;
             return d;
           });
}
