#include "mars/batch_env.hpp"

#include <stdexcept>

#include "mars/renderer.hpp"

namespace mars {

BatchEnv::BatchEnv(int num_envs, EnvConfig config) {
  if (num_envs <= 0) {
    throw std::invalid_argument("num_envs must be positive");
  }
  envs_.reserve(static_cast<size_t>(num_envs));
  for (int i = 0; i < num_envs; ++i) {
    envs_.emplace_back(config);
  }
}

void BatchEnv::reset_all(uint64_t seed, float* obs_out) {
  for (int i = 0; i < num_envs(); ++i) {
    envs_[static_cast<size_t>(i)].reset(seed + static_cast<uint64_t>(i) * 9973ULL, true,
                                       obs_out + i * obs_dim());
  }
}

void BatchEnv::reset_at(int env_id, uint64_t seed, bool trial_start, float* obs_out) {
  envs_.at(static_cast<size_t>(env_id)).reset(seed, trial_start, obs_out);
}

void BatchEnv::step_batch(const int* actions, float* obs_out, float* rewards_out,
                          uint8_t* terminated_out, uint8_t* truncated_out) {
  for (int i = 0; i < num_envs(); ++i) {
    const auto out =
        envs_[static_cast<size_t>(i)].step(actions[i], obs_out + i * obs_dim());
    rewards_out[i] = out.reward;
    terminated_out[i] = out.terminated ? 1 : 0;
    truncated_out[i] = out.truncated ? 1 : 0;
  }
}

void BatchEnv::render_rgb(int env_id, uint8_t* rgb_out, int width, int height,
                          bool debug_overlay) const {
  RenderConfig config{};
  config.width = width;
  config.height = height;
  config.debug_overlay = debug_overlay;
  Renderer renderer(config);
  renderer.render_rgb(envs_.at(static_cast<size_t>(env_id)), rgb_out, width, height);
}

}  // namespace mars
