#include "mars/terrain.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace mars {

void Terrain::configure(const TerrainConfig& config) {
  dx_ = config.dx;
  inv_dx_ = 1.0f / config.dx;
  base_height_ = config.base_height;
  amplitude_ = config.amplitude;
  roughness_ = config.roughness;
  crater_count_ = config.crater_count;
  step_count_ = config.step_count;
  heights_.assign(static_cast<size_t>(config.sample_count), config.base_height);
  solid_.assign(static_cast<size_t>(config.sample_count), 1u);
}

void Terrain::generate(uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> phase_dist(0.0f, 6.2831853f);
  std::uniform_real_distribution<float> unit_dist(0.0f, 1.0f);
  const float p0 = phase_dist(rng);
  const float p1 = phase_dist(rng);
  const float p2 = phase_dist(rng);
  for (int i = 0; i < static_cast<int>(heights_.size()); ++i) {
    const float x = static_cast<float>(i) * dx_;
    heights_[static_cast<size_t>(i)] =
        base_height_ +
        amplitude_ * (0.45f * std::sin(0.13f * x + p0) +
                      0.22f * std::sin(0.53f * x + p1) +
                      0.10f * std::sin(1.37f * x + p2));
  }

  const float max_x = length();
  for (int n = 0; n < step_count_; ++n) {
    const float x0 = 5.0f + unit_dist(rng) * std::max(1.0f, max_x - 8.0f);
    const float width = 0.6f + unit_dist(rng) * 2.8f;
    const float height = (unit_dist(rng) * 2.0f - 1.0f) * amplitude_ * (0.35f + roughness_);
    for (int i = 0; i < static_cast<int>(heights_.size()); ++i) {
      const float x = static_cast<float>(i) * dx_;
      const float t = clamp((x - x0) / width, 0.0f, 1.0f);
      const float smooth = t * t * (3.0f - 2.0f * t);
      heights_[static_cast<size_t>(i)] += height * smooth;
    }
  }

  for (int n = 0; n < crater_count_; ++n) {
    const float cx = 7.0f + unit_dist(rng) * std::max(1.0f, max_x - 10.0f);
    const float radius = 0.8f + unit_dist(rng) * 2.5f;
    const float depth = amplitude_ * (0.25f + 0.55f * unit_dist(rng));
    for (int i = 0; i < static_cast<int>(heights_.size()); ++i) {
      const float x = static_cast<float>(i) * dx_;
      const float d = std::abs(x - cx) / radius;
      if (d < 1.0f) {
        const float bowl = 0.5f + 0.5f * std::cos(d * 3.14159265f);
        heights_[static_cast<size_t>(i)] -= depth * bowl;
      } else if (d < 1.45f) {
        const float rim_t = (d - 1.0f) / 0.45f;
        heights_[static_cast<size_t>(i)] += depth * 0.22f * (1.0f - rim_t);
      }
    }
  }

  for (int pass = 0; pass < 2; ++pass) {
    float prev = heights_.front();
    for (int i = 1; i < static_cast<int>(heights_.size()) - 1; ++i) {
      const float current = heights_[static_cast<size_t>(i)];
      heights_[static_cast<size_t>(i)] =
          prev * 0.18f + current * 0.64f + heights_[static_cast<size_t>(i + 1)] * 0.18f;
      prev = current;
    }
  }

  const int spawn_safe_samples = std::min(static_cast<int>(heights_.size()), static_cast<int>(5.0f / dx_));
  for (int i = 0; i < spawn_safe_samples; ++i) {
    const float t = static_cast<float>(i) / std::max(1.0f, static_cast<float>(spawn_safe_samples - 1));
    heights_[static_cast<size_t>(i)] =
        base_height_ * (1.0f - t) + heights_[static_cast<size_t>(i)] * t;
  }
}

TerrainSample Terrain::query(float x) const {
  if (heights_.empty()) {
    return {};
  }
  const int last = static_cast<int>(heights_.size()) - 1;
  const float fx = clamp(x * inv_dx_, 0.0f, static_cast<float>(last));
  const int i0 = static_cast<int>(std::floor(fx));
  const int i1 = std::min(i0 + 1, last);
  if (solid_[static_cast<size_t>(i0)] == 0u || solid_[static_cast<size_t>(i1)] == 0u) {
    return {-1000.0f, 0.0f, {0.0f, 1.0f}, false};
  }
  float h, slope;
  query_height_slope(x, h, slope);
  const float inv_len = 1.0f / std::sqrt(1.0f + slope * slope);
  return {h, slope, {-slope * inv_len, inv_len}, true};
}

void Terrain::query_height_slope(float x, float& h, float& slope) const {
  if (heights_.empty()) {
    h = 0.0f;
    slope = 0.0f;
    return;
  }
  const int last = static_cast<int>(heights_.size()) - 1;
  const float fx = clamp(x * inv_dx_, 0.0f, static_cast<float>(last));
  const int i0 = static_cast<int>(std::floor(fx));
  const int i1 = std::min(i0 + 1, last);
  const float t = fx - static_cast<float>(i0);
  h = heights_[static_cast<size_t>(i0)] * (1.0f - t) + heights_[static_cast<size_t>(i1)] * t;
  const int im = std::max(i0 - 1, 0);
  const int ip = std::min(i0 + 1, last);
  slope = (heights_[static_cast<size_t>(ip)] - heights_[static_cast<size_t>(im)]) /
          (static_cast<float>(ip - im) * dx_ + 1.0e-6f);
}

void Terrain::deform(float x, float radius, float amount) {
  if (heights_.empty() || radius <= 0.0f) {
    return;
  }
  const int first = std::max(0, static_cast<int>(std::floor((x - radius) / dx_)));
  const int last = std::min(static_cast<int>(heights_.size() - 1),
                            static_cast<int>(std::ceil((x + radius) / dx_)));
  for (int i = first; i <= last; ++i) {
    const float sx = static_cast<float>(i) * dx_;
    const float d = std::abs(sx - x) / radius;
    const float w = clamp(1.0f - d, 0.0f, 1.0f);
    if (solid_[static_cast<size_t>(i)] != 0u) {
      heights_[static_cast<size_t>(i)] -= amount * w;
    }
  }
}

float Terrain::carve_basin(float begin_x, float end_x, float depth) {
  if (heights_.empty() || end_x <= begin_x || depth <= 0.0f) {
    return base_height_;
  }
  const float left_height = query(begin_x).height;
  const float right_height = query(end_x).height;
  const float water_level = std::min(left_height, right_height);
  const int first = std::max(0, static_cast<int>(std::floor(begin_x * inv_dx_)));
  const int last = std::min(static_cast<int>(heights_.size() - 1),
                            static_cast<int>(std::ceil(end_x * inv_dx_)));
  const float inv_width = 1.0f / (end_x - begin_x);
  for (int i = first; i <= last; ++i) {
    const float x = static_cast<float>(i) * dx_;
    const float t = clamp((x - begin_x) * inv_width, 0.0f, 1.0f);
    const float bowl = std::sin(t * 3.14159265f);
    const float bank = left_height * (1.0f - t) + right_height * t;
    heights_[static_cast<size_t>(i)] = bank - depth * bowl * bowl;
  }
  return water_level;
}

void Terrain::carve_ledge(float begin_x, float end_x, float ramp_length,
                          float ramp_height) {
  if (heights_.empty() || end_x <= begin_x) return;
  const float ramp_begin = std::max(0.0f, begin_x - std::max(0.0f, ramp_length));
  const int ramp_first = std::max(0, static_cast<int>(std::floor(ramp_begin * inv_dx_)));
  const int ramp_last = std::min(static_cast<int>(heights_.size() - 1),
                                 static_cast<int>(std::floor(begin_x * inv_dx_)));
  for (int i = ramp_first; i <= ramp_last; ++i) {
    const float x = static_cast<float>(i) * dx_;
    const float t = clamp((x - ramp_begin) / std::max(dx_, begin_x - ramp_begin), 0.0f, 1.0f);
    // Smooth launch ramp with a non-zero lip slope. The ramp changes only raw
    // geometry; it does not encode whether the gap should be jumped or avoided.
    heights_[static_cast<size_t>(i)] += ramp_height * t * t;
  }
  const int first = std::max(0, static_cast<int>(std::ceil(begin_x * inv_dx_)));
  const int last = std::min(static_cast<int>(heights_.size() - 1),
                            static_cast<int>(std::floor(end_x * inv_dx_)));
  for (int i = first; i <= last; ++i) solid_[static_cast<size_t>(i)] = 0u;
}

float Terrain::height_at_index(int i) const {
  return heights_[static_cast<size_t>(std::clamp(i, 0, static_cast<int>(heights_.size() - 1)))];
}

}  // namespace mars
