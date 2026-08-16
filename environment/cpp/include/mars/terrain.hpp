#pragma once

#include <cstdint>
#include <vector>

#include "mars/math.hpp"

namespace mars {

struct TerrainSample {
  float height = 0.0f;
  float slope = 0.0f;
  Vec2 normal{0.0f, 1.0f};
  bool solid = true;
};

struct TerrainConfig {
  int sample_count = 2048;
  float dx = 0.25f;
  float base_height = 0.0f;
  float amplitude = 0.5f;
  float roughness = 0.35f;
  int crater_count = 18;
  int step_count = 16;
  float length = 512.0f;
};

class Terrain {
 public:
  void configure(const TerrainConfig& config);
  void generate(uint64_t seed);

  TerrainSample query(float x) const;
  void query_height_slope(float x, float& height, float& slope) const;
  void deform(float x, float radius, float amount);
  void add_height_at_index(int index, float amount);
  float carve_basin(float begin_x, float end_x, float depth);
  void carve_ledge(float begin_x, float end_x, float ramp_length, float ramp_height);

  int sample_count() const { return static_cast<int>(heights_.size()); }
  float dx() const { return dx_; }
  float length() const { return dx_ * static_cast<float>(heights_.size() - 1); }

 private:
  float height_at_index(int i) const;

  std::vector<float> heights_;
  std::vector<uint8_t> solid_;
  float dx_ = 0.25f;
  float inv_dx_ = 4.0f;
  float base_height_ = 0.0f;
  float amplitude_ = 0.5f;
  float roughness_ = 0.35f;
  int crater_count_ = 18;
  int step_count_ = 16;
};

}
