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

struct TerrainSurface {
  float begin_x = 0.0f;
  float end_x = 0.0f;
  float begin_height = 0.0f;
  float end_height = 0.0f;
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
  float safe_start_fraction = 0.02f;
  float difficulty_exponent = 0.7125f;
  // Virtual source coordinate for curriculum sampling.  It lets a generated
  // test world use the same distribution as a very distant part of a course.
  float difficulty_distance_offset = 0.0f;
  bool preserve_spawn_safety = true;
};

class Terrain {
 public:
  void configure(const TerrainConfig& config);
  void generate(uint64_t seed);

  TerrainSample query(float x) const;
  // Finds the highest collision surface reachable from a reference point.
  // This permits an upper dry route and a lower water/floor route at one X.
  TerrainSample query_near(float x, float reference_y) const;
  void query_height_slope(float x, float& height, float& slope) const;
  void deform(float x, float radius, float amount);
  void add_height_at_index(int index, float amount);
  float carve_basin(float begin_x, float end_x, float depth, uint64_t seed);
  void carve_ledge(float begin_x, float end_x, float ramp_length, float ramp_height);
  void add_surface(float begin_x, float end_x, float begin_height, float end_height);

  int sample_count() const { return static_cast<int>(heights_.size()); }
  float dx() const { return dx_; }
  float length() const { return dx_ * static_cast<float>(heights_.size() - 1); }
  float difficulty_at(float x) const;
  float safe_start_fraction() const { return safe_start_fraction_; }

 private:
  float height_at_index(int i) const;

  std::vector<float> heights_;
  std::vector<uint8_t> solid_;
  std::vector<TerrainSurface> surfaces_;
  float dx_ = 0.25f;
  float inv_dx_ = 4.0f;
  float base_height_ = 0.0f;
  float amplitude_ = 0.5f;
  float roughness_ = 0.35f;
  int crater_count_ = 18;
  int step_count_ = 16;
  float safe_start_fraction_ = 0.02f;
  float difficulty_exponent_ = 0.7125f;
  float difficulty_distance_offset_ = 0.0f;
  bool preserve_spawn_safety_ = true;
};

}
