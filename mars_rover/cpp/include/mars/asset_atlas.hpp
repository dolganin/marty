#pragma once

#include <string>
#include <unordered_map>

#include "mars/math.hpp"

namespace mars {

struct SpriteRect {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct SpriteMetadata {
  SpriteRect rect{};
  Vec2 pivot{0.5f, 0.5f};
  float pixels_per_meter = 100.0f;
};

struct AssetAtlas {
  std::string image;
  std::unordered_map<std::string, SpriteMetadata> sprites;
};

}  // namespace mars
