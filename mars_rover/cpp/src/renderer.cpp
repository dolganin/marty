#include "mars/renderer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace mars {

namespace {

struct Color {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

void put_pixel(uint8_t* rgb, int width, int height, int x, int y, Color c) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  const size_t idx = static_cast<size_t>((y * width + x) * 3);
  rgb[idx + 0] = c.r;
  rgb[idx + 1] = c.g;
  rgb[idx + 2] = c.b;
}

void draw_line(uint8_t* rgb, int width, int height, int x0, int y0, int x1, int y1, Color c) {
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  while (true) {
    put_pixel(rgb, width, height, x0, y0, c);
    if (x0 == x1 && y0 == y1) {
      break;
    }
    const int e2 = 2 * err;
    if (e2 >= dy) {
      err += dy;
      x0 += sx;
    }
    if (e2 <= dx) {
      err += dx;
      y0 += sy;
    }
  }
}

void draw_circle(uint8_t* rgb, int width, int height, int cx, int cy, int radius, Color c) {
  const int r2 = radius * radius;
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        put_pixel(rgb, width, height, x, y, c);
      }
    }
  }
}

void draw_rotated_box(uint8_t* rgb, int width, int height, int cx, int cy, float angle, float hw,
                      float hh, Color c) {
  const float ca = std::cos(-angle);
  const float sa = std::sin(-angle);
  const int radius = static_cast<int>(std::ceil(std::sqrt(hw * hw + hh * hh)));
  for (int y = cy - radius; y <= cy + radius; ++y) {
    for (int x = cx - radius; x <= cx + radius; ++x) {
      const float dx = static_cast<float>(x - cx);
      const float dy = static_cast<float>(y - cy);
      const float lx = ca * dx - sa * dy;
      const float ly = sa * dx + ca * dy;
      if (std::abs(lx) <= hw && std::abs(ly) <= hh) {
        put_pixel(rgb, width, height, x, y, c);
      }
    }
  }
}

}  // namespace

Renderer::Renderer(RenderConfig config) : config_(config) {}

void Renderer::load_atlas(const AssetAtlas& atlas) {
  atlas_ = atlas;
}

void Renderer::render_rgb(const Env& env, uint8_t* rgb_out, int width, int height) {
  if (!rgb_out || width <= 0 || height <= 0) {
    return;
  }
  for (int i = 0; i < width * height; ++i) {
    rgb_out[i * 3 + 0] = 255;
    rgb_out[i * 3 + 1] = 255;
    rgb_out[i * 3 + 2] = 255;
  }
  const auto& terrain = env.terrain();
  const auto& state = env.state();
  const auto& rig = env.config().rig;
  const float ppm = config_.pixels_per_meter;
  const float camera_x = state.body.position.x - 3.0f;
  const float camera_y = state.body.position.y - 1.8f;

  auto world_to_screen = [&](Vec2 p) {
    const int sx = static_cast<int>((p.x - camera_x) * ppm);
    const int sy = height - 1 - static_cast<int>((p.y - camera_y) * ppm);
    return Vec2{static_cast<float>(sx), static_cast<float>(sy)};
  };

  for (int px = 0; px < width; ++px) {
    const float wx = camera_x + static_cast<float>(px) / ppm;
    const float h = terrain.query(wx).height;
    const int ground_y = height - 1 - static_cast<int>((h - camera_y) * ppm);
    for (int py = std::max(0, ground_y); py < height; ++py) {
      const size_t idx = static_cast<size_t>((py * width + px) * 3);
      rgb_out[idx + 0] = 0;
      rgb_out[idx + 1] = 0;
      rgb_out[idx + 2] = 0;
    }
  }

  for (int px = 1; px < width; ++px) {
    const float wx0 = camera_x + static_cast<float>(px - 1) / ppm;
    const float wx1 = camera_x + static_cast<float>(px) / ppm;
    const int y0 = height - 1 - static_cast<int>((terrain.query(wx0).height - camera_y) * ppm);
    const int y1 = height - 1 - static_cast<int>((terrain.query(wx1).height - camera_y) * ppm);
    draw_line(rgb_out, width, height, px - 1, y0, px, y1, {0, 0, 0});
  }

  for (int i = 0; i < state.wheel_count; ++i) {
    const auto& wheel = state.wheels[static_cast<size_t>(i)];
    const Vec2 sp = world_to_screen(wheel.position);
    const int radius = std::max(2, static_cast<int>(wheel.radius * ppm));
    draw_circle(rgb_out, width, height, static_cast<int>(sp.x), static_cast<int>(sp.y), radius,
                {18, 18, 18});
    draw_circle(rgb_out, width, height, static_cast<int>(sp.x), static_cast<int>(sp.y),
                std::max(1, radius / 2), {110, 110, 110});

    if (i < static_cast<int>(rig.wheels.size())) {
      const Vec2 anchor = state.body.position + rotate(rig.wheels[static_cast<size_t>(i)].local_anchor,
                                                       state.body.angle);
      const Vec2 asp = world_to_screen(anchor);
      draw_line(rgb_out, width, height, static_cast<int>(asp.x), static_cast<int>(asp.y),
                static_cast<int>(sp.x), static_cast<int>(sp.y), {35, 35, 35});
    }
  }

  const Vec2 bsp = world_to_screen(state.body.position);
  draw_rotated_box(rgb_out, width, height, static_cast<int>(bsp.x), static_cast<int>(bsp.y),
                   -state.body.angle, rig.body.size.x * 0.5f * ppm, rig.body.size.y * 0.5f * ppm,
                   {60, 60, 60});

  for (const auto& part : rig.visual_parts) {
    const Vec2 p = state.body.position + rotate(part.local_position, state.body.angle);
    const Vec2 sp = world_to_screen(p);
    const int size = std::max(2, static_cast<int>(0.08f * ppm));
    draw_rotated_box(rgb_out, width, height, static_cast<int>(sp.x), static_cast<int>(sp.y),
                     -(state.body.angle + part.local_rotation), size, size, {25, 25, 25});
  }

  if (config_.debug_overlay) {
    for (int i = 0; i < state.wheel_count; ++i) {
      const auto& wheel = state.wheels[static_cast<size_t>(i)];
      const Vec2 p = world_to_screen(wheel.position);
      const auto sample = terrain.query(wheel.position.x);
      const Vec2 ground = world_to_screen({wheel.position.x, sample.height});
      draw_line(rgb_out, width, height, static_cast<int>(p.x), static_cast<int>(p.y),
                static_cast<int>(ground.x), static_cast<int>(ground.y), {0, 128, 255});
    }
  }
}

}  // namespace mars
