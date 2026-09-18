#include "mars/renderer.hpp"
#include "mars/biome_bank.hpp"

#include <algorithm>
#include <cmath>

namespace mars {
namespace {

struct Color { uint8_t r, g, b; };

struct SurfaceStyle {
  Color ground;
  Color dust;
  Color liquid_color;
  BiomeVisuals particles;
  bool liquid;
};

Color color(BiomeColor c) { return {c.r, c.g, c.b}; }

SurfaceStyle surface_style(int biome_id) {
  const auto visuals = biome_by_id(biome_id).visuals();
  return {color(visuals.ground), color(visuals.particles), color(visuals.liquid),
          visuals, visuals.liquid_surface};
}

inline void put_pixel(uint8_t* rgb, int width, int height, int x, int y, Color c) {
  if (static_cast<unsigned>(x) >= static_cast<unsigned>(width) ||
      static_cast<unsigned>(y) >= static_cast<unsigned>(height)) return;
  const size_t idx = static_cast<size_t>((y * width + x) * 3);
  rgb[idx] = c.r; rgb[idx + 1] = c.g; rgb[idx + 2] = c.b;
}

void draw_line(uint8_t* rgb, int width, int height, int x0, int y0, int x1, int y1, Color c) {
  const int dx = std::abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;
  for (;;) {
    put_pixel(rgb, width, height, x0, y0, c);
    if (x0 == x1 && y0 == y1) break;
    const int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

void draw_circle(uint8_t* rgb, int width, int height, int cx, int cy, int radius, Color c) {
  const int r2 = radius * radius;
  const int y0 = std::max(0, cy - radius), y1 = std::min(height - 1, cy + radius);
  for (int y = y0; y <= y1; ++y) {
    const int dy = y - cy;
    const int span = static_cast<int>(std::sqrt(static_cast<float>(r2 - dy * dy)));
    const int x0 = std::max(0, cx - span), x1 = std::min(width - 1, cx + span);
    for (int x = x0; x <= x1; ++x) put_pixel(rgb, width, height, x, y, c);
  }
}

uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

void draw_rotated_box(uint8_t* rgb, int width, int height, int cx, int cy, float angle,
                      float hw, float hh, Color c) {
  const float ca = std::cos(-angle), sa = std::sin(-angle);
  const int radius = static_cast<int>(std::ceil(std::sqrt(hw * hw + hh * hh)));
  const int y0 = std::max(0, cy - radius), y1 = std::min(height - 1, cy + radius);
  const int x0 = std::max(0, cx - radius), x1 = std::min(width - 1, cx + radius);
  for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) {
    const float dx = static_cast<float>(x - cx), dy = static_cast<float>(y - cy);
    const float lx = ca * dx - sa * dy, ly = sa * dx + ca * dy;
    if (std::abs(lx) <= hw && std::abs(ly) <= hh) put_pixel(rgb, width, height, x, y, c);
  }
}

}  

Renderer::Renderer(RenderConfig config) : config_(config) {}

void Renderer::render_rgb(const Env& env, uint8_t* rgb, int width, int height) {
  if (!rgb || width <= 0 || height <= 0) return;
  const auto& terrain = env.terrain();
  const auto& state = env.state();
  const auto& body_zone = env.mechanic_at(state.body.position.x);
  const auto body_style = surface_style(body_zone.biome_id);
  const Color sky = color(body_style.particles.sky);
  for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i) {
    rgb[i * 3] = sky.r; rgb[i * 3 + 1] = sky.g; rgb[i * 3 + 2] = sky.b;
  }
  const auto& rig = env.config().rig;
  const float ppm = config_.pixels_per_meter;
  const float camera_x = state.body.position.x - 3.0f;
  const float camera_y = state.body.position.y - 1.8f;
  auto screen = [&](Vec2 p) {
    return Vec2{(p.x - camera_x) * ppm,
                static_cast<float>(height - 1) - (p.y - camera_y) * ppm};
  };

  for (int x = 0; x < width; ++x) {
    const float wx = camera_x + static_cast<float>(x) / ppm;
    const auto& zone = env.mechanic_at(wx);
    const float ground_height = terrain.query(wx).height;
    SurfaceStyle style = surface_style(zone.biome_id);
    if (zone.type == MechanicType::Liquid && zone.liquid_level - ground_height < 0.08f) {
      style = surface_style(builtin_biome_id(MechanicType::Sand));
    }
    const int gy = height - 1 - static_cast<int>((ground_height - camera_y) * ppm);
    const int start = std::max(0, gy);
    if (style.liquid) {
      const int water_start = std::max(
          0, height - 1 - static_cast<int>((zone.liquid_level - camera_y) * ppm));
      const int water_end = std::min(height, std::max(water_start, gy));
      for (int y = water_start; y < water_end; ++y) {
        const size_t i = static_cast<size_t>((y * width + x) * 3);
        rgb[i] = style.liquid_color.r; rgb[i + 1] = style.liquid_color.g;
        rgb[i + 2] = style.liquid_color.b;
      }
    }
    for (int y = start; y < height; ++y) {
      const size_t i = static_cast<size_t>((y * width + x) * 3);
      rgb[i] = style.ground.r; rgb[i + 1] = style.ground.g; rgb[i + 2] = style.ground.b;
    }
  }

  if (body_style.particles.ambient_particles > 0) {
    
    
    for (int p = 0; p < body_style.particles.ambient_particles; ++p) {
      const uint32_t h = hash_u32(static_cast<uint32_t>(p * 977 + state.step_index / 2));
      const int drift = static_cast<int>(state.step_index * body_style.particles.ambient_drift);
      const int x = (static_cast<int>(h & 1023u) + drift) % (width + 24) - 12;
      const int y = static_cast<int>((h >> 10) % static_cast<uint32_t>(std::max(1, height)));
      const int length = 2 + static_cast<int>((h >> 20) & 7u);
      draw_line(rgb, width, height, x, y, x + length, y - 1, body_style.dust);
    }
  }

  for (int i = 0; i < state.wheel_count; ++i) {
    const auto& wheel = state.wheels[static_cast<size_t>(i)];
    const Vec2 wheel_position = wheel.position;
    if (i < static_cast<int>(rig.wheels.size())) {
      const auto& wr = rig.wheels[static_cast<size_t>(i)];
      const Vec2 anchor =
          screen(state.body.position + rotate(wr.local_anchor, state.body.angle));
      const Vec2 wp = screen(wheel_position);
      draw_line(rgb, width, height, static_cast<int>(anchor.x), static_cast<int>(anchor.y),
                static_cast<int>(wp.x), static_cast<int>(wp.y), {70, 70, 70});
    }
    const Vec2 wp = screen(wheel_position);
    const int radius = std::max(2, static_cast<int>(wheel.radius * ppm));
    draw_circle(rgb, width, height, static_cast<int>(wp.x), static_cast<int>(wp.y), radius,
                {25, 25, 25});
    draw_circle(rgb, width, height, static_cast<int>(wp.x), static_cast<int>(wp.y),
                std::max(1, radius / 2), {120, 120, 120});
    const float spoke_angle = wheel.angle;
    const int sx = static_cast<int>(wp.x + std::cos(spoke_angle) * radius);
    const int sy = static_cast<int>(wp.y - std::sin(spoke_angle) * radius);
    draw_line(rgb, width, height, static_cast<int>(wp.x), static_cast<int>(wp.y), sx, sy,
              {230, 230, 230});

    if (wheel.in_contact) {
      const auto& wheel_zone = env.mechanic_at(wheel_position.x);
      const float wheel_ground_height = terrain.query(wheel_position.x).height;
      const int wheel_biome_id =
          wheel_zone.type == MechanicType::Liquid &&
                  wheel_zone.liquid_level - wheel_ground_height < 0.08f
              ? builtin_biome_id(MechanicType::Sand)
              : wheel_zone.biome_id;
      const auto wheel_style = surface_style(wheel_biome_id);
      const Color dust_color = wheel_style.dust;
      const auto ground = terrain.query(wheel_position.x);
      const Vec2 tangent = normalized({ground.normal.y, -ground.normal.x});
      const float rolling_speed = dot(state.body.velocity, tangent);
      const float dust_speed = std::abs(rolling_speed);
      const int particle_count =
          std::clamp(wheel_style.particles.base_particles +
                         static_cast<int>(dust_speed * wheel_style.particles.particle_rate),
                     0, wheel_style.particles.max_particles);
      const Vec2 spray_dir = tangent * (rolling_speed >= 0.0f ? -1.0f : 1.0f);
      const Vec2 origin{wheel_position.x, ground.height + wheel.radius * 0.12f};
      for (int p = 0; p < particle_count; ++p) {
        const uint32_t h = hash_u32(static_cast<uint32_t>(state.step_index * 37 + i * 101 + p * 17));
        const float rx = static_cast<float>(h & 255u) / 255.0f;
        const float ry = static_cast<float>((h >> 8) & 255u) / 255.0f;
        const float distance = wheel.radius + 0.08f +
                               rx * (0.18f + dust_speed * 0.10f) *
                                   wheel_style.particles.particle_spread;
        const float lift = ry * (0.05f + dust_speed * 0.035f) *
                           wheel_style.particles.particle_lift;
        const Vec2 particle_world = origin + spray_dir * distance + ground.normal * lift;
        const Vec2 pp = screen(particle_world);
        const int px = static_cast<int>(pp.x);
        const int py = static_cast<int>(pp.y);
        for (int size = 0; size < wheel_style.particles.particle_size; ++size)
          put_pixel(rgb, width, height, px + size, py, dust_color);
      }
    }
  }

  const Vec2 bp = screen(state.body.position);
  draw_rotated_box(rgb, width, height, static_cast<int>(bp.x), static_cast<int>(bp.y),
                   -state.body.angle, rig.body.size.x * 0.5f * ppm,
                   rig.body.size.y * 0.5f * ppm, {210, 105, 25});

  if (state.solar_panel_deployment > 0.001f) {
    const float deployment = state.solar_panel_deployment;
    const Vec2 mast_base_world =
        state.body.position + rotate({0.0f, rig.body.size.y * 0.5f}, state.body.angle);
    const Vec2 panel_center_world =
        state.body.position +
        rotate({0.0f, rig.body.size.y * 0.5f + 0.08f + 0.34f * deployment},
               state.body.angle);
    const Vec2 mast_base = screen(mast_base_world);
    const Vec2 panel_center = screen(panel_center_world);
    draw_line(rgb, width, height, static_cast<int>(mast_base.x), static_cast<int>(mast_base.y),
              static_cast<int>(panel_center.x), static_cast<int>(panel_center.y), {80, 80, 88});
    const float panel_half_width = (0.12f + 0.72f * deployment) * ppm;
    draw_rotated_box(rgb, width, height, static_cast<int>(panel_center.x),
                     static_cast<int>(panel_center.y), -state.body.angle,
                     panel_half_width, std::max(2.0f, 0.055f * ppm), {36, 82, 142});
    const Vec2 panel_left =
        screen(panel_center_world + rotate({-panel_half_width / ppm, 0.0f}, state.body.angle));
    const Vec2 panel_right =
        screen(panel_center_world + rotate({panel_half_width / ppm, 0.0f}, state.body.angle));
    draw_line(rgb, width, height, static_cast<int>(panel_left.x), static_cast<int>(panel_left.y),
              static_cast<int>(panel_right.x), static_cast<int>(panel_right.y), {105, 175, 225});
  }

  
  
  for (int x = 0; x < width; ++x) {
    const float wx = camera_x + static_cast<float>(x) / ppm;
    const auto& zone = env.mechanic_at(wx);
    if (zone.type != MechanicType::Liquid) continue;
    const Color liquid_tint = surface_style(zone.biome_id).liquid_color;
    const int surface_y = std::max(
        0, height - 1 - static_cast<int>((zone.liquid_level - camera_y) * ppm));
    const int ground_y = std::min(
        height, height - 1 - static_cast<int>((terrain.query(wx).height - camera_y) * ppm));
    for (int y = surface_y; y < ground_y; ++y) {
      const size_t idx = static_cast<size_t>((y * width + x) * 3);
      rgb[idx] = static_cast<uint8_t>((static_cast<int>(rgb[idx]) * 3 + liquid_tint.r) / 4);
      rgb[idx + 1] = static_cast<uint8_t>((static_cast<int>(rgb[idx + 1]) * 3 + liquid_tint.g) / 4);
      rgb[idx + 2] = static_cast<uint8_t>((static_cast<int>(rgb[idx + 2]) * 3 + liquid_tint.b) / 4);
    }
  }

  
  
  
  
  constexpr float kMinVisibleBrightness = 0.70f;
  const float brightness =
      clamp(body_style.particles.screen_brightness, kMinVisibleBrightness, 1.5f);
  for (size_t i = 0; i < static_cast<size_t>(width) * height * 3; ++i) {
    rgb[i] = static_cast<uint8_t>(clamp(static_cast<float>(rgb[i]) * brightness, 0.0f, 255.0f));
  }

  if (state.lidar_active_steps > 0 && state.lidar_range > 0.0f) {
    const Vec2 origin = screen(state.body.position + rotate({rig.body.size.x * 0.45f, 0.0f},
                                                             state.body.angle));
    constexpr int kRays = 18;
    for (int ray = 1; ray <= kRays; ++ray) {
      const float distance = state.lidar_range * static_cast<float>(ray) / kRays;
      const float x = state.body.position.x + distance;
      const Vec2 hit = screen({x, terrain.query(x).height});
      draw_line(rgb, width, height, static_cast<int>(origin.x), static_cast<int>(origin.y),
                static_cast<int>(hit.x), static_cast<int>(hit.y), {62, 255, 124});
    }
  }

}

}  
