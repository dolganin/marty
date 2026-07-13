#include "mars/biome_bank.hpp"

namespace mars {

const MechanicZone& MechanicLayout::at(float x) const {
  int lo = 0, hi = std::max(0, count - 1);
  while (lo < hi) { const int mid = (lo + hi) / 2; if (x < zones[static_cast<size_t>(mid)].end_x) hi = mid; else lo = mid + 1; }
  return zones[static_cast<size_t>(lo)];
}

MechanicParams MechanicLayout::thermal_at(float x) const { const auto& current = at(x); return thermal_at(x, current); }

MechanicParams MechanicLayout::thermal_at(float x, const MechanicZone& current) const {
  const int index = static_cast<int>(&current - zones.data()); MechanicParams result = current.params;
  const auto blend = [&](const MechanicParams& a, const MechanicParams& b, float t) {
    t = t * t * (3.0f - 2.0f * t);
    result.ambient_temperature = a.ambient_temperature + (b.ambient_temperature - a.ambient_temperature) * t;
    result.thermal_transfer = a.thermal_transfer + (b.thermal_transfer - a.thermal_transfer) * t;
  };
  if (index > 0) { const auto& previous = zones[static_cast<size_t>(index - 1)];
    const float width = std::min(4.0f, std::min(previous.end_x - previous.begin_x, current.end_x - current.begin_x) * 0.25f);
    if (width > 0.0f && x < current.begin_x + width) { blend(previous.params, current.params, clamp((x - (current.begin_x - width)) / (2.0f * width), 0.0f, 1.0f)); return result; }
  }
  if (index + 1 < count) { const auto& next = zones[static_cast<size_t>(index + 1)];
    const float width = std::min(4.0f, std::min(current.end_x - current.begin_x, next.end_x - next.begin_x) * 0.25f);
    if (width > 0.0f && x > current.end_x - width) blend(current.params, next.params, clamp((x - (current.end_x - width)) / (2.0f * width), 0.0f, 1.0f));
  }
  return result;
}

int builtin_biome_id(MechanicType type) { return static_cast<int>(type); }
float mechanic_friction_scale(int biome_id, const MechanicParams& params) { return biome_by_id(biome_id).friction_scale(params); }
void apply_mechanic(int biome_id, const MechanicParams& params, MechanicContext& ctx) { biome_by_id(biome_id).apply(params, ctx); }
void apply_body_mechanic(int biome_id, const MechanicParams& params, MechanicBodyContext& ctx) {
  biome_by_id(biome_id).apply_body_effects(params, ctx);
}

}  // namespace mars
