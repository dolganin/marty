# Mars Rover RL Environment Architecture

## Target Architecture

The C++20 core owns physics, contacts, terrain queries, hidden ground mechanics, reward,
termination, observation building, trial logic, batch stepping, and native rendering. Python is a
thin Gymnasium/SB3/CleanRL integration layer and must not run physics in the step loop.

Rendering is an optional module. Headless training should not construct `Renderer` or load atlas
images. `rgb_array` and human/debug rendering read simulation state after stepping.

## Asset And Rig Model

PNG assets are visual references only. Physics is configured numerically through simple shapes:
circles for wheels, a box for the body in MVP, optional simple shapes for later parts.

Atlas metadata defines sprite rectangles, pivots, and visual scale:

```json
{
  "image": "rover_atlas.png",
  "sprites": {
    "rover_body": {
      "x": 0,
      "y": 0,
      "w": 512,
      "h": 256,
      "pivot": [0.5, 0.5],
      "pixels_per_meter": 100
    }
  }
}
```

Rig metadata defines physical parts and visual attachments:

```yaml
rover:
  body:
    sprite: rover_body
    mass: 12.0
    inertia: 4.0
    size: [1.8, 0.5]
    collision: {type: box, size: [1.8, 0.5]}
  wheels:
    - name: wheel_front
      sprite: rover_wheel
      radius: 0.24
      mass: 1.0
      local_anchor: [0.65, -0.35]
      suspension: {rest_length: 0.28, stiffness: 120.0, damping: 12.0}
```

All simulation coordinates are meters. `pixels_per_meter` affects only rendering.

Sprite transform:

```text
body_sprite_world_position = body.position + rotate(body.local_visual_offset, body.angle)
wheel_sprite_world_position = wheel.position
visual_part_world_position = parent.position + rotate(local_position, parent.angle)
visual_part_world_rotation = parent.angle + local_rotation
```

Pivot is normalized in sprite-local image coordinates. `[0.5, 0.5]` means the sprite center is
placed at the computed world transform. `[0.5, 1.0]` means bottom-center is placed at the computed
world transform, useful for masts/rocks.

Collision must not be inferred from alpha masks or image contours because visual silhouettes are
unstable for physics: artists change them, transparent padding changes them, and collision meshes
from images are too expensive and noisy for deterministic high-throughput RL.

## Step Loop

1. Decode discrete action into motor torque, brake, body torque.
2. Query terrain under each wheel.
3. Compute wheel-ground penetration.
4. Compute normal force.
5. Compute suspension spring-damper force between body anchor and wheel.
6. Compute traction/friction and slip.
7. Apply hidden ground mechanic through enum + params + switch.
8. Integrate wheels and body.
9. Update energy.
10. Apply deforming terrain for crust mechanics.
11. Compute reward.
12. Check termination/truncation.
13. Write observation into caller-provided buffer.

## Hidden Mechanics

Hot path uses `MechanicType` + `MechanicParams` + `apply_mechanic(...)`, not virtual calls per
contact. Mechanics modify traction, sink, viscosity, wind force, gravity multiplier, energy drain,
or terrain deformation.

The observation must not expose mechanic id or raw hidden params. Debug info may expose them only
behind an explicit debug mode.

## Trial Protocol

A trial is `K` episodes under the same hidden mechanics. Terrain may change between episodes inside
the trial. At trial boundary hidden mechanics are resampled. Agent recurrent memory should reset
only when `trial_start == true`.

## BatchEnv

`BatchEnv` owns `N` independent `Env` instances and steps all of them through one pybind call:

```cpp
void step_batch(
  const int* actions,
  float* obs_out,
  float* rewards_out,
  uint8_t* terminated_out,
  uint8_t* truncated_out
);
```

Buffers are allocated by Python once and reused. The C++ step loop does not allocate.

## MVP Milestones

1. Build core data model, terrain heightmap, two-wheel suspension rover, wheel-ground contacts.
2. Add pybind batch stepping and Gymnasium wrapper.
3. Add YAML/JSON loaders for env, rig, mechanics, and atlas metadata.
4. Add SDL2/OpenGL renderer as optional target, with `rgb_array` buffer output.
5. Add deterministic mechanics train/test sampling and trial protocol controls.
6. Add body-ground collision, obstacles, damage/hard-contact penalties.
7. Add parallel `BatchEnv` backend using a fixed thread pool or TBB/OpenMP toggle.

## Performance Rules

- No per-step heap allocation.
- Use SoA or cache-friendly arrays if wheel/contact counts grow.
- Avoid Python loops over envs during training.
- Release the GIL inside pybind batch step.
- Keep each env RNG local and seedable.
- Do not use global random state.
- Do not load renderer or image assets in headless mode.
- Keep observation writes linear into contiguous buffers.
- Use simple deterministic collision primitives before considering Box2D.
