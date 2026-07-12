# Mars Rover Env

Native C++ 2D side-view Mars rover RL environment with a thin Gymnasium wrapper.

## Build

Use a Python environment that can install build dependencies:

```bash
cd mars_rover
python -m pip install -e .
```

The editable install builds `_mars_rover_cpp` with pybind11.

## Manual Play

```bash
mars-rover-play --debug
```

Controls:

- `Right` / `D`: gas
- `Left` / `A`: reverse
- `Down` / `S` / `Space`: brake
- `J` / `L`: tilt while holding gas
- `R`: reset with a new hidden mechanic trial
- `N`: reset terrain while keeping the same hidden mechanic
- `Q` / `Esc`: quit

Rendering is native RGB output from C++. The temporary debug visual style is white background,
black terrain, and grayscale rover primitives. Texture atlas rendering can be layered onto the
same transforms later.
