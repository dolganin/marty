# Mars Rover

Mars Rover is a native reinforcement-learning environment and a set of agent
baselines for procedural multi-mechanic courses.

## Repository layout

```text
environment/            Native environment distribution
  cpp/                   C++20 simulation and pybind11 bindings
  python/mars_rover_env/ Python API, configs and environment tools
  tests/                 Environment and physics tests
baselines/               Training and evaluation distribution
  python/mars_rover_agents/
  tests/
assets/
  textures/source/       Original rover art inputs
  textures/processed/    Runtime-ready textures
  screenshots/           Repository screenshots and captures
docs/                    Architecture, task definitions and experiment reports
scripts/                 Cross-platform setup and training workflows
artifacts/               Evaluation records and model artifacts
```

## Setup

```bash
scripts/setup.sh Release
```

## Test

```bash
.venv/bin/python -m pytest environment/tests baselines/tests -q
```

## Run

```bash
.venv/bin/mars-rover-play
.venv/bin/mars-agent-train-curriculum
```

Detailed documentation lives in [`docs/`](docs/).
