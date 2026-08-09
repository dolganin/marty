# Mars Rover Environment

Native C++20 2D Mars rover environment with a thin Python/Gymnasium wrapper. The same commands
build the native extension, run smoke tests and start the manual debugger on Windows and Linux.

## Prerequisites

- Python 3.10+ (64-bit)
- Windows: GCC/MinGW-w64 or Visual Studio 2022 Build Tools
- Linux (Debian/Ubuntu): `sudo apt install python3-venv python3-dev build-essential`
- Linux GUI play mode: `sudo apt install python3-tk`

The Windows setup script automatically finds an installed GCC/MinGW-w64 toolchain and also supports
MSVC. Linux uses the system GCC toolchain. Set `MARS_ROVER_OPENMP=0` only if the compiler does not
provide OpenMP; vectorized environments use it by default.

## Complete Linux start-to-finish guide

All commands below are run from the `mars_rover` directory. In a container running as root, omit
`sudo` from the package installation command.

1. Install the compiler, Python tooling, Tk GUI support and TLS certificates:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3 python3-venv python3-dev python3-tk build-essential ca-certificates curl
   sudo update-ca-certificates
   ```

2. Create the virtual environment and build the native extension:

   ```bash
   bash scripts/setup.sh Release
   ```

3. To generate LLM biomes, create the ignored local configuration:

   ```bash
   cp python/mars_rover_env/configs/biome_generator.yaml \
      python/mars_rover_env/configs/biome_generator.local.yaml
   chmod 600 python/mars_rover_env/configs/biome_generator.local.yaml
   ```

   Edit `biome_generator.local.yaml` and override the connection settings:

   ```yaml
   openai:
     address: "https://api.openai.com/v1"
     api_mode: "responses"
     model: "gpt-5-mini"
     token: "YOUR_TOKEN"
   ```

   For an OpenAI-compatible `/chat/completions` endpoint, set `api_mode: "chat_completions"`.
   The local file is excluded from Git. `OPENAI_API_KEY` may be used instead of storing `token`.

4. Validate the endpoint settings without making an API request, then generate the biome bank:

   ```bash
   .venv/bin/python -m mars_rover_env.tools.generate_biomes \
     --split train --count 12 --replace --dry-run
   .venv/bin/python -m mars_rover_env.tools.generate_biomes \
     --split train --count 12 --replace
   ```

5. Rebuild after generation because biomes are compiled into the native module:

   ```bash
   bash scripts/setup.sh Release
   ```

6. Start the interactive environment:

   ```bash
   .venv/bin/mars-rover-play --debug
   ```

For later launches, only step 6 is required. Repeat steps 4 and 5 when replacing the generated
biome bank. If no LLM-generated biomes are needed, skip steps 3-5 and start immediately after the
initial build.

## One-command setup

Run from the `mars_rover` directory.

Windows PowerShell:

```powershell
.\scripts\setup.ps1                 # Auto: MSVC if active, otherwise GCC/MinGW
.\scripts\setup.ps1 -Toolchain GCC  # explicitly use installed MinGW GCC
.\scripts\setup.ps1 -BuildType Release
```

Linux:

```bash
bash scripts/setup.sh Debug
bash scripts/setup.sh Release
```

Both scripts create `.venv`, install all dependencies in editable mode, compile the C++ module,
and run `mars-rover-doctor`. No global Python packages are required.

## Run and verify

Windows:

```powershell
.\.venv\Scripts\mars-rover-play.exe --debug
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\mars-rover-doctor.exe
```

Linux:

```bash
.venv/bin/mars-rover-play --debug
.venv/bin/python -m pytest
.venv/bin/mars-rover-doctor
```

Controls: `Right/D` gas, `Left/A` reverse, `Down/S/Space` brake, `C/Shift` clutch,
`Z/X` shift down/up, `J/L` tilt, `E` ignition, `V` cycle RWD/FWD/AWD, `F` deploy or
stow the solar panel, `G` lidar scan, `R` restart the episode, `Q/Esc` quit.

## Generated biome bank

Copy `python/mars_rover_env/configs/biome_generator.yaml` to
`biome_generator.local.yaml`, place the API endpoint, model and token there, then generate a bank:

```bash
.venv/bin/python -m mars_rover_env.tools.generate_biomes --split train --count 12 --replace
```

The local configuration is ignored by Git. The player prints the compiled biome catalog before
opening the window. Re-run the setup command after generation so the native extension includes the
new bank.

The benchmark bank is split and frozen by construction:

- `train` and held-out `test` generation each enforce the six skill strata and deterministic
  behavioral fingerprint deduplication. Every environment slot samples one biome for the complete
  episode; test is held out, while train sampling intentionally includes the common anchors used
  for equating. Anchor-only evaluation remains a separate mode.
- Run the train difficulty gate before training the reference policy:

  ```bash
  .venv/bin/mars-rover-gate-biomes --split train --solve-min 0.05
  .venv/bin/mars-rover-audit-bank
  ```

- `mars-rover-train-robust-ppo` refuses to run without CUDA. It uses the native batched C++ env
  (`--num-envs 256` by default) and trains only on frozen train + common anchors, never on test. It
  never falls back to CPU. `mars-rover-train-robust-ppo --preflight` constructs the exact model and
  executes one CUDA batch without calling `learn()` or writing an artifact. Actual runs checkpoint
  every 250k transitions; `--resume` accepts only a checkpoint whose immutable run specification,
  train hash, environment version, config hash, seed and hyperparameters still match.
- After freezing that artifact, run the test gate with `--robust-model`; its model hash,
  `train_version`, environment contract, and train/test provenance are checked before any rollout.
  Rejected items can be removed with `mars-rover-generate-biomes --prune-rejected` and replaced in
  their missing strata. `mars-rover-audit-bank --require-reference --require-test-gate` is the final
  scoring preflight. Evaluate the same agents on
  anchors with `mars-rover-anchor-eval`, and fit the common-item transform with
  `mars-rover-equate-banks`. `mars-rover-score-bank` then evaluates random, scripted, PPO, or an
  imported `module:function` policy on every test item, applies the stored random/robust
  normalization and writes the score JSON. Callable policies are instantiated once per trial and
  retain recurrent state across its four episodes, so RL² adaptation is measured rather than reset
  away. For 2-3 regenerated banks, pass those files to
  `mars-rover-ranking-stability`. Before replacing a bank,
  `mars-rover-snapshot-bank` archives the exact C++ header, manifest state and environment config
  under its full hash; rerunning it verifies immutability instead of overwriting the snapshot.

## VS Code debugging

Open the repository root in VS Code, install the Python and C/C++ extensions, and run the
`Mars Rover: setup (Debug)` task once. Launch configurations are included for:

- Python manual player (`F5`, Python debugger)
- Native C++ extension on Windows (`cppvsdbg`)
- Native C++ extension on Linux (`gdb`)
- pytest

The native configurations start Python with `-m mars_rover_env.tools.play --debug`; breakpoints in
both `cpp/src/*.cpp` and `cpp/bindings/pybind_module.cpp` work with a Debug build. Re-run setup after
changing C++ code. Python edits are immediately visible because the package is installed editable.

## Manual build

The scripts are only convenience wrappers. The underlying portable build is:

```bash
python -m venv .venv
# activate the venv, then:
python -m pip install -U pip setuptools wheel pybind11
MARS_ROVER_BUILD_TYPE=Debug python -m pip install -e '.[dev]' --no-build-isolation
python -m mars_rover_env.tools.doctor
```

In PowerShell set the build type with `$env:MARS_ROVER_BUILD_TYPE = "Debug"` before `pip install`.
The standalone `cpp/CMakeLists.txt` is also portable and useful for IDE indexing, but the supported
Python package build goes through `pip` so the extension lands in the correct environment.
