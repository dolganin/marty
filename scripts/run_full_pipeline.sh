#!/usr/bin/env bash









set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVIRONMENT_ROOT="$REPO_ROOT/environment"
AGENTS_ROOT="$REPO_ROOT/baselines"
RUNS_ROOT="${MARS_ROVER_RUNS_ROOT:-$REPO_ROOT/artifacts/runs}"
TAG="${1:-pipeline_$(date -u +%Y%m%dT%H%M%SZ)}"
PY="$REPO_ROOT/.venv/bin/python"

echo "=== [0/4] environment setup $(date -u) ==="
if [ ! -x "$PY" ]; then
  python3 -m venv "$REPO_ROOT/.venv"
fi
"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install --upgrade setuptools wheel pybind11 cmake numpy gymnasium PyYAML -q
( cd "$ENVIRONMENT_ROOT" && "$PY" -m pip install -e . --no-build-isolation -q )
( cd "$AGENTS_ROOT" && "$PY" -m pip install -e . --no-build-isolation -q )
"$PY" -c "import _mars_rover_cpp; print('native extension ok:', _mars_rover_cpp.__file__)"

echo "=== [1/4] refresh LLM-generated biome slice (train + test) $(date -u) ==="
( cd "$ENVIRONMENT_ROOT" && "$PY" -m mars_rover_env.tools.bootstrap_run --count 8 --split train --skip-rebuild )
( cd "$ENVIRONMENT_ROOT" && "$PY" -m mars_rover_env.tools.bootstrap_run --count 8 --split test )

WARMSTART="${MARS_ROVER_WARMSTART:-}"
INNER_TIMESTEPS=100000
OUTER_ITERS=1000
VANILLA_TIMESTEPS=100000000

declare -A JOBS=(
  [reptile_warm]="reptile warm"
  [reptile_cold]="reptile cold"
  [vanilla_warm]="vanilla warm"
  [vanilla_cold]="vanilla cold"
)

echo "=== [2/4] training candidates, ~100M env-steps each $(date -u) ==="
cd "$AGENTS_ROOT"
CANDIDATE_ARGS=()
for name in reptile_warm reptile_cold vanilla_warm vanilla_cold; do
  algo="${JOBS[$name]%% *}"
  warm="${JOBS[$name]##* }"
  out="$RUNS_ROOT/${TAG}_${name}"
  init_args=()
  if [ "$warm" = "warm" ] && [ -n "$WARMSTART" ]; then
    init_args=(--init-model "$WARMSTART")
  fi
  echo "--- $name -> $out $(date -u) ---"
  if [ "$algo" = "reptile" ]; then
    "$PY" -m mars_rover_agents.train_reptile_ppo \
      --outer-iterations "$OUTER_ITERS" --inner-timesteps "$INNER_TIMESTEPS" --num-envs 256 \
      --reptile-lr 0.3 --reptile-lr-final 0.03 --checkpoint-every 100 \
      --eval-episodes 4 --eval-max-steps 1200 --seed "$RANDOM" \
      "${init_args[@]}" \
      --output "$out" --run-name "${TAG}-${name}" 2>&1
  else
    "$PY" -m mars_rover_agents.train_reptile_ppo \
      --outer-iterations 1 --inner-timesteps "$VANILLA_TIMESTEPS" --num-envs 256 \
      --reptile-lr 1.0 --reptile-lr-final 1.0 --checkpoint-every 1 \
      --eval-episodes 4 --eval-max-steps 1200 --seed "$RANDOM" \
      "${init_args[@]}" \
      --output "$out" --run-name "${TAG}-${name}" 2>&1
  fi
  CANDIDATE_ARGS+=(--candidate "${name}=${out}/final/model.zip")
done

echo "=== [3/4] adaptation-speed evaluation on 40 fresh trials $(date -u) ==="
"$PY" -m mars_rover_agents.evaluate_adaptation_speed \
  "${CANDIDATE_ARGS[@]}" \
  --trials 40 --seed-begin "$RANDOM" --target-distance 60 --max-attempts 5 \
  --max-episode-steps 1200 --finetune-steps 8192 \
  --output "$RUNS_ROOT/${TAG}_adaptation_speed.json" 2>&1

echo "=== [4/4] done $(date -u); results: $RUNS_ROOT/${TAG}_adaptation_speed.json ==="
