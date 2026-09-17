#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p artifacts/training_logs

.venv/bin/python -m mars_rover_agents.suite_runner \
    --minutes 110 \
    --skip reptile \
    --allow-unfrozen-bank \
    --robust-envs 32 \
    --rl2-envs 12 \
    --render-trials 3 \
    > artifacts/training_logs/suite_rl2_robust.log 2>&1 &

echo "запущено, идентификатор процесса: $!"
echo "журнал: artifacts/training_logs/suite_rl2_robust.log"
