# Meta-RL solution

Everything in this directory belongs to the Codex experiment; the environment is
left unchanged. `src/meta_ppo.py` implements recurrent PPO in the RL-squared style.

Why recurrent meta-RL: one trial is a fixed hidden biome with a shared 120-second
clock. Death starts another attempt but does not restore time. The GRU state is
retained across all respawns and cleared only when the clock starts a new trial. Its
input includes observation, previous action/reward, respawn flag, and elapsed trial
fraction. Evaluation reports the best attempt, total distance, number of attempts,
and improvement of later attempts over the first one.

The actor starts with a weak, fully trainable control prior favoring gas and gear
changes. A uniform random policy repeatedly toggles charging, ignition, and drive
mode and almost never discovers locomotion; spending the exploration budget on that
universal prerequisite obscures the actual meta-learning problem.

Train and evaluate from the repository root. This is the fast v12 configuration
used for the recorded run: 64 independently generated train tracks, one recurrent
PPO batch per update, and a frontier bonus aligned with maximum distance reached.

```bash
.venv/bin/python codex/src/meta_ppo.py train --total-frames 1500000 \
  --num-envs 64 --rollout-steps 128 --frame-skip 8 --hidden-size 128 \
  --epochs 1 --envs-per-batch 64 --frontier-bonus-scale 2 \
  --output codex/artifacts/v12_trained_1p5m_fast \
  --mlflow-tracking-uri codex/artifacts/mlflow.db \
  --mlflow-experiment mars-rover-v12 --run-name v12-train-1p5m-fast-seed-2026
.venv/bin/python codex/src/meta_ppo.py benchmark \
  --checkpoint codex/artifacts/v12_trained_1p5m_fast/checkpoint.pt \
  --biome-split 2 --trials-per-repeat 20 --repeats 5 \
  --base-seed 50000 --seed-stride 1000 \
  --output codex/artifacts/v12_trained_1p5m_fast/test_benchmark.json \
  --mlflow-tracking-uri codex/artifacts/mlflow.db \
  --mlflow-experiment mars-rover-v12 --run-name v12-test-eval-5x20
```

Omitting `--stochastic` is intentional for deployment: the learned distribution's
argmax controller is much stronger under the hard two-minute deadline. Add
`--stochastic` only to measure exploratory behavior during training analysis.

Every training update, repeated evaluation, aggregate metric, source snapshot,
checkpoint, and JSON report is logged to the local MLflow SQLite store. Open it with:

```bash
.venv/bin/mlflow ui \
  --backend-store-uri sqlite:////workspace/code/Marty/codex/artifacts/mlflow.db \
  --host 127.0.0.1 --port 5000
```

Use `--reset-memory-on-attempt` for the required memory ablation. A positive
later-versus-first distance is not enough by itself, because taking the maximum of
several random attempts creates the same apparent improvement without adaptation.

Quick integration check:

```bash
.venv/bin/python codex/src/meta_ppo.py train --total-frames 8192 \
  --num-envs 8 --rollout-steps 32 --output codex/artifacts/smoke
```

The PPO discount and GAE horizon are lengthened for credit across attempts. The
runner repeats continuous controls for a few physics ticks, but releases
edge-triggered gear/toggle/lidar controls after the first tick. It stops the whole
parallel batch early if any episode ends, preventing a transition from crossing a
reset. Checkpoints, JSONL metrics, summaries, the MLflow database, and MLflow
artifacts stay under `codex/artifacts/`.
