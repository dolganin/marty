# Mars Rover agents

This sibling project contains comparable baselines and the benchmark harness.
The simulation remains in `../mars_rover`; agent code never receives an env
object, biome id, debug state, or mechanic parameters.

The only adaptation contract is:

```python
class Agent:
    def reset(self, trial_start: bool) -> None: ...
    def act(self, obs) -> int: ...
    def observe(self, reward, done, info) -> None: ...
```

`reset(False)` preserves trial memory. The common evaluator produces raw and
oracle-normalized episode returns, trial AUC, adaptation delta, reliability,
action entropy, JSONL trajectories, and a behavior GIF. Evaluator-only physics
diagnostics may be recorded in artifacts, but are never passed to an agent.

Training is GPU-only and refuses a CPU fallback. MLflow defaults to experiment
`Marty/mars_rover_agents` at `http://swagstation.netcraze.pro:4249` (the same
server is reachable inside the Docker network as `http://TalosViz:4249`). Every
run is tagged with bank/train/config/source hashes, seed, split, architecture,
GPU, and run identity.

```bash
../mars_rover/.venv/bin/pip install -e '.[train,dev]'
../mars_rover/.venv/bin/mars-agent-train-robust-ppo --preflight
../mars_rover/.venv/bin/mars-agent-train-robust-ppo --timesteps 1000000
../mars_rover/.venv/bin/mars-agent-train-rl2 --total-transitions 5000000
../mars_rover/.venv/bin/mars-agent-finalize-rl2 runs/<completed-rl2-run>
```

JSONL and summaries remain under `runs/<run>/evaluations/` for auditing and
resume. Behavior GIFs use local staging only: after upload they are retained
exclusively under `evaluations/step_*` in MLflow.
The RL2 finalizer uses five held-out trials by default for both anchors and test,
uploads the full summaries/JSONL/GIF records to the original training run, and
then removes its local GIF staging files.
