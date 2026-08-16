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

Train and evaluate from the repository root:

```bash
.venv/bin/python codex/src/meta_ppo.py train --total-frames 20000000 \
  --num-envs 64 --output codex/artifacts/meta_ppo
.venv/bin/python codex/src/meta_ppo.py evaluate \
  --checkpoint codex/artifacts/meta_ppo/checkpoint.pt --biome-split 2 --trials 50
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
reset. Checkpoints, JSONL metrics, and summaries stay under `codex/artifacts/`.
