# Meta-RL solution

Everything in this directory belongs to the Codex experiment; the environment is
left unchanged. `src/meta_ppo.py` implements recurrent PPO in the RL-squared style.

Why recurrent meta-RL: one trial contains four episodes with the same hidden biome
layout. The GRU state is retained across those episode resets and cleared only at a
new trial. Its input includes observation, previous action/reward, an episode reset
flag, and the episode number. Evaluation reports each of episodes 1..4 separately;
improvement after episode 1 measures online adaptation rather than memorization.

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

Quick integration check:

```bash
.venv/bin/python codex/src/meta_ppo.py train --total-frames 8192 \
  --num-envs 8 --rollout-steps 32 --output codex/artifacts/smoke
```

The runner repeats continuous controls for a few physics ticks, but releases
edge-triggered gear/toggle/lidar controls after the first tick. It stops the whole
parallel batch early if any episode ends, preventing a transition from crossing a
reset. Checkpoints, JSONL metrics, and summaries stay under `codex/artifacts/`.
