# Experiment log

## 2026-08-16 — integration and 200k-frame probe

All runs used the currently compiled biome bank. These are engineering probes, not
final benchmark numbers.

| run | frames | purpose | outcome |
|---|---:|---|---|
| `smoke` | 1,024 | end-to-end checkpoint/evaluation | passed; four-episode trial evaluated |
| `probe_200k` | 205,344 | uniform initial policy | collapsed into stuck terminations; roughly 0.2% progress |
| `probe_prior_200k` | 205,152 | trainable locomotion prior | learning remained stable; fresh-train progress roughly 1.2–1.5% |

Evaluation of `probe_prior_200k` on five fresh trials:

- train split: episode returns `-21.44, -21.57, -19.94, -13.31`; adaptation
  return delta `+3.17`; no finishes;
- held-out test split: episode returns `-28.87, -30.37, -29.88, -29.70`;
  adaptation return delta `-1.12`; no finishes.

The train result is too small and noisy to claim learned adaptation. The held-out
result shows that 200k frames are insufficient. A serious run should use at least
20M frames and multiple evaluation seeds; compare episode 1 against episodes 2–4,
not just aggregate return. The initial uniform-policy probe motivated the trainable
action prior now present in the implementation.

## 2026-08-16 — 2M-frame training run

`trained_2m` completed 2,008,768 physics frames. Evaluation used 50 fresh trials
(200 episodes) per split. Distances exclude the rover's 1 m spawn offset.

| policy | split | mean distance | median range | maximum | finishes |
|---|---|---:|---:|---:|---:|
| stochastic PPO | train | 18.10 m | 8.93–11.24 m | 90.03 m | 0/200 |
| stochastic PPO | test | 14.13 m | 6.47–9.18 m | 70.68 m | 0/200 |
| deterministic argmax | train | 6.31 m | 0.23–0.41 m | 57.67 m | 0/200 |
| deterministic argmax | test | 4.28 m | 0.29–0.36 m | 51.67 m | 0/200 |

The large stochastic/deterministic gap means the learned distribution is still
multimodal and has not converged to a reliable controller. There is no statistically
useful improvement across episodes 1–4 on held-out biomes, so this run learned basic
locomotion but not robust meta-adaptation.

## 2026-08-16 — v11 shared 120-second trial

The environment changed to unlimited attempts under one 7,200-tick clock. The agent
now retains memory until that clock expires, observes elapsed trial fraction, and is
evaluated by best attempt plus adaptation after its first attempt.

`v11_trained_5m` completed 5,057,472 physics frames. Reproducible stochastic
evaluation used 50 fresh trials per split:

| split | mean best | median best | p90 | maximum | attempts | later − first |
|---|---:|---:|---:|---:|---:|---:|
| train | 31.82 m | 28.59 m | 65.07 m | 103.42 m | 4.02 | +5.00 m |
| held-out test | 29.91 m | 23.59 m | 62.76 m | 92.34 m | 4.56 | +6.60 m |

The raw later-minus-first delta is positive, but a matched memory ablation shows it
is mostly the statistical benefit of taking the best of several attempts. Clearing
the GRU after every death scores 32.00 m on train and 31.04 m on test, versus 31.82 m
and 29.91 m with memory. Thus this 5M-frame run learned a better general controller,
but **did not yet learn useful cross-attempt adaptation**. No 800 m finish occurred.
