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

## 2026-08-16 — v12 thermal/heater terrain, 120-second trials

The action vocabulary was extended from 15 to 19 macros so the policy can use the
new heater alone or together with throttle/tilt. Training used 64 independently
generated train tracks, 1,536,960 physics ticks, a GRU retained across deaths, and a
frontier reward for newly reached distance. All runs, weights, metrics, source
snapshots, and benchmark JSON files are in the `mars-rover-v12` MLflow experiment.

Repeated stochastic evaluation uses fresh tracks and reports the maximum distance
reached anywhere within each fixed 120-second trial:

| evaluation | repeats × tracks | mean best | median best | p90 | max observed | later − first |
|---|---:|---:|---:|---:|---:|---:|
| train split | 3 × 20 | 14.26 m | 9.11 m | 28.91 m | 56.57 m | +3.86 m |
| held-out test | 5 × 20 | 15.31 m | 11.81 m | 31.09 m | 60.46 m | +3.60 m |
| test, memory cleared at death | 5 × 20 | 15.38 m | 12.38 m | 31.01 m | 60.46 m | +3.72 m |
| **held-out test, deterministic deploy** | **5 × 20** | **81.84 m** | **78.79 m** | **141.79 m** | **204.85 m** | **−17.77 m** |

The memory ablation is indistinguishable from the recurrent result, so the positive
later-minus-first number is explained by repeated stochastic attempts rather than
learned online adaptation. This run is a valid general controller baseline for the
harder v12 environment, but it is not yet a successful meta-adaptive policy and did
not finish the 800 m course.

The high-entropy sampling policy is unsuitable for deployment: its action changes
cause avoidable crashes. Argmax inference reaches 81.84 m on average and 204.85 m at
best. As a diagnostic, constant throttle on 20 additional test tracks reached
72.94 m on average and 228.80 m at best, confirming that the recurrent checkpoint's
main current skill is a robust general driving prior, not learned online adaptation.
