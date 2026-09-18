# Rehearsal curriculum PPO result

Date: 2026-08-16. Bank: `sha256:025b0573176fba0eec6287959e9ecce026014bc0b5dfbefe6e1ba7d8bc24b930`.

## Method

The first experiment used a progressive chain/consequence curriculum:

1. 1 zone, 5% crash penalty;
2. 2 zones, 15%;
3. 4 zones, 35%;
4. 8 zones, 65%;
5. 14 zones and the unmodified target reward.

The stage ablation exposed catastrophic forgetting: the 8-zone checkpoint
covered 36.9 m on average on fresh full-chain trials, while the final 14-zone
checkpoint fell to 8.0 m.  The final method therefore starts from stage 4 and
interleaves 4/8/14-zone PPO blocks in a 25/25/50 ratio for six cycles, with the
original crash penalties and a reduced `1e-4` learning rate.  The last block in
every cycle is the exact 14-zone target environment.

No biome id, debug state, or mechanic parameters are exposed to the policy.

## Corrected evaluation

Two evaluator bugs were fixed before measuring the method:

- raw SB3 action indices were being passed to `MarsRoverEnv` instead of their
  public bit-mask macros;
- a 4096-step fine-tuning burst changed the chain after four episodes, despite
  the protocol claiming that the chain stayed fixed.

The reported protocol uses 30 fresh trials (seeds 50000-50029), target distance
60 m, at most six attempts, 1500 rollout steps per attempt, and 4096 PPO
fine-tuning steps between attempts.  Each learner is now pinned to one chain for
the complete trial.

| Model | Success | Mean attempts | Mean env steps | Mean best distance |
|---|---:|---:|---:|---:|
| Interleaved rehearsal curriculum | **20.0%** | **5.17** | **23,116** | **41.1 m** |
| Progressive curriculum final | 10.0% | 5.73 | 26,045 | 25.1 m |
| Robust PPO reference | 10.0% | 5.67 | 26,705 | 32.7 m |

Without test-time fine-tuning, rehearsal reaches 60 m on 16.7% of trials and
averages 38.5 m; the robust reference reaches 0% and averages 9.6 m on the same
seeds.  The robust checkpoint was trained against an older generated bank, so
the comparison measures transfer to the current bank rather than a same-bank
training-budget ablation.

## Artifacts

- Progressive run: `runs/curriculum-ppo_s2042_20260816T023402Z/`
- Rehearsal run: `runs/rehearsal-ppo_s2043_20260816T025630Z/`
- Final model: `runs/rehearsal-ppo_s2043_20260816T025630Z/model.zip`
- Full adaptation result: `runs/rehearsal-ppo_s2043_20260816T025630Z/adaptation_speed.json`
- Initial-policy ablation: `runs/rehearsal-ppo_s2043_20260816T025630Z/initial_ablation.json`

The result is positive but not solved: 80% of fresh chains still fail to reach
60 m.  The strongest next experiment is mixed-chain training inside every PPO
rollout (rather than switching environments between blocks), combined with a
KL anchor to the 8-zone locomotion teacher.
