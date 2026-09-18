# Stack bank growth

## Statistics

- Batches processed: 10
- Candidates processed: 500
- Accepted in this run: 0
- Current bank: 11 train, 2 test

## Accepted stacks

| Name | Components | Split | Gap |
|---|---|---|---:|
| [existing_stack_01](#) | normal, wind | anchor | 0.000 |
| [existing_stack_02](#) | sand, wind | train | 0.000 |
| [existing_stack_03](#) | mud, crust, wind | train | 0.000 |
| [existing_stack_04](#) | ice, low_gravity | train | 0.000 |
| [existing_stack_05](#) | liquid, wind | test | 0.000 |
| [existing_stack_06](#) | sand, mud, wind, crust | test | 0.000 |
| [existing_stack_07](#) | ice, wind | train | 0.000 |
| [existing_stack_08](#) | sand, crust | train | 0.000 |
| [existing_stack_09](#) | wind, ice | train | 0.000 |
| [existing_stack_10](#) | crust, wind | train | 0.000 |
| [existing_stack_11](#) | crust, low_gravity | train | 0.000 |
| [existing_stack_12](#) | crust, sand | train | 0.000 |
| [existing_stack_13](#) | mud, wind | train | 0.000 |
| [existing_stack_14](#) | ice, wind, low_gravity | train | 0.000 |

## Rejections

- filter 1: 11 (2.2% of rejected candidates)
- filter 2: 362 (73.6% of rejected candidates)
- filter 3: 103 (20.9% of rejected candidates)
- filter 4: 13 (2.6% of rejected candidates)
- filter 5: 3 (0.6% of rejected candidates)

## Near-boundary rejections

| Candidate | Stage | Reason |
|---|---:|---|

## Mechanic coverage

| Mechanic | Train | Test |
|---|---:|---:|
| normal | 0 | 0 |
| sand | 3 | 1 |
| ice | 4 | 0 |
| mud | 2 | 1 |
| wind | 7 | 2 |
| low_gravity | 3 | 0 |
| crust | 5 | 1 |
| liquid | 0 | 1 |

## Open questions

- Candidate exhaustion reached: 500 candidates were screened and the final 50 had no full pass. The recurrent-vs-robust signed gate, not missing checkpoints, is the dominant limiting condition.
