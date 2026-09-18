# Frozen mechanic-stack contract

This document records the contract already implemented by
`mars::FrozenMechanismStack`; it is not a second runtime format.

## Candidate

A candidate has a stable lowercase `name` and one ordered `components` list.
The list contains from one to four distinct values from:

`normal`, `sand`, `ice`, `mud`, `wind`, `low_gravity`, `crust`, `liquid`.

Order is meaningful. A candidate must not repeat a component, contain a
mechanism outside this list, or duplicate an accepted stack in the same order.
The generated C++ representation is a `FrozenMechanismStack` initializer:

```cpp
{{MechanicType::Ice, MechanicType::LowGravity,
  MechanicType::Wind, MechanicType::Normal}, 3, BiomeSplit::Train}
```

`Normal` is only padding after `count`; it is not inserted into the declared
component list unless it was explicitly proposed.

## Evaluation protocol

The bank-growth tool evaluates a candidate with its stack forced into every
mechanic region. This override is available only through the offline
`EnvConfig.evaluation_stack_*` fields and defaults to disabled in the released
environment.

Progress is measured against a fixed 250 m scoring window from spawn. The
window is deliberately independent of the physical terrain backing array, so
the 5%/95% thresholds remain meaningful on the long fixed-horizon course.

The model-free filters use the public 24-action macro vocabulary. The robust
and recurrent filters use the same public observation and action contract as
their shipped checkpoints; debug information is never passed to them.

## LLM response

One request must return a JSON object in this form, with at most the requested
number of entries:

```json
{"candidates": [{"name": "ice_low_gravity_wind", "components": ["ice", "low_gravity", "wind"]}]}
```

The request is only a proposal source. Every response is persisted verbatim in
the batch directory and nothing is accepted until all five local filters pass.
