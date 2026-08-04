# Cayley program tree non-finite diagnosis preregistration

## Question

The frozen one-step GPU smoke fit in device memory but returned non-finite
language loss, router auxiliary, and gradient norm.  Locate the first
non-finite activation without changing the candidate.

## Frozen candidate

- `experiments/cayley_program_tree.py` SHA-256:
  `166ce56d7e619d1d9a9ea35346d3b474d1f0dd05babbb735f9beff6f6725ff8b`
- `experiments/cayley_program_tree_lm_pilot.py` SHA-256:
  `ace68e73e32bd1efd712b1850ea8425e155edf5d4075afb00f6f84b083463235`
- Seed: 815.
- Data: the exact frozen training shard, starting at batch zero.
- No weights, initialization scales, route decisions, topology, precision
  rules, or model equations may be changed by this diagnosis.

## Cases

1. BF16 autocast, train mode, batch 1.
2. BF16 autocast, eval mode, batch 1.
3. FP32, eval mode, batch 1.
4. BF16 autocast, train mode, batch 8, matching the failed smoke.

All cases are forward-only.  For every decoder layer, record MLP input, MLP
output, decoder output, logits, and cross-entropy statistics.  Then replay the
exact tree equations on each captured MLP input and record every depth's
projected state, multiplicative operands, activation, delta, and accumulated
hidden state.

## Decision

- First non-finite in BF16 but not FP32: numerical instability under the
  serving/training precision contract.
- First non-finite in both: algebraic or initialization instability independent
  of BF16.
- All forward cases finite: the smoke failure lies in autograd/backward and a
  separate backward trace is required.

This experiment diagnoses the failed frozen candidate.  It cannot promote a
modified candidate or authorize the language pilot.
